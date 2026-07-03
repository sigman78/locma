from __future__ import annotations

import importlib.metadata

import typer
from rich.console import Console

from locma.cli.depot import depot_app
from locma.cli.render import GameRenderer
from locma.cli.textfmt import ascii_table
from locma.harness.ceiling_eval import _disjoint_eval_seeds
from locma.harness.draft_bench import draft_names, make_battle, make_draft, round_robin
from locma.harness.match import run_match
from locma.harness.tournament import run_tournament
from locma.harness.trace import (
    read_game_log,
    record_game,
    serialize_trace,
    trace_hash,
    write_game_log,
)
from locma.policies.registry import make_policy as registry_make_policy
from locma.stats.intervals import binomial_test, wilson_ci
from locma.stats.openskill_ratings import openskill_from_results, ordinal
from locma.stats.sprt import sprt as sprt_test

# rich_markup_mode=None keeps typer's help/usage-error output plain Click text
# (no framed panels); pretty_exceptions_enable=False does the same for tracebacks.
app = typer.Typer(
    help="Legends of Code & Magic 1.2 explore kit",
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
)
app.add_typer(depot_app, name="depot")
# rich is used ONLY by the interactive game renderer; CLI reporting is plain
# text (see locma/cli/textfmt.py for the rationale).
console = Console()


def _version() -> str:
    try:
        return importlib.metadata.version("locma")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


def make_policy(spec: str):
    try:
        return registry_make_policy(spec)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


@app.command()
def play(
    a: str,
    b: str,
    games: int = 100,
    seed: int = 0,
    render: bool = typer.Option(False, help="render each game turn-by-turn as played"),
    log: str = typer.Option(None, help="write a game-log JSONL (enables replay)"),
):
    """Run a mirrored match A vs B; optionally render and/or log each game."""
    if games < 1:
        raise typer.BadParameter("games must be >= 1")
    pa, pb = make_policy(a), make_policy(b)
    wins_a = total = 0
    if render or log:
        records: list[dict] = []
        renderer = GameRenderer(console) if render else None
        for k in range(games):
            s = seed + k
            # mirrored pair: game1 A=seat0, game2 A=seat1
            for a_seat in (0, 1):
                p0, p1 = (pa, pb) if a_seat == 0 else (pb, pa)
                if renderer:
                    print(f"---- game seed={s} a_seat={a_seat} ----")
                    res, trace = _recorded_with_render(p0, p1, s, renderer)
                else:
                    res, trace = record_game(p0, p1, seed=s)
                won = (res.winner == 0) if a_seat == 0 else (res.winner == 1)
                wins_a += int(won)
                total += 1
                if log:
                    records.append(
                        {
                            "format": 1,
                            "engine_version": _version(),
                            "policy_a": a,
                            "policy_b": b,
                            "seed": s,
                            "a_seat": a_seat,
                            "actions": serialize_trace(trace),
                            "winner": res.winner,
                            "turns": res.turns,
                            "hash": trace_hash(trace, res.winner, res.turns),
                        }
                    )
        if log:
            write_game_log(log, records)
    else:
        res = run_match(pa, pb, games=games, seed=seed)
        wins_a, total = res.wins_a, res.games

    lo, hi = wilson_ci(wins_a, total)
    p = binomial_test(wins_a, total, 0.5)
    print(
        f"{a} vs {b}  win rate A = {wins_a / total:.3f} "
        f"(95% CI {lo:.3f}-{hi:.3f}), p={p:.4g}, n={total}"
    )


def _recorded_with_render(p0, p1, seed, renderer):
    """Run a game, rendering each step and recording the trace."""
    from locma.core.engine import run_game  # noqa: PLC0415 — local to keep import graph flat
    from locma.harness.trace import Recorder  # noqa: PLC0415

    rec = Recorder()

    def on_step(seat, action, gs):
        rec.record(seat, action, gs)
        renderer.on_step(seat, action, gs)

    result = run_game(p0, p1, seed, on_step=on_step)
    return result, rec.trace


@app.command()
def tournament(
    names: list[str],
    games: int = 50,
    seed: int = 0,
    reference: str = "random",
    matrix: bool = typer.Option(False, help="print the pair-score matrix"),
):
    """Round-robin tournament with openskill (primary) and Elo ratings."""
    pols = [make_policy(n) for n in names]
    res = run_tournament(pols, games=games, seed=seed, reference=reference)

    # openskill from the same win matrix (reconstruct per-game results from win rates)
    pairs: list[tuple[str, str, float]] = []
    seen: set[frozenset[str]] = set()
    for (x, y), rate in res.win_matrix.items():
        key = frozenset((x, y))
        if key in seen:
            continue
        seen.add(key)
        wins_x = round(rate * games * 2)
        for _ in range(wins_x):
            pairs.append((x, y, 1.0))
        for _ in range(games * 2 - wins_x):
            pairs.append((x, y, 0.0))
    osk = openskill_from_results(pairs)

    order = sorted(res.ratings, key=lambda k: -ordinal(*osk.get(k, (25.0, 8.333))))
    rows = []
    for n in order:
        mu, sigma = osk.get(n, (25.0, 8.333))
        rows.append(
            (
                n,
                f"{ordinal(mu, sigma):.2f}",
                f"{res.ratings[n]:.0f}",
                f"{res.p_vs_reference.get(n, float('nan')):.4g}",
            )
        )
    print(ascii_table(("policy", "openskill", "elo", "p vs ref"), rows, align="lrrr"))

    if matrix:
        cells = [
            tuple(
                [row]
                + [
                    "--" if row == col else f"{res.win_matrix.get((row, col), float('nan')):.2f}"
                    for col in names
                ]
            )
            for row in names
        ]
        print()
        print(
            ascii_table(
                ("", *names),
                cells,
                align="l" + "r" * len(names),
                title="Pair-score matrix (row win rate vs column)",
            )
        )


@app.command("noise-floor")
def noise_floor(a: str, games: int = 200, seed: int = 0):
    """Play policy A against an independent copy of itself: the luck baseline."""
    if games < 1:
        raise typer.BadParameter("games must be >= 1")
    res = run_match(make_policy(a), make_policy(a), games=games, seed=seed)
    lo, hi = wilson_ci(res.wins_a, res.games)
    half = (hi - lo) / 2
    print(
        f"{a} vs itself  win rate = {res.win_rate_a:.3f} "
        f"(95% CI {lo:.3f}-{hi:.3f}), n={res.games}\n"
        f"resolution limit: +/-{half:.3f}  "
        f"(edges smaller than this are indistinguishable from luck)"
    )


@app.command("draft-bench")
def draft_bench_cmd(
    drafts: list[str] = typer.Argument(  # noqa: B008
        None, help="draft names (default: all built-in drafts)"
    ),
    battle: str = typer.Option(
        "ground",
        help="battle policy piloting BOTH seats: ground, greedy, scripted, "
        "azlite:100, dmcts:K,I, ppo:PATH",
    ),
    games: int = typer.Option(100, help="mirrored game pairs per draft pair (2x this total)"),
    seed: int = 0,
    workers: int = typer.Option(
        1, help="process-pool workers for the pair grid (0 = all CPUs minus one)"
    ),
):
    """Rank draft (deck-building) policies in isolation.

    Both seats are piloted by the SAME ``--battle`` policy and differ only in their
    draft, so the win-rate edge is pure deck quality (the draft deals both seats
    identical offers on a fixed seed; a self-duel is exactly 0.500). A ``+rndK``
    suffix (e.g. ``balanced+rnd4``) makes K of that draft's 30 picks uniformly
    random. Prints a ranking by average win rate vs the field and the pair-score
    matrix. See docs/experiments.md.
    """
    if games < 1:
        raise typer.BadParameter("games must be >= 1")
    names = list(drafts) if drafts else draft_names()
    if len(names) < 2:
        raise typer.BadParameter("need at least 2 drafts to rank")
    for n in names:
        try:
            make_draft(n)  # validate (incl. +rndK specs) up front for a clean error
        except ValueError as e:
            raise typer.BadParameter(str(e)) from e
    try:
        make_battle(battle)  # validate the battle spec up front for a clean error
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    from locma.harness.parallel import resolve_workers  # noqa: PLC0415

    s = round_robin(names, battle=battle, games=games, seed=seed, workers=resolve_workers(workers))

    # Resolution limit: Wilson half-width of a single cell at this n.
    lo, hi = wilson_ci(games, 2 * games)
    half = (hi - lo) / 2
    print(f"battle={battle}  n={2 * games}/pair  resolution +/-{half:.3f}")

    rank_rows = [
        (n, f"{s.avg_win_rate[n]:.3f}") for n in sorted(names, key=lambda k: -s.avg_win_rate[k])
    ]
    print(
        ascii_table(
            ("draft", "avg"),
            rank_rows,
            align="lr",
            title=f"Draft ranking (avg win rate vs field, battle={battle})",
        )
    )

    cells = [
        tuple([row] + ["--" if row == col else f"{s.win_matrix[(row, col)]:.2f}" for col in names])
        for row in names
    ]
    print()
    print(
        ascii_table(
            ("", *names),
            cells,
            align="l" + "r" * len(names),
            title="Pair-score matrix (row win rate vs column)",
        )
    )


@app.command()
def sprt(
    x: str,
    vs: str = "random",
    p0: float = 0.5,
    p1: float = 0.6,
    max_games: int = 1000,
    batch: int = 20,
    seed: int = 0,
):
    """Sequential probability ratio test; stops as soon as evidence decides."""
    if max_games < 1:
        raise typer.BadParameter("max-games must be >= 1")
    px, py = make_policy(x), make_policy(vs)
    wins = n = k = 0
    r = None
    while n < max_games:
        res = run_match(px, py, games=batch, seed=seed + k)
        k += batch
        wins += res.wins_a
        n += res.games
        r = sprt_test(wins, n, p0, p1)
        if r.decision != "continue":
            break
    lo, hi = wilson_ci(wins, n)
    print(f"verdict: {r.decision}  winrate={wins / n:.3f} (CI {lo:.3f}-{hi:.3f}), n={n}")


@app.command()
def replay(
    file: str,
    assert_hash: bool = typer.Option(
        False, "--assert-hash", help="fail if recomputed hash differs"
    ),
    render: bool = typer.Option(False, help="render each replayed game"),
):
    """Re-simulate a logged game and (optionally) assert byte-identical hash."""
    rows = read_game_log(file)
    mismatches = 0
    for i, row in enumerate(rows):
        pa, pb = make_policy(row["policy_a"]), make_policy(row["policy_b"])
        p0, p1 = (pa, pb) if row["a_seat"] == 0 else (pb, pa)
        if render:
            renderer = GameRenderer(console)
            print(f"---- replay game {i} seed={row['seed']} ----")
            result, trace = _recorded_with_render(p0, p1, row["seed"], renderer)
        else:
            result, trace = record_game(p0, p1, seed=row["seed"])
        h = trace_hash(trace, result.winner, result.turns)
        ok = h == row.get("hash")
        if not ok:
            mismatches += 1
            print(f"game {i}: hash MISMATCH stored={row.get('hash')} got={h}")
        else:
            print(f"game {i}: ok ({h})")
    if assert_hash and mismatches:
        raise typer.Exit(code=1)


@app.command()
def train(
    steps: int = 50_000,
    out: str = "model.zip",
    opponent: str = "random",
    seed: int = 0,
    n_envs: int = typer.Option(1, help="parallel envs (CPU speedup)"),
    checkpoints: str = typer.Option(
        None, help="comma-separated step marks to save checkpoints at (one trajectory)"
    ),
    ent_coef: float = typer.Option(0.02, help="entropy coefficient for MaskablePPO"),
    both_seat: bool = typer.Option(True, help="train as both first AND second player"),
    obs_mode: str = typer.Option(
        "flat", help="obs encoding: 'flat' (default), 'token', or 'token-v1' (tokenized)"
    ),
    learning_rate: float = typer.Option(3e-4, help="PPO learning rate"),
    target_kl: float | None = typer.Option(None, help="PPO target KL early-stop (None = off)"),
    n_steps: int = typer.Option(2048, help="PPO rollout length (per env) before each update"),
    batch_size: int = typer.Option(64, help="PPO minibatch size"),
    n_epochs: int = typer.Option(10, help="PPO epochs per update"),
    gamma: float = typer.Option(0.99, help="PPO discount factor"),
    gae_lambda: float = typer.Option(0.95, help="PPO GAE lambda"),
    clip_range: float = typer.Option(0.2, help="PPO clip range"),
    vf_coef: float = typer.Option(0.5, help="PPO value function loss coefficient"),
    max_grad_norm: float = typer.Option(0.5, help="PPO max gradient norm for clipping"),
    device: str = typer.Option("auto", help="torch device: 'auto', 'cpu', or 'cuda'"),
    tensorboard_log: str | None = typer.Option(None, help="tensorboard log directory"),
    draft_noise: int = typer.Option(
        0, help="make K of each deck's 30 draft picks uniformly random (deck diversity)"
    ),
):
    """Train a MaskablePPO agent on the battle env (requires the [ml] extra)."""
    if steps < 1:
        raise typer.BadParameter("steps must be >= 1")
    if n_envs < 1:
        raise typer.BadParameter("n_envs must be >= 1")
    if not 0 <= draft_noise <= 30:
        raise typer.BadParameter("draft-noise must be in [0, 30]")
    if obs_mode not in ("flat", "token", "token-v1"):
        raise typer.BadParameter("obs_mode must be 'flat', 'token', or 'token-v1'")
    marks = None
    if checkpoints:
        try:
            marks = [int(x) for x in checkpoints.split(",")]
        except ValueError as e:
            raise typer.BadParameter("checkpoints must be comma-separated integers") from e
    # `opponent` is passed as a spec string: the trainer rebuilds it per env.
    make_policy(opponent)  # validate the spec up front for a friendly error
    try:
        from locma.envs.training import train_agent  # noqa: PLC0415 — optional [ml] dep

        saved = train_agent(
            opponent,
            steps=steps,
            out=out,
            seed=seed,
            n_envs=n_envs,
            checkpoints=marks,
            ent_coef=ent_coef,
            both_seat=both_seat,
            obs_mode=obs_mode,
            learning_rate=learning_rate,
            target_kl=target_kl,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            device=device,
            tensorboard_log=tensorboard_log,
            draft_noise=draft_noise,
        )
    except ImportError as e:
        raise typer.BadParameter("training requires the [ml] extra: uv sync --extra ml") from e
    print(f"saved {saved}")


@app.command("train-zoo")
def train_zoo_cmd(
    steps_per_opponent: int = typer.Option(200_000, help="timesteps per opponent phase"),
    out: str = "model.zip",
    seed: int = 0,
    ent_coef: float = typer.Option(0.02, help="entropy coefficient for MaskablePPO"),
    both_seat: bool = typer.Option(True, help="train as both first AND second player"),
    obs_mode: str = typer.Option(
        "flat", help="obs encoding: 'flat' (default), 'token', or 'token-v1' (tokenized)"
    ),
    learning_rate: float = typer.Option(3e-4, help="PPO learning rate"),
    target_kl: float | None = typer.Option(None, help="PPO target KL early-stop (None = off)"),
    n_steps: int = typer.Option(2048, help="PPO rollout length (per env) before each update"),
    batch_size: int = typer.Option(64, help="PPO minibatch size"),
    n_epochs: int = typer.Option(10, help="PPO epochs per update"),
    gamma: float = typer.Option(0.99, help="PPO discount factor"),
    gae_lambda: float = typer.Option(0.95, help="PPO GAE lambda"),
    clip_range: float = typer.Option(0.2, help="PPO clip range"),
    vf_coef: float = typer.Option(0.5, help="PPO value function loss coefficient"),
    max_grad_norm: float = typer.Option(0.5, help="PPO max gradient norm for clipping"),
    device: str = typer.Option("auto", help="torch device: 'auto', 'cpu', or 'cuda'"),
    n_envs: int = typer.Option(1, help="parallel envs per opponent phase (CPU speedup)"),
    tensorboard_log: str | None = typer.Option(None, help="tensorboard log directory"),
    draft_noise: int = typer.Option(
        0, help="make K of each deck's 30 draft picks uniformly random (deck diversity)"
    ),
):
    """Train one MaskablePPO agent back-to-back against the code-declared opponent
    zoo (a curriculum; see ZOO_OPPONENTS in locma/envs/training.py). Requires the
    [ml] extra."""
    if steps_per_opponent < 1:
        raise typer.BadParameter("steps-per-opponent must be >= 1")
    if n_envs < 1:
        raise typer.BadParameter("n_envs must be >= 1")
    if not 0 <= draft_noise <= 30:
        raise typer.BadParameter("draft-noise must be in [0, 30]")
    if obs_mode not in ("flat", "token", "token-v1"):
        raise typer.BadParameter("obs_mode must be 'flat', 'token', or 'token-v1'")
    from locma.envs.training import ZOO_OPPONENTS  # noqa: PLC0415 — constant, no [ml] needed

    for o in ZOO_OPPONENTS:
        make_policy(o)  # validate each declared opponent spec up front
    print(f"zoo curriculum: {' -> '.join(ZOO_OPPONENTS)}")
    try:
        from locma.envs.training import train_zoo  # noqa: PLC0415 — optional [ml] dep

        saved = train_zoo(
            steps_per_opponent=steps_per_opponent,
            out=out,
            seed=seed,
            ent_coef=ent_coef,
            both_seat=both_seat,
            obs_mode=obs_mode,
            learning_rate=learning_rate,
            target_kl=target_kl,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            device=device,
            n_envs=n_envs,
            tensorboard_log=tensorboard_log,
            draft_noise=draft_noise,
        )
    except ImportError as e:
        raise typer.BadParameter("training requires the [ml] extra: uv sync --extra ml") from e
    print(f"saved {saved}")


@app.command("record-practicum")
def record_practicum_cmd(
    teacher: str = typer.Option("mcts:100", help="teacher policy spec to clone"),
    opponents: list[str] = typer.Option(  # noqa: B008
        ["random", "scripted", "greedy", "max-guard", "max-attack"],
        help="opponent specs the teacher plays against",
    ),
    games: int = typer.Option(200, help="games per opponent (each played in both seats)"),
    out: str = typer.Option("practicum.npz", help="output practicum .npz path"),
    seed: int = 0,
    obs_mode: str = typer.Option(
        "flat", help="observation encoding: 'flat' (default), 'token', or 'token-v1'"
    ),
):
    """Record a practicum of teacher battle decisions for distillation."""
    if games < 1:
        raise typer.BadParameter("games must be >= 1")
    if obs_mode not in ("flat", "token", "token-v1"):
        raise typer.BadParameter("obs_mode must be 'flat', 'token', or 'token-v1'")
    make_policy(teacher)  # validate up front for a friendly error
    for o in opponents:
        make_policy(o)
    from locma.envs.practicum import record_practicum  # noqa: PLC0415

    manifest = record_practicum(
        teacher=teacher,
        opponents=tuple(opponents),
        games=games,
        out=out,
        seed=seed,
        obs_mode=obs_mode,
    )
    print(
        f"recorded {manifest['n_examples']} examples "
        f"({manifest['n_dropped_overflow']} dropped) -> {out}"
    )


@app.command()
def distill(
    data: str = typer.Option("practicum.npz", help="practicum .npz to clone"),
    out: str = typer.Option("model.zip", help="output MaskablePPO model path"),
    epochs: int = 10,
    batch: int = 256,
    lr: float = 3e-4,
    val_frac: float = typer.Option(0.1, help="fraction of games held out for agreement"),
    seed: int = 0,
    obs_mode: str | None = typer.Option(
        None, help="obs encoding; defaults to the practicum's manifest mode"
    ),
):
    """Behavior-clone a practicum into a MaskablePPO model.zip (requires the [ml] extra)."""
    if epochs < 1:
        raise typer.BadParameter("epochs must be >= 1")
    if not 0.0 <= val_frac < 1.0:
        raise typer.BadParameter("val-frac must be in [0, 1)")
    if obs_mode is not None and obs_mode not in ("flat", "token", "token-v1"):
        raise typer.BadParameter("obs_mode must be 'flat', 'token', or 'token-v1'")
    try:
        from locma.envs.distill import behavior_clone  # noqa: PLC0415 — optional [ml] dep

        info = behavior_clone(
            data=data,
            out=out,
            epochs=epochs,
            batch=batch,
            lr=lr,
            val_frac=val_frac,
            seed=seed,
            obs_mode=obs_mode,
        )
    except ImportError as e:
        raise typer.BadParameter("distill requires the [ml] extra: uv sync --extra ml") from e
    print(
        f"saved {info['out']}  val_agreement={info['val_agreement']:.3f} "
        f"(train={info['n_train']}, val={info['n_val']})"
    )


@app.command("record-selfplay")
def record_selfplay_cmd(
    oracle_path: str = typer.Option(..., help="path to the warm/oracle .zip model"),
    out: str = typer.Option("selfplay.npz", help="output .npz path"),
    self_play_games: int = typer.Option(240, help="number of self-play games"),
    baseline_games: int = typer.Option(100, help="number of baseline games vs scripted opponents"),
    k: int = typer.Option(6, help="determinizations per decision (PUCT worlds)"),
    i: int = typer.Option(40, help="PUCT iterations per world"),
    c_puct: float = typer.Option(1.5, help="PUCT exploration constant"),
    eps: float = typer.Option(0.25, help="Dirichlet noise mixing coefficient"),
    alpha: float = typer.Option(0.3, help="Dirichlet concentration parameter"),
    temp_moves: int = typer.Option(10, help="plies with temperature sampling"),
    seed: int = typer.Option(0, help="RNG seed"),
):
    """Generate AlphaZero self-play data (requires the [ml] extra)."""
    if self_play_games < 0:
        raise typer.BadParameter("self-play-games must be >= 0")
    if baseline_games < 0:
        raise typer.BadParameter("baseline-games must be >= 0")
    if self_play_games + baseline_games < 1:
        raise typer.BadParameter("self-play-games + baseline-games must be >= 1")
    if k < 1:
        raise typer.BadParameter("k must be >= 1")
    if i < 1:
        raise typer.BadParameter("i must be >= 1")
    try:
        import locma.envs.selfplay as _selfplay  # noqa: PLC0415 — optional [ml] dep

        manifest = _selfplay.record_selfplay(
            oracle_path=oracle_path,
            out=out,
            self_play_games=self_play_games,
            baseline_games=baseline_games,
            K=k,
            I=i,
            c_puct=c_puct,
            eps=eps,
            alpha=alpha,
            temp_moves=temp_moves,
            seed=seed,
        )
    except ImportError as e:
        raise typer.BadParameter(
            "record-selfplay requires the [ml] extra: uv sync --extra ml"
        ) from e
    print(
        f"recorded {manifest['n_examples']} examples "
        f"({manifest['failed_games']} failed games) -> {out}"
    )


@app.command("az-train")
def az_train_cmd(
    data: list[str] = typer.Option(  # noqa: B008
        [], help="self-play .npz path(s) — repeatable for the sliding window"
    ),
    warm_start: str = typer.Option(..., help="path to warm-start token MaskablePPO .zip"),
    out: str = typer.Option("az.zip", help="output model .zip path"),
    epochs: int = typer.Option(10, help="training epochs"),
    batch: int = typer.Option(256, help="mini-batch size"),
    lr: float = typer.Option(1e-4, help="Adam learning rate"),
    c_v: float = typer.Option(0.5, help="value loss coefficient"),
    val_frac: float = typer.Option(0.1, help="fraction of games held out for validation"),
    seed: int = typer.Option(0, help="RNG seed"),
):
    """Fine-tune an AlphaZero net on self-play data (requires the [ml] extra)."""
    if not data:
        raise typer.BadParameter("--data must be provided at least once")
    if epochs < 1:
        raise typer.BadParameter("epochs must be >= 1")
    if batch < 1:
        raise typer.BadParameter("batch must be >= 1")
    if not 0.0 <= val_frac < 1.0:
        raise typer.BadParameter("val-frac must be in [0, 1)")
    try:
        import locma.envs.az_train as _az_train  # noqa: PLC0415 — optional [ml] dep

        info = _az_train.az_train(
            data=data,
            warm_start=warm_start,
            out=out,
            epochs=epochs,
            batch=batch,
            lr=lr,
            c_v=c_v,
            val_frac=val_frac,
            seed=seed,
        )
    except ImportError as e:
        raise typer.BadParameter("az-train requires the [ml] extra: uv sync --extra ml") from e
    print(
        f"saved {info['out']}  "
        f"val_policy_ce={info['val_policy_ce']:.4f} "
        f"val_value_mse={info['val_value_mse']:.4f} "
        f"(train={info['n_train']}, val={info['n_val']})"
    )


@app.command("az-selfplay")
def az_selfplay_cmd(
    warm_start: str = typer.Option("runs/selfplay-r2.zip", help="initial oracle net path"),
    prefix: str = typer.Option("runs/az", help="output path prefix"),
    iterations: int = typer.Option(4, help="maximum self-play/train/eval rounds"),
    window: int = typer.Option(2, help="rolling window of datasets passed to az-train"),
    base_seed: int = typer.Option(0, help="base RNG seed"),
    self_play_games: int = typer.Option(240, help="self-play games per iteration"),
    baseline_games: int = typer.Option(100, help="baseline games per iteration"),
    k_gen: int = typer.Option(6, help="PUCT determinizations for generation"),
    i_gen: int = typer.Option(40, help="PUCT iterations per world for generation"),
    c_puct: float = typer.Option(1.5, help="PUCT exploration constant"),
    eps: float = typer.Option(0.25, help="Dirichlet noise mixing coefficient"),
    alpha: float = typer.Option(0.3, help="Dirichlet concentration parameter"),
    temp_moves: int = typer.Option(10, help="temperature plies for generation"),
    epochs: int = typer.Option(10, help="training epochs"),
    batch: int = typer.Option(256, help="mini-batch size"),
    lr: float = typer.Option(1e-4, help="Adam learning rate"),
    c_v: float = typer.Option(0.5, help="value loss coefficient"),
    k_eval: int = typer.Option(8, help="PUCT determinizations for evaluation"),
    i_eval: int = typer.Option(40, help="PUCT iterations per world for evaluation"),
    games_per_opp: int = typer.Option(20, help="games per opponent in avg_hard3"),
    h2h_games: int = typer.Option(40, help="games for head-to-head evaluation"),
    h2h_thresh: float = typer.Option(0.53, help="minimum h2h win-rate to adopt candidate"),
    hard3_eps: float = typer.Option(0.02, help="max score regression allowed for adoption"),
    max_rejects: int = typer.Option(2, help="early-stop after this many consecutive rejects"),
):
    """Run the AlphaZero iterated self-play loop (requires the [ml] extra)."""
    if iterations < 1:
        raise typer.BadParameter("iterations must be >= 1")
    if window < 1:
        raise typer.BadParameter("window must be >= 1")
    try:
        import locma.envs.azloop as _azloop  # noqa: PLC0415 — optional [ml] dep

        res = _azloop.az_selfplay(
            warm_start=warm_start,
            prefix=prefix,
            iterations=iterations,
            window=window,
            base_seed=base_seed,
            self_play_games=self_play_games,
            baseline_games=baseline_games,
            K_gen=k_gen,
            I_gen=i_gen,
            c_puct=c_puct,
            eps=eps,
            alpha=alpha,
            temp_moves=temp_moves,
            epochs=epochs,
            batch=batch,
            lr=lr,
            c_v=c_v,
            K_eval=k_eval,
            I_eval=i_eval,
            games_per_opp=games_per_opp,
            h2h_games=h2h_games,
            h2h_thresh=h2h_thresh,
            hard3_eps=hard3_eps,
            max_rejects=max_rejects,
        )
    except ImportError as e:
        raise typer.BadParameter("az-selfplay requires the [ml] extra: uv sync --extra ml") from e
    print(
        f"best_net={res['best_net']} "
        f"best_score={res['best_score']:.3f} "
        f"final_hard3={res['final_hard3']:.3f} "
        f"final_h2h={res['final_h2h']:.3f}"
    )


# _disjoint_eval_seeds moved to locma/harness/ceiling_eval.py (shared with the
# web panel's experiments); imported at the top and re-exported for tests.


@app.command("ceiling-eval")
def ceiling_eval_cmd(
    candidates: str = typer.Option(
        ...,
        help="comma-separated candidates: model paths, depot: refs, or policy "
        "specs (e.g. vbeam:depot:b0/b0_s0.zip)",
    ),
    baselines: str = typer.Option(
        ..., help="comma-separated B0 model paths or depot: refs (e.g. depot:b0/b0_s0.zip)"
    ),
    seeds: int = typer.Option(40, help="number of held-out eval seeds (from 1_000_000)"),
    games_per_seed: int = typer.Option(25, help="paired games per opponent per seed"),
    threshold: float = typer.Option(0.03, help="avg-hard3 lift required for 'headroom'"),
    workers: int = typer.Option(
        1, help="process-pool workers over the (model, seed) grid (0 = all CPUs minus one)"
    ),
):
    """Rigorous paired-difference verdict for the PPO ceiling study (requires [ml])."""
    try:
        from locma.harness.ceiling_eval import run_verdict  # noqa: PLC0415
    except ImportError as e:
        raise typer.BadParameter("ceiling-eval requires the [ml] extra") from e
    from locma.harness.parallel import resolve_workers  # noqa: PLC0415

    seed_list = _disjoint_eval_seeds(seeds, games_per_seed)
    out = run_verdict(
        candidates.split(","),
        baselines.split(","),
        seeds=seed_list,
        games_per_seed=games_per_seed,
        threshold=threshold,
        workers=resolve_workers(workers),
    )
    print(
        f"cand={out['cand_avg']:.3f}  B0={out['b0_avg']:.3f}  "
        f"delta={out['mean_delta']:+.3f}  95% CI [{out['ci_lo']:+.3f}, {out['ci_hi']:+.3f}]"
    )
    print(f"VERDICT: {out['verdict']}")


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    replay_dir: str = "replays",
    asset_dir: str = "locma/data/assets",
    gamelog_dir: str = ".",
    presets_dir: str = typer.Option("experiments/presets", help="experiment preset JSON files"),
    results_dir: str = typer.Option("runs/experiments", help="finished experiment job results"),
    workers: int = typer.Option(
        0, help="process-pool workers for experiment jobs (0 = all CPUs minus one)"
    ),
):
    """Run the local web panel: experiments, depot, replays, play-testing
    (requires the [server] extra)."""
    try:
        import uvicorn  # noqa: PLC0415 — optional [server] dep

        from locma.server.app import create_app  # noqa: PLC0415
    except ImportError as e:
        raise typer.BadParameter("serve requires the [server] extra: uv sync --extra server") from e
    app_ = create_app(
        replay_dir=replay_dir,
        asset_dir=asset_dir,
        gamelog_dir=gamelog_dir,
        presets_dir=presets_dir,
        results_dir=results_dir,
        workers=workers,
    )
    print(f"serving on http://{host}:{port}")
    uvicorn.run(app_, host=host, port=port)


@app.command("fetch-cards")
def fetch_cards_cmd():
    from locma.data.fetch import fetch_cards  # noqa: PLC0415 — lazy import

    path = fetch_cards()
    print(f"cards at {path}")


@app.command("fetch-art")
def fetch_art_cmd(
    force: bool = typer.Option(False, "--force", help="re-download even if cached"),
):
    """Download card portrait art into the local (gitignored) cache. Opt-in."""
    from importlib import resources  # noqa: PLC0415

    from locma.data.fetch import fetch_art  # noqa: PLC0415 — lazy import

    n = fetch_art(force=force)
    cache_dir = resources.files("locma.data").joinpath("assets")
    print(f"fetched {n} art assets into {cache_dir}")
    print(
        "Card art is downloaded for local use only; "
        "seek permission from the authors before redistribution."
    )
