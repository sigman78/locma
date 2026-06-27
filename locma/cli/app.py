from __future__ import annotations

import importlib.metadata

import typer
from rich.console import Console
from rich.table import Table

from locma.cli.render import GameRenderer
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

app = typer.Typer(help="Legends of Code & Magic 1.2 explore kit")
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
                    console.rule(f"game seed={s} a_seat={a_seat}")
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
    console.print(
        f"[bold]{a}[/] vs [bold]{b}[/]  win rate A = {wins_a / total:.3f} "
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

    t = Table(title="Ratings", box=None)
    t.add_column("policy")
    t.add_column("openskill", justify="right")
    t.add_column("elo", justify="right")
    t.add_column("p vs ref", justify="right")
    order = sorted(res.ratings, key=lambda k: -ordinal(*osk.get(k, (25.0, 8.333))))
    for n in order:
        mu, sigma = osk.get(n, (25.0, 8.333))
        t.add_row(
            n,
            f"{ordinal(mu, sigma):.2f}",
            f"{res.ratings[n]:.0f}",
            f"{res.p_vs_reference.get(n, float('nan')):.4g}",
        )
    console.print(t)

    if matrix:
        m = Table(title="Pair-score matrix (row win rate vs column)", box=None)
        m.add_column("")
        for n in names:
            m.add_column(n, justify="right")
        for row in names:
            cells = [row]
            for col in names:
                if row == col:
                    cells.append("--")
                else:
                    cells.append(f"{res.win_matrix.get((row, col), float('nan')):.2f}")
            m.add_row(*cells)
        console.print(m)


@app.command("noise-floor")
def noise_floor(a: str, games: int = 200, seed: int = 0):
    """Play policy A against an independent copy of itself: the luck baseline."""
    if games < 1:
        raise typer.BadParameter("games must be >= 1")
    res = run_match(make_policy(a), make_policy(a), games=games, seed=seed)
    lo, hi = wilson_ci(res.wins_a, res.games)
    half = (hi - lo) / 2
    console.print(
        f"[bold]{a}[/] vs itself  win rate = {res.win_rate_a:.3f} "
        f"(95% CI {lo:.3f}-{hi:.3f}), n={res.games}\n"
        f"resolution limit: +/-{half:.3f}  "
        f"[dim](edges smaller than this are indistinguishable from luck)[/]"
    )


@app.command("action-stats")
def action_stats(policy: str, opponent: str = "mixed", games: int = 100, seed: int = 0):
    """Print tactical action histograms for policy A in mirrored matches."""
    if games < 1:
        raise typer.BadParameter("games must be >= 1")
    from locma.harness.action_stats import policy_action_stats  # noqa: PLC0415

    make_policy(policy)
    make_policy(opponent)
    stats = policy_action_stats(policy, opponent, games=games, seed=seed)
    rates = stats.as_rates()
    t = Table(title=f"Action stats: {policy} vs {opponent}", box=None)
    t.add_column("metric")
    t.add_column("value", justify="right")
    for k, v in rates.items():
        t.add_row(k, f"{int(v)}" if k == "decisions" else f"{v:.3f}")
    console.print(t)


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
    console.print(
        f"verdict: [bold]{r.decision}[/]  winrate={wins / n:.3f} (CI {lo:.3f}-{hi:.3f}), n={n}"
    )


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
            console.rule(f"replay game {i} seed={row['seed']}")
            result, trace = _recorded_with_render(p0, p1, row["seed"], renderer)
        else:
            result, trace = record_game(p0, p1, seed=row["seed"])
        h = trace_hash(trace, result.winner, result.turns)
        ok = h == row.get("hash")
        if not ok:
            mismatches += 1
            console.print(f"[red]game {i}: hash MISMATCH[/] stored={row.get('hash')} got={h}")
        else:
            console.print(f"game {i}: ok ({h})")
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
        "flat", help="obs encoding: flat/base, token, or tactical"
    ),
    reward_mode: str = typer.Option("sparse", help="reward mode: sparse, health, or board"),
    learning_rate: float = typer.Option(3e-4, help="PPO learning rate"),
    target_kl: float | None = typer.Option(None, help="PPO target KL early-stop (None = off)"),
    init_model: str = typer.Option(None, help="warm-start from an existing model zip"),
):
    """Train a MaskablePPO agent on the battle env (requires the [ml] extra)."""
    if steps < 1:
        raise typer.BadParameter("steps must be >= 1")
    if n_envs < 1:
        raise typer.BadParameter("n_envs must be >= 1")
    if obs_mode not in ("flat", "base", "token", "tactical"):
        raise typer.BadParameter("obs_mode must be flat/base, token, or tactical")
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
            reward_mode=reward_mode,
            learning_rate=learning_rate,
            target_kl=target_kl,
            init_model=init_model,
        )
    except ImportError as e:
        raise typer.BadParameter("training requires the [ml] extra: uv sync --extra ml") from e
    console.print(f"saved {saved}")


@app.command("train-zoo")
def train_zoo_cmd(
    steps_per_opponent: int = typer.Option(200_000, help="timesteps per opponent phase"),
    out: str = "model.zip",
    seed: int = 0,
    ent_coef: float = typer.Option(0.02, help="entropy coefficient for MaskablePPO"),
    both_seat: bool = typer.Option(True, help="train as both first AND second player"),
    obs_mode: str = typer.Option(
        "flat", help="obs encoding: flat/base, token, or tactical"
    ),
    reward_mode: str = typer.Option("sparse", help="reward mode: sparse, health, or board"),
    learning_rate: float = typer.Option(3e-4, help="PPO learning rate"),
    target_kl: float | None = typer.Option(None, help="PPO target KL early-stop (None = off)"),
    init_model: str = typer.Option(None, help="warm-start from an existing model zip"),
):
    """Train one MaskablePPO agent back-to-back against the code-declared opponent
    zoo (a curriculum; see ZOO_OPPONENTS in locma/envs/training.py). Requires the
    [ml] extra."""
    if steps_per_opponent < 1:
        raise typer.BadParameter("steps-per-opponent must be >= 1")
    if obs_mode not in ("flat", "base", "token", "tactical"):
        raise typer.BadParameter("obs_mode must be flat/base, token, or tactical")
    from locma.envs.training import ZOO_OPPONENTS  # noqa: PLC0415 — constant, no [ml] needed

    for o in ZOO_OPPONENTS:
        make_policy(o)  # validate each declared opponent spec up front
    console.print(f"zoo curriculum: {' -> '.join(ZOO_OPPONENTS)}")
    try:
        from locma.envs.training import train_zoo  # noqa: PLC0415 — optional [ml] dep

        saved = train_zoo(
            steps_per_opponent=steps_per_opponent,
            out=out,
            seed=seed,
            ent_coef=ent_coef,
            both_seat=both_seat,
            obs_mode=obs_mode,
            reward_mode=reward_mode,
            learning_rate=learning_rate,
            target_kl=target_kl,
            init_model=init_model,
        )
    except ImportError as e:
        raise typer.BadParameter("training requires the [ml] extra: uv sync --extra ml") from e
    console.print(f"saved {saved}")


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
        "flat", help="observation encoding: 'flat' (default) or 'token' (tokenized for PPO2)"
    ),
):
    """Record a practicum of teacher battle decisions for distillation."""
    if games < 1:
        raise typer.BadParameter("games must be >= 1")
    if obs_mode not in ("flat", "token"):
        raise typer.BadParameter("obs_mode must be 'flat' or 'token'")
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
    console.print(
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
    if obs_mode is not None and obs_mode not in ("flat", "token"):
        raise typer.BadParameter("obs_mode must be 'flat' or 'token'")
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
    console.print(
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
    console.print(
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
    console.print(
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
    console.print(
        f"best_net={res['best_net']} "
        f"best_score={res['best_score']:.3f} "
        f"final_hard3={res['final_hard3']:.3f} "
        f"final_h2h={res['final_h2h']:.3f}"
    )


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    replay_dir: str = "replays",
    asset_dir: str = "locma/data/assets",
    gamelog_dir: str = ".",
):
    """Run the local replay-viewer web server (requires the [server] extra)."""
    try:
        import uvicorn  # noqa: PLC0415 — optional [server] dep

        from locma.server.app import create_app  # noqa: PLC0415
    except ImportError as e:
        raise typer.BadParameter("serve requires the [server] extra: uv sync --extra server") from e
    app_ = create_app(replay_dir=replay_dir, asset_dir=asset_dir, gamelog_dir=gamelog_dir)
    console.print(f"serving on http://{host}:{port}")
    uvicorn.run(app_, host=host, port=port)


@app.command("fetch-cards")
def fetch_cards_cmd():
    from locma.data.fetch import fetch_cards  # noqa: PLC0415 — lazy import

    path = fetch_cards()
    console.print(f"cards at {path}")


@app.command("fetch-art")
def fetch_art_cmd(
    force: bool = typer.Option(False, "--force", help="re-download even if cached"),
):
    """Download card portrait art into the local (gitignored) cache. Opt-in."""
    from importlib import resources  # noqa: PLC0415

    from locma.data.fetch import fetch_art  # noqa: PLC0415 — lazy import

    n = fetch_art(force=force)
    cache_dir = resources.files("locma.data").joinpath("assets")
    console.print(f"fetched {n} art assets into {cache_dir}")
    console.print(
        "[yellow]Card art is downloaded for local use only; "
        "seek permission from the authors before redistribution.[/]"
    )
