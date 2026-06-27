"""AlphaZero self-play loop orchestrator — P5 (netdmcts Phase 2).

Ties ``record_selfplay`` (P3) and ``az_train`` (P4) into an iterated loop
with a composite adoption gate:

    adopt iff  (h2h > h2h_thresh)  AND  (score >= best_score - hard3_eps)

``best_score`` tracks the high-water mark (``max`` on each adopt).

Import-safe without the ``[ml]`` extra: torch/sb3 are only pulled in
transitively inside the real ``record_selfplay`` / ``az_train`` bodies,
which are bypassed by monkeypatching in tests.

All four pipeline steps (``record_selfplay``, ``az_train``, ``avg_hard3``,
``h2h_winrate``) are module-level names so tests can monkeypatch them and
exercise the gate logic without any ``[ml]`` dependency.
"""

from __future__ import annotations

import json

from locma.envs.az_train import az_train  # module-level — monkeypatchable
from locma.envs.selfplay import record_selfplay  # module-level — monkeypatchable
from locma.harness.match import run_match  # module-level — monkeypatchable
from locma.policies.registry import make_policy  # module-level — monkeypatchable


def avg_hard3(
    net_path: str,
    games_per_opp: int = 20,
    K: int = 8,
    I: int = 40,  # noqa: E741
    c_puct: float = 1.5,
    seed: int = 0,
) -> float:
    """Mean win-rate of *net_path* vs the three hard baselines.

    Runs ``games_per_opp`` games against each of
    ``("scripted", "max-guard", "max-attack")`` and returns the mean of the
    three ``win_rate_a`` values from ``run_match``.
    """
    baselines = ("scripted", "max-guard", "max-attack")
    rates = []
    for baseline in baselines:
        netdmcts = make_policy(f"netdmcts:{K},{I},{c_puct},{net_path}")
        opp = make_policy(baseline)
        res = run_match(netdmcts, opp, games=games_per_opp, seed=seed)
        rates.append(res.win_rate_a)
    return sum(rates) / len(rates)


def h2h_winrate(
    new_path: str,
    best_path: str,
    games: int = 40,
    K: int = 8,
    I: int = 40,  # noqa: E741
    c_puct: float = 1.5,
    seed: int = 0,
) -> float:
    """Head-to-head win-rate of the *new* net vs the current *best* net.

    Returns ``win_rate_a`` from ``run_match``, which is the *new* net's
    fraction of wins when it is policy A.
    """
    res = run_match(
        make_policy(f"netdmcts:{K},{I},{c_puct},{new_path}"),
        make_policy(f"netdmcts:{K},{I},{c_puct},{best_path}"),
        games=games,
        seed=seed,
    )
    return res.win_rate_a


def az_selfplay(
    warm_start: str = "runs/selfplay-r2.zip",
    prefix: str = "runs/az",
    iterations: int = 4,
    window: int = 2,
    base_seed: int = 0,
    self_play_games: int = 240,
    baseline_games: int = 100,
    K_gen: int = 6,
    I_gen: int = 40,
    c_puct: float = 1.5,
    eps: float = 0.25,
    alpha: float = 0.3,
    temp_moves: int = 10,
    epochs: int = 10,
    batch: int = 256,
    lr: float = 1e-4,
    c_v: float = 0.5,
    K_eval: int = 8,
    I_eval: int = 40,
    games_per_opp: int = 20,
    h2h_games: int = 40,
    h2h_thresh: float = 0.53,
    hard3_eps: float = 0.02,
    max_rejects: int = 2,
    verbose: int = 1,
) -> dict:
    """AlphaZero iterated self-play loop with a composite adoption gate.

    Each iteration:

    1. Generates self-play data with the current ``best_net`` as oracle.
    2. Trains a candidate net warm-started from ``best_net``.
    3. Evaluates the candidate on two axes:
       - ``score``: mean win-rate vs the three hard baselines (``avg_hard3``).
       - ``h2h``: head-to-head win-rate vs ``best_net`` (``h2h_winrate``).
    4. **Composite adopt:** ``(h2h > h2h_thresh) and (score >= best_score - hard3_eps)``.
       - On adopt: ``best_net`` advances; ``best_score`` tracks the high-water.
       - On reject: ``best_net`` is retained (rejected net is discarded).
    5. If ``rejects >= max_rejects``, stops early.

    After the loop, runs a final confirmation evaluation (50 games vs baselines,
    100-game h2h vs the original ``warm_start``).

    Parameters
    ----------
    warm_start:
        Initial net path.  Used as first oracle and as the reference for the
        final h2h confirmation.
    prefix:
        Output path prefix.  Generates ``{prefix}-data-{it}.npz``,
        ``{prefix}-net-{it}.zip``, and ``{prefix}-results.jsonl``.
    iterations:
        Maximum self-play/train/eval rounds.
    window:
        Rolling-window size: only the most recent ``window`` datasets are
        passed to ``az_train``.
    base_seed:
        Base RNG seed; iteration ``it`` uses ``base_seed + it``.
    self_play_games:
        Self-play games per iteration.
    baseline_games:
        Baseline games per iteration.
    K_gen, I_gen:
        PUCT parameters for self-play generation.
    c_puct:
        PUCT exploration constant (shared between generation and evaluation).
    eps, alpha:
        Dirichlet noise parameters for generation.
    temp_moves:
        Temperature plies for generation.
    epochs, batch, lr, c_v:
        Training hyperparameters.
    K_eval, I_eval:
        PUCT parameters for evaluation.
    games_per_opp:
        Games per opponent in ``avg_hard3``.
    h2h_games:
        Games for head-to-head evaluation.
    h2h_thresh:
        Minimum h2h win-rate for adoption (strictly greater than).
    hard3_eps:
        Maximum allowed drop below ``best_score`` for adoption.
    max_rejects:
        Early-stop threshold: stop after this many consecutive rejections.
    verbose:
        Print per-iteration summary when nonzero.

    Returns
    -------
    dict
        ``best_net``, ``best_score``, ``history``, ``final_hard3``,
        ``final_h2h``.
    """
    best_net: str = warm_start
    best_score: float = avg_hard3(best_net, games_per_opp, K_eval, I_eval, c_puct, base_seed)
    datasets: list[str] = []
    rejects: int = 0
    history: list[dict] = []
    results_path = f"{prefix}-results.jsonl"

    for it in range(iterations):
        # --- Step a: generate self-play data ---
        npz = f"{prefix}-data-{it}.npz"
        record_selfplay(
            best_net,
            out=npz,
            self_play_games=self_play_games,
            baseline_games=baseline_games,
            K=K_gen,
            I=I_gen,
            c_puct=c_puct,
            eps=eps,
            alpha=alpha,
            temp_moves=temp_moves,
            seed=base_seed + it,
        )
        datasets.append(npz)

        # --- Step b: train candidate net ---
        new_net = f"{prefix}-net-{it}.zip"
        az_train(
            datasets[-window:],
            warm_start=best_net,
            out=new_net,
            epochs=epochs,
            batch=batch,
            lr=lr,
            c_v=c_v,
            seed=base_seed + it,
        )

        # --- Step c: evaluate candidate ---
        score = avg_hard3(new_net, games_per_opp, K_eval, I_eval, c_puct, base_seed + it)
        h2h = h2h_winrate(new_net, best_net, h2h_games, K_eval, I_eval, c_puct, base_seed + it)

        # --- Step d: composite adopt gate ---
        adopt = (h2h > h2h_thresh) and (score >= best_score - hard3_eps)
        if adopt:
            best_net = new_net
            best_score = max(score, best_score)
            rejects = 0
        else:
            rejects += 1

        # --- Step e: record iteration ---
        entry: dict = {
            "it": it,
            "score": score,
            "h2h": h2h,
            "adopted": adopt,
            "best_score": best_score,
            "net": new_net,
        }
        history.append(entry)
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        if verbose:
            status = "ADOPT" if adopt else "REJECT"
            print(
                f"[azloop] it={it}  score={score:.4f}  h2h={h2h:.4f}"
                f"  best_score={best_score:.4f}  {status}"
            )

        # --- Step f: early stop ---
        if rejects >= max_rejects:
            break

    # --- Final confirmation ---
    final_hard3 = avg_hard3(best_net, 50, K_eval, I_eval, c_puct, base_seed + 1000)
    final_h2h = h2h_winrate(best_net, warm_start, 100, K_eval, I_eval, c_puct, base_seed + 2000)

    return {
        "best_net": best_net,
        "best_score": best_score,
        "history": history,
        "final_hard3": final_hard3,
        "final_h2h": final_h2h,
    }
