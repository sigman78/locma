"""Verify the draw-blue crossover with more terminal-rollout games + CIs.

The horizon sweep (scripts/blue_oracle_horizon.py) found the cheating-MCTS
oracle's card-draw-blue play rate rises 0.133 -> 0.258 as the rollout
extends to terminal — but terminal draw-n was only 31 (rate CI ~[0.10,
0.41]). This reruns the two endpoints (heur3 base, terminal) at a FRESH
disjoint seed and ~7x the games, and puts a bootstrap-over-games 95% CI on
each draw/non-draw rate so the crossover claim is testable:

  claim A: terminal draw-blue rate > the base (heur3) draw-blue rate
           (the horizon lifts draw specifically).
  claim B: terminal draw-blue ~>= terminal non-draw (the crossover).

Bootstrap resamples whole games (opportunities are correlated within a
game), so the CI reflects the real effective sample size.

Output: runs/blue-oracle-verify.json.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

DIET = "runs/e31a_diet.json"
GAMES = 150
SEED = 55_000_000  # disjoint from the sweep's 54M
OPPS = ("scripted", "max-guard", "max-attack")
ITERS = 300
USE_LO, USE_HI, STRIDE, MAX_HAND = 9, 113, 13, 8
N_BOOT = 4000

REGIMES = {
    "heur3": f"mcts:{ITERS},1.41,0,3,{DIET}",
    "terminal": f"mcts:{ITERS},1.41,0,0,{DIET}",
}

summary: dict = {}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _blue_split():
    from locma.data.cards_db import load_cards  # noqa: PLC0415

    cards = {c.id: c for c in load_cards()}
    blue = [cid for cid, c in cards.items() if c.type == 3]
    draw = {cid for cid in blue if cards[cid].card_draw > 0}
    return set(blue), draw


def _per_game(npz_path, blue, draw):
    """Per-game (draw_played, draw_opp, nondraw_played, nondraw_opp) arrays."""
    d = np.load(npz_path)
    act, mask, cids, gid = d["action"], d["mask"], d["obs_card_ids"], d["game_id"]
    n = len(act)
    use_legal = mask[:, USE_LO:USE_HI].reshape(n, MAX_HAND, STRIDE).any(axis=2)
    played = (act >= USE_LO) & (act < USE_HI)
    pslot = np.where(played, (act - USE_LO) // STRIDE, -1)
    acc: dict[int, list[int]] = {}
    for i in range(n):
        for s in range(MAX_HAND):
            cid = int(cids[i, s])
            if cid in blue and use_legal[i, s]:
                g = acc.setdefault(int(gid[i]), [0, 0, 0, 0])
                is_draw = cid in draw
                did = pslot[i] == s
                g[0] += is_draw and did
                g[1] += is_draw
                g[2] += (not is_draw) and did
                g[3] += not is_draw
    return np.array(list(acc.values()), dtype=float)  # (n_games, 4)


def _rate_ci(games, played_col, opp_col, rng):
    pl = games[:, played_col].sum()
    op = games[:, opp_col].sum()
    if op == 0:
        return None, None, 0
    boot = []
    ng = len(games)
    for _ in range(N_BOOT):
        idx = rng.integers(0, ng, ng)
        o = games[idx, opp_col].sum()
        if o > 0:
            boot.append(games[idx, played_col].sum() / o)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return round(pl / op, 3), [round(float(lo), 3), round(float(hi), 3)], int(op)


def main() -> None:
    from locma.envs.practicum import record_practicum  # noqa: PLC0415

    os.makedirs("runs", exist_ok=True)
    if os.path.exists("runs/blue-oracle-verify.json"):
        with open("runs/blue-oracle-verify.json", encoding="utf-8") as f:
            summary.update(json.load(f))
    rng = np.random.default_rng(0)
    blue, draw = _blue_split()
    log(f"=== oracle horizon VERIFY (games={GAMES}, seed={SEED}) === draw-blues {sorted(draw)}")
    for tag, spec in REGIMES.items():
        if tag in summary:
            log(f"{tag}: exists, skip")
            continue
        out = f"runs/bohv_{tag}.npz"
        t0 = time.time()
        if not os.path.exists(out):
            record_practicum(
                teacher=spec, opponents=OPPS, games=GAMES, out=out, seed=SEED, obs_mode="token"
            )
        g = _per_game(out, blue, draw)
        dr, dci, dn = _rate_ci(g, 0, 1, rng)
        nr, nci, nn = _rate_ci(g, 2, 3, rng)
        res = {
            "spec": spec,
            "draw_blue_rate": dr,
            "draw_blue_ci": dci,
            "draw_blue_n": dn,
            "nondraw_blue_rate": nr,
            "nondraw_blue_ci": nci,
            "nondraw_blue_n": nn,
            "minutes": round((time.time() - t0) / 60, 1),
        }
        summary[tag] = res
        with open("runs/blue-oracle-verify.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
        log(f"{tag}: {res}")

    # bootstrap the terminal - heur3 draw-blue difference (paired by nothing;
    # independent seeds/games, so difference of independent bootstrap draws).
    if "heur3" in summary and "terminal" in summary:
        gh = _per_game("runs/bohv_heur3.npz", blue, draw)
        gt = _per_game("runs/bohv_terminal.npz", blue, draw)
        diffs = []
        for _ in range(N_BOOT):
            ih = rng.integers(0, len(gh), len(gh))
            it = rng.integers(0, len(gt), len(gt))
            oh, ot = gh[ih, 1].sum(), gt[it, 1].sum()
            if oh > 0 and ot > 0:
                diffs.append(gt[it, 0].sum() / ot - gh[ih, 0].sum() / oh)
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        summary["terminal_minus_heur3_draw"] = {
            "delta": round(float(np.mean(diffs)), 3),
            "ci": [round(float(lo), 3), round(float(hi), 3)],
            "excludes_zero": bool(lo > 0 or hi < 0),
        }
        with open("runs/blue-oracle-verify.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
        log(f"terminal-heur3 draw diff: {summary['terminal_minus_heur3_draw']}")
    log("done")


if __name__ == "__main__":
    main()
