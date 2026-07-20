"""Is the oracle WRONG about blues, or are blues weak? Rollout-horizon test.

The blue-value diagnostic used a cheating MCTS oracle (rollout_turns=3,
heuristic leaf) as the "true value" reference. But its leaf value
(mcts._leaf_value) is health-lead + 0.5*board-power-lead — it has NO
card-advantage term, and a 3-turn RANDOM rollout barely converts card
draw into anything. So the oracle is structurally blind to the card-draw
value of blues (154, 157). The base read already shows it: oracle plays
draw-blues 0.152 vs non-draw 0.234 — declining exactly the items its
heuristic can't price.

Decisive test: extend the rollout horizon so card advantage can actually
convert. Same iterations/seed/opponents/decks, three leaf regimes:
  - heur3  : rollout_turns=3 heuristic leaf (the base oracle)
  - heur8  : rollout_turns=8 heuristic leaf (longer; more conversion)
  - terminal: rollout_turns=0 = random rollout to GAME END (reward is the
    actual win/loss — no material heuristic at all)
If the oracle's DRAW-blue rate rises toward/above its non-draw rate as the
horizon grows, the low draw rate was a heuristic-horizon artifact (the
oracle undervalues draw blues) — NOT evidence those blues are weak. If it
stays flat, 0.15 is robust and draw blues really are weak.

Output: runs/blue-oracle-horizon.json.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

DIET = "runs/e31a_diet.json"
GAMES = 20
SEED = 54_000_000
OPPS = ("scripted", "max-guard", "max-attack")
ITERS = 300
USE_LO, USE_HI, STRIDE, MAX_HAND = 9, 113, 13, 8

REGIMES = {
    "heur3": f"mcts:{ITERS},1.41,0,3,{DIET}",
    "heur8": f"mcts:{ITERS},1.41,0,8,{DIET}",
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


def _rates(npz_path, blue, draw):
    d = np.load(npz_path)
    act, mask, cids = d["action"], d["mask"], d["obs_card_ids"]
    n = len(act)
    use_legal = mask[:, USE_LO:USE_HI].reshape(n, MAX_HAND, STRIDE).any(axis=2)
    played = (act >= USE_LO) & (act < USE_HI)
    pslot = np.where(played, (act - USE_LO) // STRIDE, -1)
    opp = {"draw": 0, "nondraw": 0}
    pl = {"draw": 0, "nondraw": 0}
    for i in range(n):
        for s in range(MAX_HAND):
            cid = int(cids[i, s])
            if cid in blue and use_legal[i, s]:
                key = "draw" if cid in draw else "nondraw"
                opp[key] += 1
                if pslot[i] == s:
                    pl[key] += 1
    rate = lambda k: round(pl[k] / opp[k], 3) if opp[k] else None  # noqa: E731
    return {
        "n_decisions": int(n),
        "draw_blue_rate": rate("draw"),
        "draw_blue_n": opp["draw"],
        "nondraw_blue_rate": rate("nondraw"),
        "nondraw_blue_n": opp["nondraw"],
    }


def main() -> None:
    from locma.envs.practicum import record_practicum  # noqa: PLC0415

    os.makedirs("runs", exist_ok=True)
    if os.path.exists("runs/blue-oracle-horizon.json"):
        with open("runs/blue-oracle-horizon.json", encoding="utf-8") as f:
            summary.update(json.load(f))
    blue, draw = _blue_split()
    log(f"=== oracle horizon sweep === draw-blues {sorted(draw)}")
    for tag, spec in REGIMES.items():
        if tag in summary:
            log(f"{tag}: exists, skip")
            continue
        out = f"runs/boh_{tag}.npz"
        t0 = time.time()
        if not os.path.exists(out):
            record_practicum(
                teacher=spec, opponents=OPPS, games=GAMES, out=out, seed=SEED, obs_mode="token"
            )
        mins = round((time.time() - t0) / 60, 1)
        res = {"spec": spec, **_rates(out, blue, draw), "minutes": mins}
        summary[tag] = res
        with open("runs/blue-oracle-horizon.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
        log(f"{tag}: {res}")
    log("done")


if __name__ == "__main__":
    main()
