"""E28d bench: fx nets trained on item-rich decks vs the e28c recipe of record.

The escalation arm motivated by E28c (worklog 2026-07-19): effect visibility
was CI-positive and moved item behavior DESPITE an item-starved training
diet (zoo decks ~blue-free; ~7k green-item hand appearances total). E28d
retrains at the EXACT e28c recipe with ONE more change: training decks come
from the E31a diet table (`train-zoo ... --draft-override runs/e31a_diet.json`
— 6.28 items/deck, 1.13 blues, 72% of decks carry blue vs ~0 before), the
first training data with blue-item exposure in the program's history.

Pre-registered reads (before the bench ran):
  - PRIMARY: paired ruler at the RoR recipe (ldraft deploy decks). Caveat
    cuts BOTH ways now: ldraft is blue-alias-blind (E31 plan) so deploy
    decks are blue-light — blue-specific learning may not show on
    avg-hard3. A positive promotes-candidate as usual.
  - MECHANISM: item rate per opportunity on ldraft decks (comparable to
    e28c's 0.226/0.194 and e28p's 0.142), PLUS the same read on diet decks
    (blue-rich) for e28d AND e28c nets — with a BLUE-specific rate
    (obs_card_ids slots 0-7 x the slot-major Use block, 9+s*13+tc). A
    ruler-null with a positive blue read is NOT a kill: it says the net
    converts blues it never sees at deploy — that routes to E31c (teach
    ldraft to draft blues), not to closing the diet lever.
  - Guard-rail: boardkeep vs singles (5M CRN, e28c band 0.25-0.30).

Stages (idempotent via runs/e28d-bench-summary.json; needs e28d_s{0,1}):
  B. full paired ruler: [ppo:e28d_sX,ldraft_sX] vs
     [ppo:depot:e28c_sX,ldraft_sX], 40x25 @ 50M anchors.
  C. fresh-anchor confirm @ 51M iff B ci_lo > 0.
  D. item reads @ 52M: e28d on ldraft decks; e28d and e28c on diet decks.
  E. boardkeep guard-rail iff B+C pass.

Anchor bookkeeping: 43-49M spent (e28p ladder, e28c pair+ladder); 50M/51M
fresh for verdicts, 52M for item reads (47M was the e28c item seed).

Smoke: E28D_SMOKE=1 -> tiny grids, separate summary path.
"""

from __future__ import annotations

import json
import os
import time

SMOKE = os.environ.get("E28D_SMOKE") == "1"
WORKERS = 19
SEEDS = (0, 1)
SUMMARY_PATH = "runs/e28d-bench-smoke.json" if SMOKE else "runs/e28d-bench-summary.json"
FULL = (2, 2) if SMOKE else (40, 25)
PRIMARY_START = 50_000_000
CONFIRM_START = 51_000_000
ITEM_GAMES = 2 if SMOKE else 60
ITEM_SEED = 52_000_000
GUARD_GAMES = 20 if SMOKE else 1000
GUARD_SEED = 5_000_000

E28D = [f"runs/e28d_s{s}.zip" for s in SEEDS]
E28C = [f"depot:e28c/e28c_s{s}.zip" for s in SEEDS]
LDRAFT = [f"depot:ldraft/ldraft_s{s}.zip" for s in SEEDS]
DIET = "runs/e31a_diet.json"
CANDS = [f"ppo:{m},{ld}" for m, ld in zip(E28D, LDRAFT, strict=True)]
BASES = [f"ppo:{m},{ld}" for m, ld in zip(E28C, LDRAFT, strict=True)]

USE_LO, USE_HI = 9, 113
USE_STRIDE = 13  # Use block is slot-major: 9 + hand_slot*13 + target_code
MAX_HAND = 8

summary: dict = {}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def record(key: str, value) -> None:
    summary[key] = value
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)


def verdict(tag: str, candidates, baselines, grid, start: int) -> dict:
    from locma.harness.ceiling_eval import (  # noqa: PLC0415 — lazy heavy import
        _disjoint_eval_seeds,
        run_verdict,
    )

    if tag in summary:
        log(f"{tag}: exists, skip")
        return summary[tag]
    n_seeds, gps = grid
    t0 = time.time()
    out = run_verdict(
        candidates,
        baselines,
        seeds=_disjoint_eval_seeds(n_seeds, gps, start=start),
        games_per_seed=gps,
        workers=WORKERS,
    )
    out = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
    out["minutes"] = round((time.time() - t0) / 60, 1)
    record(tag, out)
    log(f"{tag}: {out}")
    return out


def _blue_ids() -> set[int]:
    from locma.data.cards_db import load_cards  # noqa: PLC0415

    return {c.id for c in load_cards() if c.type == 3}


def item_read(tag: str, teacher_spec: str) -> dict:
    """Play the net as its own practicum teacher; measure item + blue behavior."""
    import numpy as np  # noqa: PLC0415

    from locma.envs.practicum import record_practicum  # noqa: PLC0415

    if tag in summary:
        log(f"{tag}: exists, skip")
        return summary[tag]
    out_npz = f"runs/{tag}.npz"
    record_practicum(
        teacher=teacher_spec,
        games=ITEM_GAMES,
        out=out_npz,
        seed=ITEM_SEED,
        obs_mode="token",
    )
    d = np.load(out_npz)
    act, mask, card_ids = d["action"], d["mask"], d["obs_card_ids"]
    can_item = mask[:, USE_LO:USE_HI].any(axis=1)
    played = (act >= USE_LO) & (act < USE_HI)

    blue = _blue_ids()
    hand_blue = np.isin(card_ids[:, :MAX_HAND], list(blue))  # (n, 8)
    use_legal = mask[:, USE_LO:USE_HI].reshape(len(act), MAX_HAND, USE_STRIDE).any(axis=2)
    can_blue = (hand_blue & use_legal).any(axis=1)
    played_slot = np.where(played, (act - USE_LO) // USE_STRIDE, -1)
    played_blue = np.array([s >= 0 and bool(hand_blue[i, s]) for i, s in enumerate(played_slot)])

    res = {
        "teacher": teacher_spec,
        "n_decisions": int(len(act)),
        "item_opportunities": int(can_item.sum()),
        "item_rate_per_opportunity": round(float(played[can_item].mean()), 4),
        "blue_opportunities": int(can_blue.sum()),
        "blue_rate_per_opportunity": round(float(played_blue[can_blue].mean()), 4)
        if can_blue.any()
        else None,
    }
    record(tag, res)
    log(f"{tag}: {res}")
    return res


def guardrail(tag: str, spec: str) -> dict:
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    if tag in summary:
        log(f"{tag}: exists, skip")
        return summary[tag]
    t0 = time.time()
    r = run_match(make_policy("boardkeep"), make_policy(spec), games=GUARD_GAMES, seed=GUARD_SEED)
    wr = r.wins_a / (r.wins_a + r.wins_b)
    res = {
        "spec": spec,
        "boardkeep_wr": round(wr, 4),
        "games": 2 * GUARD_GAMES,
        "minutes": round((time.time() - t0) / 60, 1),
    }
    record(tag, res)
    log(f"{tag}: {res}")
    return res


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary.update(json.load(f))
    log("=== E28d bench start ===")
    for m in E28D:
        if not os.path.exists(m):
            raise SystemExit(f"missing trained model {m} — run train-zoo --draft-override first")

    full = verdict("full_vs_e28c", CANDS, BASES, FULL, start=PRIMARY_START)

    confirm = None
    if full["ci_lo"] > 0:
        confirm = verdict("confirm_vs_e28c_51M", CANDS, BASES, FULL, start=CONFIRM_START)
    else:
        record("confirm_skipped", "full ci_lo <= 0")

    for s in SEEDS:
        item_read(f"item_read_e28d_s{s}_ldraft", CANDS[s])
        item_read(f"item_read_e28d_s{s}_diet", f"ppo:{E28D[s]},{DIET}")
    item_read("item_read_e28c_s0_diet", f"ppo:{E28C[0]},{DIET}")

    diet_pass = bool(full["ci_lo"] > 0 and confirm is not None and confirm["ci_lo"] > 0)
    if diet_pass:
        for s in SEEDS:
            guardrail(f"guardrail_e28d_s{s}", CANDS[s])

    record(
        "gates",
        {
            "full_delta": full["mean_delta"],
            "full_ci": [full["ci_lo"], full["ci_hi"]],
            "confirm_ci": None if confirm is None else [confirm["ci_lo"], confirm["ci_hi"]],
            "diet_pass": diet_pass,
            "headroom": bool(full["mean_delta"] >= 0.03),
            "regression": bool(full["ci_hi"] < 0),
        },
    )
    log(f"gates: {summary['gates']}")


if __name__ == "__main__":
    main()
