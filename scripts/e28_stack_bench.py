"""E28 follow-up: 3-seed pointer artifact + the lens-over-pointer stack.

Gate 2 (worklog 2026-07-19) gave the pointer-head retrain headroom +0.073
over the reactive RoR pair with two seeds. This driver completes the
standard 3-seed artifact and asks the promotion-relevant question: does the
E26 lens (lethal-guard DFS + policy-head ensemble) stacked OVER the pointer
trio beat the current guarded recipe of record (lens over the b0k trio)?

Stages (idempotent, resumable via runs/e28-stack-summary.json; requires
runs/e28p_s{0,1,2}.zip):
  A. pure3: [ppo:e28p_sX,ldraft_sX] x3 vs [ppo:b0k_sX,ldraft_sX] x3,
     40x25 @ 43M — the 3-seed completion of gate 2's 2-seed ruler.
  B. stack_vs_ror: [lppo:e28p trio,ldraft_sX] x3 vs
     [lppo:b0k trio,ldraft_sX] x3 (the guarded RoR), 40x25 @ 43M — the
     promotion comparison.
  C. lens_increment: [lppo:e28p trio,ldraft_sX] x3 vs
     [ppo:e28p_sX,ldraft_sX] x3, 40x25 @ 43M — does the lens still add on
     top of pointer nets (E26 mechanisms were disjoint on b0k; the pointer
     nets already fixed targeting, so the increment may shrink)?
  D. boardkeep guard-rail on the stack (1000 mirrored @ 5M CRN).
  E. confirm of stage B on fresh 44M anchors iff B ci_lo > 0.

Smoke: E28_SMOKE=1 -> tiny grids, separate artifact paths.
"""

from __future__ import annotations

import json
import os
import time

SMOKE = os.environ.get("E28_SMOKE") == "1"
WORKERS = 19
SEEDS = (0, 1, 2)
SUMMARY_PATH = "runs/e28-stack-smoke.json" if SMOKE else "runs/e28-stack-summary.json"
FULL = (2, 2) if SMOKE else (40, 25)
PRIMARY_START = 43_000_000
CONFIRM_START = 44_000_000
GUARD_GAMES = 20 if SMOKE else 1000
GUARD_SEED = 5_000_000

E28P = [f"runs/e28p_s{s}.zip" for s in SEEDS]
B0K = [f"depot:b0k/b0k_s{s}.zip" for s in SEEDS]
LDRAFT = [f"depot:ldraft/ldraft_s{s}.zip" for s in SEEDS]
E28P_TRIO = "|".join(E28P)
B0K_TRIO = "|".join(B0K)

PURE_CANDS = [f"ppo:{m},{ld}" for m, ld in zip(E28P, LDRAFT, strict=True)]
PURE_BASES = [f"ppo:{m},{ld}" for m, ld in zip(B0K, LDRAFT, strict=True)]
STACK_CANDS = [f"lppo:{E28P_TRIO},{ld}" for ld in LDRAFT]
STACK_BASES = [f"lppo:{B0K_TRIO},{ld}" for ld in LDRAFT]

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
    log("=== E28 stack bench start ===")
    for m in E28P:
        if not os.path.exists(m):
            raise SystemExit(f"missing trained model {m}")

    pure3 = verdict("pure3_reactive", PURE_CANDS, PURE_BASES, FULL, start=PRIMARY_START)
    stack = verdict("stack_vs_ror", STACK_CANDS, STACK_BASES, FULL, start=PRIMARY_START)
    incr = verdict("lens_increment", STACK_CANDS, PURE_CANDS, FULL, start=PRIMARY_START)

    guardrail("guardrail_stack", STACK_CANDS[0])

    confirm = None
    if stack["ci_lo"] > 0:
        confirm = verdict("confirm_stack_44M", STACK_CANDS, STACK_BASES, FULL, start=CONFIRM_START)
    else:
        record("confirm_skipped", "stack ci_lo <= 0")

    record(
        "gates",
        {
            "pure3_delta": pure3["mean_delta"],
            "pure3_ci": [pure3["ci_lo"], pure3["ci_hi"]],
            "stack_delta": stack["mean_delta"],
            "stack_ci": [stack["ci_lo"], stack["ci_hi"]],
            "lens_increment_delta": incr["mean_delta"],
            "lens_increment_ci": [incr["ci_lo"], incr["ci_hi"]],
            "confirm_ci": None if confirm is None else [confirm["ci_lo"], confirm["ci_hi"]],
            "stack_beats_ror": bool(
                stack["ci_lo"] > 0 and confirm is not None and confirm["ci_lo"] > 0
            ),
        },
    )
    log(f"gates: {summary['gates']}")


if __name__ == "__main__":
    main()
