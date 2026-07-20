"""E28c stack bench: 3-seed token-fx artifact vs the E28 promotion ladder.

The e28c bench (runs/e28c-bench-summary.json, commit 47d0ab0) showed the fx
obs variant beats the e28p RoR pair +0.0212 [+0.0140, +0.0282], replicated
at fresh anchors (+0.0215 [+0.0157, +0.0273]) — sub-headroom but
zero-excluding twice, the E7 promotion pattern. With e28c_s2 trained, this
driver mirrors scripts/e28_stack_bench.py stage-for-stage to decide
promotion on both rungs (pure reactive artifact + guarded lppo stack).

Stages (idempotent, resumable via runs/e28c-stack-summary.json; requires
runs/e28c_s{0,1,2}.zip):
  A. pure3: [ppo:e28c_sX,ldraft_sX] x3 vs [ppo:depot:e28p_sX,ldraft_sX] x3,
     40x25 @ 48M — 3-seed completion of the 2-seed paired ruler.
  B. stack_vs_ror: [lppo:e28c trio,ldraft_sX] x3 vs
     [lppo:e28p trio,ldraft_sX] x3 (the 0.908 guarded RoR), 40x25 @ 48M —
     the promotion comparison.
  C. lens_increment: [lppo:e28c trio,ldraft_sX] x3 vs
     [ppo:e28c_sX,ldraft_sX] x3, 40x25 @ 48M — does the lens still add on
     top of fx nets?
  D. boardkeep guard-rail on the stack (1000 mirrored @ 5M CRN — same seed
     as every prior guard read, e28p stack band 0.221).
  E. confirm of stage B on fresh 49M anchors iff B ci_lo > 0.

Anchor bookkeeping: 43M/44M spent by e28_stack_bench, 45M/46M/47M by
e28c_bench — 48M/49M are fresh.

Smoke: E28C_SMOKE=1 -> tiny grids, separate summary path.
"""

from __future__ import annotations

import json
import os
import time

SMOKE = os.environ.get("E28C_SMOKE") == "1"
WORKERS = 19
SEEDS = (0, 1, 2)
SUMMARY_PATH = "runs/e28c-stack-smoke.json" if SMOKE else "runs/e28c-stack-summary.json"
FULL = (2, 2) if SMOKE else (40, 25)
PRIMARY_START = 48_000_000
CONFIRM_START = 49_000_000
GUARD_GAMES = 20 if SMOKE else 1000
GUARD_SEED = 5_000_000

E28C = [f"runs/e28c_s{s}.zip" for s in SEEDS]
E28P = [f"depot:e28p/e28p_s{s}.zip" for s in SEEDS]
LDRAFT = [f"depot:ldraft/ldraft_s{s}.zip" for s in SEEDS]
E28C_TRIO = "|".join(E28C)
E28P_TRIO = "|".join(E28P)

PURE_CANDS = [f"ppo:{m},{ld}" for m, ld in zip(E28C, LDRAFT, strict=True)]
PURE_BASES = [f"ppo:{m},{ld}" for m, ld in zip(E28P, LDRAFT, strict=True)]
STACK_CANDS = [f"lppo:{E28C_TRIO},{ld}" for ld in LDRAFT]
STACK_BASES = [f"lppo:{E28P_TRIO},{ld}" for ld in LDRAFT]

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
    log("=== E28c stack bench start ===")
    for m in E28C:
        if not os.path.exists(m):
            raise SystemExit(f"missing trained model {m}")

    pure3 = verdict("pure3_reactive", PURE_CANDS, PURE_BASES, FULL, start=PRIMARY_START)
    stack = verdict("stack_vs_ror", STACK_CANDS, STACK_BASES, FULL, start=PRIMARY_START)
    incr = verdict("lens_increment", STACK_CANDS, PURE_CANDS, FULL, start=PRIMARY_START)

    guardrail("guardrail_stack", STACK_CANDS[0])

    confirm = None
    if stack["ci_lo"] > 0:
        confirm = verdict("confirm_stack_49M", STACK_CANDS, STACK_BASES, FULL, start=CONFIRM_START)
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
