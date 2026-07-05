"""E12 mixed-ensemble driver: b0k critics join the ensemble; b0k reactive confirm.

E11 produced b0k (B0 recipe on the boardkeep-extended zoo): dominant over
depot:b0 on every axis (avg-hard3 +0.0084 CI-positive at 3M+ seeds, better vs
all five exploit archetypes, critic +0.0186 under vbeam at 0.886 ~ shared's
0.890). Its critic diversity axis (a disciplined opponent) is orthogonal to
shared's (asymmetric decks), and E8 showed ensembling de-noises the beam's
sibling orderings -- so mixing families is the natural next lever on the
0.926 recipe of record.

Stages (idempotent -- each skips when its summary key exists):
  A. reactive fresh-anchor confirm: [b0k_s0..2] vs [depot:b0 s0..2], 40x25
     on a fresh 4M+ eval-seed range (E11's +0.0084 ran at 3M+).
  B. ensemble pilots (10x10, 4M+) vs the ensemble recipe of record
     (vbeam:shared_s0|s1|s2, 0.926):
       ens_b0k3   = b0k_s0|s1|s2            (are b0k critics as ensemblable?)
       ens_mixed3 = shared_s0|shared_s1|b0k_s0  (mixed at compute parity)
       ens6       = all six                 (2x evaluator compute)
  C. full 40x25 (4M+) for every pilot with ci_hi > 0.
  D. fresh-anchor confirm (6M+) for every full with ci_lo > 0.

Pre-registered gates:
  reactive: b0k replaces depot:b0 as reactive recipe of record iff stage A
    ci_lo > 0 (dominance swap: two independent CI-positive seed ranges plus
    the E11 exploit/critic dominance; the +0.03 headroom bar is NOT claimed).
  ens_b0k3 / ens_mixed3 (compute parity with the RoR): promote iff full AND
    confirm both ci_lo > 0.
  ens6 (2x compute): promote iff confirmed mean_delta >= +0.03 (the same
    compute-for-headroom trade the 3-critic ensemble itself cleared in E8).

Progress in runs/mixens-overnight.log, results in runs/mixens-summary.json.
Smoke mode: MIXENS_SMOKE=1 -> 2x2 grids, separate summary/log paths.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

SMOKE = os.environ.get("MIXENS_SMOKE") == "1"
WORKERS = 19
SUMMARY_PATH = "runs/mixens-smoke.json" if SMOKE else "runs/mixens-summary.json"
LOG_PATH = "runs/mixens-smoke.log" if SMOKE else "runs/mixens-overnight.log"
FULL = (2, 2) if SMOKE else (40, 25)  # (eval seeds, games_per_seed)
PILOT = (2, 2) if SMOKE else (10, 10)
PRIMARY_START = 4_000_000  # 1M standard / 2M E8 confirm / 3M E11 already used
CONFIRM_START = 6_000_000  # 5M+ holds the E10/E11 exploit-bench match seeds

SEEDS = (0, 1, 2)
SHARED = [f"depot:shared/shared_s{s}.zip" for s in SEEDS]
B0K = [f"runs/b0k_s{s}.zip" for s in SEEDS]
B0 = [f"depot:b0/b0_s{s}.zip" for s in SEEDS]
ENS_ROR = "vbeam:" + "|".join(SHARED)  # the 0.926 recipe of record

ARMS = {
    "ens_b0k3": "vbeam:" + "|".join(B0K),
    "ens_mixed3": "vbeam:" + "|".join([SHARED[0], SHARED[1], B0K[0]]),
    "ens6": "vbeam:" + "|".join(SHARED + B0K),
}

summary: dict = {}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def record(key: str, value) -> None:
    summary[key] = value
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"{key}: {json.dumps(value)}")


def verdict(tag: str, candidates: list[str], baselines: list[str], grid, start: int) -> dict:
    from locma.harness.ceiling_eval import (  # noqa: PLC0415 -- lazy heavy import
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
    return out


def gates() -> None:
    if "e12_gates" in summary:
        log("e12_gates: exists, skip")
        return
    g: dict = {}
    rc = summary["confirm_reactive_b0k_4M"]
    g["reactive_promote_b0k"] = bool(rc["ci_lo"] > 0)
    g["reactive_confirm_delta"] = rc["mean_delta"]
    for arm in ARMS:
        full = summary.get(f"full_{arm}")
        conf = summary.get(f"confirm_{arm}_6M")
        if full is None:
            g[arm] = "pilot-gated out"
        elif full["ci_lo"] <= 0:
            g[arm] = f"full not CI-positive ({full['mean_delta']})"
        elif conf is None:
            g[arm] = "full CI-positive but unconfirmed"
        elif arm == "ens6":
            ok = conf["ci_lo"] > 0 and conf["mean_delta"] >= 0.03
            verdict_txt = "PROMOTE" if ok else "below the +0.03 2x-compute bar"
            g[arm] = f"{verdict_txt} ({conf['mean_delta']})"
        else:
            ok = conf["ci_lo"] > 0
            g[arm] = f"{'PROMOTE' if ok else 'confirm failed'} ({conf['mean_delta']})"
    record("e12_gates", g)


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary.update(json.load(f))
    log("=== E12 mixed-ensemble driver start ===")

    # Stage A: reactive fresh-anchor confirm (the b0k promotion decision).
    verdict("confirm_reactive_b0k_4M", B0K, B0, FULL, start=PRIMARY_START)

    # Stage B: ensemble pilots vs the recipe of record.
    for arm, spec in ARMS.items():
        verdict(f"pilot_{arm}", [spec], [ENS_ROR], PILOT, start=PRIMARY_START)

    # Stage C: fulls for pilots that are not clearly negative.
    for arm, spec in ARMS.items():
        if summary[f"pilot_{arm}"]["ci_hi"] > 0:
            verdict(f"full_{arm}", [spec], [ENS_ROR], FULL, start=PRIMARY_START)
        else:
            log(f"full_{arm}: pilot ci_hi <= 0, not promoted")

    # Stage D: fresh-anchor confirms for CI-positive fulls.
    for arm, spec in ARMS.items():
        full = summary.get(f"full_{arm}")
        if full and full["ci_lo"] > 0:
            verdict(f"confirm_{arm}_6M", [spec], [ENS_ROR], FULL, start=CONFIRM_START)

    gates()
    log("=== E12 mixed-ensemble driver DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
