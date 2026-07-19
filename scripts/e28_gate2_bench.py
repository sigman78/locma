"""E28 gate 2: pointer-head PPO retrain vs the reactive recipe of record.

Gate 1 (worklog 2026-07-19) showed the pointer action head breaks the BC
imitation cap at matched parameters. Gate 2 asks whether that converts into
WIN RATE when trained by PPO at the exact b0k recipe (token V0, lr=1e-4,
target_kl=0.025, dropout=0.1, n_envs=16, 5-phase zoo x 200k = 1M steps),
and whether PPO converts slot access into item usage where BC could not.

Stages (idempotent, resumable via runs/e28-gate2-summary.json; models
runs/e28p_s{0,1}.zip are trained separately via `locma train-zoo
--pointer-head`):
  B. full reactive ruler: [ppo:e28p_sX,ldraft_sX] vs [ppo:b0k_sX,ldraft_sX],
     40x25 @ 40M anchors (per-eval-seed avg-hard3, paired bootstrap CI).
  C. fresh-anchor confirm @ 41M iff B ci_lo > 0 (E12 promotion pattern).
  D. item-behavior read: record-practicum with each net as its own teacher
     (60 games @ 42M seeds) -> item plays per item-legal decision. The E14a
     baseline underuse is 3.3x; gate-1 BC could not move it — can PPO?
  E. boardkeep guard-rail iff B+C pass: 1000 mirrored games @ 5M common
     random numbers, boardkeep vs each arm (E10/E18c protocol) — a
     promotion must not open an adversarial hole (b0k reference: 0.512).

Pre-registered gates: gate2_pass iff B ci_lo > 0 AND C ci_lo > 0;
headroom iff B mean_delta >= +0.03. Item read is informational (mechanism),
not a gate.

Smoke: E28_SMOKE=1 -> tiny grids, separate artifact paths.
"""

from __future__ import annotations

import json
import os
import time

SMOKE = os.environ.get("E28_SMOKE") == "1"
WORKERS = 19
SEEDS = (0, 1)
SUMMARY_PATH = "runs/e28-gate2-smoke.json" if SMOKE else "runs/e28-gate2-summary.json"
FULL = (2, 2) if SMOKE else (40, 25)
PRIMARY_START = 40_000_000
CONFIRM_START = 41_000_000
ITEM_GAMES = 2 if SMOKE else 60
ITEM_SEED = 42_000_000
GUARD_GAMES = 20 if SMOKE else 1000
GUARD_SEED = 5_000_000

E28P = [f"runs/e28p_s{s}.zip" for s in SEEDS]
B0K = [f"depot:b0k/b0k_s{s}.zip" for s in SEEDS]
LDRAFT = [f"depot:ldraft/ldraft_s{s}.zip" for s in SEEDS]
CANDS = [f"ppo:{m},{ld}" for m, ld in zip(E28P, LDRAFT, strict=True)]
BASES = [f"ppo:{m},{ld}" for m, ld in zip(B0K, LDRAFT, strict=True)]

USE_LO, USE_HI = 9, 113

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


def item_read(tag: str, teacher_spec: str) -> dict:
    """Play the net as its own practicum teacher; measure item behavior."""
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
    act, mask = d["action"], d["mask"]
    can_item = mask[:, USE_LO:USE_HI].any(axis=1)
    played = (act >= USE_LO) & (act < USE_HI)
    res = {
        "teacher": teacher_spec,
        "n_decisions": int(len(act)),
        "item_opportunities": int(can_item.sum()),
        "item_rate_per_opportunity": round(float(played[can_item].mean()), 4),
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
    log("=== E28 gate 2 bench start ===")
    for m in E28P:
        if not os.path.exists(m):
            raise SystemExit(f"missing trained model {m} — run train-zoo --pointer-head first")

    full = verdict("full_reactive", CANDS, BASES, FULL, start=PRIMARY_START)

    confirm = None
    if full["ci_lo"] > 0:
        confirm = verdict("confirm_reactive_41M", CANDS, BASES, FULL, start=CONFIRM_START)
    else:
        record("confirm_skipped", "full ci_lo <= 0")

    # Item-behavior read (mechanism, not a gate): candidate seeds + b0k s0 ref.
    for s in SEEDS:
        item_read(f"item_read_e28p_s{s}", CANDS[s])
    item_read("item_read_b0k_s0", BASES[0])

    gate2_pass = bool(full["ci_lo"] > 0 and confirm is not None and confirm["ci_lo"] > 0)
    if gate2_pass:
        for s in SEEDS:
            guardrail(f"guardrail_e28p_s{s}", CANDS[s])
        guardrail("guardrail_b0k_s0", BASES[0])

    record(
        "gates",
        {
            "full_delta": full["mean_delta"],
            "full_ci": [full["ci_lo"], full["ci_hi"]],
            "confirm_ci": None if confirm is None else [confirm["ci_lo"], confirm["ci_hi"]],
            "gate2_pass": gate2_pass,
            "headroom": bool(full["mean_delta"] >= 0.03),
        },
    )
    log(f"gates: {summary['gates']}")


if __name__ == "__main__":
    main()
