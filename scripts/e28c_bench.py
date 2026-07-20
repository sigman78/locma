"""E28c bench: pointer-head nets on token-fx obs vs the e28p recipe of record.

Feature completion (docs/reactive-limits-program.md E28c): the fx token
variant adds the 3 play-effect columns (player_hp/enemy_hp/card_draw) for
hand cards — information otherwise reachable only through the untrained
card-id embedding. e28c_s{0,1} are trained at the EXACT e28p recipe with
obs_mode=token-fx (`locma train-zoo --pointer-head --obs-mode token-fx
--learning-rate 1e-4 --target-kl 0.025 --n-envs 16`); this driver runs the
gate-2 protocol against the CURRENT reactive RoR pair (e28p, promoted
2026-07-19).

Dose caveat (pre-registered before the bench ran): the zoo training decks
are essentially blue-item-free (practicum census: 0 blue items in 33k
decisions' hands; 7/8 blue items carry effects, only 4/24 green do), so
this arm measures fx at a WEAK training dose. A null here reads "fx alone
at the item-light recipe is not enough", NOT "effect visibility is
worthless" — the follow-up arm is fx + item-rich training decks
(draft_override, E19 machinery).

Stages (idempotent via runs/e28c-bench-summary.json):
  B. full paired ruler: [ppo:e28c_sX,ldraft_sX] vs [ppo:e28p_sX,ldraft_sX],
     40x25 @ 45M anchors.
  C. fresh-anchor confirm @ 46M iff B ci_lo > 0.
  D. item-behavior read (mechanism): each net as its own practicum teacher.
  E. boardkeep guard-rail iff B+C pass.

Smoke: E28C_SMOKE=1 -> tiny grids, separate summary path.
"""

from __future__ import annotations

import json
import os
import time

SMOKE = os.environ.get("E28C_SMOKE") == "1"
WORKERS = 19
SEEDS = (0, 1)
SUMMARY_PATH = "runs/e28c-bench-smoke.json" if SMOKE else "runs/e28c-bench-summary.json"
FULL = (2, 2) if SMOKE else (40, 25)
PRIMARY_START = 45_000_000
CONFIRM_START = 46_000_000
ITEM_GAMES = 2 if SMOKE else 60
ITEM_SEED = 47_000_000
GUARD_GAMES = 20 if SMOKE else 1000
GUARD_SEED = 5_000_000

E28C = [f"runs/e28c_s{s}.zip" for s in SEEDS]
E28P = [f"depot:e28p/e28p_s{s}.zip" for s in SEEDS]
LDRAFT = [f"depot:ldraft/ldraft_s{s}.zip" for s in SEEDS]
CANDS = [f"ppo:{m},{ld}" for m, ld in zip(E28C, LDRAFT, strict=True)]
BASES = [f"ppo:{m},{ld}" for m, ld in zip(E28P, LDRAFT, strict=True)]

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
    log("=== E28c bench start ===")
    for m in E28C:
        if not os.path.exists(m):
            raise SystemExit(f"missing trained model {m} — run train-zoo --obs-mode token-fx first")

    full = verdict("full_vs_e28p", CANDS, BASES, FULL, start=PRIMARY_START)

    confirm = None
    if full["ci_lo"] > 0:
        confirm = verdict("confirm_vs_e28p_46M", CANDS, BASES, FULL, start=CONFIRM_START)
    else:
        record("confirm_skipped", "full ci_lo <= 0")

    for s in SEEDS:
        item_read(f"item_read_e28c_s{s}", CANDS[s])
    item_read("item_read_e28p_s0", BASES[0])

    fx_pass = bool(full["ci_lo"] > 0 and confirm is not None and confirm["ci_lo"] > 0)
    if fx_pass:
        for s in SEEDS:
            guardrail(f"guardrail_e28c_s{s}", CANDS[s])

    record(
        "gates",
        {
            "full_delta": full["mean_delta"],
            "full_ci": [full["ci_lo"], full["ci_hi"]],
            "confirm_ci": None if confirm is None else [confirm["ci_lo"], confirm["ci_hi"]],
            "fx_pass": fx_pass,
            "headroom": bool(full["mean_delta"] >= 0.03),
            "regression": bool(full["ci_hi"] < 0),
        },
    )
    log(f"gates: {summary['gates']}")


if __name__ == "__main__":
    main()
