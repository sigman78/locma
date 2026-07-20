"""E29 slim-extractor bench: does the transformer-free net MATCH e28c cheaper?

The slim arm (SlimTokenExtractor) is a CHEAPNESS lever, not a strength lever:
E28b showed the transformer mixing is unnecessary for the pointer gather, so
dropping it (56.7k extractor params vs 418.5k, 7.4x fewer) should MATCH the
e28c recipe of record at materially less inference cost. So the decision rule
is inverted from a normal bench: success = NOT a regression (ci_hi >= 0), i.e.
slim is statistically indistinguishable from — or beats — e28c; a clear loss
(ci_hi < 0) kills it.

slim_s{0,1} = `train-zoo --pointer-head --obs-mode token-fx --slim-extractor`
at the EXACT e28c recipe (lr 1e-4, target_kl 0.025, n_envs 16, 5-phase zoo).

Stages (idempotent via runs/e29-slim-summary.json; needs runs/e29slim_s{0,1}):
  A. full paired ruler: [ppo:slim_sX,ldraft_sX] vs [ppo:depot:e28c_sX,ldraft_sX]
     40x25 @ 58M.
  B. confirm @ 59M (fresh) iff A ci_hi >= 0 (i.e. not a clear loss).
  C. compute: extractor param counts + wall-clock s/game (slim vs e28c vs
     greedy control) — the cheapness payoff.
  D. boardkeep guard-rail on slim (2000 mirrored @ 5M CRN, e28c band 0.25-0.30).

Anchor bookkeeping: 45-52M (e28c/e28d benches), 57M (e29 winrate) spent;
58M/59M fresh. Smoke: E29SLIM_SMOKE=1 -> tiny grids.
"""

from __future__ import annotations

import json
import os
import time

SMOKE = os.environ.get("E29SLIM_SMOKE") == "1"
WORKERS = 19
SEEDS = (0, 1)
SUMMARY_PATH = "runs/e29-slim-smoke.json" if SMOKE else "runs/e29-slim-summary.json"
FULL = (2, 2) if SMOKE else (40, 25)
PRIMARY_START = 58_000_000
CONFIRM_START = 59_000_000
GUARD_GAMES = 20 if SMOKE else 1000
GUARD_SEED = 5_000_000
COMPUTE_GAMES = 4 if SMOKE else 60
COMPUTE_SEED = 60_000_000

SLIM = [f"runs/e29slim_s{s}.zip" for s in SEEDS]
E28C = [f"depot:e28c/e28c_s{s}.zip" for s in SEEDS]
LDRAFT = [f"depot:ldraft/ldraft_s{s}.zip" for s in SEEDS]
CANDS = [f"ppo:{m},{ld}" for m, ld in zip(SLIM, LDRAFT, strict=True)]
BASES = [f"ppo:{m},{ld}" for m, ld in zip(E28C, LDRAFT, strict=True)]

summary: dict = {}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def record(key: str, value) -> None:
    summary[key] = value
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)


def verdict(tag: str, candidates, baselines, grid, start: int) -> dict:
    from locma.harness.ceiling_eval import (  # noqa: PLC0415
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


def compute_read() -> dict:
    """Extractor param counts + wall-clock s/game, slim vs e28c."""
    import time as _t  # noqa: PLC0415

    from locma.envs.extractor import SlimTokenExtractor, TokenSetExtractor  # noqa: PLC0415
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    if "compute" in summary:
        return summary["compute"]
    from locma.envs.battle_env import BattleEnv  # noqa: PLC0415

    space = BattleEnv(opponent=make_policy("greedy"), seed=0, obs_mode="token-fx").observation_space
    params = {
        "slim_extractor": sum(p.numel() for p in SlimTokenExtractor(space).parameters()),
        "full_extractor": sum(p.numel() for p in TokenSetExtractor(space).parameters()),
    }
    params["ratio"] = round(params["full_extractor"] / params["slim_extractor"], 2)

    sgame = {}
    for tag, spec in (("slim", CANDS[0]), ("e28c", BASES[0])):
        pol = make_policy(spec)
        # Warm one game so the timed loop excludes the first lazy model load
        # (both sides pay it identically, but warming tightens the ratio).
        run_match(pol, make_policy("greedy"), games=1, seed=COMPUTE_SEED - 1)
        t0 = _t.time()
        run_match(pol, make_policy("greedy"), games=COMPUTE_GAMES, seed=COMPUTE_SEED)
        sgame[tag] = round((_t.time() - t0) / (2 * COMPUTE_GAMES), 4)
    sgame["speedup"] = round(sgame["e28c"] / sgame["slim"], 2) if sgame["slim"] else None
    res = {"params": params, "s_per_game": sgame}
    record("compute", res)
    log(f"compute: {res}")
    return res


def guardrail(tag: str, spec: str) -> dict:
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    if tag in summary:
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
    log("=== E29 slim bench start ===")
    for m in SLIM:
        if not os.path.exists(m):
            raise SystemExit(f"missing {m} — train with train-zoo --slim-extractor first")

    full = verdict("full_vs_e28c", CANDS, BASES, FULL, start=PRIMARY_START)

    confirm = None
    not_a_loss = full["ci_hi"] >= 0
    if not_a_loss:
        confirm = verdict("confirm_vs_e28c_59M", CANDS, BASES, FULL, start=CONFIRM_START)
    else:
        record("confirm_skipped", "full ci_hi < 0 (clear regression)")

    compute_read()
    for s in SEEDS:
        guardrail(f"guardrail_slim_s{s}", CANDS[s])

    matches = bool(not_a_loss and confirm is not None and confirm["ci_hi"] >= 0)
    record(
        "gates",
        {
            "full_delta": full["mean_delta"],
            "full_ci": [full["ci_lo"], full["ci_hi"]],
            "confirm_ci": None if confirm is None else [confirm["ci_lo"], confirm["ci_hi"]],
            "matches_e28c": matches,  # success = statistically not-worse, both anchors
            "regression": bool(full["ci_hi"] < 0),
            "param_ratio": summary["compute"]["params"]["ratio"],
            "speedup": summary["compute"]["s_per_game"]["speedup"],
        },
    )
    log(f"gates: {summary['gates']}")


if __name__ == "__main__":
    main()
