"""E42 Gate 1: matched-harness training readout (P1/P3/P3b/P4).

Pre-registered 2026-09-25, docs/e42-public-history-plan.md "Gate 1 — matched-
harness training". Two arms warm from ``depot:e36/e36_gen7.zip`` under the
identical PFSP regime (``scripts/e36_pfsp.py --start-gen 8 --generations 2
--steps 1000000 --n-envs 6``, seed base 27_000_000):

  e42h — ``token-fxh`` obs (fx tokens + the public-history ``hist`` branch),
         widened warm start.
  e42c — ``token-fx`` obs (control), same warm-start path (fresh optimizer),
         byte-identical pool.

This script runs the reactive-vs-reactive cells (P1, P3, the extra P3b
self-ablation, P4). P2 (the dmcts hard-rung sensitivity check) and G (the
boardkeep exploit guard) are NOT run here — see the exact commands below.

Cells:
  P1  (primary)  ppo:H,ldraft            vs ppo:C,ldraft            1000 pairs, seed 80_000_000
  P3  (mechanism) ppo:H,ldraft,nohist    vs ppo:C,ldraft             500 pairs, seed 81_000_000
  P3b (extra)     ppo:H,ldraft           vs ppo:H,ldraft,nohist      500 pairs, seed 81_500_000
  P4  (sanity)    avg-hard3 (scripted, max-guard, max-attack) for H and C,
                  150 games each (75 pairs/opp), seed 83_000_000 + opp_idx*10_000,
                  common random numbers across H and C.

Verdict words (glossary, applied to the absolute win rate against the 0.50
H2H baseline): positive = Wilson CI excludes 0.50 upward (ci_lo > 0.50);
headroom = positive AND mean win rate >= 0.53; null = CI straddles 0.50;
negative = CI entirely below 0.50 (ci_hi < 0.50).

Outcome synthesis follows the plan's table:
  P1 positive AND P3 loses the gain (null/negative)  -> public-history memory
      IS a lever; escalate (3-seed promotion track, token-level history, LSTM retest).
  P1 positive but P3 KEEPS the gain                  -> branch capacity / optimizer
      restart, not information; NULL for the hypothesis, do not promote.
  P1 null                                             -> check Gate 0: if it PASSED,
      one scale check (2 more gens both arms) before closing; else close (memory
      over public history is not a lever at this regime; LSTM retest is dead).
  P1 negative                                         -> branch/shift tax; close.
  (always flagged) if P4's avg-hard3 delta is comparable in size to P1's edge,
      report as a generic obs-completion effect, not a memory-specific result.

Usage:
    .venv/Scripts/python scripts/e42_gate1.py \\
        --h runs/e36_e42h_gen9.zip --c runs/e36_e42c_gen9.zip \\
        --workers 16
    .venv/Scripts/python scripts/e42_gate1.py --smoke

Out: runs/e42/gate1.json (runs/e42/gate1_smoke.json for --smoke),
log runs/e42/gate1.log (gate1_smoke.log for --smoke).

--- P2 (run separately, via the ladder's --nets + this module's new --extra-net
    flag on scripts/e36_dmcts_ladder.py — the hard rung, gen7 anchor + both arms,
    200 pairs each, CRN across nets within a rung):

    .venv/Scripts/python scripts/e36_dmcts_ladder.py --rungs hard_2250sim \\
        --nets e36_gen7,e42h_gen9,e42c_gen9 \\
        --extra-net e42h_gen9=ppo:runs/e36_e42h_gen9.zip,depot:ldraft/ldraft_s0.zip \\
        --extra-net e42c_gen9=ppo:runs/e36_e42c_gen9.zip,depot:ldraft/ldraft_s0.zip \\
        --pairs 200 --workers 16 --out runs/e42/p2.json

--- G (boardkeep exploit guard, promotion-track guard-rail only, run separately
    via scripts/e36_exploit.py). That script hardcodes its DEFENDERS/CANDIDATES
    dict rather than taking a CLI net override, so gating e42h_gen9 needs one
    additive edit there first (out of this script's ownership) — add

        "e42h_gen9": f"ppo:runs/e36_e42h_gen9.zip,{LDRAFT}",

    to DEFENDERS and "e42h_gen9" to CANDIDATES in scripts/e36_exploit.py, then run:

    .venv/Scripts/python scripts/e36_exploit.py
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from locma.stats.intervals import wilson_ci

LDRAFT = "depot:ldraft/ldraft_s0.zip"

DEFAULT_H = "runs/e36_e42h_gen9.zip"
DEFAULT_C = "runs/e36_e42c_gen9.zip"
SMOKE_CKPT = "depot:e36/e36_gen7.zip"  # both arms, for --smoke (needs two real checkpoints)

HARD3_OPPS = ("scripted", "max-guard", "max-attack")

P1_SEED = 80_000_000
P3_SEED = 81_000_000
P3B_SEED = 81_500_000
P4_SEED = 83_000_000

_CACHE: dict = {}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def ppo_spec(path: str, nohist: bool = False) -> str:
    """``ppo:PATH,LDRAFT[,nohist]`` — the trailing ``nohist`` flag zeroes the
    hist vector at play time (the E42 mechanism ablation instrument)."""
    spec = f"ppo:{path},{LDRAFT}"
    if nohist:
        spec += ",nohist"
    return spec


def _noop() -> None:
    try:
        from locma.harness.parallel import init_eval_worker  # noqa: PLC0415

        init_eval_worker()
    except Exception:  # noqa: BLE001
        pass


def _cell(a_spec: str, b_spec: str, seed: int, pairs: int) -> tuple[int, int]:
    """Picklable pool unit: one seed block of run_match(a, b) -> (a_wins, games)."""
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    for spec in (a_spec, b_spec):
        if spec not in _CACHE:
            _CACHE[spec] = make_policy(spec)
    res = run_match(_CACHE[a_spec], _CACHE[b_spec], games=pairs, seed=seed)
    return res.wins_a, res.games


def verdict(wr: float, ci: list[float]) -> str:
    """Glossary verdict for an absolute win rate against the 0.50 H2H baseline."""
    if ci[1] < 0.50:
        return "negative"
    if ci[0] > 0.50:
        return "headroom" if wr >= 0.53 else "positive"
    return "null"


def run_h2h(
    label: str, a_spec: str, b_spec: str, seed0: int, pairs: int, block: int, workers: int
) -> dict:
    """Blocked H2H over seed pairs: reports a's win rate (a_wr) with Wilson CI."""
    units = []
    off = 0
    while off < pairs:
        n = min(block, pairs - off)
        units.append((seed0 + off, n))
        off += n
    print(f"  [{label}] {len(units)} blocks x up to {block} pairs on {workers} workers", flush=True)

    wins = games = 0
    if workers > 1 and len(units) > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_noop) as ex:
            futs = {ex.submit(_cell, a_spec, b_spec, s, n): (s, n) for s, n in units}
            for f in as_completed(futs):
                s, n = futs[f]
                w, g = f.result()
                wins += w
                games += g
                print(f"    [{label}] seed {s}: {w}/{g}", flush=True)
    else:
        for s, n in units:
            w, g = _cell(a_spec, b_spec, s, n)
            wins += w
            games += g

    wr = wins / games if games else 0.0
    lo, hi = wilson_ci(wins, games)
    ci = [round(lo, 4), round(hi, 4)]
    return {
        "a": a_spec,
        "b": b_spec,
        "wins": wins,
        "games": games,
        "wr": round(wr, 4),
        "ci": ci,
        "verdict": verdict(wr, ci),
    }


def run_p4(nets: dict[str, str], games: int, workers: int) -> dict:
    """avg-hard3 for each net in ``nets``, CRN across nets within each opponent
    (same seed for every net vs a given opponent -> a paired H-vs-C read)."""
    units = []  # (net_label, net_spec, opp, seed, games)
    for i, opp in enumerate(HARD3_OPPS):
        seed = P4_SEED + i * 10_000
        for label, spec in nets.items():
            units.append((label, spec, opp, seed, games))
    print(f"  [hard3] {len(units)} matches ({games} pairs each) on {workers} workers", flush=True)

    per: dict[str, dict[str, float]] = {label: {} for label in nets}
    if workers > 1 and len(units) > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_noop) as ex:
            futs = {ex.submit(_cell, u[1], u[2], u[3], u[4]): (u[0], u[2], u[3]) for u in units}
            for f in as_completed(futs):
                label, opp, seed = futs[f]
                w, g = f.result()
                per[label][opp] = w / g if g else 0.0
                print(f"    [hard3] {label:10s} vs {opp:10s} seed {seed}: {w}/{g}", flush=True)
    else:
        for label, spec, opp, seed, n in units:
            w, g = _cell(spec, opp, seed, n)
            per[label][opp] = w / g if g else 0.0

    out = {}
    for label in nets:
        rates = [per[label][o] for o in HARD3_OPPS]
        out[label] = {
            "avg_hard3": round(sum(rates) / len(rates), 4),
            "per_opp": {o: round(per[label][o], 4) for o in HARD3_OPPS},
        }
    return out


def synthesize_outcome(p1: dict, p3: dict, p4_delta: float) -> str:
    p1v, p3v = p1["verdict"], p3["verdict"]
    p1_edge = p1["wr"] - 0.50
    note = ""
    if p1_edge != 0 and abs(p4_delta) >= 0.5 * abs(p1_edge):
        note = (
            f" [CAVEAT: P4 avg-hard3 delta ({p4_delta:+.4f}) is comparable in size "
            f"to P1's edge ({p1_edge:+.4f}) -- if this holds, report as a generic "
            f"obs-completion effect, not memory-specific.]"
        )
    if p1v == "negative":
        return "P1 NEGATIVE -> branch/shift tax; close." + note
    if p1v in ("positive", "headroom"):
        if p3v in ("null", "negative"):
            return (
                "P1 positive AND P3 loses the gain (nohist arm ~parity/negative vs "
                "control) -> PUBLIC-HISTORY MEMORY IS A LEVER. Escalate: 3-seed "
                "promotion track with guards; token-level history variant; the "
                "LSTM retest (learned memory vs hand features) is justified." + note
            )
        return (
            "P1 positive but P3 KEEPS the gain -> the gain is branch capacity / "
            "optimizer restart, not information. NULL for the hypothesis; do not "
            "promote." + note
        )
    return (
        "P1 NULL -> check Gate 0: if Gate 0 PASSED, run one scale check (2 more "
        "generations both arms) before closing; if still null, memory over public "
        "history is not a reactive-net lever at this regime, and the LSTM retest "
        "is dead (its information set is a subset of this one)." + note
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--h", default=DEFAULT_H, help="e42h arm checkpoint (token-fxh)")
    ap.add_argument("--c", default=DEFAULT_C, help="e42c arm checkpoint (token-fx control)")
    ap.add_argument("--pairs", type=int, default=1000, help="P1 seed pairs (n = 2*pairs games)")
    ap.add_argument("--p3-pairs", type=int, default=500)
    ap.add_argument("--p3b-pairs", type=int, default=500)
    ap.add_argument("--p4-games", type=int, default=75, help="pairs per hard3 opponent (n=2*pairs)")
    ap.add_argument("--block-pairs", type=int, default=50)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=f"2 pairs/games per cell, serial, both --h/--c overridden to {SMOKE_CKPT}",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    smoke = args.smoke
    pairs = 2 if smoke else args.pairs
    p3_pairs = 2 if smoke else args.p3_pairs
    p3b_pairs = 2 if smoke else args.p3b_pairs
    p4_games = 2 if smoke else args.p4_games
    block = args.block_pairs
    workers = 1 if smoke else args.workers
    out = args.out or ("runs/e42/gate1_smoke.json" if smoke else "runs/e42/gate1.json")

    h_ckpt, c_ckpt = args.h, args.c
    if smoke:
        h_ckpt = c_ckpt = SMOKE_CKPT

    print(f"E42 Gate 1 — {utc_now()}  h={h_ckpt}  c={c_ckpt}  smoke={smoke}")
    if smoke:
        print(
            f"  smoke: --h/--c overridden to {SMOKE_CKPT} for both arms "
            "(need two existing checkpoints); nohist must be a harmless no-op on it"
        )

    H = ppo_spec(h_ckpt)
    C = ppo_spec(c_ckpt)
    H_NOHIST = ppo_spec(h_ckpt, nohist=True)

    t0 = time.perf_counter()

    print("\n--- P1 (primary): H vs C, matched ldraft ---")
    p1 = run_h2h("P1", H, C, P1_SEED, pairs, block, workers)

    print("\n--- P3 (mechanism): H,nohist vs C ---")
    p3 = run_h2h("P3", H_NOHIST, C, P3_SEED, p3_pairs, block, workers)

    print("\n--- P3b (extra, self-ablation): H vs H,nohist ---")
    p3b = run_h2h("P3b", H, H_NOHIST, P3B_SEED, p3b_pairs, block, workers)

    print("\n--- P4 (sanity): avg-hard3, H and C, CRN ---")
    p4 = run_p4({"h": H, "c": C}, p4_games, workers)
    p4_delta = p4["h"]["avg_hard3"] - p4["c"]["avg_hard3"]

    outcome = synthesize_outcome(p1, p3, p4_delta)

    payload = {
        "generated": utc_now(),
        "checkpoints": {"h": h_ckpt, "c": c_ckpt},
        "smoke": smoke,
        "seeds": {"p1": P1_SEED, "p3": P3_SEED, "p3b": P3B_SEED, "p4": P4_SEED},
        "p1": p1,
        "p3": p3,
        "p3b": p3b,
        "p4": {**p4, "delta_avg_hard3_h_minus_c": round(p4_delta, 4)},
        "outcome": outcome,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2))

    print("\n============ E42 GATE 1 SUMMARY ============")
    print(f"{'cell':6s} {'a_wr':>8s} {'ci_lo':>8s} {'ci_hi':>8s} {'n':>6s}  verdict")
    for label, cell in (("P1", p1), ("P3", p3), ("P3b", p3b)):
        print(
            f"{label:6s} {cell['wr']:8.4f} {cell['ci'][0]:8.4f} {cell['ci'][1]:8.4f} "
            f"{cell['games']:6d}  {cell['verdict']}"
        )
    print(
        f"\nP4 avg-hard3   h={p4['h']['avg_hard3']:.4f}  c={p4['c']['avg_hard3']:.4f}  "
        f"delta={p4_delta:+.4f}"
    )
    for label in ("h", "c"):
        print(f"  {label}: {json.dumps(p4[label]['per_opp'])}")
    print(f"\nOUTCOME: {outcome}")
    print(f"\nwrote {out}  ({payload['seconds']}s)")


if __name__ == "__main__":
    main()
