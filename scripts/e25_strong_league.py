"""E25 strong-opponent league: verify the promoted rbeam recipe against strong search.

The HARD3 pool saturates (rbeam 0.983, planner 0.978 — both near 1.0), so it no
longer discriminates. This league pits the rbeam-4x4 recipe of record against a
panel of STRONG opponents at matched ``ldraft`` (the E22-E24 convention that
isolates battle search from deck quality), mirrored, with per-opponent Wilson CIs
and a FAIR-league mean:

  - vbeam planner ensemble   (fair; the planner recipe of record)
  - dmcts:15,100             (fair MCTS, 1500 sims; beat the planner in E22)
  - azlite:100               (CHEATS: perfect foresight over future draws) --
                              reported separately as an unfair upper reference,
                              NOT folded into the fair-league mean.

Two stages (pilot then confirm at a fresh seed), resumable per opponent: results
are checkpointed to ``runs/e25-strong-league.json`` after every cell, keyed by
(stage, opponent), so a crash resumes without redoing finished cells.

Run:
    uv run --extra ml python scripts/e25_strong_league.py --stage pilot
    uv run --extra ml python scripts/e25_strong_league.py --stage confirm --only "dmcts:15,100"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from locma.harness.match import run_match
from locma.policies.registry import make_policy
from locma.stats.intervals import wilson_ci

SHARED = "depot:shared/shared_s0.zip|depot:shared/shared_s1.zip|depot:shared/shared_s2.zip"
LDRAFT = "depot:ldraft/ldraft_s0.zip"

# The promoted play-time-search recipe of record (rbeam 4x4).
CANDIDATE = f"rbeam:{SHARED},8,20,4,4,{LDRAFT}"

# (label, spec, fair). ``fair`` opponents are averaged into the league mean;
# azlite cheats (perfect foresight), so it is an upper-reference row only.
OPPONENTS: list[tuple[str, str, bool]] = [
    ("vbeam", f"vbeam:{SHARED},8,20,{LDRAFT}", True),
    ("dmcts:15,100", f"dmcts:15,100,0,3,{LDRAFT}", True),
    ("azlite:100", f"azlite:100,1.5,0,0,{LDRAFT}", False),
]
# netdmcts:1,320 (prior search RoR) is intentionally NOT in the league: already
# confirmed vs rbeam at 0.548 (E24, 500 games), and at ~17 s/game it would roughly
# double this run for a known result. Cite that number alongside this table.

PILOT_SEED0 = 32_000_000
CONFIRM_SEED0 = 33_000_000
OUT = Path("runs/e25-strong-league.json")


def _run_cell(cand, spec: str, games: int, seed: int) -> dict:
    opp = make_policy(spec)
    t0 = time.time()
    res = run_match(cand, opp, games=games, seed=seed)
    dt = time.time() - t0
    n = res.wins_a + res.wins_b
    lo, hi = wilson_ci(res.wins_a, n)
    return {
        "spec": spec,
        "wins": res.wins_a,
        "n": n,
        "win_rate": res.win_rate_a,
        "ci": [lo, hi],
        "sec_per_game": dt / n,
        "beats_50": lo > 0.5,
        "loses_50": hi < 0.5,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("pilot", "confirm"), default="pilot")
    ap.add_argument("--games", type=int, default=None, help="mirrored pairs/opp (2N games)")
    ap.add_argument(
        "--only",
        default="",
        help="';'-separated opponent labels to (re)run; labels may contain commas "
        "(e.g. --only 'vbeam;dmcts:15,100')",
    )
    args = ap.parse_args()

    games = args.games if args.games is not None else (100 if args.stage == "pilot" else 250)
    seed0 = PILOT_SEED0 if args.stage == "pilot" else CONFIRM_SEED0
    only = {s for s in args.only.split(";") if s}

    state = json.loads(OUT.read_text()) if OUT.exists() else {"candidate": CANDIDATE, "stages": {}}
    stage = state["stages"].setdefault(args.stage, {})

    cand = make_policy(CANDIDATE)
    for oi, (label, spec, fair) in enumerate(OPPONENTS):
        if only and label not in only:
            continue
        if label in stage and not (only and label in only):
            print(f"[skip] {args.stage}/{label} already done", flush=True)
            continue
        cell = _run_cell(cand, spec, games, seed0 + oi * 1_000_000)
        cell["fair"] = fair
        stage[label] = cell
        tag = "" if fair else "  (CHEATS: perfect foresight)"
        verdict = "rbeam ahead" if cell["beats_50"] else "opp ahead" if cell["loses_50"] else "wash"
        print(
            f"[done] {args.stage}/{label}: {cell['win_rate']:.3f} "
            f"[{cell['ci'][0]:.3f},{cell['ci'][1]:.3f}] {cell['wins']}/{cell['n']} "
            f"({verdict}) @ {cell['sec_per_game']:.2f}s/game{tag}",
            flush=True,
        )
        OUT.write_text(json.dumps(state, indent=2))

    fair = [c for c in stage.values() if c.get("fair")]
    if fair:
        stage["_fair_league_mean"] = sum(c["win_rate"] for c in fair) / len(fair)
        OUT.write_text(json.dumps(state, indent=2))
        print(
            f"\n{args.stage} fair-league mean = {stage['_fair_league_mean']:.4f} "
            f"(over {len(fair)} fair opponents)",
            flush=True,
        )


if __name__ == "__main__":
    main()
