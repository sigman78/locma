"""Throughput benchmark: turns/sec (and games/sec) for every battle policy in
the current registry, including a few config points for the parameterized
searchers (mcts/dmcts/netdmcts iteration counts, vbeam beam width) since their
wall-clock cost is dose-dependent, not a single number.

Each policy under test plays as BOTH seats against a fixed cheap opponent
(`greedy`, near-instant) so wall-clock time is dominated by the policy under
test, not the opponent -- this isolates per-policy decision cost rather than
measuring a specific matchup's combined cost. Game counts are chosen per
policy (not uniform) to keep expensive cells (mcts:20000, dmcts:15,2000)
finishing in a couple of minutes rather than the hour+ an accuracy-grade
run would take (see worklog "E22" for how wrong a stale throughput guess
can be) -- this is a THROUGHPUT reading, not a win-rate result.

Run: F:/WorkDir/locma/.venv/Scripts/python.exe scripts/bench_policies.py
"""

from __future__ import annotations

import json
import math
import time

from locma.core.engine import run_game
from locma.policies.registry import make_policy

OPP = "greedy"  # cheap, near-instant fixed opponent

SHARED = [f"depot:shared/shared_s{s}.zip" for s in (0, 1, 2)]
LDRAFT0 = "depot:ldraft/ldraft_s0.zip"
B0K0 = "depot:b0k/b0k_s0.zip"
ENS = "vbeam:" + "|".join(SHARED) + f",8,20,{LDRAFT0}"

# (spec, games) -- games sized so slow cells stay in the minutes range.
CELLS: list[tuple[str, int]] = [
    ("random", 200),
    ("scripted", 200),
    ("greedy", 200),
    ("max-guard", 200),
    ("max-attack", 200),
    ("boardkeep", 200),
    ("shell", 200),
    ("guardwall", 200),
    ("bufface", 200),
    ("rnddeck", 200),
    ("azlite:100", 20),
    ("azlite:1000", 5),
    ("mcts:100", 20),
    ("mcts:1000", 10),
    ("mcts:5000", 5),
    ("mcts:20000", 3),
    ("dmcts:15,30", 10),
    ("dmcts:15,100", 5),
    ("dmcts:15,500", 3),
    ("dmcts:15,2000", 2),
    (f"netdmcts:8,40,1.5,{B0K0}", 10),
    (f"netdmcts:8,160,1.5,{B0K0}", 3),
    (f"ppo:{B0K0}", 50),
    (f"vbeam:{B0K0},8,20", 20),
    (ENS, 10),
]

results: list[dict] = []

for spec, games in CELLS:
    under_test = make_policy(spec)
    opp = make_policy(OPP)
    total_turns = 0
    t0 = time.time()
    for g in range(games):
        # alternate seats so both A-as-p0 and A-as-p1 decision paths are timed
        r = run_game(under_test, opp, seed=g) if g % 2 == 0 else run_game(opp, under_test, seed=g)
        total_turns += r.turns
    dt = time.time() - t0
    tps = total_turns / dt if dt > 0 else math.inf
    gps = games / dt if dt > 0 else math.inf
    row = {
        "spec": spec,
        "games": games,
        "turns": total_turns,
        "seconds": round(dt, 3),
        "turns_per_sec": round(tps, 2),
        "games_per_sec": round(gps, 4),
        "sec_per_game": round(dt / games, 4) if games else None,
    }
    results.append(row)
    print(
        f"{spec:<40} games={games:<4} turns={total_turns:<6} "
        f"{dt:>8.2f}s  {tps:>8.2f} turns/s  {dt / games:>8.3f} s/game",
        flush=True,
    )

with open("runs/bench-policies.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
