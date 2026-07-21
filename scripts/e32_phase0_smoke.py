"""Phase 0 feasibility + cost smoke for the search re-baseline on depot:e29slim.

For each search policy (vbeam / rbeam / netdmcts) run a few mirrored games with
the e29slim trio as evaluator AND with the incumbent shared trio, matched
config, same seeds, vs a cheap greedy opponent. Reports: ran-without-crash,
win-rate-vs-greedy (sanity: search should crush greedy), s/game, and the
e29slim/shared cost ratio (the Phase-2 "cost-freed depth" multiplier).

The de-risk: netdmcts uses the POLICY head as PUCT priors — the one path not
yet exercised on a pointer/slim net.
"""

from __future__ import annotations

import time
import traceback

from locma.harness.match import run_match
from locma.policies.registry import make_policy

E29 = "depot:e29slim/e29slim_s0.zip|depot:e29slim/e29slim_s1.zip|depot:e29slim/e29slim_s2.zip"
SHARED = "depot:shared/shared_s0.zip|depot:shared/shared_s1.zip|depot:shared/shared_s2.zip"
LDRAFT = "depot:ldraft/ldraft_s0.zip"
OPP = "greedy"
GAMES = 4  # mirrored pairs -> 2x total
SEED = 30_000_000

# (label, net_ensemble, single_net) -> spec builders per policy
CONFIGS = {
    "vbeam(8,20)": lambda net, one: f"vbeam:{net},8,20,{LDRAFT}",
    "rbeam(8,20,4,4)": lambda net, one: f"rbeam:{net},8,20,4,4,{LDRAFT}",
    "netdmcts(1,320,1.5)": lambda net, one: f"netdmcts:1,320,1.5,{one},{LDRAFT}",
}
NETS = {
    "e29slim": (E29, "depot:e29slim/e29slim_s0.zip"),
    "shared": (SHARED, "depot:shared/shared_s0.zip"),
}


def smoke(spec: str) -> dict:
    try:
        pol = make_policy(spec)
        opp = make_policy(OPP)
        t0 = time.perf_counter()
        res = run_match(pol, opp, games=GAMES, seed=SEED)
        dt = time.perf_counter() - t0
        n = res.games
        return {
            "ok": True,
            "games": n,
            "wr_vs_greedy": round(res.win_rate_a, 3),
            "s_per_game": round(dt / n, 2),
        }
    except Exception as e:  # noqa: BLE001 — smoke wants the message, not a raise
        return {"ok": False, "err": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}


def main() -> None:
    rows = []
    for cfg_label, build in CONFIGS.items():
        cost = {}
        for net_label, (ens, one) in NETS.items():
            spec = build(ens, one)
            print(f"\n>>> {cfg_label} / {net_label}\n    {spec}", flush=True)
            r = smoke(spec)
            if r["ok"]:
                cost[net_label] = r["s_per_game"]
                print(
                    f"    OK  games={r['games']}  wr_vs_greedy={r['wr_vs_greedy']}"
                    f"  s/game={r['s_per_game']}",
                    flush=True,
                )
            else:
                print(f"    FAIL  {r['err']}", flush=True)
                print(r["tb"], flush=True)
            rows.append((cfg_label, net_label, r))
        if "e29slim" in cost and "shared" in cost and cost["shared"]:
            ratio = cost["e29slim"] / cost["shared"]
            print(f"    >> cost ratio e29slim/shared = {ratio:.2f}x", flush=True)

    print("\n================ PHASE 0 SUMMARY ================")
    for cfg_label, net_label, r in rows:
        status = (
            f"OK  wr={r['wr_vs_greedy']}  s/game={r['s_per_game']}"
            if r["ok"]
            else f"FAIL {r['err']}"
        )
        print(f"{cfg_label:22s} {net_label:9s} {status}")


if __name__ == "__main__":
    main()
