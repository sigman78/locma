"""E41 part B — how much search quality does the OPPONENT-reply beam width buy?

Pre-registered 2026-07-25, before any run.

Post-E41 batching, ``rbeam``'s binding cost is VIEW COUNT (~956 views/turn, ~93%
of them from the opponent-reply beams), not call count — 85% of views already
arrive in batches >=129 where us/item is at the floor, so no further batching
helps. The one lever that reduces view count is the reply beam's WIDTH.

It runs at ``width=8``, the same as our own turn, but the two are not the same
kind of search: our own-turn plan is the move we COMMIT to, while the reply is one
ply of an expectiminimax AVERAGE over sampled worlds. A cheaper reply may cost
almost nothing. E24's frontier swept ``n_plans`` x ``n_worlds`` and never touched
this.

Unlike everything else in E41 this CHANGES PLAY (verified: opp_width=2 altered 1
of 25 root plans), so it is gated on the RULER, not on byte-identity.

Two measurements per width, both needed — a speedup that costs strength is not a
win, and a strength change inside noise is not a regression:

  speed     ms/decision on a fixed set of real decision states (paired: the exact
            same states for every width, so the comparison is not confounded by
            which positions each arm happened to see).
  strength  win rate over the re-pinned primary ruler ``dmcts:15,150`` (2250 sims,
            matched ``ldraft``, CRN across widths). Reference: the trio at the
            default width scored 0.4525 pure / 0.4675 guarded on this rung.

Pre-registered read: adopt a narrower reply beam iff its ruler CI overlaps the
default's AND it is materially faster. A width whose CI sits below the default's
is a real regression regardless of speed.

    python scripts/e41_oppwidth_sweep.py --arm speed
    python scripts/e41_oppwidth_sweep.py --arm ruler --pairs 200 --workers 8
    python scripts/e41_oppwidth_sweep.py --smoke
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from locma.stats.intervals import wilson_ci

LDRAFT = "depot:ldraft/ldraft_s0.zip"
TRIO = "depot:e36/e36_gen7.zip|depot:e36m1/e36_m1_gen7.zip|depot:e36s22/e36_s22_gen7.zip"
GEN7 = "depot:e36/e36_gen7.zip"
DMCTS_HARD = f"dmcts:15,150,0,3,{LDRAFT}"

WIDTHS = (1, 2, 4, 6, 8)  # 8 == the default (opp_width=None); the control
SEED_SPEED = 91_000_000
SEED_RULER = 92_000_000

_CACHE: dict = {}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _rbeam_with_opp_width(net: str, opp_width: int | None):
    """The standard rbeam recipe, with the reply-beam width overridden.

    Built through the registry so the spec, evaluator wiring and draft half are
    exactly the recipe of record, then the knob is set on the battle half —
    ``opp_width`` is deliberately NOT a registry parameter until a value is
    promoted (a spec param would imply a supported config).
    """
    from locma.policies.registry import make_policy  # noqa: PLC0415

    pol = make_policy(f"rbeam:{net},8,20,4,4,{LDRAFT}")
    pol.battle.opp_width = opp_width
    return pol


# ---------------------------------------------------------------------------
# Arm 1 — speed, paired over identical decision states
# ---------------------------------------------------------------------------


def arm_speed(net: str, n_states: int, out: str) -> dict:
    from locma.core import battle as battlemod  # noqa: PLC0415
    from locma.core.engine import run_game  # noqa: PLC0415
    from locma.core.state import Phase  # noqa: PLC0415
    from locma.data.cards_db import load_cards  # noqa: PLC0415
    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.policies.mcts import _clone_battle  # noqa: PLC0415
    from locma.policies.rbeam import plan_turn_reply_aware  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415
    from locma.policies.vbeam import EnsembleValueEvaluator, NetValueEvaluator  # noqa: PLC0415

    paths = [resolve_path(p) for p in net.split("|")]
    ev = NetValueEvaluator(paths[0]) if len(paths) == 1 else EnsembleValueEvaluator(paths)
    cards = load_cards()

    states: list = []

    def hook(seat, action, gs):
        if (
            gs.phase == Phase.BATTLE
            and len(list(battlemod.battle_legal(gs))) >= 3
            and len(states) < n_states
        ):
            states.append(_clone_battle(gs))

    s = SEED_SPEED
    while len(states) < n_states:
        run_game(make_policy("greedy"), make_policy("scripted"), s, on_pre_step=hook)
        s += 1
    print(f"[speed] {len(states)} decision states, net={'trio' if '|' in net else 'single'}")

    rows: dict = {}
    plans_by_width: dict = {}
    for w in WIDTHS:
        ow = None if w == 8 else w
        t0 = time.perf_counter()
        plans = [
            plan_turn_reply_aware(
                st,
                ev,
                cards=cards,
                rng=random.Random(11),
                width=8,
                max_actions=20,
                n_plans=4,
                n_worlds=4,
                opp_width=ow,
            )
            for st in states
        ]
        el = time.perf_counter() - t0
        plans_by_width[w] = plans
        rows[w] = {"ms_per_decision": round(el / len(states) * 1000, 1), "seconds": round(el, 2)}
        print(f"  opp_width={w}: {rows[w]['ms_per_decision']:>7.1f} ms/decision", flush=True)

    base = plans_by_width[8]
    for w in WIDTHS:
        same = sum(a == b for a, b in zip(base, plans_by_width[w], strict=True))
        rows[w]["plans_same_as_default"] = f"{same}/{len(base)}"
        rows[w]["speedup_vs_default"] = round(
            rows[8]["ms_per_decision"] / rows[w]["ms_per_decision"], 3
        )

    payload = {
        "generated": utc_now(),
        "arm": "speed",
        "net": net,
        "n_states": len(states),
        "widths": rows,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2))
    print(f"\n{'opp_width':>10}{'ms/decision':>14}{'speedup':>10}{'plans==default':>17}")
    for w in WIDTHS:
        r = rows[w]
        print(
            f"{w:>10}{r['ms_per_decision']:>14.1f}{r['speedup_vs_default']:>10.2f}x"
            f"{r['plans_same_as_default']:>17}"
        )
    print(f"wrote {out}")
    return payload


# ---------------------------------------------------------------------------
# Arm 2 — strength on the re-pinned primary ruler
# ---------------------------------------------------------------------------


def _ruler_cell(net: str, opp_width, seed: int, pairs: int) -> tuple[int, int, float]:
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    key = (net, opp_width)
    if key not in _CACHE:
        _CACHE[key] = _rbeam_with_opp_width(net, opp_width)
    if DMCTS_HARD not in _CACHE:
        _CACHE[DMCTS_HARD] = make_policy(DMCTS_HARD)
    t0 = time.perf_counter()
    res = run_match(_CACHE[key], _CACHE[DMCTS_HARD], games=pairs, seed=seed)
    return res.wins_a, res.games, time.perf_counter() - t0


def _noop() -> None:
    try:
        from locma.harness.parallel import init_eval_worker  # noqa: PLC0415

        init_eval_worker()
    except Exception:  # noqa: BLE001
        pass


def arm_ruler(net: str, widths, pairs: int, block: int, workers: int, out: str) -> dict:
    units = []
    for w in widths:
        ow = None if w == 8 else w
        off = 0
        while off < pairs:
            n = min(block, pairs - off)
            # CRN: every width shares the same game seeds, so the comparison is paired.
            units.append((w, net, ow, SEED_RULER + off, n))
            off += n
    print(f"[ruler] widths={list(widths)} x {pairs} pairs = {len(units)} blocks on {workers}")

    agg = {w: [0, 0, 0.0] for w in widths}
    if workers > 1 and len(units) > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_noop) as ex:
            futs = {ex.submit(_ruler_cell, u[1], u[2], u[3], u[4]): (u[0], u[3]) for u in units}
            for f in as_completed(futs):
                w, seed = futs[f]
                wins, games, sec = f.result()
                agg[w][0] += wins
                agg[w][1] += games
                agg[w][2] += sec
                print(f"    [opp_width={w}] seed {seed}: {wins}/{games} ({sec:.0f}s)", flush=True)
    else:
        for u in units:
            wins, games, sec = _ruler_cell(u[1], u[2], u[3], u[4])
            agg[u[0]][0] += wins
            agg[u[0]][1] += games
            agg[u[0]][2] += sec

    rows = {}
    for w in widths:
        wins, games, sec = agg[w]
        lo, hi = wilson_ci(int(wins), int(games))
        rows[w] = {
            "net_wr_vs_dmcts_hard": round(wins / games, 4),
            "ci": [round(lo, 4), round(hi, 4)],
            "n": int(games),
            "s_per_game": round(sec / games, 2),
        }
    payload = {
        "generated": utc_now(),
        "arm": "ruler",
        "net": net,
        "ruler": DMCTS_HARD,
        "reference_default_width": {"pure_trio": 0.4525, "guarded_trio": 0.4675},
        "widths": rows,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2))

    print(f"\n{'opp_width':>10}{'WR vs dmcts:15,150':>20}{'CI':>20}{'s/game':>9}")
    for w in widths:
        r = rows[w]
        ci = "[{:.3f},{:.3f}]".format(r["ci"][0], r["ci"][1])
        print(f"{w:>10}{r['net_wr_vs_dmcts_hard']:>20.4f}{ci:>20}{r['s_per_game']:>9.2f}")
    print(f"\nwrote {out}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", default="speed", choices=("speed", "ruler"))
    ap.add_argument("--net", default=TRIO, help="evaluator net(s); default = the e36 trio RoR")
    ap.add_argument("--single", action="store_true", help="use the single gen7 net instead")
    ap.add_argument("--states", type=int, default=40, help="speed arm: decision states")
    ap.add_argument("--pairs", type=int, default=200, help="ruler arm: seed pairs/width")
    ap.add_argument("--block-pairs", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--widths", default="", help="comma subset of the swept widths")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    net = GEN7 if args.single else args.net
    widths = [int(x) for x in args.widths.split(",") if x.strip()] if args.widths else list(WIDTHS)
    out = args.out or f"runs/e41/oppwidth_{args.arm}.json"
    if args.arm == "speed":
        arm_speed(net, 4 if args.smoke else args.states, out)
    else:
        arm_ruler(
            net,
            widths,
            3 if args.smoke else args.pairs,
            args.block_pairs,
            1 if args.smoke else args.workers,
            out,
        )


if __name__ == "__main__":
    main()
