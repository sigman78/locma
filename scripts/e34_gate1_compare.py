"""E34 Gate 1: did board-potential shaping move trade behavior toward lookahead?

Reads E33-harness summaries for the shaped and control nets (trained at matched
budget/seed, the ONLY difference is --board-potential-weight) and prints the
behavioral deltas the shaping targets. The oracle (vbeam over the e29slim trio)
is the fixed target both are compared against; the shaped-minus-control delta is
the clean isolation of the shaping effect.

Gate 1 PASSES if the shaped net moves toward the oracle relative to control on:
higher trade frequency, lower face-share, higher per-trade dphi / favorable rate.

Usage:
  python scripts/e34_gate1_compare.py runs/e33-e34control.json runs/e33-e34shaped.json \
      [runs/e33-lppo.json]   # optional e29slim RoR reference
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(p):
    d = json.loads(Path(p).read_text())
    u = d["card_usage_per_turn"]
    te = d["trade_efficiency"]["subject"] or {}
    plan = d["trade_efficiency"]["plan"] or {}
    return {
        "name": Path(p).stem,
        "trades_per_turn": u["A_trade"]["subject"],
        "face_per_turn": u["A_face"]["subject"],
        "face_share": d["attack_face_share"]["subject"],
        "mean_dphi": te.get("mean_dphi"),
        "favorable": te.get("favorable"),
        "unfavorable": te.get("unfavorable"),
        "overkill": te.get("mean_overkill"),
        "item_rate": d["item_play_rate"]["subject"],
        # oracle targets (same in every run up to sampling)
        "T_trades": u["A_trade"]["plan"],
        "T_face_share": d["attack_face_share"]["plan"],
        "T_dphi": plan.get("mean_dphi"),
        "T_favorable": plan.get("favorable"),
    }


def main() -> None:
    paths = sys.argv[1:]
    if len(paths) < 2:
        raise SystemExit(__doc__)
    rows = [load(p) for p in paths]
    ctrl, shaped = rows[0], rows[1]
    tgt = shaped  # oracle targets carried on each row

    cols = [
        "trades_per_turn",
        "face_share",
        "mean_dphi",
        "favorable",
        "unfavorable",
        "overkill",
        "item_rate",
    ]
    print(f"{'metric':16s}" + "".join(f"{r['name'][:14]:>16s}" for r in rows) + f"{'ORACLE':>12s}")
    print("-" * (16 + 16 * len(rows) + 12))
    oracle = {
        "trades_per_turn": tgt["T_trades"],
        "face_share": tgt["T_face_share"],
        "mean_dphi": tgt["T_dphi"],
        "favorable": tgt["T_favorable"],
        "unfavorable": None,
        "overkill": None,
        "item_rate": None,
    }
    for c in cols:
        line = f"{c:16s}" + "".join(
            f"{(r[c] if r[c] is not None else float('nan')):>16.3f}" for r in rows
        )
        o = oracle.get(c)
        line += f"{o:>12.3f}" if o is not None else f"{'-':>12s}"
        print(line)

    print("\n--- Gate 1: shaped - control (toward-oracle direction in parens) ---")
    checks = [
        ("trades_per_turn", +1, "more trading"),
        ("face_share", -1, "less face-greed"),
        ("mean_dphi", +1, "better trades"),
        ("favorable", +1, "more favorable"),
        ("unfavorable", -1, "fewer bad trades"),
    ]
    passed = 0
    for c, want, label in checks:
        if shaped[c] is None or ctrl[c] is None:
            continue
        delta = shaped[c] - ctrl[c]
        good = (delta > 0) == (want > 0)
        passed += good
        arrow = "toward" if good else "AWAY"
        print(f"  {c:16s} {delta:+.3f}  ({label}) -> {arrow} oracle")
    print(
        f"\n  {passed}/{len(checks)} metrics move toward the oracle. "
        f"{'GATE 1 SUPPORTED' if passed >= 4 else 'GATE 1 WEAK/FAILED'}"
    )


if __name__ == "__main__":
    main()
