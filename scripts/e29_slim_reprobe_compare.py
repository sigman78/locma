"""E29 slim re-probe: compare concept retention down the towers across nets.

Reads the per-model JSONs written by ``e27_concept_probe.py probe`` and prints,
for each concept of interest, the probe metric (AUC for binary, R2 for
continuous) at each layer in forward order plus the raw->min-tower RETENTION
DROP. The question: did dropping the transformer (e28c TokenSetExtractor ->
e29slim SlimTokenExtractor) clean up the can_item representation that the E27
prestudy found collapsing down the towers (0.98 raw -> 0.61)?

Usage:
  python scripts/e29_slim_reprobe_compare.py \
      runs/netprobe/e27_e28c_s0.json runs/netprobe/e27_e29slim_s0.json \
      [runs/netprobe/e27_b0k_s0.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# forward-order layers; nets miss some (b0k/e28c have "slots", slim does not).
LAYER_ORDER = ["raw", "slots", "features", "pi_a1", "pi_a2", "vf_a1", "vf_a2", "logits"]
TOWER_LAYERS = ["pi_a1", "pi_a2", "vf_a1", "vf_a2"]
CONCEPTS = ["can_item", "item_now", "winner_side", "lethal_now"]


def _metric(entry: dict, lname: str):
    layer = entry["layers"].get(lname)
    if layer is None:
        return None
    return layer.get("auc", layer.get("r2"))


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit(__doc__)
    reports = {p.stem.replace("e27_", ""): json.loads(p.read_text()) for p in paths}

    for concept in CONCEPTS:
        print(f"\n=== {concept} ===")
        present = [
            ln
            for ln in LAYER_ORDER
            if any(_metric(r["concepts"][concept], ln) is not None for r in reports.values())
        ]
        header = "net".ljust(14) + "".join(ln.rjust(9) for ln in present) + "   raw->minTower"
        print(header)
        print("-" * len(header))
        for name, rep in reports.items():
            e = rep["concepts"][concept]
            row = name.ljust(14)
            for ln in present:
                v = _metric(e, ln)
                row += "  n/a".rjust(9) if v is None else f"{v:>9.3f}"
            raw = _metric(e, "raw")
            towers = [_metric(e, ln) for ln in TOWER_LAYERS if _metric(e, ln) is not None]
            drop = (raw - min(towers)) if (raw is not None and towers) else None
            row += "   " + ("n/a" if drop is None else f"{drop:>+.3f}")
            print(row)
        # base rate for context
        brs = {name: rep["concepts"][concept].get("base_rate") for name, rep in reports.items()}
        if any(v is not None for v in brs.values()):
            parts = "  ".join(f"{n}={v:.3f}" for n, v in brs.items() if v is not None)
            print("base_rate: " + parts)


if __name__ == "__main__":
    main()
