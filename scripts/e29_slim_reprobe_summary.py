"""E29 slim re-probe: seed-averaged concept retention, e28c vs e29slim.

Aggregates the per-seed JSONs from ``e27_concept_probe.py probe`` (run on the
SAME token-fx concept data, so the states are identical across nets) into a
mean-over-seeds table and a summary JSON. Answers: did dropping the transformer
(TokenSetExtractor -> SlimTokenExtractor) preserve the can_item signal that the
E27 prestudy found collapsing down the towers?

The headline reads are ``features`` (the extractor output = tower input) and the
min over the four tower layers (worst-case retention). can_item = "an item play
is legal" (presence); item_now = "the teacher plays an item here" (choice).

Usage:
  python scripts/e29_slim_reprobe_summary.py \
      --net e28c runs/netprobe/e27_e28c_s0.json runs/netprobe/e27_e28c_s1.json ... \
      --net e29slim runs/netprobe/e27_e29slim_s0.json ...
  (or rely on the default glob over runs/netprobe/e27_<net>_s*.json)
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

TOWER_LAYERS = ["pi_a1", "pi_a2", "vf_a1", "vf_a2"]
REPORT_LAYERS = ["raw", "features", "vf_a1", "vf_a2"]
CONCEPTS = ["can_item", "item_now", "winner_side", "lethal_now"]


def _metric(entry: dict, lname: str):
    layer = entry["layers"].get(lname)
    return None if layer is None else layer.get("auc", layer.get("r2"))


def _seed_files(net: str) -> list[str]:
    return sorted(glob.glob(f"runs/netprobe/e27_{net}_s*.json"))


def _agg_net(paths: list[str]) -> dict:
    reps = [json.loads(Path(p).read_text()) for p in paths]
    out: dict = {"seeds": len(reps), "concepts": {}}
    for concept in CONCEPTS:
        entries = [r["concepts"][concept] for r in reps]
        layers: dict = {}
        all_layers = {ln for e in entries for ln in e["layers"]}
        for ln in all_layers:
            vals = [_metric(e, ln) for e in entries if _metric(e, ln) is not None]
            if vals:
                layers[ln] = {"mean": statistics.fmean(vals), "n": len(vals)}
        # min-over-towers per seed, then mean of those minima
        seed_min = []
        for e in entries:
            tv = [_metric(e, ln) for ln in TOWER_LAYERS if _metric(e, ln) is not None]
            if tv:
                seed_min.append(min(tv))
        raw_mean = layers.get("raw", {}).get("mean")
        min_tower_mean = statistics.fmean(seed_min) if seed_min else None
        out["concepts"][concept] = {
            "layers": layers,
            "min_tower_mean": min_tower_mean,
            "raw_to_min_drop": (
                None if (raw_mean is None or min_tower_mean is None) else raw_mean - min_tower_mean
            ),
            "base_rate": statistics.fmean(
                [e["base_rate"] for e in entries if e.get("base_rate") is not None]
            )
            if any(e.get("base_rate") is not None for e in entries)
            else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", action="append", nargs="+", metavar=("NAME", "JSON"), default=None)
    ap.add_argument("--out", default="runs/netprobe/e29_reprobe_summary.json")
    args = ap.parse_args()

    if args.net:
        nets = {grp[0]: (grp[1:] if len(grp) > 1 else _seed_files(grp[0])) for grp in args.net}
    else:
        nets = {n: _seed_files(n) for n in ("e28c", "e29slim")}

    agg = {name: _agg_net(paths) for name, paths in nets.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"nets": agg}, indent=1))

    for concept in CONCEPTS:
        print(f"\n=== {concept}  (mean over seeds) ===")
        head = "net".ljust(12) + "seeds".rjust(6) + "".join(ln.rjust(10) for ln in REPORT_LAYERS)
        head += "min_tower".rjust(11) + "raw->min".rjust(10)
        print(head)
        print("-" * len(head))
        for name, a in agg.items():
            c = a["concepts"][concept]
            row = name.ljust(12) + str(a["seeds"]).rjust(6)
            for ln in REPORT_LAYERS:
                m = c["layers"].get(ln, {}).get("mean")
                row += ("n/a".rjust(10)) if m is None else f"{m:>10.3f}"
            mt = c["min_tower_mean"]
            dr = c["raw_to_min_drop"]
            row += f"{mt:>11.3f}" if mt is not None else "n/a".rjust(11)
            row += f"{dr:>+10.3f}" if dr is not None else "n/a".rjust(10)
            print(row)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
