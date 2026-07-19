"""E27: concept probes — which concepts does the trained net actually compute?

The net-probe prestudy (PR #83) showed no layer beats the raw observation at
decoding the TEACHER'S ACTION. This experiment asks the finer question, per
concept: when the net plays badly around a concept (missed lethals, unused
items, threat blindness — E14a), is that concept MISSING from the
representation, or PRESENT but unused by the head? A concept is "computed"
only where some layer beats the SAME linear probe on the raw observation
(the prestudy's control); trivial near-linear concepts (mana, hp diff) are
included as positive controls that should saturate everywhere.

Concepts: ground-truth labels from the full game state at record time
(locma.stats.concepts — lethal_now needs lguard's exhaustive DFS from E26),
plus three derived from arrays already recorded: winner_side (did the
mover's seat win the game), item_now (teacher plays an item here, probed on
the states where an item is legal), can_item (an item play is legal).

Usage:
  python scripts/e27_concept_probe.py record --games 150 --seed 1000000 \
      --out runs/e27-concepts.npz
  python scripts/e27_concept_probe.py probe --model depot:b0k/b0k_s0.zip \
      --data runs/e27-concepts.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

USE_LO, USE_HI = 9, 113  # semantic Use-action index range [9, 113)


def cmd_record(args) -> None:
    from locma.envs.practicum import record_practicum  # noqa: PLC0415
    from locma.stats.concepts import concept_labels  # noqa: PLC0415

    manifest = record_practicum(
        teacher=args.teacher,
        games=args.games,
        out=args.out,
        seed=args.seed,
        obs_mode="token",
        labeler=lambda gs: concept_labels(gs, node_cap=args.node_cap),
    )
    print(f"recorded {manifest['n_examples']} labeled examples -> {args.out}")
    print(f"concepts: {manifest.get('concepts')}")


def _raw_obs_features(arrays: dict) -> np.ndarray:
    n = len(arrays["obs_tokens"])
    return np.concatenate(
        [
            arrays["obs_tokens"].reshape(n, -1).astype(np.float64),
            arrays["obs_scalars"].astype(np.float64),
            arrays["obs_token_mask"].astype(np.float64),
        ],
        axis=1,
    )


def cmd_probe(args) -> None:
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.envs.distill import load_practicum, split_by_game  # noqa: PLC0415
    from locma.stats.activations import collect_activations, practicum_obs  # noqa: PLC0415
    from locma.stats.concepts import BINARY_CONCEPTS, CONTINUOUS_CONCEPTS  # noqa: PLC0415
    from locma.stats.netdiag import probe_classify, probe_regression  # noqa: PLC0415

    arrays, manifest = load_practicum(args.data)
    n = len(arrays["action"])
    print(f"model={args.model}  data={args.data}  n={n}")

    model = MaskablePPO.load(resolve_path(args.model), device="cpu")
    obs = practicum_obs(arrays, "token")
    acts, _ = collect_activations(model.policy, obs, batch_size=args.batch_size)
    feats = {"raw": _raw_obs_features(arrays), **acts}

    # concept targets: recorded labels + derived ones
    action = arrays["action"]
    is_use_legal = arrays["mask"][:, USE_LO:USE_HI].any(axis=1)
    concepts: dict[str, dict] = {}
    for key in BINARY_CONCEPTS:
        concepts[key] = {"y": arrays[f"concept_{key}"], "kind": "binary"}
    concepts["winner_side"] = {
        "y": (arrays["winner"] == arrays["seat"]).astype(np.float32),
        "kind": "binary",
    }
    concepts["item_now"] = {
        "y": ((action >= USE_LO) & (action < USE_HI)).astype(np.float32),
        "kind": "binary",
        "row_filter": is_use_legal,  # only where an item play is legal
    }
    concepts["can_item"] = {"y": is_use_legal.astype(np.float32), "kind": "binary"}
    for key in CONTINUOUS_CONCEPTS:
        concepts[key] = {"y": arrays[f"concept_{key}"], "kind": "continuous"}

    train_idx, test_idx = split_by_game(arrays["game_id"], args.val_frac, args.seed)
    train_idx, test_idx = np.asarray(train_idx), np.asarray(test_idx)
    rng = np.random.default_rng(args.seed)

    results: dict[str, dict] = {}
    for cname, spec in concepts.items():
        y = np.asarray(spec["y"], dtype=np.float64)
        # binary labels use -1 for "unknown" (lethal_now cap hit: absence not
        # established); continuous concepts are legitimately negative.
        keep = (y >= 0) if spec["kind"] == "binary" else np.ones(len(y), dtype=bool)
        if "row_filter" in spec:
            keep &= np.asarray(spec["row_filter"], dtype=bool)
        tr = train_idx[keep[train_idx]]
        te = test_idx[keep[test_idx]]
        entry: dict = {
            "kind": spec["kind"],
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "base_rate": float(y[te].mean()) if spec["kind"] == "binary" else None,
            "layers": {},
        }
        y_shuf = y.copy()
        y_shuf[te] = rng.permutation(y[te])
        for lname, x in feats.items():
            if spec["kind"] == "binary":
                r = probe_classify(x[tr], y[tr].astype(int), x[te], y[te].astype(int), 2, args.l2)
                entry["layers"][lname] = {"auc": r["auc"], "acc": r["accuracy"]}
            else:
                r = probe_regression(x[tr], y[tr], x[te], y[te], args.l2)
                entry["layers"][lname] = {"r2": r["r2"]}
        # shuffled-label control on the strongest non-raw layer's features
        if spec["kind"] == "binary":
            r = probe_classify(
                feats["features"][tr],
                y[tr].astype(int),
                feats["features"][te],
                y_shuf[te].astype(int),
                2,
                args.l2,
            )
            entry["shuffled_auc"] = r["auc"]
        results[cname] = entry
        print(f"{cname}: done (n_test={entry['n_test']})")

    out = args.out or f"runs/netprobe/e27_{Path(resolve_path(args.model)).stem}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "data": args.data,
        "teacher": manifest.get("teacher"),
        "n_examples": n,
        "seed": args.seed,
        "l2": args.l2,
        "layer_order": list(feats),
        "concepts": results,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {out}")

    metric = lambda e, ln: e["layers"][ln].get("auc", e["layers"][ln].get("r2"))  # noqa: E731
    names = list(feats)
    head = "concept".ljust(18) + "".join(ln.rjust(10) for ln in names) + "  best-raw"
    print("\n" + head)
    for cname, e in results.items():
        raw_v = metric(e, "raw")
        best = max(metric(e, ln) for ln in names if ln != "raw")
        row = cname.ljust(18) + "".join(f"{metric(e, ln):>10.3f}" for ln in names)
        print(row + f"  {best - raw_v:>+8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="record a concept-labeled token practicum")
    rec.add_argument("--teacher", default="mcts:100")
    rec.add_argument("--games", type=int, default=150)
    rec.add_argument("--seed", type=int, default=1_000_000)
    rec.add_argument("--node-cap", type=int, default=3000)
    rec.add_argument("--out", default="runs/e27-concepts.npz")
    rec.set_defaults(fn=cmd_record)

    pr = sub.add_parser("probe", help="probe concepts per layer vs the raw-obs control")
    pr.add_argument("--model", required=True)
    pr.add_argument("--data", default="runs/e27-concepts.npz")
    pr.add_argument("--out", default=None)
    pr.add_argument("--val-frac", type=float, default=0.2)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--l2", type=float, default=1.0)
    pr.add_argument("--batch-size", type=int, default=2048)
    pr.set_defaults(fn=cmd_probe)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
