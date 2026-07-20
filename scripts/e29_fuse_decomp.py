"""E29 pre-read: fuse-layer loss decomposition — width bottleneck or learned discarding?

The net-probe study (PR #83 + held-out replication) shows the token extractor's
fuse layer (``features_extractor.head``, Linear 1344->256 + ReLU) is the largest
single step-drop in teacher-action decodability along the trunk. Two mechanisms
would explain it, with opposite implications for the E29 arm:

  1. STRUCTURAL: any 1344->256 ReLU projection loses that much at the linear-
     probe level — the fix is a bypass (skip path / pointer-style access) or a
     wider ``features_dim``; retraining/conditioning the fuse alone cannot help.
  2. LEARNED DISCARDING: PPO trained the fuse to throw tactical detail away —
     objective/conditioning work on the fuse itself is on the table.

Discriminating control: apply a RANDOM-INIT Linear+ReLU of the fuse's exact
shape to the fuse's exact input ``[slots ; scalar_mlp(scalars)]`` and probe the
teacher action from both. Trained ~= random -> structural; trained << random
-> learned discarding; trained >> random -> the fuse actively preserves.

Usage (token models only — the flat path has no fuse):
  python scripts/e29_fuse_decomp.py --model depot:b0k/b0k_s0.zip \
      --data runs/practicum-dmcts.npz

Requires the [ml] extra. Runs on CPU in a couple of minutes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="model.zip path or depot: ref (token obs)")
    ap.add_argument("--data", required=True, help="token-mode practicum .npz")
    ap.add_argument(
        "--out", default=None, help="output JSON (default runs/netprobe/<model>__fuse-decomp.json)"
    )
    ap.add_argument("--max-examples", type=int, default=40_000)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--rand-seeds", type=int, default=3, help="random-fuse control replicates")
    args = ap.parse_args()

    import torch  # noqa: PLC0415 — lazy so --help works without [ml]
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.envs.distill import load_practicum, split_by_game  # noqa: PLC0415
    from locma.stats.activations import collect_activations, practicum_obs  # noqa: PLC0415
    from locma.stats.netdiag import probe_classify  # noqa: PLC0415

    arrays, manifest = load_practicum(args.data)
    if manifest.get("obs_mode") != "token":
        raise SystemExit("fuse decomposition needs a token-mode practicum (flat has no fuse)")

    model = MaskablePPO.load(resolve_path(args.model), device="cpu")
    policy = model.policy
    if not hasattr(policy.features_extractor, "scalar_mlp"):
        raise SystemExit("model has no TokenSetExtractor (flat model?) — nothing to decompose")

    n_total = len(arrays["action"])
    if n_total > args.max_examples:
        # Prefix truncation keeps games contiguous, so the game-level split stays clean.
        arrays = {k: v[: args.max_examples] for k, v in arrays.items()}
    n = len(arrays["action"])
    print(f"model={args.model}  data={args.data}  n={n}/{n_total}")

    obs = practicum_obs(arrays, "token")
    acts, _ = collect_activations(policy, obs, batch_size=args.batch_size)
    slots, features = acts["slots"], acts["features"]

    def batched(fn, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            outs = [
                fn(torch.as_tensor(x[i : i + args.batch_size], dtype=torch.float32)).numpy()
                for i in range(0, n, args.batch_size)
            ]
        return np.concatenate(outs, axis=0)

    # The fuse's exact input: flattened per-slot outputs + the scalar branch.
    scalar_out = batched(policy.features_extractor.scalar_mlp, obs["scalars"])
    fuse_in = np.concatenate([slots, scalar_out], axis=1)

    raw = np.concatenate([obs["tokens"].reshape(n, -1), obs["scalars"], obs["token_mask"]], axis=1)

    train_idx, val_idx = split_by_game(arrays["game_id"], args.val_frac, args.seed)
    tr, va = np.asarray(train_idx), np.asarray(val_idx)
    labels = arrays["action"].astype(np.int64)
    mask_va = arrays["mask"][va].astype(bool)

    results: dict[str, dict] = {}

    def run(name: str, x: np.ndarray) -> None:
        r = probe_classify(x[tr], labels[tr], x[va], labels[va], 155, l2=args.l2, mask_test=mask_va)
        r["width"] = int(x.shape[1])
        results[name] = r
        print(
            f"{name:22s} width={x.shape[1]:5d}  acc={r['accuracy']:.4f}"
            f"  (majority={r['majority_accuracy']:.4f})"
        )

    run("raw_obs", raw)
    run("slots", slots)
    run("fuse_in", fuse_in)
    run("features_trained", features)
    fuse_width = features.shape[1]
    for rs in range(args.rand_seeds):
        torch.manual_seed(rs)
        lin = torch.nn.Linear(fuse_in.shape[1], fuse_width)
        run(f"rand_fuse_s{rs}", batched(lambda t, lin=lin: torch.relu(lin(t)), fuse_in))

    rand_accs = [results[f"rand_fuse_s{rs}"]["accuracy"] for rs in range(args.rand_seeds)]
    out = {
        "model": args.model,
        "data": args.data,
        "n_examples": n,
        "teacher": manifest.get("teacher"),
        "seed": args.seed,
        "l2": args.l2,
        "val_frac": args.val_frac,
        "notes": (
            "teacher-action ridge probes on the fuse's exact input/output vs a random-init "
            "Linear+ReLU of the same shape; trained ~= random -> structural compression, "
            "trained << random -> learned discarding"
        ),
        "probes": results,
        "rand_fuse_mean_accuracy": float(np.mean(rand_accs)),
    }
    if args.out is None:
        stem_m = Path(resolve_path(args.model)).stem
        args.out = f"runs/netprobe/{stem_m}__fuse-decomp.json"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
