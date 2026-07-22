"""E36 generational weight-SVD evolution — what self-play did to the spectra.

Loads the PFSP chain checkpoints (gen0..genN) plus the e29slim warm-start
parent, and for every named Linear in the policy computes the weight-matrix
singular-value spectrum and two scale-free summaries per generation:

  - stable rank    = ||W||_F^2 / ||W||_2^2  (energy spread; higher = more
                     directions carry variance, lower = spikier/low-rank)
  - effective rank = exp(entropy of the normalized singular-value distribution)
                     (Roy & Vetterli 2007; a soft count of active directions)

The held-out search gap closes monotonically gen0->gen7 while avg-hard3 stays
pinned at ceiling; this bundle is the weight-space companion to that behavioural
curve. Output feeds docs/notes/netviz-e36-evolution.html (embedded as const D).

Usage:
  python scripts/e36_weight_evolution.py \
      --gens runs/e36_gen0.zip,...,runs/e36_gen7.zip \
      --parent depot:e29slim/e29slim_s0.zip \
      --out runs/netviz-e36-evolution-data.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def _rnd(a, d: int = 5) -> list[float]:
    return [round(float(x), d) for x in a]


def named_linears(policy):
    import torch.nn as nn  # noqa: PLC0415

    out = []
    fe = policy.features_extractor
    if hasattr(fe, "head"):
        linears = [m for m in fe.head.modules() if isinstance(m, nn.Linear)]
        out.extend((f"extractor_head_{i + 1}", m) for i, m in enumerate(linears))
    for tower, seq in (
        ("pi", policy.mlp_extractor.policy_net),
        ("vf", policy.mlp_extractor.value_net),
    ):
        linears = [m for m in seq if isinstance(m, nn.Linear)]
        out.extend((f"{tower}_l{i + 1}", m) for i, m in enumerate(linears))
    # E28 pointer-head nets wrap a per-slot scoring MLP; plain nets are Linear.
    an = policy.action_net
    if isinstance(an, nn.Linear):
        out.append(("action_net", an))
    else:
        an_linears = [m for m in an.modules() if isinstance(m, nn.Linear)]
        out.extend((f"action_net_{i + 1}", m) for i, m in enumerate(an_linears))
    return out


def _summaries(sv: np.ndarray) -> dict:
    sv = sv[sv > 0]
    fro2 = float((sv**2).sum())
    spec = float(sv.max())
    stable = fro2 / (spec**2) if spec > 0 else 0.0
    p = (sv**2) / max(fro2, 1e-12)
    entropy = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
    eff = float(np.exp(entropy))
    return {
        "stable_rank": round(stable, 3),
        "eff_rank": round(eff, 3),
        "spectral_norm": round(spec, 4),
        "n": int(len(sv)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--gens",
        default=",".join(f"runs/e36_gen{g}.zip" for g in range(8)),
        help="comma list of chain checkpoints, gen0 first",
    )
    ap.add_argument("--parent", default="depot:e29slim/e29slim_s0.zip", help="warm-start net")
    ap.add_argument("--out", default="runs/netviz-e36-evolution-data.json")
    args = ap.parse_args()

    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.depot import resolve_path  # noqa: PLC0415

    specs = [("parent", args.parent)]
    specs += [(f"gen{i}", p) for i, p in enumerate(args.gens.split(","))]

    layer_order: list[str] = []
    models_out = {}
    for label, spec in specs:
        model = MaskablePPO.load(resolve_path(spec), device="cpu")
        entry = {"spec": spec, "layers": {}}
        for wname, m in named_linears(model.policy):
            w = m.weight.detach().numpy().astype(np.float64)
            sv = np.linalg.svd(w, compute_uv=False)
            svn = sv / max(sv.max(), 1e-12)
            entry["layers"][wname] = {
                "shape": list(m.weight.shape),
                "sv_norm": _rnd(svn, 5),
                **_summaries(sv),
            }
            if label == "parent":
                layer_order.append(wname)
        models_out[label] = entry
        print(f"{label}: {spec} — {len(entry['layers'])} layers done")

    bundle = {
        "labels": [lbl for lbl, _ in specs],
        "layer_order": layer_order,
        "models": models_out,
    }
    out = json.dumps(bundle, separators=(",", ":"))
    with open(args.out, "w") as f:
        f.write(out)
    print(f"wrote {args.out}  {len(out) / 1e6:.3f} MB")


if __name__ == "__main__":
    main()
