"""Extract the data bundle behind docs/notes/netviz-b0k.html (requires [ml]).

Forwards a saved MaskablePPO token model (and a same-architecture random
re-init) over a held-out practicum and bundles everything the interactive
anatomy page needs: per-layer covariance eigenspectra (trained + init),
top-3 PC projections of a fixed state sample with per-state metadata
(critic value, outcome, seat, opponent, game progress, action group),
per-unit health, pairwise linear CKA, and weight-matrix SVDs.

Usage:
  python scripts/netviz_extract.py --model depot:b0k/b0k_s0.zip \
      --data runs/probe-mcts-token.npz --out runs/netviz-data.json

To refresh the committed page, splice the bundle over the previous one in
docs/notes/netviz-b0k.html (it is embedded as `const D = {...};`).
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def _rnd(a, d: int = 5) -> list[float]:
    return [round(float(x), d) for x in a]


def _cov_eig(x: np.ndarray):
    """Descending covariance eigenvalues, eigenvectors, and centered data."""
    xc = x - x.mean(axis=0, keepdims=True)
    c = (xc.T @ xc) / max(len(xc) - 1, 1)
    w, v = np.linalg.eigh(c)
    order = np.argsort(w)[::-1]
    return np.maximum(w[order], 0.0), v[:, order], xc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="depot:b0k/b0k_s0.zip", help="token model.zip or depot: ref")
    ap.add_argument("--data", default="runs/probe-mcts-token.npz", help="token practicum .npz")
    ap.add_argument(
        "--obs-mode",
        default="token",
        choices=("token", "token-fx"),
        help="token (17-wide) or token-fx (20-wide; fx cols re-derived from card_ids)",
    )
    ap.add_argument("--out", default="runs/netviz-data.json")
    ap.add_argument("--max-examples", type=int, default=30_000)
    ap.add_argument("--n-scatter", type=int, default=2_600)
    ap.add_argument("--n-cka", type=int, default=4_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch  # noqa: PLC0415 — lazy so --help works without [ml]
    import torch.nn as nn  # noqa: PLC0415
    from gymnasium import spaces  # noqa: PLC0415
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.envs.distill import load_practicum  # noqa: PLC0415
    from locma.stats.activations import (  # noqa: PLC0415
        collect_activations,
        practicum_obs,
        reinit_clone,
    )
    from locma.stats.netdiag import linear_cka  # noqa: PLC0415

    rng = np.random.default_rng(args.seed)

    arrays, manifest = load_practicum(args.data)
    arrays = {k: v[: args.max_examples] for k, v in arrays.items()}
    n = len(arrays["action"])
    print(f"n={n} states")

    model = MaskablePPO.load(resolve_path(args.model), device="cpu")
    if not isinstance(model.observation_space, spaces.Dict):
        raise SystemExit("netviz_extract expects a token (Dict-obs) model")
    obs = practicum_obs(arrays, args.obs_mode)

    print("collecting trained activations...")
    acts, kinds = collect_activations(model.policy, obs, batch_size=2048)
    print("collecting reinit activations...")
    clone = reinit_clone(model.policy, seed=args.seed)
    acts0, _ = collect_activations(clone, obs, batch_size=2048)

    # raw-obs baseline features (matches scripts/net_probe.py)
    raw = np.concatenate(
        [
            arrays["obs_tokens"].reshape(n, -1).astype(np.float64),
            arrays["obs_scalars"].astype(np.float64),
            arrays["obs_token_mask"].astype(np.float64),
        ],
        axis=1,
    )

    vals = []
    with torch.no_grad():
        for s in range(0, n, 2048):
            batch = {k: v[s : s + 2048] for k, v in obs.items()}
            obs_t, _ = model.policy.obs_to_tensor(batch)
            vals.append(model.policy.predict_values(obs_t).cpu().numpy().ravel())
    values = np.concatenate(vals)

    game_id = arrays["game_id"]
    progress = np.zeros(n, dtype=np.float64)
    for g in np.unique(game_id):
        idx = np.where(game_id == g)[0]
        progress[idx] = np.linspace(0, 1, len(idx))

    act_idx = arrays["action"].astype(int)
    # semantic layout (locma.envs.encode): 0=pass, 1..8 summon, 9..112 use, rest attack
    agroup = np.full(n, 3, dtype=int)
    agroup[act_idx == 0] = 0
    agroup[(act_idx >= 1) & (act_idx <= 8)] = 1
    agroup[(act_idx >= 9) & (act_idx <= 112)] = 2

    sample = np.sort(rng.choice(n, size=args.n_scatter, replace=False))
    meta = {
        "winner": arrays["winner"][sample].astype(int).tolist(),
        "seat": arrays["seat"][sample].astype(int).tolist(),
        "opp": arrays["opponent_id"][sample].astype(int).tolist(),
        "opp_names": list(manifest.get("opponents", [])),
        "progress": _rnd(progress[sample], 3),
        "value": _rnd(values[sample], 3),
        "agroup": agroup[sample].tolist(),
        "agroup_names": ["pass", "summon", "use", "attack"],
    }

    layer_names = ["raw", *acts.keys()]
    layers_out = {}
    for name in layer_names:
        x = raw if name == "raw" else acts[name].astype(np.float64)
        w, v, xc = _cov_eig(x)
        ev = w / max(w.sum(), 1e-12)
        proj = xc[sample] @ v[:, :3]
        entry = {
            "width": int(x.shape[1]),
            "kind": "input" if name == "raw" else kinds[name],
            "evr": _rnd(ev[: min(len(ev), 512)], 6),
            "pc_var": _rnd(ev[:3], 4),
            "pc": [_rnd(proj[:, i], 3) for i in range(3)],
        }
        if name != "raw":
            x0 = acts0[name].astype(np.float64)
            w0, _, _ = _cov_eig(x0)
            entry["evr0"] = _rnd(w0 / max(w0.sum(), 1e-12), 6)[:512]
            kind = kinds[name]
            if kind == "tanh":
                entry["unit"] = _rnd((np.abs(x) > 0.99).mean(axis=0), 3)
                entry["unit_metric"] = "saturation rate"
            elif kind == "relu":
                entry["unit"] = _rnd((x > 0).mean(axis=0), 3)
                entry["unit_metric"] = "duty cycle"
            else:
                sd = x.std(axis=0)
                entry["unit"] = _rnd(sd / max(sd.max(), 1e-12), 3)
                entry["unit_metric"] = "relative std"
            entry["cka_init"] = round(linear_cka(x[: args.n_cka // 2], x0[: args.n_cka // 2]), 3)
        layers_out[name] = entry
        print(f"{name}: done")

    ck_idx = np.sort(rng.choice(n, size=min(args.n_cka, n), replace=False))
    mats = {ln: (raw if ln == "raw" else acts[ln].astype(np.float64))[ck_idx] for ln in layer_names}
    cka = [[0.0] * len(layer_names) for _ in layer_names]
    for i, a in enumerate(layer_names):
        for j, b in enumerate(layer_names):
            cka[i][j] = cka[j][i] if j < i else round(linear_cka(mats[a], mats[b]), 3)
    print("cka: done")

    def named_linears(policy):
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
        # Plain nets have a Linear action_net; E28 pointer-head nets wrap a
        # per-slot scoring MLP (PointerActionNet) — extract its Linears.
        an = policy.action_net
        if isinstance(an, nn.Linear):
            out.append(("action_net", an))
        else:
            an_linears = [m for m in an.modules() if isinstance(m, nn.Linear)]
            out.extend((f"action_net_{i + 1}", m) for i, m in enumerate(an_linears))
        return out

    weights = {}
    for (wname, m), (_, m0) in zip(named_linears(model.policy), named_linears(clone), strict=True):
        s = np.linalg.svd(m.weight.detach().numpy().astype(np.float64), compute_uv=False)
        s0 = np.linalg.svd(m0.weight.detach().numpy().astype(np.float64), compute_uv=False)
        weights[wname] = {"shape": list(m.weight.shape), "sv": _rnd(s, 5), "sv0": _rnd(s0, 5)}
    print("weights: done")

    bundle = {
        "model": args.model,
        "data": args.data,
        "n_states": n,
        "n_scatter": args.n_scatter,
        "meta": meta,
        "layers": layers_out,
        "layer_order": layer_names,
        "cka": cka,
        "weights": weights,
    }
    out = json.dumps(bundle, separators=(",", ":"))
    with open(args.out, "w") as f:
        f.write(out)
    print(f"wrote {args.out}  {len(out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
