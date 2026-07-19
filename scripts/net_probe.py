"""Network-utilization probe: how much of a PPO net's capacity is in use?

Prestudy instrument for the architecture sweep. Forwards a frozen practicum's
observations through a saved MaskablePPO model, captures per-layer activations
(locma.stats.activations) and reports (locma.stats.netdiag):

  - spectrum stats per layer (participation ratio, effective rank, n99) for
    the TRAINED net and a same-architecture RANDOM-INIT clone — training
    typically collapses dimensionality toward task subspaces; the delta from
    init is the interesting number, not the absolute PR.
  - unit health (tanh saturation / relu death) — direct capacity-pressure
    evidence: a saturated 64-wide tower argues for width, a mostly-idle one
    does not.
  - linear probes per layer vs the SAME probe on the raw observation:
      * outcome (did the teacher's seat win) — value-like signal, nonlinear
        in the input;
      * teacher action (masked 155-class) — tactical signal from the
        practicum's search teacher.
    A layer only "computes" what it decodes BETTER than the raw obs.
  - linear CKA: trained-vs-reinit per layer (how far training moved the
    representation) and pairwise between layers (redundancy across depth).
  - the model's own masked-argmax agreement with the teacher action (no
    probe — direct head readout, for calibrating the probe numbers).

Usage (see docs/worklog.md for the study):
  python scripts/net_probe.py --model runs/ab-flat-s0.zip \
      --data runs/practicum-flat.npz --out runs/netprobe/ab-flat-s0.json
  python scripts/net_probe.py --model depot:b0k/b0k_s0.zip \
      --data runs/practicum-dmcts.npz

Requires the [ml] extra. Runs on CPU in a couple of minutes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _raw_obs_features(arrays: dict, obs_mode: str) -> np.ndarray:
    """Raw-observation feature matrix for the probe baseline.

    Token mode: numeric token features + scalars + token mask, flattened.
    card_ids are omitted (categorical; a linear probe on raw ids is
    meaningless) — noted in the output JSON.
    """
    if obs_mode == "token":
        n = len(arrays["obs_tokens"])
        return np.concatenate(
            [
                arrays["obs_tokens"].reshape(n, -1).astype(np.float64),
                arrays["obs_scalars"].astype(np.float64),
                arrays["obs_token_mask"].astype(np.float64),
            ],
            axis=1,
        )
    return arrays["obs"].astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="model.zip path or depot: ref")
    ap.add_argument("--data", required=True, help="practicum .npz (obs mode must match model)")
    ap.add_argument(
        "--out", default=None, help="output JSON (default runs/netprobe/<model>__<data>.json)"
    )
    ap.add_argument("--max-examples", type=int, default=40_000)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--no-reinit", action="store_true", help="skip the random-init baseline")
    args = ap.parse_args()

    from gymnasium import spaces  # noqa: PLC0415 — lazy so --help works without [ml]
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.envs.distill import load_practicum, split_by_game  # noqa: PLC0415
    from locma.stats.activations import (  # noqa: PLC0415
        collect_activations,
        practicum_obs,
        reinit_clone,
    )
    from locma.stats.netdiag import (  # noqa: PLC0415
        linear_cka,
        probe_classify,
        spectrum_stats,
        unit_health,
    )

    arrays, manifest = load_practicum(args.data)
    obs_mode = manifest.get("obs_mode", "flat")

    model = MaskablePPO.load(resolve_path(args.model), device="cpu")
    model_is_token = isinstance(model.observation_space, spaces.Dict)
    if model_is_token != (obs_mode == "token"):
        raise SystemExit(
            f"obs-family mismatch: model is {'token' if model_is_token else 'flat'}, "
            f"practicum is {obs_mode}"
        )

    n_total = len(arrays["action"])
    if n_total > args.max_examples:
        # Prefix truncation keeps games contiguous, so the game-level split stays clean.
        arrays = {k: v[: args.max_examples] for k, v in arrays.items()}
    n = len(arrays["action"])
    print(f"model={args.model}  data={args.data}  obs_mode={obs_mode}  n={n}/{n_total}")

    obs = practicum_obs(arrays, obs_mode)
    acts, kinds = collect_activations(model.policy, obs, batch_size=args.batch_size)

    reinit_acts: dict = {}
    if not args.no_reinit:
        clone = reinit_clone(model.policy, seed=args.seed)
        reinit_acts, _ = collect_activations(clone, obs, batch_size=args.batch_size)

    train_idx, val_idx = split_by_game(arrays["game_id"], args.val_frac, args.seed)
    tr = np.asarray(train_idx)
    va = np.asarray(val_idx)
    won = (arrays["winner"] == arrays["seat"]).astype(np.int64)
    action = arrays["action"].astype(np.int64)
    mask = arrays["mask"].astype(bool)
    n_classes = mask.shape[1]

    def probes_for(x: np.ndarray) -> dict:
        return {
            "outcome": probe_classify(x[tr], won[tr], x[va], won[va], n_classes=2, l2=args.l2),
            "teacher_action": probe_classify(
                x[tr],
                action[tr],
                x[va],
                action[va],
                n_classes=n_classes,
                l2=args.l2,
                mask_test=mask[va],
            ),
        }

    raw = _raw_obs_features(arrays, obs_mode)
    layers: dict[str, dict] = {}
    print("computing raw-obs baseline probes...")
    raw_report = {
        "spectrum": spectrum_stats(raw),
        "probes": probes_for(raw),
    }
    for name, a in acts.items():
        print(f"layer {name}: spectrum + probes ({a.shape[1]}d)...")
        layers[name] = {
            "spectrum": spectrum_stats(a),
            "unit_health": unit_health(a, kind=kinds[name]),
            "probes": probes_for(a),
        }
        if name in reinit_acts:
            layers[name]["spectrum_reinit"] = spectrum_stats(reinit_acts[name])
            layers[name]["cka_vs_reinit"] = linear_cka(a, reinit_acts[name])

    names = list(acts)
    cka_pairs = {
        f"{a}|{b}": linear_cka(acts[a], acts[b])
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }

    # Direct head readout: the model's own masked argmax vs the teacher.
    masked_logits = np.where(mask[va], acts["logits"][va], -np.inf)
    model_agreement = float((masked_logits.argmax(axis=1) == action[va]).mean())

    out = args.out or str(
        Path("runs/netprobe") / f"{Path(args.model).stem}__{Path(args.data).stem}.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.model,
        "data": args.data,
        "obs_mode": obs_mode,
        "teacher": manifest.get("teacher"),
        "n_examples": n,
        "seed": args.seed,
        "l2": args.l2,
        "val_frac": args.val_frac,
        "notes": "raw_obs token baseline = tokens+scalars+token_mask (card_ids omitted)",
        "model_teacher_agreement": model_agreement,
        "raw_obs": raw_report,
        "layers": layers,
        "cka_pairs": cka_pairs,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out}")

    # ------------------------------------------------------------------ table
    from rich.console import Console  # noqa: PLC0415 — report-only dep, keep with the table
    from rich.table import Table  # noqa: PLC0415

    t = Table(title=f"{args.model} on {Path(args.data).name} (teacher={manifest.get('teacher')})")
    for col in (
        "layer",
        "width",
        "PR",
        "PR/w",
        "PR/w init",
        "erank",
        "n99",
        "sat/dead",
        "CKA vs init",
        "outcome AUC",
        "action acc",
    ):
        t.add_column(col, justify="right")

    def row(name: str, rep: dict, is_raw: bool = False) -> None:
        s = rep["spectrum"]
        h = rep.get("unit_health", {})
        si = rep.get("spectrum_reinit")
        sat = h.get("saturated_unit_frac", h.get("dead_frac"))
        p = rep["probes"]
        t.add_row(
            name,
            str(s["width"]),
            f"{s['participation_ratio']:.1f}",
            f"{s['pr_frac']:.3f}",
            f"{si['pr_frac']:.3f}" if si else "-",
            f"{s['effective_rank']:.1f}",
            str(s["n99"]),
            "-" if is_raw or sat is None else f"{sat:.2f}",
            f"{rep['cka_vs_reinit']:.2f}" if "cka_vs_reinit" in rep else "-",
            f"{p['outcome']['auc']:.3f}",
            f"{p['teacher_action']['accuracy']:.3f}",
        )

    row("raw_obs", raw_report, is_raw=True)
    for name in names:
        row(name, layers[name])
    console = Console()
    console.print(t)
    console.print(
        f"outcome majority baseline: {raw_report['probes']['outcome']['majority_accuracy']:.3f}   "
        f"masked-majority action baseline: "
        f"{raw_report['probes']['teacher_action']['majority_accuracy']:.3f}   "
        f"model argmax agreement with teacher: {model_agreement:.3f}"
    )


if __name__ == "__main__":
    main()
