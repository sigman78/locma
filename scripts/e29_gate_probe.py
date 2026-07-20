"""E29 conditioned-trunk GATE: does feature_ln reduce tower saturation AND
restore can_item retention? (probe-based falsification, before any bench).

Background (docs/reactive-limits-program.md E29): the net-probe prestudy
found the SB3 policy/value towers' first Tanh layer ~94% saturated — the
mechanism behind E27's can_item info-loss (0.98 raw obs -> 0.61 after the
towers). E29a adds an opt-in LayerNorm on the extractor output (the tower
input; `TokenSetExtractor(feature_ln=True)`, CLI `--feature-ln`). This gate
trains a conditioned net and a matched control at a REDUCED budget on the
e28c recipe (token-fx + pointer), then probes both on a FROZEN set:

  gate 1: first-Tanh saturation (pi_a1 / vf_a1) must DROP, conditioned vs
          control.
  gate 2: can_item retention at the tower layers must IMPROVE toward the
          raw-obs number (0.98), conditioned vs control.

Pass both -> earn the full 3-seed bench (+ the ranking-loss critic rerun,
E15's frozen-extractor wall). Fail gate 1 -> the lever does not do what it
claims; kill cheaply.

Train the two nets first (concurrent, box allows 2 trainers), e.g.:
  locma train-zoo --pointer-head --obs-mode token-fx --learning-rate 1e-4 \
     --target-kl 0.025 --n-envs 16 --steps-per-opponent 100000 --seed 0 \
     --out runs/e29_ctrl_s0.zip
  locma train-zoo ... --feature-ln --out runs/e29_ln_s0.zip
then: python scripts/e29_gate_probe.py

Output: runs/e29-gate-summary.json.
"""

from __future__ import annotations

import json
import os

import numpy as np

CTRL = "runs/e29_ctrl_s0.zip"
LN = "runs/e29_ln_s0.zip"
PROBE_NPZ = "runs/e29_probe.npz"
PROBE_TEACHER = "ppo:depot:e28c/e28c_s0.zip,depot:ldraft/ldraft_s0.zip"
PROBE_GAMES = 40
PROBE_SEED = 56_000_000
USE_LO, USE_HI, MAX_HAND = 9, 113, 8
L2 = 1.0
VAL_FRAC = 0.3
TOWER_LAYERS = ("pi_a1", "pi_a2", "vf_a1", "vf_a2")

summary: dict = {}


def _to_fx(tokens_v0, card_ids):
    """Reconstruct the token-fx observation from a v0 token practicum
    (byte-faithful: fx cols are a deterministic card_id -> effect lookup)."""
    from locma.envs.encode import _fx_table  # noqa: PLC0415

    fxt = _fx_table()
    n = len(tokens_v0)
    fx = np.zeros((n, tokens_v0.shape[1], 3), dtype=np.float32)
    hand = card_ids[:, :MAX_HAND].astype(int)
    for i in range(n):
        for s in range(MAX_HAND):
            cid = hand[i, s]
            if cid > 0:
                fx[i, s] = fxt[cid]
    return np.concatenate([tokens_v0, fx], axis=2)


def _probe_one(model_path, arrays, fx_obs, tr, te):
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.stats.activations import collect_activations  # noqa: PLC0415
    from locma.stats.netdiag import probe_classify, unit_health  # noqa: PLC0415

    model = MaskablePPO.load(resolve_path(model_path), device="cpu")
    acts, kinds = collect_activations(model.policy, fx_obs)

    sat = {
        lyr: round(float(unit_health(acts[lyr], "tanh")["saturation_rate"]), 4)
        for lyr in ("pi_a1", "vf_a1")
        if lyr in acts
    }

    y = (arrays["mask"][:, USE_LO:USE_HI].any(axis=1)).astype(int)
    raw = np.concatenate(
        [
            arrays["obs_tokens"].reshape(len(y), -1).astype(np.float64),
            arrays["obs_scalars"].astype(np.float64),
            arrays["obs_token_mask"].astype(np.float64),
        ],
        axis=1,
    )
    feats = {"raw": raw, "features": acts["features"], **{k: acts[k] for k in TOWER_LAYERS}}
    can_item = {}
    for lname, x in feats.items():
        r = probe_classify(x[tr], y[tr], x[te], y[te], 2, L2)
        can_item[lname] = round(float(r["accuracy"]), 4)
    return {"saturation": sat, "can_item_acc": can_item, "base_rate": round(float(y[te].mean()), 4)}


def main() -> None:
    from locma.envs.distill import split_by_game  # noqa: PLC0415
    from locma.envs.practicum import record_practicum  # noqa: PLC0415

    os.makedirs("runs", exist_ok=True)
    for p in (CTRL, LN):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} — train the two gate nets first (see module docstring)")

    if not os.path.exists(PROBE_NPZ):
        record_practicum(
            teacher=PROBE_TEACHER,
            games=PROBE_GAMES,
            out=PROBE_NPZ,
            seed=PROBE_SEED,
            obs_mode="token",
        )
    arrays = dict(np.load(PROBE_NPZ))
    fx_tokens = _to_fx(arrays["obs_tokens"].astype(np.float32), arrays["obs_card_ids"])
    fx_obs = {
        "tokens": fx_tokens,
        "card_ids": arrays["obs_card_ids"].astype(np.float32),
        "token_mask": arrays["obs_token_mask"].astype(np.float32),
        "scalars": arrays["obs_scalars"].astype(np.float32),
    }
    tr, te = split_by_game(arrays["game_id"], VAL_FRAC, PROBE_SEED)
    tr, te = np.asarray(tr), np.asarray(te)
    print(f"probe set n={len(arrays['action'])}  train={len(tr)} test={len(te)}")

    summary["control"] = _probe_one(CTRL, arrays, fx_obs, tr, te)
    summary["conditioned"] = _probe_one(LN, arrays, fx_obs, tr, te)

    c, l = summary["control"], summary["conditioned"]  # noqa: E741
    tower_ret = lambda d: max(d["can_item_acc"][k] for k in ("pi_a1", "pi_a2"))  # noqa: E731
    summary["gate"] = {
        "sat_pi_a1_ctrl": c["saturation"].get("pi_a1"),
        "sat_pi_a1_cond": l["saturation"].get("pi_a1"),
        "sat_vf_a1_ctrl": c["saturation"].get("vf_a1"),
        "sat_vf_a1_cond": l["saturation"].get("vf_a1"),
        "can_item_tower_ctrl": round(tower_ret(c), 4),
        "can_item_tower_cond": round(tower_ret(l), 4),
        "can_item_raw": c["can_item_acc"]["raw"],
        "gate1_saturation_drops": bool(
            l["saturation"].get("pi_a1", 1) < c["saturation"].get("pi_a1", 0)
        ),
        "gate2_can_item_improves": bool(tower_ret(l) > tower_ret(c)),
    }
    summary["gate"]["PASS"] = bool(
        summary["gate"]["gate1_saturation_drops"] and summary["gate"]["gate2_can_item_improves"]
    )
    with open("runs/e29-gate-summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary["gate"], indent=1))


if __name__ == "__main__":
    main()
