"""Blue-item diagnostic: are blues WEAK, or can the net not PLAY them?

The E28d null (worklog 2026-07-20) showed item-rich training did not raise
item/blue conversion over the item-light-trained e28c and cost win rate.
That was read as "consequence valuation, not exposure" — but it does NOT
distinguish two hypotheses this script separates:

  H1 (blues are weak): in the current balance, declining a blue is often
     the correct play, so a low blue-play rate is optimal, not a defect.
  H2 (can't value them): the net under-plays blues it should play, and
     would keep under-playing stronger blues too.

Two net-independent reads:

1. ORACLE blue-play rate (H1). A CHEATING perfect-information MCTS
   (net-independent, engine-grounded heuristic rollouts) plays the same
   blue-rich diet decks; measure ITS blue rate per opportunity. If even
   the perfect-info cheater declines blues at ~the net's rate, blues are
   contextually weak (H1) and the net is roughly correct. If the oracle
   plays blues far more, the net genuinely under-plays them (H2).

2. MAGNITUDE-DOSE probe (H2 + the "stronger new blues" question). Hold a
   real blue-in-hand decision fixed and SCALE that card's fx effect
   columns (player_hp/enemy_hp/card_draw) by k in {0, 0.5, 1, 2, 3} —
   i.e. the same card made weaker/stronger, "similarly designed". Read
   the fx net's own play-probability for that card (pointer Use logit for
   the slot). Monotone-increasing in k = the net VALUES effect magnitude
   and would play stronger blues (the defect, if any, is that today's
   blues are too weak to clear its bar — points at card design, not the
   net). Flat = the net is magnitude-blind on blues (a real H2 defect).
   A green-item control curve (greens were plentiful in training) says
   whether any magnitude response is blue-specific.

Reads e28c (RoR) and e28d (item-rich trained) so we also see whether
item-rich training bought any magnitude sensitivity it failed to convert
to win rate.

Outputs: runs/blue-value-diag.json (+ practicum npzs under runs/).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

DIET = "runs/e31a_diet.json"
GAMES = 40
SEED = 53_000_000
ORACLE = "mcts:300,1.41,0,3," + DIET  # cheating perfect-info, diet decks
NETS = {"e28c_s0": "depot:e28c/e28c_s0.zip", "e28d_s0": "runs/e28d_s0.zip"}
LDRAFT0 = "depot:ldraft/ldraft_s0.zip"

USE_LO, USE_HI, USE_STRIDE, MAX_HAND = 9, 113, 13, 8
DOSES = (0.0, 0.5, 1.0, 2.0, 3.0)
FX_LO = 17  # fx effect columns are token cols 17:20 (base features 0:17)

summary: dict = {}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def record(key, value) -> None:
    summary[key] = value
    with open("runs/blue-value-diag.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)


def _color_ids():
    from locma.data.cards_db import load_cards  # noqa: PLC0415

    cards = load_cards()
    return (
        {c.id for c in cards if c.type == 3},  # blue
        {c.id for c in cards if c.type == 1},  # green
    )


def _rates(npz_path, blue):
    """item + blue rate per opportunity from a token/token-fx practicum."""
    d = np.load(npz_path)
    act, mask, card_ids = d["action"], d["mask"], d["obs_card_ids"]
    can_item = mask[:, USE_LO:USE_HI].any(axis=1)
    played = (act >= USE_LO) & (act < USE_HI)
    hand_blue = np.isin(card_ids[:, :MAX_HAND], list(blue))
    use_legal = mask[:, USE_LO:USE_HI].reshape(len(act), MAX_HAND, USE_STRIDE).any(axis=2)
    can_blue = (hand_blue & use_legal).any(axis=1)
    slot = np.where(played, (act - USE_LO) // USE_STRIDE, -1)
    played_blue = np.array([s >= 0 and bool(hand_blue[i, s]) for i, s in enumerate(slot)])
    return {
        "n_decisions": int(len(act)),
        "item_opportunities": int(can_item.sum()),
        "item_rate_per_opportunity": round(float(played[can_item].mean()), 4),
        "blue_opportunities": int(can_blue.sum()),
        "blue_rate_per_opportunity": (
            round(float(played_blue[can_blue].mean()), 4) if can_blue.any() else None
        ),
    }


def oracle_read():
    from locma.envs.practicum import record_practicum  # noqa: PLC0415

    blue, _ = _color_ids()
    reads = {}
    specs = {
        "oracle_mcts": ORACLE,
        "e28c_s0_diet": f"ppo:{NETS['e28c_s0']},{DIET}",
        "e28d_s0_diet": f"ppo:{NETS['e28d_s0']},{DIET}",
    }
    for tag, spec in specs.items():
        out = f"runs/bvd_{tag}.npz"
        if not os.path.exists(out):
            record_practicum(teacher=spec, games=GAMES, out=out, seed=SEED, obs_mode="token")
        reads[tag] = {"teacher": spec, **_rates(out, blue)}
        log(f"{tag}: {reads[tag]}")
    record("oracle_vs_net", reads)
    return reads


def _forward_use_prob(policy, batch, mask, focal_slot):
    """Masked play-probability for each decision's focal-slot Use block."""
    import torch  # noqa: PLC0415

    obs_t, _ = policy.obs_to_tensor(batch)
    with torch.no_grad():
        feats = policy.extract_features(obs_t)  # fires the pointer slot hook
        latent_pi, _ = policy.mlp_extractor(feats)
        dist = policy._get_action_dist_from_latent(latent_pi)
        dist.apply_masking(mask)
        probs = dist.distribution.probs.cpu().numpy()  # (B, 155)
    out = np.zeros(len(focal_slot), dtype=np.float64)
    for i, s in enumerate(focal_slot):
        lo = USE_LO + s * USE_STRIDE
        out[i] = probs[i, lo : lo + USE_STRIDE].sum()
    return out


def _dose_curve(model, tokens, card_ids, token_mask, scalars, mask, focal_slot):
    """Mean focal-slot Use prob as fx columns are scaled by each dose k."""
    B = 512
    curve = {}
    for k in DOSES:
        vals = []
        for lo in range(0, len(focal_slot), B):
            hi = min(lo + B, len(focal_slot))
            tok = tokens[lo:hi].copy()
            fs = focal_slot[lo:hi]
            rows = np.arange(hi - lo)
            tok[rows, fs, FX_LO:] = tokens[lo:hi][rows, fs, FX_LO:] * k
            batch = {
                "tokens": tok,
                "card_ids": card_ids[lo:hi],
                "token_mask": token_mask[lo:hi],
                "scalars": scalars[lo:hi],
            }
            vals.append(_forward_use_prob(model.policy, batch, mask[lo:hi], fs))
        curve[f"k{k:g}"] = round(float(np.concatenate(vals).mean()), 4)
    return curve


def _to_fx(tokens_v0, card_ids):
    """Reconstruct the token-fx observation from a v0 token practicum.

    The fx variant appends [player_hp, enemy_hp, card_draw] for HAND slots
    (0..7), zeros elsewhere — a deterministic card_id -> effect lookup
    (encode._fx_table), identical to what encode_battle_tokens(view, "fx")
    would have produced. So concat(v0 row, fx row) is byte-faithful to the
    net's real fx input, no recorder change needed.
    """
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
    return np.concatenate([tokens_v0, fx], axis=2)  # (n, 20, 20)


def magnitude_probe():
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.depot import resolve_path  # noqa: PLC0415

    blue, green = _color_ids()
    probe = {}
    for tag, ref in NETS.items():
        # Reuse the token practicum captured by oracle_read (byte-faithful fx
        # is reconstructed from card_ids — no separate fx recording).
        d = np.load(f"runs/bvd_{tag}_diet.npz")
        card_ids = d["obs_card_ids"].astype(np.float32)
        tokens = _to_fx(d["obs_tokens"].astype(np.float32), card_ids)
        token_mask = d["obs_token_mask"].astype(np.float32)
        scalars = d["obs_scalars"].astype(np.float32)
        mask = d["mask"]
        n = len(tokens)
        use_legal = mask[:, USE_LO:USE_HI].reshape(n, MAX_HAND, USE_STRIDE).any(axis=2)  # (n,8)
        model = MaskablePPO.load(resolve_path(ref))

        entry = {}
        for color_tag, ids in (("blue", blue), ("green", green)):
            hand_color = np.isin(card_ids[:, :MAX_HAND], list(ids)) & use_legal  # (n,8)
            has = hand_color.any(axis=1)
            focal = np.argmax(hand_color, axis=1)  # first color Use-legal hand slot
            idx = np.where(has)[0]
            if len(idx) == 0:
                entry[color_tag] = {"n": 0}
                continue
            curve = _dose_curve(
                model,
                tokens[idx],
                card_ids[idx],
                token_mask[idx],
                scalars[idx],
                mask[idx],
                focal[idx].astype(int),
            )
            entry[color_tag] = {"n": int(len(idx)), "use_prob_by_dose": curve}
            log(f"{tag} {color_tag}: {entry[color_tag]}")
        probe[tag] = entry
    record("magnitude_probe", probe)
    return probe


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    if os.path.exists("runs/blue-value-diag.json"):
        with open("runs/blue-value-diag.json", encoding="utf-8") as f:
            summary.update(json.load(f))
    log("=== blue-value diagnostic start ===")
    oracle_read()
    magnitude_probe()
    log("done")


if __name__ == "__main__":
    main()
