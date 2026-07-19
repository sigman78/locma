"""Extract architecture + demo forward-pass data for docs/notes/model-explorer.html.

Models: depot:b0k/b0k_s0.zip (token) + runs/ab-flat-s0.zip (flat).
Demo states come from runs/e27-concepts.npz (token) and the SAME row indices
in runs/probe-mcts-flat.npz (both were recorded from the identical seed-1M
game stream, so row i is the same decision point in both encodings).

Outputs runs/archviz-data.json: parameter tensors for both checkpoints plus,
per demo state, the decoded board (card names/stats), tactical scalars, true
head-averaged attention maps per transformer layer, per-slot output norms,
tower activations, masked top-6 action distribution, teacher action, critic
value and E27 concept labels. To refresh the committed page, splice the
bundle over the previous one (embedded as ``const D = {...};``).

Requires the [ml] extra.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from locma.data.cards_db import card_by_id, load_cards
from locma.depot import resolve_path

CARDS = card_by_id(load_cards())

SCALAR_NAMES = [
    "turn",
    "me_health",
    "op_health",
    "me_mana",
    "summonable_count",
    "op_hand_count",
    "my_board_count",
    "op_board_count",
    "opp_guard_count",
    "my_total_attack",
    "my_total_defense",
    "reachable_face_damage",
    "lethal_available",
]
FLAT_SCALAR_NAMES = [
    "turn",
    "me_health",
    "me_mana",
    "op_health",
    "op_hand_count",
    "my_board_count",
    "op_board_count",
    "my_hand_count",
]
ZONES = ["hand"] * 8 + ["my_board"] * 6 + ["op_board"] * 6


def decode_action(idx: int) -> str:
    if idx == 0:
        return "Pass"
    if 1 <= idx <= 8:
        return f"Summon hand[{idx - 1}]"
    if 9 <= idx <= 112:
        s, tc = divmod(idx - 9, 13)
        tgt = f"my_board[{tc}]" if tc < 6 else (f"op_board[{tc - 6}]" if tc < 12 else "face/none")
        return f"Use hand[{s}] -> {tgt}"
    a, tc = divmod(idx - 113, 7)
    tgt = f"op_board[{tc}]" if tc < 6 else "face"
    return f"Attack my_board[{a}] -> {tgt}"


def rnd(a, d=3):
    return [round(float(x), d) for x in np.asarray(a).ravel()]


def param_blocks(policy) -> list[dict]:
    out = []
    for name, p in policy.named_parameters():
        out.append({"name": name, "shape": list(p.shape), "params": int(p.numel())})
    return out


# ---------------------------------------------------------------- token model
tok = MaskablePPO.load(resolve_path("depot:b0k/b0k_s0.zip"), device="cpu")
tok.policy.set_training_mode(False)
flat_m = MaskablePPO.load("runs/ab-flat-s0.zip", device="cpu")
flat_m.policy.set_training_mode(False)

td = np.load("runs/e27-concepts.npz")
fd = np.load("runs/probe-mcts-flat.npz")
assert len(td["action"]) == len(fd["action"])

# demo state selection
lethal = td["concept_lethal_now"]
use_legal = td["mask"][:, 9:113].any(axis=1)
act = td["action"]
turn = td["obs_scalars"][:, 0]

i_lethal = int(np.where((lethal == 1) & (turn > 6) & (turn < 20))[0][0])
i_item = int(np.where(use_legal & (act >= 9) & (act < 113) & (turn > 4))[0][0])
i_early = int(np.where((turn >= 3) & (turn <= 4) & (lethal == 0) & ~use_legal)[0][0])
DEMO = [
    ("lethal", i_lethal, "a position with an engine-verified forced win this turn"),
    ("item", i_item, "a position where the teacher plays an item"),
    ("early", i_early, "a quiet early-game development turn"),
]

fe = tok.policy.features_extractor

# capture per-layer transformer inputs to recompute true attention maps
layer_inputs: list[torch.Tensor] = []
hooks = [
    layer.register_forward_pre_hook(lambda mod, args: layer_inputs.append(args[0].detach()))
    for layer in fe.transformer.layers
]


def token_trace(i: int) -> dict:
    obs = {
        "tokens": torch.as_tensor(td["obs_tokens"][i : i + 1]),
        "card_ids": torch.as_tensor(td["obs_card_ids"][i : i + 1]),
        "token_mask": torch.as_tensor(td["obs_token_mask"][i : i + 1]),
        "scalars": torch.as_tensor(td["obs_scalars"][i : i + 1]),
    }
    layer_inputs.clear()
    with torch.no_grad():
        obs_t, _ = tok.policy.obs_to_tensor({k: v.numpy() for k, v in obs.items()})
        feats = tok.policy.extract_features(obs_t)
        acts = {"features": feats}
        x = feats
        names = iter(["pi_a1", "pi_a2"])
        for m in tok.policy.mlp_extractor.policy_net:
            x = m(x)
            if not isinstance(m, torch.nn.Linear):
                acts[next(names)] = x
        latent_pi = x
        x = feats
        names = iter(["vf_a1", "vf_a2"])
        for m in tok.policy.mlp_extractor.value_net:
            x = m(x)
            if not isinstance(m, torch.nn.Linear):
                acts[next(names)] = x
        logits = tok.policy.action_net(latent_pi)
        value = tok.policy.value_net(x)

        # true attention maps: rerun each captured layer input through self_attn
        kpm = obs_t["token_mask"] == 0
        if bool(kpm.all()):
            kpm = kpm & False
        attn = []
        for layer, inp in zip(fe.transformer.layers, layer_inputs, strict=True):
            _, w = layer.self_attn(
                inp,
                inp,
                inp,
                key_padding_mask=kpm,
                need_weights=True,
                average_attn_weights=True,
            )
            attn.append([rnd(row, 3) for row in w[0]])

        # per-slot transformer output norms
        ids = obs_t["card_ids"].long()
        id_embed = fe.id_embed(ids)
        x0 = fe.token_ln(fe.proj(torch.cat([obs_t["tokens"], id_embed], dim=-1)))
        x0 = x0 + fe.pos_embed
        z = fe.transformer(x0, src_key_padding_mask=kpm)
        slot_norms = z[0].norm(dim=-1)

    mask = td["mask"][i]
    lg = logits[0].numpy()
    masked = np.where(mask, lg, -np.inf)
    order = np.argsort(masked)[::-1][:6]
    p = np.exp(masked - masked.max())
    p = p / p.sum()

    cards = []
    for s in range(20):
        cid = int(td["obs_card_ids"][i][s])
        if cid == 0:
            cards.append(None)
            continue
        c = CARDS[cid]
        t = td["obs_tokens"][i][s]
        cards.append(
            {
                "slot": s,
                "zone": ZONES[s],
                "id": cid,
                "name": c.name,
                "cost": int(c.cost),
                "atk": int(t[8]),
                "def": int(t[9]),
                "abilities": c.abilities,
                "ready": bool(t[16] > 0),
            }
        )
    return {
        "cards": cards,
        "scalars": {
            n: round(float(v), 1) for n, v in zip(SCALAR_NAMES, td["obs_scalars"][i], strict=True)
        },
        "attn": attn,
        "slot_norms": rnd(slot_norms, 2),
        "acts": {k: rnd(v[0], 3) for k, v in acts.items()},
        "top_actions": [
            {
                "idx": int(j),
                "name": decode_action(int(j)),
                "p": round(float(p[j]), 3),
                "logit": round(float(lg[j]), 2),
            }
            for j in order
        ],
        "teacher_action": {"idx": int(act[i]), "name": decode_action(int(act[i]))},
        "value": round(float(value[0, 0]), 3),
        "labels": {
            "lethal_now": float(td["concept_lethal_now"][i]),
            "opp_threat_lethal": float(td["concept_opp_threat_lethal"][i]),
            "winner_side": int(td["winner"][i] == td["seat"][i]),
        },
    }


def flat_trace(i: int) -> dict:
    obs = fd["obs"][i : i + 1].astype(np.float32)
    with torch.no_grad():
        obs_t, _ = flat_m.policy.obs_to_tensor(obs)
        feats = flat_m.policy.extract_features(obs_t)
        acts = {}
        x = feats
        names = iter(["pi_a1", "pi_a2"])
        for m in flat_m.policy.mlp_extractor.policy_net:
            x = m(x)
            if not isinstance(m, torch.nn.Linear):
                acts[next(names)] = x
        latent_pi = x
        x = feats
        names = iter(["vf_a1", "vf_a2"])
        for m in flat_m.policy.mlp_extractor.value_net:
            x = m(x)
            if not isinstance(m, torch.nn.Linear):
                acts[next(names)] = x
        logits = flat_m.policy.action_net(latent_pi)
        value = flat_m.policy.value_net(x)
    mask = fd["mask"][i]
    lg = logits[0].numpy()
    masked = np.where(mask, lg, -np.inf)
    order = np.argsort(masked)[::-1][:6]
    p = np.exp(masked - masked.max())
    p = p / p.sum()
    return {
        "scalars": {
            n: round(float(v), 1) for n, v in zip(FLAT_SCALAR_NAMES, obs[0][:8], strict=True)
        },
        "acts": {k: rnd(v[0], 3) for k, v in acts.items()},
        "top_actions": [
            {
                "idx": int(j),
                "name": decode_action(int(j)),
                "p": round(float(p[j]), 3),
                "logit": round(float(lg[j]), 2),
            }
            for j in order
        ],
        "value": round(float(value[0, 0]), 3),
    }


demos = {}
for tag, i, blurb in DEMO:
    demos[tag] = {"index": i, "blurb": blurb, "token": token_trace(i), "flat": flat_trace(i)}
    print(f"demo {tag}: state {i} done")
for h in hooks:
    h.remove()

bundle = {
    "token_params": param_blocks(tok.policy),
    "flat_params": param_blocks(flat_m.policy),
    "token_total": int(sum(p.numel() for p in tok.policy.parameters())),
    "flat_total": int(sum(p.numel() for p in flat_m.policy.parameters())),
    "demos": demos,
}
out = json.dumps(bundle, separators=(",", ":"))
with open("runs/archviz-data.json", "w") as f:
    f.write(out)
print(f"wrote runs/archviz-data.json  {len(out) / 1e3:.0f} KB")
