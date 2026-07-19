"""Export encoder-introspection data for the E28 encoder-viz page.

For b0k_s0 and e28p_s0: card-id embedding PCA + property probes (vs a
random-init control), pos_embed cosine geometry, per-layer/per-head attention
on real game states (manual transformer forward, verified against the module
output), and a z-token PCA cloud. Writes runs/netprobe/encviz-data.json; the
JSON is inlined into docs/notes/e28-encoder-viz.html.

Run from the repo root: ``uv run --extra ml python scripts/e28_encviz_export.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

import locma.envs.pointer_head  # noqa: F401  (e28p checkpoints pickle this class)
from locma.data.cards_db import load_cards
from locma.depot.core import resolve_path
from locma.envs.battle_env import BattleEnv
from locma.policies.registry import make_policy

OUT = Path("runs/netprobe/encviz-data.json")
N_ATT_STATES = 6
CLOUD_CAP = 600
RNG = np.random.default_rng(0)

NETS = {
    "b0k": "depot:b0k/b0k_s0.zip",
    "e28p": "depot:e28p/e28p_s0.zip",
}


def r(x, nd=4):
    if isinstance(x, (list, tuple)):
        return [r(v, nd) for v in x]
    return round(float(x), nd)


# ---------------------------------------------------------------- cards
cards = load_cards()
card_by_id = {c.id: c for c in cards}
cards_json = [
    {
        "id": c.id,
        "name": c.name,
        "t": int(c.type),
        "cost": c.cost,
        "atk": c.attack,
        "def": c.defense,
        "ab": c.abilities,
    }
    for c in sorted(cards, key=lambda c: c.id)
]
assert len(cards_json) == 160


# ---------------------------------------------------------------- probes
def cv_r2(X, y, folds=5):
    """5-fold CV R^2 of ridge (lstsq + tiny l2) predicting y from X."""
    n = len(y)
    idx = RNG.permutation(n)
    scores = []
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        Xtr = np.c_[X[tr], np.ones(len(tr))]
        Xte = np.c_[X[te], np.ones(len(te))]
        A = Xtr.T @ Xtr + 1e-3 * np.eye(Xtr.shape[1])
        w = np.linalg.solve(A, Xtr.T @ y[tr])
        pred = Xte @ w
        ss_res = np.sum((y[te] - pred) ** 2)
        ss_tot = np.sum((y[te] - y[tr].mean()) ** 2)
        scores.append(1 - ss_res / max(ss_tot, 1e-9))
    return float(np.mean(scores))


def cv_acc(X, y, folds=5):
    """5-fold CV accuracy of a one-vs-rest ridge classifier (argmax over
    per-class ridge regressions on one-hot targets) — sklearn-free linear probe."""
    classes = np.unique(y)
    Y = (y[:, None] == classes[None, :]).astype(float)
    n = len(y)
    idx = RNG.permutation(n)
    scores = []
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        Xtr = np.c_[X[tr], np.ones(len(tr))]
        Xte = np.c_[X[te], np.ones(len(te))]
        A = Xtr.T @ Xtr + 1e-2 * np.eye(Xtr.shape[1])
        W = np.linalg.solve(A, Xtr.T @ Y[tr])
        pred = classes[np.argmax(Xte @ W, axis=1)]
        scores.append(float(np.mean(pred == y[te])))
    return float(np.mean(scores))


def pca2(X):
    mu = X.mean(0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    xy = Xc @ Vt[:2].T
    var = (S**2) / np.sum(S**2)
    return xy, var[:2]


def probe_block(E):
    """Probe card properties from an embedding matrix E (160, d)."""
    cost = np.array([c["cost"] for c in cards_json], float)
    atk = np.array([c["atk"] for c in cards_json], float)
    dfn = np.array([c["def"] for c in cards_json], float)
    typ = np.array([c["t"] for c in cards_json])
    grd = np.array([1 if "G" in c["ab"] else 0 for c in cards_json])
    return {
        "type_acc": r(cv_acc(E, typ)),
        "guard_acc": r(cv_acc(E, grd)),
        "cost_r2": r(cv_r2(E, cost)),
        "atk_r2": r(cv_r2(E, atk)),
        "def_r2": r(cv_r2(E, dfn)),
        "base": {
            "type_majority": r(np.mean(typ == np.bincount(typ).argmax())),
            "guard_majority": r(np.mean(grd == np.bincount(grd).argmax())),
        },
    }


# ---------------------------------------------------------------- game states
print("playing a probe game (e28p_s0 vs scripted, token obs)...")
drv = MaskablePPO.load(resolve_path(NETS["e28p"]), device="cpu")
env = BattleEnv(make_policy("scripted"), seed=424242, obs_mode="token")
obs, _ = env.reset()
states = []
done = False
while not done:
    mask = env.action_masks()
    hp = [p.health for p in env.gs.players]
    states.append(
        {
            "obs": {k: np.array(v, copy=True) for k, v in obs.items()},
            "my_hp": int(hp[0]),
            "op_hp": int(hp[1]),
            "n_tokens": int(obs["token_mask"].sum()),
        }
    )
    act, _ = drv.predict(obs, action_masks=mask, deterministic=True)
    states[-1]["action"] = int(act)
    obs, rew, term, trunc, _ = env.step(int(act))
    done = term or trunc
print(f"  {len(states)} agent decisions, result reward={rew}")

att_idx = sorted(set(np.linspace(0, len(states) - 1, N_ATT_STATES).round().astype(int)))


def slot_zone(s):
    return "hand" if s < 8 else ("my" if s < 14 else "op")


def state_names(st):
    ids = st["obs"]["card_ids"].astype(int)
    msk = st["obs"]["token_mask"]
    out = []
    for s in range(20):
        if msk[s] < 0.5 or ids[s] == 0:
            out.append(None)
        else:
            c = card_by_id[int(ids[s])]
            out.append(c.name)
    return out


# ---------------------------------------------------------------- per-net export
def export_net(tag, ref):
    print(f"[{tag}] loading {ref}")
    model = MaskablePPO.load(resolve_path(ref), device="cpu")
    fe = model.policy.features_extractor
    fe.eval()

    # --- card-id embedding (rows 1..160; row 0 is PAD)
    E = fe.id_embed.weight.detach().numpy()[1:161].copy()
    xy, var = pca2(E)
    Ectl = RNG.normal(size=E.shape)
    probes = probe_block(E)
    probes_ctl = probe_block(Ectl)

    # --- pos_embed geometry
    P = fe.pos_embed.detach().numpy()[0]  # (20, 64)
    norms = np.linalg.norm(P, axis=1)
    Pn = P / np.maximum(norms[:, None], 1e-9)
    cos = Pn @ Pn.T

    # --- manual transformer forward with attention capture
    def fwd(obs_batch):
        with torch.no_grad():
            tokens = torch.as_tensor(
                np.stack([o["tokens"] for o in obs_batch]), dtype=torch.float32
            )
            ids = torch.as_tensor(np.stack([o["card_ids"] for o in obs_batch])).long()
            tmask = torch.as_tensor(
                np.stack([o["token_mask"] for o in obs_batch]), dtype=torch.float32
            )
            x = fe.token_ln(fe.proj(torch.cat([tokens, fe.id_embed(ids)], dim=-1)))
            x = x + fe.pos_embed
            x0 = x.clone()
            kpm = tmask == 0
            att_all = []
            for layer in fe.transformer.layers:
                sa, w = layer.self_attn(
                    x,
                    x,
                    x,
                    key_padding_mask=kpm,
                    need_weights=True,
                    average_attn_weights=False,
                )
                att_all.append(w)  # (B, H, 20, 20)
                x = layer.norm1(x + sa)
                ff = layer.linear2(layer.activation(layer.linear1(x)))
                x = layer.norm2(x + ff)
            zref = fe.transformer(x0, src_key_padding_mask=kpm)
            diff = (x - zref).abs().max().item()
            return x0.numpy(), x.numpy(), [a.numpy() for a in att_all], diff

    all_obs = [st["obs"] for st in states]
    x0, z, atts, sanity = fwd(all_obs)
    print(f"  sanity: manual forward vs module max|diff| = {sanity:.2e}")
    assert sanity < 1e-4, "manual transformer forward drifted from module output"

    # --- attention states
    att_states = []
    for si in att_idx:
        st = states[si]
        msk = st["obs"]["token_mask"].astype(int).tolist()
        names = state_names(st)
        mats = [
            [r(atts[li][si, h].tolist(), 3) for h in range(atts[li].shape[1])]
            for li in range(len(atts))
        ]
        # per-slot mixing: 1 - cos(x0, z)
        a, b = x0[si], z[si]
        cosd = np.sum(a * b, 1) / np.maximum(
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-9
        )
        att_states.append(
            {
                "label": f"decision {si + 1}/{len(states)} · hp {st['my_hp']}-"
                f"{st['op_hp']} · {st['n_tokens']} cards",
                "ids": st["obs"]["card_ids"].astype(int).tolist(),
                "mask": msk,
                "names": names,
                "att": mats,
                "mix": r((1 - cosd).tolist(), 3),
            }
        )

    # --- z-token cloud over all states
    pts, metas = [], []
    for si, st in enumerate(states):
        msk = st["obs"]["token_mask"]
        ids = st["obs"]["card_ids"].astype(int)
        for s in range(20):
            if msk[s] < 0.5:
                continue
            pts.append(z[si, s])
            metas.append({"zone": slot_zone(s), "tid": int(ids[s]), "st": si})
    pts = np.array(pts)
    if len(pts) > CLOUD_CAP:
        keep = RNG.choice(len(pts), CLOUD_CAP, replace=False)
        pts, metas = pts[keep], [metas[i] for i in keep]
    cxy, cvar = pca2(pts)
    cloud = [{"x": r(cxy[i, 0], 3), "y": r(cxy[i, 1], 3), **metas[i]} for i in range(len(metas))]

    return {
        "emb_xy": r(xy.tolist(), 3),
        "emb_var": r(var.tolist(), 3),
        "probes": probes,
        "probes_ctl": probes_ctl,
        "pos_cos": r(cos.tolist(), 3),
        "pos_norm": r(norms.tolist(), 3),
        "att_states": att_states,
        "cloud": cloud,
        "cloud_var": r(cvar.tolist(), 3),
        "sanity": r(sanity, 8),
    }


data = {
    "cards": cards_json,
    "nets": {tag: export_net(tag, ref) for tag, ref in NETS.items()},
    "meta": {
        "n_states": len(states),
        "att_idx": [int(i) for i in att_idx],
        "game": "e28p_s0 (deterministic) vs scripted, balanced draft, seed 424242",
    },
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, separators=(",", ":")))
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
for tag in NETS:
    p, c = data["nets"][tag]["probes"], data["nets"][tag]["probes_ctl"]
    print(f"[{tag}] probes trained vs random-ctl:")
    for k in ("type_acc", "guard_acc", "cost_r2", "atk_r2", "def_r2"):
        print(f"    {k:10s} {p[k]:+.3f}  vs ctl {c[k]:+.3f}")
    print(f"    (majority: type {p['base']['type_majority']}, guard {p['base']['guard_majority']})")
