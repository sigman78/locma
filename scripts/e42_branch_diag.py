"""E42 mechanism diagnostic: does e42h_gen9 USE its hist branch?

Branch magnitude, argmax agreement and value shift with the hist vector zeroed,
on 40 e42h-vs-e42c games (seed 90M). Result 2026-09-26: branch/scalar magnitude
0.115, argmax agreement 0.975, TV 0.025, |dV| 0.072 (|V| 0.41) over 1774 decisions.
"""

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from locma.core import battle as battlemod
from locma.core.engine import make_battle_view, run_game
from locma.envs.encode import action_mask, encode_battle_tokens
from locma.policies.registry import make_policy

m = MaskablePPO.load("runs/e36_e42h_gen9.zip", device="cpu")
fe = m.policy.features_extractor
print(
    "hist last-layer |W|_F:",
    float(fe.hist_mlp[-1].weight.norm()),
    " scalar_mlp Linear |W|_F:",
    float(fe.scalar_mlp[1].weight.norm()),
)
obs_rows, masks = [], []
subj = make_policy("ppo:runs/e36_e42h_gen9.zip,depot:ldraft/ldraft_s0.zip")
opp = make_policy("ppo:runs/e36_e42c_gen9.zip,depot:ldraft/ldraft_s0.zip")


def hook(seat, action, gs):
    if seat == 0:
        v = make_battle_view(gs)
        obs_rows.append(encode_battle_tokens(v, "fxh"))
        masks.append(action_mask(v, battlemod.battle_legal(gs)))


for s in range(40):
    run_game(subj, opp, seed=90_000_000 + s, on_pre_step=hook)
keys = obs_rows[0].keys()
obs = {k: np.stack([r[k] for r in obs_rows]) for k in keys}
mask = torch.as_tensor(np.stack(masks)).bool()


def logits(o):
    with torch.no_grad():
        obs_t, _ = m.policy.obs_to_tensor(o)
        dist = m.policy.get_distribution(obs_t)
        lg = dist.distribution.logits.clone()
        lg[~mask] = -1e9
        return lg, obs_t


lr, obs_t = logits(obs)
o0 = dict(obs)
o0["hist"] = np.zeros_like(obs["hist"])
l0, obs0_t = logits(o0)
with torch.no_grad():
    s = fe.scalar_mlp(obs_t["scalars"])
    h = fe.hist_mlp(obs_t["hist"])
    vr = m.policy.predict_values(obs_t)
    v0 = m.policy.predict_values(obs0_t)
print("decisions:", len(obs_rows))
print("mean |hist branch| / |scalar branch|:", float(h.norm(dim=1).mean() / s.norm(dim=1).mean()))
agree = float((lr.argmax(1) == l0.argmax(1)).float().mean())
print("argmax agreement real-hist vs zero-hist:", agree)
pr = torch.softmax(lr, 1)
p0 = torch.softmax(l0, 1)
print("mean TV distance of action dists:", float(0.5 * (pr - p0).abs().sum(1).mean()))
dv = float((vr - v0).abs().mean())
print("mean |dV| real vs zero-hist:", dv, " (|V| mean", float(vr.abs().mean()), ")")
