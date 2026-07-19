"""E28b: does the pointer head need the transformer's mixing? (BC ablation gate)

The E28 gate-1 pointer head reads the transformer's per-slot output ``z``.
The encoder introspection (docs/notes/e28-encoder-viz.html) found attention is
near-uniform context pooling and the card-id embedding never trained — so is
the cross-card mixing load-bearing for the pointer gather, or is slot-
structured access alone the whole mechanism?

Gate design (E28 gate-1 protocol, one variable changed; frozen b0k extractor,
trainable pi-tower clone + fresh pointer head, identical BC on the e27
practicum, identical split/epochs/lr/batch; context ``latent_pi`` comes
through the full frozen trunk in EVERY arm, so only the gathered tokens vary):

  - ``pointer``  gathers z            (post-attention; gate-1 reference re-run)
  - ``premix``   gathers x0           (token_ln(proj([feats,id_embed]))+pos_embed,
                                       BEFORE the transformer)
  - ``raw``      gathers [feats|id_embed]  (33-d, bypassing even the learned
                                       projection; hidden widened to match
                                       trainable params)

Pre-registered decision rule (validation masked-argmax agreement, seeds 0/1):
  - premix >= pointer - 0.005 on both seeds  -> mixing adds nothing to the
    gather; the transformer is droppable from the policy path (slim-extractor
    retrain arm opens, informs E29).
  - premix <= pointer - 0.02 on both seeds   -> mixing is real; keep it.
  - otherwise ambiguous — judge with the item/lethal breakdowns.

Scope caveat: latent_pi still uses the transformer in all arms. This gate
isolates the GATHERED-token question; a fully transformer-free trunk is a
retrain question (and touches the critic).

Usage:
  python scripts/e28b_mix_ablation.py --data runs/e27-concepts.npz \
      --seeds 0,1 --out runs/netprobe/e28b_mix_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

USE_LO, USE_HI = 9, 113  # semantic Use range; families: 0 pass, 1 summon, 2 use, 3 attack
NONE_SLOT = 20  # virtual "no slot" index -> zero token row


def action_family(idx: np.ndarray) -> np.ndarray:
    fam = np.full(len(idx), 3, dtype=np.int64)
    fam[idx == 0] = 0
    fam[(idx >= 1) & (idx <= 8)] = 1
    fam[(idx >= USE_LO) & (idx < USE_HI)] = 2
    return fam


def build_action_table():
    """Per-action (src_slot, tgt_slot, family) over the 155 semantic indices."""
    src = np.full(155, NONE_SLOT, dtype=np.int64)
    tgt = np.full(155, NONE_SLOT, dtype=np.int64)
    fam = np.zeros(155, dtype=np.int64)
    for i in range(1, 9):
        src[i], fam[i] = i - 1, 1
    for i in range(USE_LO, USE_HI):
        s, tc = divmod(i - USE_LO, 13)
        src[i], fam[i] = s, 2
        if tc < 6:
            tgt[i] = 8 + tc
        elif tc < 12:
            tgt[i] = 14 + (tc - 6)
    for i in range(USE_HI, 155):
        a, tc = divmod(i - USE_HI, 7)
        src[i], fam[i] = 8 + a, 3
        if tc < 6:
            tgt[i] = 14 + tc
    return src, tgt, fam


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="depot:b0k/b0k_s0.zip")
    ap.add_argument("--data", default="runs/e27-concepts.npz")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--arms", default="pointer,premix,raw")
    ap.add_argument("--out", default="runs/netprobe/e28b_mix_summary.json")
    args = ap.parse_args()

    import torch  # noqa: PLC0415 — [ml] extra
    import torch.nn as nn  # noqa: PLC0415
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.envs.distill import load_practicum, split_by_game  # noqa: PLC0415

    torch.set_num_threads(16)

    arrays, _ = load_practicum(args.data)
    n = len(arrays["action"])

    obs_np = {
        "tokens": arrays["obs_tokens"].astype(np.float32),
        "card_ids": arrays["obs_card_ids"].astype(np.float32),
        "token_mask": arrays["obs_token_mask"].astype(np.float32),
        "scalars": arrays["obs_scalars"].astype(np.float32),
    }
    act_np = arrays["action"].astype(np.int64)
    mask_np = arrays["mask"].astype(bool)

    base = MaskablePPO.load(resolve_path(args.model), device="cpu")
    base.policy.set_training_mode(False)

    src_np, tgt_np, fam_np = build_action_table()
    src_t = torch.as_tensor(src_np)
    tgt_t = torch.as_tensor(tgt_np)
    fam_1h = torch.eye(4)[torch.as_tensor(fam_np)]  # (155, 4)

    class PointerHead(nn.Module):
        """logit(a) = MLP([g_src(a), g_tgt(a), tower_ctx, family_onehot(a)]).

        ``g`` is whatever per-slot representation the arm gathers from
        (z, x0, or raw features) — d_tok wide.
        """

        def __init__(self, d_tok: int, ctx_dim: int = 64, hidden: int = 128):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(2 * d_tok + ctx_dim + 4, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, g: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
            b = g.size(0)
            gpad = torch.cat([g, torch.zeros(b, 1, g.size(2))], dim=1)  # (B,21,d)
            src = gpad[:, src_t]
            tgt = gpad[:, tgt_t]
            c = ctx.unsqueeze(1).expand(-1, 155, -1)
            f = fam_1h.unsqueeze(0).expand(b, -1, -1)
            return self.mlp(torch.cat([src, tgt, c, f], dim=-1)).squeeze(-1)

    # capacity match for the 33-d raw arm: pointer head at d_tok=64 has
    # (196+1)*128 + 129 = 25,345 params; raw at d_tok=33 has (134+1)*h + h+1
    # = 136h + 1 -> h = 186 gives 25,297 (delta 0.2%).
    RAW_HIDDEN = 186

    class Arm(nn.Module):
        """Frozen b0k extractor + trainable pi-tower clone + pointer head.

        gather: "z" (post-attention), "x0" (pre-attention embedded slots),
        or "raw" (unprojected [token_feats | id_embed], 33-d).
        """

        def __init__(self, gather: str):
            super().__init__()
            import copy  # noqa: PLC0415

            self.fe = base.policy.features_extractor  # frozen, shared
            self.pi_tower = copy.deepcopy(base.policy.mlp_extractor.policy_net)
            self.gather = gather
            d_tok = {"z": 64, "x0": 64, "raw": 33}[gather]
            hidden = RAW_HIDDEN if gather == "raw" else args.hidden
            self.head = PointerHead(d_tok=d_tok, hidden=hidden)

        def forward(self, obs_t: dict) -> torch.Tensor:
            fe = self.fe
            with torch.no_grad():  # extractor frozen in every arm
                ids = obs_t["card_ids"].long()
                raw = torch.cat([obs_t["tokens"], fe.id_embed(ids)], dim=-1)  # (B,20,33)
                x0 = fe.token_ln(fe.proj(raw)) + fe.pos_embed
                kpm = obs_t["token_mask"] == 0
                all_pad = kpm.all(dim=1, keepdim=True)
                kpm = kpm & ~all_pad
                z = fe.transformer(x0, src_key_padding_mask=kpm)  # (B,20,64)
                feats = fe.head(
                    torch.cat([z.reshape(z.size(0), -1), fe.scalar_mlp(obs_t["scalars"])], dim=-1)
                )
            latent = self.pi_tower(feats)
            g = {"z": z, "x0": x0, "raw": raw}[self.gather]
            return self.head(g, latent)

        def trainable(self):
            return list(self.pi_tower.parameters()) + list(self.head.parameters())

    train_idx = val_idx = None  # set per seed

    def batches(idx: np.ndarray, bs: int, shuffle_rng=None):
        order = idx.copy()
        if shuffle_rng is not None:
            shuffle_rng.shuffle(order)
        for s in range(0, len(order), bs):
            sel = order[s : s + bs]
            obs_t = {k: torch.as_tensor(v[sel]) for k, v in obs_np.items()}
            yield obs_t, torch.as_tensor(act_np[sel]), torch.as_tensor(mask_np[sel]), sel

    def evaluate(arm: Arm) -> dict:
        arm.eval()
        preds = np.zeros(len(val_idx), dtype=np.int64)
        with torch.no_grad():
            p = 0
            for obs_t, _, m, sel in batches(val_idx, 2048):
                lg = arm(obs_t).masked_fill(~m, -1e9)
                preds[p : p + len(sel)] = lg.argmax(dim=1).numpy()
                p += len(sel)
        teach = act_np[val_idx]
        agree = float((preds == teach).mean())
        t_fam, p_fam = action_family(teach), action_family(preds)
        t_use = t_fam == 2
        p_use = p_fam == 2
        can_item = mask_np[val_idx][:, USE_LO:USE_HI].any(axis=1)
        lethal = arrays["concept_lethal_now"][val_idx] == 1
        return {
            "agreement": round(agree, 4),
            "item_recall": round(float(p_use[t_use].mean()), 4) if t_use.any() else None,
            "item_rate_pred": round(float(p_use[can_item].mean()), 4),
            "agreement_lethal_states": round(float((preds == teach)[lethal].mean()), 4),
            "agreement_item_states": round(float((preds == teach)[t_use].mean()), 4),
        }

    results: dict = {"config": vars(args), "n": n, "seeds": {}}
    for seed in [int(s) for s in args.seeds.split(",")]:
        train_idx, val_idx = split_by_game(arrays["game_id"], args.val_frac, seed)
        train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
        print(f"\n#### seed {seed}: n={n} train={len(train_idx)} val={len(val_idx)}")
        seed_res: dict = {}
        for arm_name in [a.strip() for a in args.arms.split(",") if a.strip()]:
            torch.manual_seed(seed)
            arm = Arm({"pointer": "z", "premix": "x0", "raw": "raw"}[arm_name])
            n_train = sum(p.numel() for p in arm.trainable())
            print(f"== seed {seed} arm {arm_name}: {n_train:,} trainable params ==")
            opt = torch.optim.Adam(arm.trainable(), lr=args.lr)
            rng = np.random.default_rng(seed)
            last = None
            for ep in range(args.epochs):
                arm.train()
                tot, nb = 0.0, 0
                for obs_t, a, m, _ in batches(train_idx, args.batch, rng):
                    lg = arm(obs_t).masked_fill(~m, -1e9)
                    loss = nn.functional.cross_entropy(lg, a)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tot += float(loss)
                    nb += 1
                last = evaluate(arm)
                print(
                    f"  ep {ep}: loss {tot / nb:.4f}  agree {last['agreement']:.4f}  "
                    f"item recall {last['item_recall']}"
                )
            seed_res[arm_name] = {"trainable_params": n_train, "final": last}
        results["seeds"][str(seed)] = seed_res

    # pre-registered verdict
    deltas = []
    for seed, sr in results["seeds"].items():
        d = sr["premix"]["final"]["agreement"] - sr["pointer"]["final"]["agreement"]
        deltas.append(d)
        print(f"seed {seed}: premix - pointer = {d:+.4f}")
    if all(d >= -0.005 for d in deltas):
        verdict = "MIXING_UNNECESSARY (premix within 0.005 of pointer on all seeds)"
    elif all(d <= -0.02 for d in deltas):
        verdict = "MIXING_REAL (premix >= 0.02 below pointer on all seeds)"
    else:
        verdict = "AMBIGUOUS (judge with breakdowns)"
    results["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
