"""E28c BC gate: do play-effect columns change behavior under BC? (raw vs raw+fx)

E28c feature completion (docs/reactive-limits-program.md): 44/160 cards — 7/8
blue items — carry play effects (player_hp/enemy_hp/card_draw) invisible to
the 17 numeric token features; the id-embedding channel that should supply
them never trains. This gate augments the e27 practicum's raw slot features
with the 3 effect columns (a pure function of card_id + zone) and re-runs the
E28b raw-gather arm against a raw+fx arm — identical everything else.

Pre-registered, ASYMMETRIC read (validation masked-argmax agreement + item
behavior, seeds 0/1): a positive on item behavior (item recall / item-state /
blue-item-state agreement) fast-tracks the PPO retrain; a NULL does NOT kill
the arm — BC against a search teacher cannot price consequence value (E27's
known residual). The decision-bearing test is the PPO retrain (gate 2).

Usage:
  python scripts/e28c_fx_bc.py --data runs/e27-concepts.npz \
      --seeds 0,1 --out runs/netprobe/e28c_fx_bc_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

USE_LO, USE_HI = 9, 113
NONE_SLOT = 20


def action_family(idx: np.ndarray) -> np.ndarray:
    fam = np.full(len(idx), 3, dtype=np.int64)
    fam[idx == 0] = 0
    fam[(idx >= 1) & (idx <= 8)] = 1
    fam[(idx >= USE_LO) & (idx < USE_HI)] = 2
    return fam


def build_action_table():
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
    ap.add_argument("--out", default="runs/netprobe/e28c_fx_bc_summary.json")
    args = ap.parse_args()

    import torch  # noqa: PLC0415 — [ml] extra
    import torch.nn as nn  # noqa: PLC0415
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.data.cards_db import load_cards  # noqa: PLC0415
    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.envs.distill import load_practicum, split_by_game  # noqa: PLC0415

    torch.set_num_threads(12)

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

    # fx columns from card_ids: hand slots only (fx encoder convention —
    # board effects are spent on play). Pure function of the stored practicum.
    cards = {c.id: c for c in load_cards()}
    fx_by_id = np.zeros((161, 3), dtype=np.float32)
    for c in cards.values():
        fx_by_id[c.id] = (c.player_hp, c.enemy_hp, c.card_draw)
    blue_ids = np.zeros(161, dtype=bool)
    for c in cards.values():
        blue_ids[c.id] = int(c.type) == 3
    ids_int = obs_np["card_ids"].astype(np.int64)  # (N, 20)
    fx_cols = fx_by_id[ids_int]  # (N, 20, 3)
    fx_cols[:, 8:, :] = 0.0  # board slots: zero
    fx_np = fx_cols.astype(np.float32)

    base = MaskablePPO.load(resolve_path(args.model), device="cpu")
    base.policy.set_training_mode(False)

    src_np, tgt_np, fam_np = build_action_table()
    src_t = torch.as_tensor(src_np)
    tgt_t = torch.as_tensor(tgt_np)
    fam_1h = torch.eye(4)[torch.as_tensor(fam_np)]

    class PointerHead(nn.Module):
        def __init__(self, d_tok: int, hidden: int, ctx_dim: int = 64):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(2 * d_tok + ctx_dim + 4, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, g: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
            b = g.size(0)
            gpad = torch.cat([g, torch.zeros(b, 1, g.size(2))], dim=1)
            src = gpad[:, src_t]
            tgt = gpad[:, tgt_t]
            c = ctx.unsqueeze(1).expand(-1, 155, -1)
            f = fam_1h.unsqueeze(0).expand(b, -1, -1)
            return self.mlp(torch.cat([src, tgt, c, f], dim=-1)).squeeze(-1)

    # capacity match: raw (33-d) hidden 186 -> 25,297 params (E28b);
    # rawfx (36-d) hidden 178 -> 25,277.
    HIDDEN = {"raw": 186, "rawfx": 178}
    D_TOK = {"raw": 33, "rawfx": 36}

    class Arm(nn.Module):
        def __init__(self, gather: str):
            super().__init__()
            import copy  # noqa: PLC0415

            self.fe = base.policy.features_extractor
            self.pi_tower = copy.deepcopy(base.policy.mlp_extractor.policy_net)
            self.gather = gather
            self.head = PointerHead(d_tok=D_TOK[gather], hidden=HIDDEN[gather])

        def forward(self, obs_t: dict, fx_t: torch.Tensor) -> torch.Tensor:
            fe = self.fe
            with torch.no_grad():
                ids = obs_t["card_ids"].long()
                emb = fe.id_embed(ids)
                raw = torch.cat([obs_t["tokens"], emb], dim=-1)  # (B,20,33)
                x0 = fe.token_ln(fe.proj(raw)) + fe.pos_embed
                kpm = obs_t["token_mask"] == 0
                all_pad = kpm.all(dim=1, keepdim=True)
                kpm = kpm & ~all_pad
                z = fe.transformer(x0, src_key_padding_mask=kpm)
                feats = fe.head(
                    torch.cat([z.reshape(z.size(0), -1), fe.scalar_mlp(obs_t["scalars"])], dim=-1)
                )
            latent = self.pi_tower(feats)
            fx36 = torch.cat([obs_t["tokens"], fx_t, emb], dim=-1)  # (B,20,36)
            g = fx36 if self.gather == "rawfx" else raw
            return self.head(g, latent)

        def trainable(self):
            return list(self.pi_tower.parameters()) + list(self.head.parameters())

    train_idx = val_idx = None

    def batches(idx: np.ndarray, bs: int, shuffle_rng=None):
        order = idx.copy()
        if shuffle_rng is not None:
            shuffle_rng.shuffle(order)
        for s in range(0, len(order), bs):
            sel = order[s : s + bs]
            obs_t = {k: torch.as_tensor(v[sel]) for k, v in obs_np.items()}
            yield (
                obs_t,
                torch.as_tensor(fx_np[sel]),
                torch.as_tensor(act_np[sel]),
                torch.as_tensor(mask_np[sel]),
                sel,
            )

    def evaluate(arm: Arm) -> dict:
        arm.eval()
        preds = np.zeros(len(val_idx), dtype=np.int64)
        with torch.no_grad():
            p = 0
            for obs_t, fx_t, _, m, sel in batches(val_idx, 2048):
                lg = arm(obs_t, fx_t).masked_fill(~m, -1e9)
                preds[p : p + len(sel)] = lg.argmax(dim=1).numpy()
                p += len(sel)
        teach = act_np[val_idx]
        agree = float((preds == teach).mean())
        t_fam, p_fam = action_family(teach), action_family(preds)
        t_use, p_use = t_fam == 2, p_fam == 2
        can_item = mask_np[val_idx][:, USE_LO:USE_HI].any(axis=1)
        # blue-item teacher states: teacher plays Use and the source card is blue
        t_src = np.where(t_use, (teach - USE_LO) // 13, 0)
        src_id = ids_int[val_idx, np.clip(t_src, 0, 7)]
        t_blue = t_use & blue_ids[src_id]
        lethal = arrays["concept_lethal_now"][val_idx] == 1
        return {
            "agreement": round(agree, 4),
            "item_recall": round(float(p_use[t_use].mean()), 4) if t_use.any() else None,
            "item_rate_pred": round(float(p_use[can_item].mean()), 4),
            "agreement_item_states": round(float((preds == teach)[t_use].mean()), 4),
            "agreement_blue_item_states": (
                round(float((preds == teach)[t_blue].mean()), 4) if t_blue.any() else None
            ),
            "n_blue_item_states": int(t_blue.sum()),
            "agreement_lethal_states": round(float((preds == teach)[lethal].mean()), 4),
        }

    results: dict = {"config": vars(args), "n": n, "seeds": {}}
    for seed in [int(s) for s in args.seeds.split(",")]:
        train_idx, val_idx = split_by_game(arrays["game_id"], args.val_frac, seed)
        train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
        print(f"\n#### seed {seed}: n={n} train={len(train_idx)} val={len(val_idx)}", flush=True)
        seed_res: dict = {}
        for arm_name in ("raw", "rawfx"):
            torch.manual_seed(seed)
            arm = Arm(arm_name)
            n_train = sum(p.numel() for p in arm.trainable())
            print(f"== seed {seed} arm {arm_name}: {n_train:,} trainable params ==", flush=True)
            opt = torch.optim.Adam(arm.trainable(), lr=args.lr)
            rng = np.random.default_rng(seed)
            last = None
            for ep in range(args.epochs):
                arm.train()
                tot, nb = 0.0, 0
                for obs_t, fx_t, a, m, _ in batches(train_idx, args.batch, rng):
                    lg = arm(obs_t, fx_t).masked_fill(~m, -1e9)
                    loss = nn.functional.cross_entropy(lg, a)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tot += float(loss.detach())
                    nb += 1
                last = evaluate(arm)
                print(
                    f"  ep {ep}: loss {tot / nb:.4f}  agree {last['agreement']:.4f}  "
                    f"item recall {last['item_recall']}  blue agree "
                    f"{last['agreement_blue_item_states']}",
                    flush=True,
                )
            seed_res[arm_name] = {"trainable_params": n_train, "final": last}
        results["seeds"][str(seed)] = seed_res

    for seed, sr in results["seeds"].items():
        a, b = sr["raw"]["final"], sr["rawfx"]["final"]
        print(
            f"seed {seed}: rawfx-raw agree {b['agreement'] - a['agreement']:+.4f}  "
            f"item_recall {a['item_recall']} -> {b['item_recall']}  "
            f"blue agree {a['agreement_blue_item_states']} -> {b['agreement_blue_item_states']}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
