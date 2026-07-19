"""E28 gate 1: pointer-style action head vs the standard dense head, under BC.

The E27 concept probes located a routing defect: hand-item information decays
0.98 (raw) -> 0.93 (slots) -> 0.76 (extractor head) -> ~0.61 (towers), so by
the time the standard dense action head (``action_net``, Linear 64->155)
decides, the net barely knows it holds a playable item. A pointer-style head
computes each action's logit FROM the slot tokens of the cards that action
involves, so per-card information reaches the decision by construction.

Gate design (cheapest falsification; no PPO spend):
  - Both arms initialize from ``depot:b0k/b0k_s0.zip``. The token extractor is
    FROZEN in both (identical features; the comparison isolates the head).
  - Arm A (control): fine-tune pi tower + the standard ``action_net``.
  - Arm B (pointer): fine-tune pi tower + a fresh PointerHead reading the
    transformer's per-slot outputs (src slot, tgt slot, tower context,
    action-family one-hot -> shared MLP -> logit).
  - Behavior-clone both on the SAME labeled held-out practicum
    (runs/e27-concepts.npz, mcts:100 teacher, seed-1M) with the same
    game-level split, epochs, lr, batch and seed.

Report (validation games only): masked-argmax agreement with the teacher,
item recall/precision on states where the teacher plays an item, agreement
on lethal_now states, per-family confusion. Gate passes if the pointer arm's
agreement is >= control AND its item behavior is clearly better.

Usage:
  python scripts/e28_pointer_bc.py --data runs/e27-concepts.npz \
      --epochs 10 --seed 0 --out runs/e28-gate1.json
"""

from __future__ import annotations

import argparse
import json

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
    """Per-action (src_slot, tgt_slot, family) over the 155 semantic indices.

    Token slots: 0-7 hand, 8-13 my board, 14-19 op board; 20 = none (zeros).
    """
    src = np.full(155, NONE_SLOT, dtype=np.int64)
    tgt = np.full(155, NONE_SLOT, dtype=np.int64)
    fam = np.zeros(155, dtype=np.int64)
    for i in range(1, 9):  # summon hand[s]
        src[i], fam[i] = i - 1, 1
    for i in range(USE_LO, USE_HI):  # use hand[s] -> target code
        s, tc = divmod(i - USE_LO, 13)
        src[i], fam[i] = s, 2
        if tc < 6:
            tgt[i] = 8 + tc
        elif tc < 12:
            tgt[i] = 14 + (tc - 6)
    for i in range(USE_HI, 155):  # attack my_board[a] -> target code
        a, tc = divmod(i - USE_HI, 7)
        src[i], fam[i] = 8 + a, 3
        if tc < 6:
            tgt[i] = 14 + tc
    return src, tgt, fam


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="depot:b0k/b0k_s0.zip")
    ap.add_argument("--data", default="runs/e27-concepts.npz")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument(
        "--arms", default="control,pointer", help="comma list: control,control_mlp,pointer"
    )
    ap.add_argument("--out", default="runs/e28-gate1.json")
    args = ap.parse_args()

    import torch  # noqa: PLC0415 — [ml] extra
    import torch.nn as nn  # noqa: PLC0415
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.envs.distill import load_practicum, split_by_game  # noqa: PLC0415

    torch.manual_seed(args.seed)
    torch.set_num_threads(16)

    arrays, _ = load_practicum(args.data)
    n = len(arrays["action"])
    train_idx, val_idx = split_by_game(arrays["game_id"], args.val_frac, args.seed)
    train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
    print(f"n={n}  train={len(train_idx)}  val={len(val_idx)}")

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

    src_t, tgt_t, fam_t = build_action_table()
    src_t = torch.as_tensor(src_t)
    tgt_t = torch.as_tensor(tgt_t)
    fam_1h = torch.eye(4)[torch.as_tensor(fam_t)]  # (155, 4)

    class PointerHead(nn.Module):
        """logit(a) = MLP([z_src(a), z_tgt(a), tower_ctx, family_onehot(a)])."""

        def __init__(self, d_model: int = 64, ctx_dim: int = 64, hidden: int = 128):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(2 * d_model + ctx_dim + 4, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, z: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
            b = z.size(0)
            zpad = torch.cat([z, torch.zeros(b, 1, z.size(2))], dim=1)  # (B,21,d)
            src = zpad[:, src_t]  # (B,155,d)
            tgt = zpad[:, tgt_t]
            c = ctx.unsqueeze(1).expand(-1, 155, -1)
            f = fam_1h.unsqueeze(0).expand(b, -1, -1)
            return self.mlp(torch.cat([src, tgt, c, f], dim=-1)).squeeze(-1)

    class Arm(nn.Module):
        """Frozen b0k extractor + trainable pi-tower clone + one of three heads.

        head_kind: "dense" = b0k's action_net (Linear 64->155, warm);
        "mlp" = fresh capacity-matched control (Linear 64->113 -> ReLU ->
        Linear 113->155, ~25k params like PointerHead — separates
        "structural slot access" from "more head parameters");
        "pointer" = fresh PointerHead.
        """

        def __init__(self, head_kind: str):
            super().__init__()
            import copy  # noqa: PLC0415

            self.fe = base.policy.features_extractor  # frozen, shared
            self.pi_tower = copy.deepcopy(base.policy.mlp_extractor.policy_net)
            self.pointer = head_kind == "pointer"
            if head_kind == "pointer":
                self.head = PointerHead(hidden=args.hidden)
            elif head_kind == "mlp":
                self.head = nn.Sequential(nn.Linear(64, 113), nn.ReLU(), nn.Linear(113, 155))
            else:
                self.head = copy.deepcopy(base.policy.action_net)

        def forward(self, obs_t: dict) -> torch.Tensor:
            fe = self.fe
            with torch.no_grad():  # extractor frozen in both arms
                ids = obs_t["card_ids"].long()
                x = fe.token_ln(fe.proj(torch.cat([obs_t["tokens"], fe.id_embed(ids)], dim=-1)))
                x = x + fe.pos_embed
                kpm = obs_t["token_mask"] == 0
                all_pad = kpm.all(dim=1, keepdim=True)
                kpm = kpm & ~all_pad
                z = fe.transformer(x, src_key_padding_mask=kpm)  # (B,20,d)
                feats = fe.head(
                    torch.cat([z.reshape(z.size(0), -1), fe.scalar_mlp(obs_t["scalars"])], dim=-1)
                )
            latent = self.pi_tower(feats)
            if self.pointer:
                return self.head(z, latent)
            return self.head(latent)

        def trainable(self):
            return list(self.pi_tower.parameters()) + list(self.head.parameters())

    def batches(idx: np.ndarray, bs: int, shuffle_rng=None):
        order = idx.copy()
        if shuffle_rng is not None:
            shuffle_rng.shuffle(order)
        for s in range(0, len(order), bs):
            sel = order[s : s + bs]
            obs_t = {k: torch.as_tensor(v[sel]) for k, v in obs_np.items()}
            yield (
                obs_t,
                torch.as_tensor(act_np[sel]),
                torch.as_tensor(mask_np[sel]),
                sel,
            )

    def evaluate(arm: Arm) -> dict:
        arm.eval()
        preds = np.zeros(len(val_idx), dtype=np.int64)
        with torch.no_grad():
            p = 0
            for obs_t, _, m, sel in batches(val_idx, 2048):
                lg = arm(obs_t)
                lg = lg.masked_fill(~m, -1e9)
                preds[p : p + len(sel)] = lg.argmax(dim=1).numpy()
                p += len(sel)
        teach = act_np[val_idx]
        agree = float((preds == teach).mean())
        t_fam, p_fam = action_family(teach), action_family(preds)
        t_use = t_fam == 2
        p_use = p_fam == 2
        can_item = mask_np[val_idx][:, USE_LO:USE_HI].any(axis=1)
        lethal = arrays["concept_lethal_now"][val_idx] == 1
        conf = [[int(((t_fam == i) & (p_fam == j)).sum()) for j in range(4)] for i in range(4)]
        return {
            "agreement": round(agree, 4),
            "item_recall": round(float(p_use[t_use].mean()), 4) if t_use.any() else None,
            "item_precision": round(float(t_use[p_use].mean()), 4) if p_use.any() else None,
            "item_rate_pred": round(float(p_use[can_item].mean()), 4),
            "item_rate_teacher": round(float(t_use[can_item].mean()), 4),
            "agreement_lethal_states": round(float((preds == teach)[lethal].mean()), 4),
            "agreement_item_states": round(float((preds == teach)[t_use].mean()), 4),
            "family_confusion_rows_teacher": conf,
        }

    results: dict = {"config": vars(args)}
    arm_specs = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm_name in arm_specs:
        torch.manual_seed(args.seed)
        arm = Arm({"control": "dense", "control_mlp": "mlp", "pointer": "pointer"}[arm_name])
        n_train = sum(p.numel() for p in arm.trainable())
        print(f"\n== arm {arm_name}: {n_train:,} trainable params ==")
        opt = torch.optim.Adam(arm.trainable(), lr=args.lr)
        rng = np.random.default_rng(args.seed)
        history = []
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
            ev = evaluate(arm)
            history.append({"epoch": ep, "train_loss": round(tot / nb, 4), **ev})
            print(
                f"ep {ep}: loss {tot / nb:.4f}  val agree {ev['agreement']:.4f}  "
                f"item recall {ev['item_recall']}"
            )
        results[arm_name] = {"trainable_params": n_train, "final": history[-1], "history": history}

    # reference: untouched b0k argmax on the same validation rows
    class Base(Arm):
        def __init__(self):
            nn.Module.__init__(self)
            self.fe = base.policy.features_extractor
            self.pi_tower = base.policy.mlp_extractor.policy_net
            self.pointer = False
            self.head = base.policy.action_net

    results["b0k_reference"] = evaluate(Base())
    print("\nb0k reference:", results["b0k_reference"])

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {args.out}")

    c, p = results["control"]["final"], results["pointer"]["final"]
    print(f"\n{'':16}{'control':>10}{'pointer':>10}")
    for k in (
        "agreement",
        "item_recall",
        "item_precision",
        "item_rate_pred",
        "agreement_lethal_states",
        "agreement_item_states",
    ):
        print(f"{k:16}{c[k]!s:>10}{p[k]!s:>10}")
    print(f"teacher item rate (where legal): {c['item_rate_teacher']}")


if __name__ == "__main__":
    main()
