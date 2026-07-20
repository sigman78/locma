"""E30 turn-plan BC diagnostic: is the ~0.37 agreement cap a FACTORIZATION
artifact, or representational?

The per-micro-action BC-agreement plateau (2026-06-27) was measured with a
head that predicts each action independently from the current state. E14a
localized the failure to turn-level branching ("starts lines, can't finish
them"). E30 asks: if the head also sees the PLAN SO FAR (the actions already
taken this turn), does agreement clear the cap?

Controlled diagnostic (one variable): both arms share identical frozen state
features (raw token obs + scalars) and identical BC training; they differ
ONLY in the plan-so-far conditioning.

  - factored:  logits = MLP(state)                       (reproduces the cap)
  - autoreg:   logits = MLP([state, GRU(plan_so_far)])   (teacher-forced over
               the actions taken earlier this turn)

Reads (held-out turns, masked-argmax agreement vs the search teacher):
  - overall per-step agreement, factored vs autoreg;
  - the MULTI-ACTION-TURN subset (turns with >=2 teacher decisions) — where a
    plan effect must concentrate if factorization is the bottleneck;
  - whole-turn exact-match rate.

Decision (pre-registered): autoreg - factored on the multi-action subset
  >= +0.10  -> factorization is the bottleneck; open an EXIT-style plan-head
              training arm.
  <= +0.03  -> the state already carries the plan; the cap is representational
              -> E30 closes.
  between   -> ambiguous; judge with the whole-turn number.

Data: a search-teacher token practicum (records are in play order; turns are
split at Pass). Default runs/practicum-token.npz (mcts:100, n~35k).

Usage: python scripts/e30_plan_bc.py [--data runs/practicum-token.npz]
Output: runs/netprobe/e30_plan_bc.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ACTION_SIZE = 155
PAD_ACTION = ACTION_SIZE  # embedding pad index for "no prior action"
MAX_PLAN = 12  # cap prior-actions per step (turns are short)


def reconstruct_turns(game_id: np.ndarray, turn_val: np.ndarray) -> list[list[int]]:
    """Group record indices into turns by (game_id, turn counter).

    Records are in play order and only the teacher's own decisions are stored,
    so a maximal run with the same game_id AND the same turn counter
    (``obs_scalars[:, 0]``, the round number) is exactly one teacher turn. Pass
    is NOT a reliable delimiter here (most turns end without an explicit Pass
    record — only ~4.5% of decisions are Pass)."""
    turns: list[list[int]] = []
    cur: list[int] = []
    prev_key = None
    for i in range(len(game_id)):
        key = (game_id[i], turn_val[i])
        if prev_key is not None and key != prev_key:
            turns.append(cur)
            cur = []
        cur.append(i)
        prev_key = key
    if cur:
        turns.append(cur)
    return turns


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="runs/practicum-token.npz")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--plan-dim", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/netprobe/e30_plan_bc.json")
    args = ap.parse_args()

    import torch  # noqa: PLC0415 — [ml] extra
    import torch.nn as nn  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    torch.manual_seed(args.seed)
    torch.set_num_threads(8)  # CPU; leave the GPU for training runs

    d = np.load(args.data)
    action = d["action"].astype(np.int64)
    mask = d["mask"].astype(bool)
    game_id = d["game_id"]
    n = len(action)
    turn_val = d["obs_scalars"][:, 0].astype(np.int64)  # round counter (raw, pre-standardize)
    state = np.concatenate(
        [d["obs_tokens"].reshape(n, -1).astype(np.float32), d["obs_scalars"].astype(np.float32)],
        axis=1,
    )
    state = (state - state.mean(0)) / (state.std(0) + 1e-6)  # standardize
    d_state = state.shape[1]

    turns = reconstruct_turns(game_id, turn_val)
    # per-record prior-action prefix (padded) + length, from turn membership.
    prefix = np.full((n, MAX_PLAN), PAD_ACTION, dtype=np.int64)
    plen = np.zeros(n, dtype=np.int64)
    turn_of = np.zeros(n, dtype=np.int64)
    for t_idx, t in enumerate(turns):
        for step, i in enumerate(t):
            turn_of[i] = t_idx
            k = min(step, MAX_PLAN)
            plen[i] = k
            if k:
                prefix[i, :k] = action[t[step - k : step]]

    rng = np.random.default_rng(args.seed)
    n_turns = len(turns)
    perm = rng.permutation(n_turns)
    n_val = int(args.val_frac * n_turns)
    val_turns = set(perm[:n_val].tolist())
    is_val = np.array([turn_of[i] in val_turns for i in range(n)])
    tr = np.where(~is_val)[0]
    te = np.where(is_val)[0]
    turn_len = np.array([len(turns[turn_of[i]]) for i in range(n)])
    multi = turn_len >= 2  # multi-action-turn subset
    print(f"n={n} turns={n_turns} train={len(tr)} test={len(te)} multi-action steps={multi.sum()}")

    st = torch.as_tensor(state)
    pf = torch.as_tensor(prefix)
    pl = torch.as_tensor(plen)
    ac = torch.as_tensor(action)
    mk = torch.as_tensor(mask)

    class Factored(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_state, args.hidden), nn.ReLU(), nn.Linear(args.hidden, ACTION_SIZE)
            )

        def forward(self, s, p, ln):
            return self.net(s)

    class Autoreg(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(ACTION_SIZE + 1, args.plan_dim, padding_idx=PAD_ACTION)
            self.gru = nn.GRU(args.plan_dim, args.plan_dim, batch_first=True)
            self.net = nn.Sequential(
                nn.Linear(d_state + args.plan_dim, args.hidden),
                nn.ReLU(),
                nn.Linear(args.hidden, ACTION_SIZE),
            )

        def forward(self, s, p, ln):
            emb = self.embed(p)  # (B, MAX_PLAN, plan_dim)
            out, _ = self.gru(emb)  # (B, MAX_PLAN, plan_dim)
            idx = (ln - 1).clamp(min=0)
            h = out[torch.arange(len(p)), idx]  # last real step
            h = torch.where((ln > 0).unsqueeze(-1), h, torch.zeros_like(h))  # k=0 -> zero plan
            return self.net(torch.cat([s, h], dim=-1))

    def masked_loss(logits, a, m):
        logits = logits.masked_fill(~m, -1e9)
        return F.cross_entropy(logits, a)

    def agreement(model, idx):
        model.eval()
        with torch.no_grad():
            logits = model(st[idx], pf[idx], pl[idx]).masked_fill(~mk[idx], -1e9)
            pred = logits.argmax(dim=1).numpy()
        return pred == action[idx]

    def train(model):
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for _ in range(args.epochs):
            model.train()
            order = rng.permutation(len(tr))
            for b in range(0, len(tr), args.batch):
                bi = tr[order[b : b + args.batch]]
                opt.zero_grad()
                loss = masked_loss(model(st[bi], pf[bi], pl[bi]), ac[bi], mk[bi])
                loss.backward()
                opt.step()
        return model

    results = {}
    for name, cls in (("factored", Factored), ("autoreg", Autoreg)):
        model = train(cls())
        correct = agreement(model, te)
        te_multi = multi[te]
        results[name] = {
            "agreement_overall": round(float(correct.mean()), 4),
            "agreement_multi_action": round(float(correct[te_multi].mean()), 4),
            "n_test": int(len(te)),
            "n_test_multi": int(te_multi.sum()),
        }
        # whole-turn exact match on test turns
        by_turn: dict[int, list[bool]] = {}
        for j, i in enumerate(te):
            by_turn.setdefault(int(turn_of[i]), []).append(bool(correct[j]))
        exact = [all(v) for v in by_turn.values()]
        results[name]["whole_turn_exact"] = round(float(np.mean(exact)), 4)
        print(f"{name}: {results[name]}")

    fa, au = results["factored"], results["autoreg"]
    d_multi = au["agreement_multi_action"] - fa["agreement_multi_action"]
    d_overall = au["agreement_overall"] - fa["agreement_overall"]
    verdict = (
        "factorization-bottleneck (open training arm)"
        if d_multi >= 0.10
        else "representational (close E30)"
        if d_multi <= 0.03
        else "ambiguous (judge by whole-turn)"
    )
    results["gate"] = {
        "data": args.data,
        "delta_multi_action": round(float(d_multi), 4),
        "delta_overall": round(float(d_overall), 4),
        "verdict": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results["gate"], indent=1))


if __name__ == "__main__":
    main()
