"""E38 Gate 1: do per-action afterstate consequence columns let a BC head
recover the search teacher's choice, where the reactive representation cannot?

Pre-registered in ``docs/e38-afterstate-columns-design.md``. E30 closed the
turn-plan arm with a REPRESENTATIONAL verdict ("the reactive obs/features do not
separably encode WHICH action a lookahead teacher picks"), which is a statement
about a fixed representation. E38 changes the representation: each candidate
action carries what the engine says it does (``locma/envs/conseq.py`` — dphi,
damage, kills/losses, overkill, mana, wins-now, enables-lethal, exposes-lethal).

Harness = the E30 controlled-BC design (``scripts/e30_plan_bc.py``): identical
frozen state features, identical masked-CE training, identical turn
reconstruction, same ``mcts:100`` teacher practicum — so the ``factored`` arm
reproduces E30's 0.373 and anchors the run.

Three arms. The third is what makes this tight:

  factored     logits = MLP(state)                                (E30 baseline)
  conseq       logit(a) = MLP([h(state), conseq(a), fam(a)])      (the columns)
  conseq_shuf  IDENTICAL architecture, consequence rows PERMUTED across actions
               within each record — same parameters, same marginal column
               distribution, action<->consequence correspondence destroyed.

A per-action head is a different function class than a dense 155-way head, so
``conseq`` > ``factored`` alone would confound information with architecture
(the E28 gate-1 lesson). The pre-registered read is ``conseq - conseq_shuf`` on
multi-action turns:

  >= +0.10  -> the columns carry decision-relevant information the reactive
              representation lacks. OPEN the PPO arm (Gate 2).
  <= +0.03  -> the teacher's choice is not recoverable even from explicit 1-ply
              consequences. E38 CLOSES and E30's verdict survives a sharper
              attack.
  between   -> ambiguous; judge on whole-turn exact match + the column ablation.

Also reported (free): a drop-one column ablation on the winning arm, to say
WHICH consequence carries the signal.

Usage:
    python scripts/e38_conseq_bc.py --data runs/e38/practicum-conseq.npz
    python scripts/e38_conseq_bc.py --seeds 0,1,2 --ablate
Output: runs/e38/conseq_bc.json (per-seed files with --seed)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ACTION_SIZE = 155
GATE_OPEN = 0.10  # conseq - conseq_shuf on multi-action turns -> open Gate 2
GATE_CLOSE = 0.03  # <= this -> E38 closes


def reconstruct_turns(game_id: np.ndarray, turn_val: np.ndarray) -> list[list[int]]:
    """Group record indices into turns by (game_id, turn counter) — E30's method.

    Records are in play order and only the teacher's own decisions are stored, so
    a maximal run with the same game_id AND the same turn counter is exactly one
    teacher turn. Pass is NOT a reliable delimiter (only ~4.5% of decisions).
    """
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


def build_family_onehot() -> np.ndarray:
    """(ACTION_SIZE, 4) one-hot over {pass, summon, use, attack}.

    Same table the pointer head uses (``locma.envs.pointer_head``), so the
    per-action arms see the family structure the real head would.
    """
    from locma.envs.pointer_head import build_action_table  # noqa: PLC0415

    _src, _tgt, fam = build_action_table()
    return np.eye(4, dtype=np.float32)[fam]


def run_seed(args, d, seed: int) -> dict:  # noqa: PLR0915 — one self-contained experiment
    import torch  # noqa: PLC0415 — [ml] extra
    import torch.nn as nn  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    torch.manual_seed(seed)
    torch.set_num_threads(args.threads)

    action = d["action"].astype(np.int64)
    mask = d["mask"].astype(bool)
    game_id = d["game_id"]
    n = len(action)
    turn_val = d["obs_scalars"][:, 0].astype(np.int64)  # round counter, pre-standardize
    state = np.concatenate(
        [d["obs_tokens"].reshape(n, -1).astype(np.float32), d["obs_scalars"].astype(np.float32)],
        axis=1,
    )
    state = (state - state.mean(0)) / (state.std(0) + 1e-6)
    d_state = state.shape[1]

    conseq = d["obs_conseq"].astype(np.float32)  # (n, ACTION_SIZE, K)
    n_col = conseq.shape[2]
    # Standardize each column over its NONZERO (legal-action) entries only —
    # zeros are the "not applicable" code, so folding them into the mean/std
    # would make scale depend on how many actions happened to be legal.
    keep = np.abs(conseq).sum(axis=2) > 0  # (n, ACTION_SIZE) legal-and-filled
    flat = conseq[keep]  # (m, K)
    c_mean, c_std = flat.mean(0), flat.std(0) + 1e-6
    conseq_z = np.zeros_like(conseq)
    conseq_z[keep] = (flat - c_mean) / c_std
    if args.columns:  # drop-one / subset ablation
        cols = [int(c) for c in args.columns.split(",")]
        keep_col = np.zeros(n_col, dtype=bool)
        keep_col[cols] = True
        conseq_z = conseq_z * keep_col.astype(np.float32)

    turns = reconstruct_turns(game_id, turn_val)
    turn_of = np.zeros(n, dtype=np.int64)
    for t_idx, t in enumerate(turns):
        for i in t:
            turn_of[i] = t_idx

    rng = np.random.default_rng(seed)
    n_turns = len(turns)
    perm = rng.permutation(n_turns)
    val_turns = set(perm[: int(args.val_frac * n_turns)].tolist())
    is_val = np.array([turn_of[i] in val_turns for i in range(n)])
    tr = np.where(~is_val)[0]
    te = np.where(is_val)[0]
    turn_len = np.array([len(turns[turn_of[i]]) for i in range(n)])
    multi = turn_len >= 2

    # conseq_shuf: permute the consequence rows AMONG THE FILLED (legal, non-Pass)
    # SLOTS ONLY, per record. Permuting all 155 slots would be a much weaker
    # control: ~150 of them are zero, so the legal slots would mostly receive
    # zeros and the arm would degenerate into "no consequence information" rather
    # than "mismatched consequence information". Restricted to the filled set,
    # every legal action sees ANOTHER legal action's real consequence — same
    # architecture, same parameter count, same per-record marginal distribution,
    # with only the action<->consequence correspondence destroyed. The permutation
    # is drawn once (not per epoch) so the arm has a stationary target.
    shuf = conseq_z.copy()
    n_degenerate = 0  # records with <2 filled rows: permutation is necessarily identity
    for i in range(n):
        idx = np.flatnonzero(keep[i])
        if len(idx) < 2:
            n_degenerate += 1
            continue
        # derangement-ish: reject an identity draw so the row really moves
        for _ in range(8):
            p = rng.permutation(len(idx))
            if np.any(p != np.arange(len(idx))):
                break
        shuf[i, idx] = conseq_z[i, idx[p]]
    assert not np.array_equal(shuf, conseq_z), "shuffle control is a no-op"

    fam = build_family_onehot()  # (ACTION_SIZE, 4)

    st = torch.as_tensor(state)
    cz = torch.as_tensor(conseq_z)
    cs = torch.as_tensor(shuf)
    fm = torch.as_tensor(fam)
    ac = torch.as_tensor(action)
    mk = torch.as_tensor(mask)

    class Factored(nn.Module):
        """E30's baseline head, unchanged."""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_state, args.hidden), nn.ReLU(), nn.Linear(args.hidden, ACTION_SIZE)
            )

        def forward(self, s, c):
            return self.net(s)

    class PerAction(nn.Module):
        """logit(a) = MLP([h(state), conseq(a), family(a)]) — the pointer shape.

        ``h(state)`` is a state summary of the same width the factored arm's
        hidden layer uses, so the two arms carry comparable state capacity; the
        conseq and conseq_shuf arms share this class exactly (only the tensor
        they are fed differs), which is what makes the shuffle a clean control.
        """

        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(d_state, args.hidden), nn.ReLU())
            self.head = nn.Sequential(
                nn.Linear(args.hidden + n_col + 4, args.hidden),
                nn.ReLU(),
                nn.Linear(args.hidden, 1),
            )

        def forward(self, s, c):
            h = self.enc(s).unsqueeze(1).expand(-1, ACTION_SIZE, -1)  # (B, A, hidden)
            f = fm.unsqueeze(0).expand(s.size(0), -1, -1)  # (B, A, 4)
            return self.head(torch.cat([h, c, f], dim=-1)).squeeze(-1)

    class FactoredPlus(nn.Module):
        """E30's dense head PLUS a small per-action consequence correction.

        ``logits = MLP(state) + g(conseq(a), family(a))``. Added after the first
        3-arm run showed the ``PerAction`` class is itself a ~0.07-agreement
        handicap versus the dense head, which would make ``conseq - factored``
        understate what a well-built consequence head can do. This arm removes
        that confound: it keeps the strong dense head byte-for-byte and adds the
        columns the way Gate 2 actually would (a per-action term on top of an
        already-good head, cf. the pointer head's per-action MLP), so
        ``factored_plus - factored`` is the honest "what do the columns buy"
        number.
        """

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_state, args.hidden), nn.ReLU(), nn.Linear(args.hidden, ACTION_SIZE)
            )
            self.corr = nn.Sequential(nn.Linear(n_col + 4, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, s, c):
            f = fm.unsqueeze(0).expand(s.size(0), -1, -1)  # (B, A, 4)
            corr = self.corr(torch.cat([c, f], dim=-1)).squeeze(-1)  # (B, A)
            return self.net(s) + corr

    def masked_loss(logits, a, m):
        return F.cross_entropy(logits.masked_fill(~m, -1e9), a)

    def agreement(model, feat, idx):
        model.eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(idx), 4096):
                bi = idx[b : b + 4096]
                logits = model(st[bi], feat[bi]).masked_fill(~mk[bi], -1e9)
                out.append(logits.argmax(dim=1).numpy())
        return np.concatenate(out) == action[idx]

    def train(model, feat):
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for _ in range(args.epochs):
            model.train()
            order = rng.permutation(len(tr))
            for b in range(0, len(tr), args.batch):
                bi = tr[order[b : b + args.batch]]
                opt.zero_grad()
                masked_loss(model(st[bi], feat[bi]), ac[bi], mk[bi]).backward()
                opt.step()
        return model

    arms = (
        ("factored", Factored, cz),  # feat unused by Factored
        ("conseq", PerAction, cz),
        ("conseq_shuf", PerAction, cs),
        ("factored_plus", FactoredPlus, cz),
        ("factored_plus_shuf", FactoredPlus, cs),
    )
    results: dict = {}
    for name, cls, feat in arms:
        model = train(cls(), feat)
        correct = agreement(model, feat, te)
        te_multi = multi[te]
        by_turn: dict[int, list[bool]] = {}
        for j, i in enumerate(te):
            by_turn.setdefault(int(turn_of[i]), []).append(bool(correct[j]))
        results[name] = {
            "agreement_overall": round(float(correct.mean()), 4),
            "agreement_multi_action": round(float(correct[te_multi].mean()), 4),
            "whole_turn_exact": round(float(np.mean([all(v) for v in by_turn.values()])), 4),
            "params": int(sum(p.numel() for p in model.parameters())),
            "n_test": int(len(te)),
            "n_test_multi": int(te_multi.sum()),
        }
        print(f"  seed {seed} {name:12s} {results[name]}", flush=True)

    results["_meta"] = {
        "seed": seed,
        "n": int(n),
        "n_turns": int(n_turns),
        "d_state": int(d_state),
        "n_conseq_columns": int(n_col),
        "columns_kept": args.columns or "all",
        "mean_filled_actions": round(float(keep.sum(axis=1).mean()), 3),
        "n_records_shuffle_degenerate": int(n_degenerate),
    }
    return results


def verdict_for(delta_multi: float) -> str:
    if delta_multi >= GATE_OPEN:
        return "columns carry signal (OPEN Gate 2)"
    if delta_multi <= GATE_CLOSE:
        return "representational verdict survives (E38 CLOSES)"
    return "ambiguous (judge on whole-turn + ablation)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="runs/e38/practicum-conseq.npz")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument(
        "--columns",
        default="",
        help="comma list of consequence column indices to KEEP (others zeroed); "
        "empty = all. Used for the drop-one ablation.",
    )
    ap.add_argument("--out", default="runs/e38/conseq_bc.json")
    args = ap.parse_args()

    d = np.load(args.data)
    if "obs_conseq" not in d:
        raise SystemExit(
            f"{args.data} has no obs_conseq — regenerate with "
            "`locma record-practicum --obs-mode token --conseq`"
        )
    from locma.envs.conseq import CONSEQ_COLUMNS  # noqa: PLC0415

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"E38 Gate 1 — data={args.data} seeds={seeds} columns={args.columns or 'all'}")
    per_seed = {str(s): run_seed(args, d, s) for s in seeds}

    def col(arm: str, key: str) -> list[float]:
        return [per_seed[str(s)][arm][key] for s in seeds]

    d_multi = [
        c - sh
        for c, sh in zip(
            col("conseq", "agreement_multi_action"),
            col("conseq_shuf", "agreement_multi_action"),
            strict=True,
        )
    ]
    d_vs_factored = [
        c - f
        for c, f in zip(
            col("conseq", "agreement_multi_action"),
            col("factored", "agreement_multi_action"),
            strict=True,
        )
    ]
    mean_d = float(np.mean(d_multi))
    payload = {
        "data": args.data,
        "seeds": seeds,
        "columns_kept": args.columns or "all",
        "conseq_columns": list(CONSEQ_COLUMNS),
        "per_seed": per_seed,
        "gate": {
            "delta_multi_conseq_minus_shuf": [round(x, 4) for x in d_multi],
            "delta_multi_mean": round(mean_d, 4),
            "delta_multi_vs_factored": [round(x, 4) for x in d_vs_factored],
            "factored_multi_action": [
                round(x, 4) for x in col("factored", "agreement_multi_action")
            ],
            # Handicap-free read: the dense head plus a per-action consequence
            # term, vs the same dense head alone and vs its own shuffle control.
            "delta_multi_fplus_minus_factored": [
                round(fp - f, 4)
                for fp, f in zip(
                    col("factored_plus", "agreement_multi_action"),
                    col("factored", "agreement_multi_action"),
                    strict=True,
                )
            ],
            "delta_multi_fplus_minus_shuf": [
                round(fp - sh, 4)
                for fp, sh in zip(
                    col("factored_plus", "agreement_multi_action"),
                    col("factored_plus_shuf", "agreement_multi_action"),
                    strict=True,
                )
            ],
            "arch_handicap_shuf_minus_factored": [
                round(sh - f, 4)
                for sh, f in zip(
                    col("conseq_shuf", "agreement_multi_action"),
                    col("factored", "agreement_multi_action"),
                    strict=True,
                )
            ],
            "bar_open": GATE_OPEN,
            "bar_close": GATE_CLOSE,
            "verdict": verdict_for(mean_d),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))

    print("\n================ E38 GATE 1 ================")
    print(f"{'arm':14s}{'multi-action':>14s}{'overall':>10s}{'whole-turn':>12s}{'params':>10s}")
    for arm in ("factored", "conseq", "conseq_shuf", "factored_plus", "factored_plus_shuf"):
        print(
            f"{arm:19s}{np.mean(col(arm, 'agreement_multi_action')):>14.4f}"
            f"{np.mean(col(arm, 'agreement_overall')):>10.4f}"
            f"{np.mean(col(arm, 'whole_turn_exact')):>12.4f}"
            f"{per_seed[str(seeds[0])][arm]['params']:>10d}"
        )
    g = payload["gate"]
    print(f"\nconseq - conseq_shuf (multi-action, per seed): {[round(x, 4) for x in d_multi]}")
    print(f"mean {mean_d:+.4f}  (open >= {GATE_OPEN}, close <= {GATE_CLOSE})")
    print(f"PRE-REGISTERED VERDICT: {g['verdict']}")
    print("\n--- handicap-free reads (the per-action MLP is itself a handicap) ---")
    print(f"  arch handicap  conseq_shuf - factored : {g['arch_handicap_shuf_minus_factored']}")
    print(f"  conseq         - factored             : {g['delta_multi_vs_factored']}")
    print(f"  factored_plus  - factored             : {g['delta_multi_fplus_minus_factored']}")
    print(f"  factored_plus  - factored_plus_shuf   : {g['delta_multi_fplus_minus_shuf']}")
    print(
        f"\nE30 anchor — factored multi-action should reproduce ~0.373; got "
        f"{np.mean(col('factored', 'agreement_multi_action')):.4f}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
