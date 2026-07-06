"""E15: sibling-ranking data + ordering-aware head fine-tunes (design doc
docs/e15-ranking-az2-design.md).

E9's lesson: matching the ensemble's VALUES (MSE) does not match its
ORDERINGS -- the beam consumes sibling margins finer than the regression
residual. This module implements the pre-registered retry:

1. ``collect_rank_data`` -- vbeam-on-the-ensemble self-play harvesting
   plan_turn's backed-up targets WITH sibling-group structure (the 5th
   collect field, E15's plan_turn extension). Each kept state carries the
   ensemble-mean label, cross-member spread, and group keys
   (``game``/``call``/``depth``/``stop_ok``/``root_act``) so the pairs the
   beam actually ranked against each other are reconstructable.
2. ``build_pairs`` -- turn group keys into ranking pairs: same-(call, depth)
   pools with depth >= 1 (the sets one beam sort ordered) plus same-call
   stop-eligible states across depths (the completed-plan comparison).
3. ``train_value_head_rank`` -- critic-branch-only fine-tune with a
   margin-weighted RankNet loss on those pairs + a small MSE anchor to the
   ensemble mean (absolute calibration must not drift: plan_turn compares
   completed plans ACROSS depths and against win/loss sentinels). Reports
   held-out PAIR ORDERING accuracy before/after, margin-stratified -- the
   G1 fidelity gate input.
4. ``train_policy_head_listwise`` (Stage 2) -- policy-branch-only fine-tune
   with soft cross-entropy toward softmax(depth-1 sibling targets / tau) at
   each root decision, weighted by branching (the E14a-indicted regime).
   The action-space projection uses ``root_act`` (the semantic index of the
   depth-1 action at the root view, computed at harvest time).

Pre-registered constants (see design doc + driver docstring): RankNet
temperature 0.05, MSE anchor weight 0.25, pair margin floor 0.01, max 15
pairs per group, listwise tau 0.05.

Imports the ML stack lazily; an ImportError means the ``[ml]`` extra is absent.
"""

from __future__ import annotations

import json
import os

import numpy as np

from locma.envs.practicum import _manifest_path
from locma.envs.vbeam_fvi import _TOKEN_KEYS, DEFAULT_OPPONENTS

RANK_TEMP: float = 0.05  # RankNet sigmoid temperature (sibling-margin scale)
ANCHOR_LAMBDA: float = 0.25  # MSE-to-ensemble-mean anchor weight
MIN_MARGIN: float = 0.01  # pairs below this label gap are noise, excluded
MAX_PAIRS_PER_GROUP: int = 15
LISTWISE_TAU: float = 0.05  # softmax temperature over sibling targets

_GROUP_KEYS = ("target", "spread", "game", "call", "depth", "stop_ok", "root_act")


def _collect_rank_shard(
    model_paths,
    opponents,
    games: int,
    out: str,
    seed: int,
    width: int,
    max_actions: int,
) -> int:
    """Top-level picklable unit of work: one grouped-harvest shard (E15).

    Plays vbeam on the ENSEMBLE evaluator vs each opponent (both seats),
    keeping EVERY collected state (all depths -- deeper pools are beam-sorted
    too) tagged with its plan-call id. Call boundaries are detected via the
    run_game ``on_step`` hook: the sink only grows during ``plan_turn``, which
    VBeamBattlePolicy invokes at most once per own turn, so each growth
    episode is exactly one call. ``root_act`` is the semantic index (at the
    call's root view) of a depth-1 state's reaching action, -1 elsewhere.
    """
    from locma.core.engine import run_game  # noqa: PLC0415
    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.envs.encode import encode_battle_tokens, sem_index  # noqa: PLC0415
    from locma.policies.composer import Composer  # noqa: PLC0415
    from locma.policies.drafts import BalancedDraftPolicy  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415
    from locma.policies.vbeam import EnsembleValueEvaluator, VBeamBattlePolicy  # noqa: PLC0415

    sink: list = []
    ev = EnsembleValueEvaluator([resolve_path(p) for p in model_paths])
    vb = VBeamBattlePolicy(
        model_path=ev.model_paths[0],
        width=width,
        max_actions=max_actions,
        evaluator=ev,
        collect=sink,
    )
    me = Composer(vb, BalancedDraftPolicy(), name="vbeam-rank-harvest")

    call_ids: list[int] = []
    state = {"call": 0, "seen": 0}

    def _on_step(_seat, _action, _gs) -> None:
        grew = len(sink) - state["seen"]
        if grew > 0:
            call_ids.extend([state["call"]] * grew)
            state["seen"] = len(sink)
            state["call"] += 1

    game_ids: list[int] = []
    game_no = 0
    for opp_spec in opponents:
        opp = make_policy(opp_spec)
        for g in range(games):
            for my_seat in (0, 1):
                before = len(sink)
                p0, p1 = (me, opp) if my_seat == 0 else (opp, me)
                run_game(p0, p1, seed + g, on_step=_on_step)
                game_ids.extend([game_no] * (len(sink) - before))
                game_no += 1

    n = len(sink)
    assert len(call_ids) == n, f"call segmentation drifted: {len(call_ids)} != {n}"

    # Per-call root view (depth 0), for root_act of the depth-1 children.
    root_view_of: dict[int, object] = {}
    for (view, _t, depth, _s, _p), cid in zip(sink, call_ids, strict=True):
        if depth == 0:
            root_view_of[cid] = view

    kept_views = [e[0] for e in sink]
    depths = np.asarray([e[2] for e in sink], dtype=np.int64)
    stop_oks = np.asarray([e[3] for e in sink], dtype=bool)
    root_act = np.full(n, -1, dtype=np.int64)
    for i, ((_v, _t, depth, _s, prefix), cid) in enumerate(zip(sink, call_ids, strict=True)):
        if depth == 1 and cid in root_view_of:
            idx = sem_index(root_view_of[cid], prefix[0])
            if idx is not None:
                root_act[i] = idx

    # Label with each member critic (chunked batched forwards, no games).
    member_vals = np.empty((len(ev.members), n), dtype=np.float32)
    chunk = 512
    for mi, member in enumerate(ev.members):
        for i in range(0, n, chunk):
            member_vals[mi, i : i + chunk] = member.values(kept_views[i : i + chunk])

    obs = [encode_battle_tokens(v) for v in kept_views]
    np.savez_compressed(
        out,
        obs_tokens=np.asarray([o["tokens"] for o in obs], dtype=np.float32),
        obs_card_ids=np.asarray([o["card_ids"] for o in obs], dtype=np.float32),
        obs_token_mask=np.asarray([o["token_mask"] for o in obs], dtype=np.float32),
        obs_scalars=np.asarray([o["scalars"] for o in obs], dtype=np.float32),
        target=member_vals.mean(axis=0),
        spread=member_vals.std(axis=0),
        game=np.asarray(game_ids, dtype=np.int64),
        call=np.asarray(call_ids, dtype=np.int64),
        depth=depths,
        stop_ok=stop_oks,
        root_act=root_act,
    )
    return n


def collect_rank_data(
    model_paths,
    out: str,
    opponents=DEFAULT_OPPONENTS,
    games: int = 150,
    seed: int = 0,
    workers: int = 1,
    width: int = 8,
    max_actions: int = 20,
) -> dict:
    """Record grouped ensemble-labeled sibling data into ``out`` (.npz).

    Shards the per-opponent game count over a process pool (like
    ``collect_ensemble_data``) and merges with per-shard ``game``/``call``
    id offsets so groups never collide. Returns a manifest with example,
    group, and spread statistics.
    """
    opponents = list(opponents)
    model_paths = list(model_paths)
    workers = max(1, min(workers, games))

    base, extra = divmod(games, workers)
    counts = [base + (1 if w < extra else 0) for w in range(workers)]
    starts = [sum(counts[:w]) for w in range(workers)]
    shard_paths = [f"{out}.shard{w}.npz" for w in range(workers)]

    if workers == 1:
        _collect_rank_shard(model_paths, opponents, games, shard_paths[0], seed, width, max_actions)
    else:
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

        from locma.harness.parallel import init_eval_worker  # noqa: PLC0415

        with ProcessPoolExecutor(max_workers=workers, initializer=init_eval_worker) as ex:
            list(
                ex.map(
                    _collect_rank_shard,
                    [model_paths] * workers,
                    [opponents] * workers,
                    counts,
                    shard_paths,
                    [seed + s for s in starts],
                    [width] * workers,
                    [max_actions] * workers,
                )
            )

    merged: dict[str, list] = {k: [] for k in (*_TOKEN_KEYS, *_GROUP_KEYS)}
    game_off = call_off = 0
    for path in shard_paths:
        with np.load(path) as d:
            for k, chunks in merged.items():
                arr = d[k]
                if k == "game":
                    arr = arr + game_off
                elif k == "call":
                    arr = arr + call_off
                chunks.append(arr)
            if len(d["game"]):
                game_off += int(d["game"].max()) + 1
                call_off += int(d["call"].max()) + 1
        os.remove(path)
    arrays = {k: np.concatenate(v) for k, v in merged.items()}
    np.savez_compressed(out, **arrays)

    n_groups = len({(c, dp) for c, dp in zip(arrays["call"], arrays["depth"], strict=True)})
    manifest = {
        "kind": "vbeam-rank-data",
        "model_paths": model_paths,
        "opponents": opponents,
        "games": games,
        "seed": seed,
        "width": width,
        "max_actions": max_actions,
        "n_examples": int(len(arrays["target"])),
        "n_calls": int(arrays["call"].max()) + 1 if len(arrays["call"]) else 0,
        "n_groups": n_groups,
        "mean_spread": float(arrays["spread"].mean()) if len(arrays["spread"]) else 0.0,
        "shards": workers,
    }
    with open(_manifest_path(out), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def build_pairs(
    call: np.ndarray,
    depth: np.ndarray,
    stop_ok: np.ndarray,
    target: np.ndarray,
    *,
    max_pairs_per_group: int = MAX_PAIRS_PER_GROUP,
    min_margin: float = MIN_MARGIN,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Ranking pairs (idx_hi, idx_lo) with target[idx_hi] > target[idx_lo].

    Groups: (a) same (call, depth) with depth >= 1 -- the pools one beam sort
    ordered against each other; (b) same call, stop_ok states across depths --
    the completed-plan comparison that picks the played plan. Pairs with a
    label gap below ``min_margin`` are excluded (noise); each group
    contributes at most ``max_pairs_per_group`` pairs (subsampled).
    """
    rng = np.random.default_rng(seed)
    groups: dict[tuple, list[int]] = {}
    for i in range(len(call)):
        if depth[i] >= 1:
            groups.setdefault(("d", int(call[i]), int(depth[i])), []).append(i)
        if stop_ok[i]:
            groups.setdefault(("s", int(call[i])), []).append(i)

    hi: list[int] = []
    lo: list[int] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        cand = [
            (a, b)
            for k, a in enumerate(members)
            for b in members[k + 1 :]
            if abs(target[a] - target[b]) >= min_margin
        ]
        if len(cand) > max_pairs_per_group:
            keep = rng.choice(len(cand), size=max_pairs_per_group, replace=False)
            cand = [cand[j] for j in keep]
        for a, b in cand:
            if target[a] >= target[b]:
                hi.append(a)
                lo.append(b)
            else:
                hi.append(b)
                lo.append(a)
    return np.asarray(hi, dtype=np.int64), np.asarray(lo, dtype=np.int64)


def _frozen_features(policy, d, n: int):
    """Precompute frozen-extractor features for all n examples, chunked."""
    import torch  # noqa: PLC0415 — optional [ml] dep

    feats_chunks = []
    chunk = 1024
    with torch.no_grad():
        for i in range(0, n, chunk):
            batch = {
                "tokens": d["obs_tokens"][i : i + chunk],
                "card_ids": d["obs_card_ids"][i : i + chunk],
                "token_mask": d["obs_token_mask"][i : i + chunk],
                "scalars": d["obs_scalars"][i : i + chunk],
            }
            obs_t, _ = policy.obs_to_tensor(batch)
            feats_chunks.append(policy.extract_features(obs_t).cpu())
    return torch.cat(feats_chunks)


def _split_games(game: np.ndarray, val_frac: float, seed: int) -> np.ndarray:
    """Boolean val mask over examples, split at GAME level (no leakage)."""
    rng = np.random.default_rng(seed)
    gids = np.unique(game)
    n_val = max(1, int(len(gids) * val_frac))
    val_games = set(rng.permutation(gids)[:n_val].tolist())
    return np.asarray([g in val_games for g in game], dtype=bool)


def train_value_head_rank(
    base_model: str,
    data: str,
    out: str,
    *,
    epochs: int = 10,
    batch_pairs: int = 512,
    lr: float = 3e-4,
    val_frac: float = 0.1,
    seed: int = 0,
    temp: float = RANK_TEMP,
    lam: float = ANCHOR_LAMBDA,
) -> dict:
    """Critic-branch-only FT with margin-weighted RankNet + MSE anchor.

    Frozen: the features extractor and the entire policy path (byte-identical
    action probabilities -- vbeam's stop rule unchanged). Trained:
    ``mlp_extractor.value_net`` + ``value_net`` -- same trainable set as E9's
    ``train_value_head``; the objective is the only moved variable.

    Returns G1 inputs: held-out sibling-pair ordering accuracy before/after,
    stratified by ensemble margin (< / >= 0.08, the E9 residual), plus val
    MSE before/after for the anchor sanity check.
    """
    import torch  # noqa: PLC0415 — optional [ml] dep
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.depot import resolve_path  # noqa: PLC0415

    model = MaskablePPO.load(resolve_path(base_model))
    policy = model.policy
    policy.set_training_mode(False)
    device = policy.device

    d = np.load(resolve_path(data))
    t_np = np.asarray(d["target"], dtype=np.float32)
    n = int(len(t_np))
    if n == 0:
        raise ValueError(f"no examples in {data}")

    val_mask = _split_games(d["game"], val_frac, seed)
    hi, lo = build_pairs(d["call"], d["depth"], d["stop_ok"], t_np, seed=seed)
    if len(hi) == 0:
        raise ValueError("no ranking pairs (groups too small or margins below floor)")
    # A pair is val iff both ends are val (game-level split keeps pairs whole:
    # all states of one call share a game).
    pair_val = val_mask[hi] & val_mask[lo]
    pair_tr = ~val_mask[hi] & ~val_mask[lo]

    feats = _frozen_features(policy, d, n)
    t = torch.from_numpy(t_np)

    for p in policy.parameters():
        p.requires_grad_(False)
    train_params = [
        *policy.mlp_extractor.value_net.parameters(),
        *policy.value_net.parameters(),
    ]
    for p in train_params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(train_params, lr=lr)

    def _v(idx) -> torch.Tensor:
        latent_vf = policy.mlp_extractor.forward_critic(feats[idx].to(device))
        return policy.value_net(latent_vf).squeeze(-1)

    hi_t = torch.from_numpy(hi)
    lo_t = torch.from_numpy(lo)
    margin = (t[hi_t] - t[lo_t]).abs()

    def _pair_metrics(mask: np.ndarray) -> dict:
        idx = np.flatnonzero(mask)
        with torch.no_grad():
            v_hi = _v(hi_t[idx]).cpu()
            v_lo = _v(lo_t[idx]).cpu()
        correct = (v_hi > v_lo).float()
        m = margin[idx]
        fine = m < 0.08
        out_m: dict = {"n_pairs": int(len(idx)), "acc": float(correct.mean())}
        if int(fine.sum()):
            out_m["acc_fine"] = float(correct[fine].mean())
        if int((~fine).sum()):
            out_m["acc_coarse"] = float(correct[~fine].mean())
        with torch.no_grad():
            vi = np.flatnonzero(val_mask)
            v_all = _v(torch.from_numpy(vi)).cpu()
        out_m["val_mse"] = float(torch.mean((v_all - t[vi]) ** 2))
        return out_m

    before = _pair_metrics(pair_val)

    tr_pairs = np.flatnonzero(pair_tr)
    gen = torch.Generator().manual_seed(seed)
    for _ep in range(epochs):
        order = torch.randperm(len(tr_pairs), generator=gen)
        for i in range(0, len(tr_pairs), batch_pairs):
            bidx = torch.from_numpy(tr_pairs[order[i : i + batch_pairs].numpy()])
            v_hi = _v(hi_t[bidx])
            v_lo = _v(lo_t[bidx])
            w = margin[bidx].to(device)
            rank = (w * torch.nn.functional.softplus(-(v_hi - v_lo) / temp)).sum() / w.sum()
            anchor = 0.5 * (
                torch.nn.functional.mse_loss(v_hi, t[hi_t[bidx]].to(device))
                + torch.nn.functional.mse_loss(v_lo, t[lo_t[bidx]].to(device))
            )
            loss = rank + lam * anchor
            opt.zero_grad()
            loss.backward()
            opt.step()

    after = _pair_metrics(pair_val)
    model.save(out)
    return {
        "n_examples": n,
        "n_pairs_total": int(len(hi)),
        "n_pairs_train": int(pair_tr.sum()),
        "epochs": epochs,
        "temp": temp,
        "lam": lam,
        "before": before,
        "after": after,
        "out": out,
    }


def train_policy_head_listwise(
    base_model: str,
    data: str,
    out: str,
    *,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 3e-4,
    val_frac: float = 0.1,
    seed: int = 0,
    tau: float = LISTWISE_TAU,
) -> dict:
    """Policy-branch-only FT: soft-CE toward softmax(sibling targets / tau).

    One training example per plan call with >= 2 depth-1 children carrying a
    valid ``root_act``: input = the ROOT state's frozen features, target = the
    softmax (temperature ``tau``) over the children's backed-up values,
    projected onto their semantic action indices; logits are masked to that
    child action set. Sample weight = ln(1 + n_children): branching-weighted
    per E14a. Frozen: extractor + critic path (byte-identical values -- the
    vbeam plan ranking is untouched; only would_pass can shift, gated by G6).

    Returns val CE and top-1 agreement with the teacher-best child,
    before/after.
    """
    import torch  # noqa: PLC0415 — optional [ml] dep
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.envs.encode import ACTION_SIZE  # noqa: PLC0415

    model = MaskablePPO.load(resolve_path(base_model))
    policy = model.policy
    policy.set_training_mode(False)
    device = policy.device

    d = np.load(resolve_path(data))
    t_np = np.asarray(d["target"], dtype=np.float32)
    call = d["call"]
    depth = d["depth"]
    root_act = d["root_act"]
    n = int(len(t_np))

    # Assemble (root example) -> (child action indices, child targets).
    root_of: dict[int, int] = {}
    kids: dict[int, list[tuple[int, float]]] = {}
    for i in range(n):
        c = int(call[i])
        if depth[i] == 0:
            root_of[c] = i
        elif depth[i] == 1 and root_act[i] >= 0:
            kids.setdefault(c, []).append((int(root_act[i]), float(t_np[i])))
    examples = [(root_of[c], ch) for c, ch in kids.items() if c in root_of and len(ch) >= 2]
    if not examples:
        raise ValueError("no listwise examples (no calls with >= 2 mapped children)")

    feats = _frozen_features(policy, d, n)
    val_mask = _split_games(d["game"], val_frac, seed)

    root_idx = np.asarray([e[0] for e in examples], dtype=np.int64)
    tgt_dist = np.zeros((len(examples), ACTION_SIZE), dtype=np.float32)
    act_mask = np.zeros((len(examples), ACTION_SIZE), dtype=bool)
    weight = np.empty(len(examples), dtype=np.float32)
    for k, (_ri, ch) in enumerate(examples):
        acts = np.asarray([a for a, _ in ch])
        vals = np.asarray([v for _, v in ch], dtype=np.float32)
        p = np.exp((vals - vals.max()) / tau)
        p /= p.sum()
        # Duplicate action indices cannot occur: depth-1 siblings differ in
        # their first action by construction (the beam expands each once).
        tgt_dist[k, acts] = p
        act_mask[k, acts] = True
        weight[k] = np.log1p(len(ch))
    ex_val = val_mask[root_idx]

    for p in policy.parameters():
        p.requires_grad_(False)
    train_params = [
        *policy.mlp_extractor.policy_net.parameters(),
        *policy.action_net.parameters(),
    ]
    for p in train_params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(train_params, lr=lr)

    feats_root = feats[torch.from_numpy(root_idx)]
    tgt_t = torch.from_numpy(tgt_dist)
    mask_t = torch.from_numpy(act_mask)
    w_t = torch.from_numpy(weight)

    def _logits(idx) -> torch.Tensor:
        latent_pi = policy.mlp_extractor.forward_actor(feats_root[idx].to(device))
        logits = policy.action_net(latent_pi)
        return logits.masked_fill(~mask_t[idx].to(device), -1e9)

    def _metrics(mask: np.ndarray) -> dict:
        idx = torch.from_numpy(np.flatnonzero(mask))
        with torch.no_grad():
            logp = torch.log_softmax(_logits(idx), dim=-1).cpu()
        tgt = tgt_t[idx]
        ce = float(-(tgt * logp.clamp(min=-30)).sum(-1).mean())
        top1 = float((logp.argmax(-1) == tgt.argmax(-1)).float().mean())
        return {"n": int(len(idx)), "ce": ce, "top1_vs_teacher": top1}

    before = _metrics(ex_val)

    tr = np.flatnonzero(~ex_val)
    gen = torch.Generator().manual_seed(seed)
    for _ep in range(epochs):
        order = torch.randperm(len(tr), generator=gen)
        for i in range(0, len(tr), batch_size):
            bidx = torch.from_numpy(tr[order[i : i + batch_size].numpy()])
            logp = torch.log_softmax(_logits(bidx), dim=-1)
            ce = -(tgt_t[bidx].to(device) * logp.clamp(min=-30)).sum(-1)
            loss = (w_t[bidx].to(device) * ce).sum() / w_t[bidx].to(device).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

    after = _metrics(ex_val)
    model.save(out)
    return {
        "n_examples_listwise": len(examples),
        "epochs": epochs,
        "tau": tau,
        "before": before,
        "after": after,
        "out": out,
    }
