"""Fitted-value iteration for the vbeam planner (E5 variant 2).

The vbeam planner (E5 variant 1, +0.206 avg-hard3) scores stopping points with
B0's critic — a value trained under the *reactive* PPO policy. Under the
planner the play distribution is better, so that critic is off-policy:
retraining it on the planner's own games should sharpen exactly the estimates
the beam ranks with.

Two-step recipe:

1. ``collect_value_data`` — play the planner against a fixed opponent pool and
   record every own-turn decision state (token obs) with the eventual game
   outcome. Reuses ``record_practicum`` (the winner/seat columns it already
   stores ARE the Monte-Carlo value labels: +1 if winner == seat else -1),
   sharded over a process pool for CPU parallelism.
2. ``train_value_head`` — fine-tune ONLY the critic branch of a saved token
   MaskablePPO (``mlp_extractor.value_net`` + ``value_net``) on those labels.
   The features extractor and the policy path are frozen and must stay
   byte-identical: vbeam's stop rule reads the policy head's masked argmax,
   and the whole point is a drop-in ``vbeam:out.zip`` with the same stopping
   behavior but better plan ranking.

Imports the ML stack lazily; an ImportError means the ``[ml]`` extra is absent.
"""

from __future__ import annotations

import json
import os

import numpy as np

from locma.envs.practicum import _manifest_path, record_practicum

# The four opponents of the B0 zoo curriculum (training distribution),
# a superset of the hard3 eval pool.
DEFAULT_OPPONENTS: tuple[str, ...] = ("greedy", "scripted", "max-guard", "max-attack")

_TOKEN_KEYS = ("obs_tokens", "obs_card_ids", "obs_token_mask", "obs_scalars")
_META_KEYS = ("action", "mask", "winner", "seat", "opponent_id")


def _collect_shard(teacher: str, opponents, games: int, out: str, seed: int) -> dict:
    """Top-level picklable unit of work: one practicum shard (token obs)."""
    return record_practicum(
        teacher=teacher, opponents=opponents, games=games, out=out, seed=seed, obs_mode="token"
    )


def collect_value_data(
    teacher: str,
    out: str,
    opponents=DEFAULT_OPPONENTS,
    games: int = 400,
    seed: int = 0,
    workers: int = 1,
) -> dict:
    """Record value-training data from ``teacher``'s own play into ``out`` (.npz).

    Splits the per-opponent game count into contiguous seed ranges, records one
    practicum shard per range (in parallel when ``workers > 1``), and merges
    the shards. With a single opponent the merged arrays are byte-identical to
    a serial run; with several, shards interleave opponents differently but
    the example *set* is identical.

    Returns the merged manifest (also written next to ``out``).
    """
    opponents = list(opponents)
    workers = max(1, min(workers, games))

    # Contiguous seed ranges: shard w covers games [start_w, start_w + count_w)
    # of every opponent, seeded seed + start_w (record_practicum uses seed + g).
    base, extra = divmod(games, workers)
    counts = [base + (1 if w < extra else 0) for w in range(workers)]
    starts = [sum(counts[:w]) for w in range(workers)]
    shard_paths = [f"{out}.shard{w}.npz" for w in range(workers)]

    if workers == 1:
        manifests = [_collect_shard(teacher, opponents, games, shard_paths[0], seed)]
    else:
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

        from locma.harness.parallel import init_eval_worker  # noqa: PLC0415

        with ProcessPoolExecutor(max_workers=workers, initializer=init_eval_worker) as ex:
            manifests = list(
                ex.map(
                    _collect_shard,
                    [teacher] * workers,
                    [opponents] * workers,
                    counts,
                    shard_paths,
                    [seed + s for s in starts],
                )
            )

    # Merge shards: concatenate arrays, offset game_id so ids stay unique.
    merged: dict[str, list] = {k: [] for k in (*_TOKEN_KEYS, *_META_KEYS, "game_id")}
    gid_offset = 0
    for path in shard_paths:
        with np.load(path) as d:
            for k in (*_TOKEN_KEYS, *_META_KEYS):
                merged[k].append(d[k])
            merged["game_id"].append(d["game_id"] + gid_offset)
            if len(d["game_id"]):
                gid_offset += int(d["game_id"].max()) + 1
        os.remove(path)
        mpath = _manifest_path(path)
        if os.path.exists(mpath):
            os.remove(mpath)

    arrays = {k: np.concatenate(v) for k, v in merged.items()}
    np.savez(out, **arrays)

    manifest = dict(manifests[0])
    manifest.update(
        teacher=teacher,
        opponents=opponents,
        games=games,
        seed=seed,
        n_examples=int(len(arrays["seat"])),
        n_dropped_overflow=sum(m["n_dropped_overflow"] for m in manifests),
        failed_games=sum(m["failed_games"] for m in manifests),
        shards=workers,
    )
    with open(_manifest_path(out), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _value_targets(data: dict) -> np.ndarray:
    """Value labels for training.

    Prefers an explicit ``target`` column (AZ-style backed-up scores from
    ``collect_backup_data``); otherwise falls back to Monte-Carlo game labels
    (+1 where the recorded seat won, else -1) from a practicum dataset.
    """
    keys = getattr(data, "files", data)
    if "target" in keys:
        return np.asarray(data["target"], dtype=np.float32)
    return np.where(data["winner"] == data["seat"], 1.0, -1.0).astype(np.float32)


def _collect_backup_shard(
    model_path: str,
    opponents,
    games: int,
    out: str,
    seed: int,
    width: int,
    max_actions: int,
) -> int:
    """Top-level picklable unit of work: one backed-up-target shard (E5 v2b).

    Plays vbeam(model) vs each opponent (both seats), harvesting plan_turn's
    backed-up targets. Keeps the states whose ranking decides play: the root,
    the depth-1 siblings (the action-choice comparison), and every
    stop-eligible state (the final plan argmax). Returns the example count.
    """
    from locma.core.engine import run_game  # noqa: PLC0415
    from locma.envs.encode import encode_battle_tokens  # noqa: PLC0415
    from locma.policies.composer import Composer  # noqa: PLC0415
    from locma.policies.drafts import BalancedDraftPolicy  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415
    from locma.policies.vbeam import VBeamBattlePolicy  # noqa: PLC0415

    sink: list = []
    vb = VBeamBattlePolicy(
        model_path=model_path, width=width, max_actions=max_actions, collect=sink
    )
    me = Composer(vb, BalancedDraftPolicy(), name="vbeam-harvest")

    kept_views: list = []
    kept_targets: list = []
    for opp_spec in opponents:
        opp = make_policy(opp_spec)
        for g in range(games):
            for my_seat in (0, 1):
                sink.clear()
                p0, p1 = (me, opp) if my_seat == 0 else (opp, me)
                run_game(p0, p1, seed + g)
                for view, target, depth, stop_ok in sink:
                    if depth <= 1 or stop_ok:
                        kept_views.append(view)
                        kept_targets.append(target)

    n = len(kept_targets)
    obs = [encode_battle_tokens(v) for v in kept_views]
    np.savez_compressed(
        out,
        obs_tokens=np.asarray([o["tokens"] for o in obs], dtype=np.float32),
        obs_card_ids=np.asarray([o["card_ids"] for o in obs], dtype=np.float32),
        obs_token_mask=np.asarray([o["token_mask"] for o in obs], dtype=np.float32),
        obs_scalars=np.asarray([o["scalars"] for o in obs], dtype=np.float32),
        target=np.asarray(kept_targets, dtype=np.float32),
    )
    return n


def collect_backup_data(
    teacher_model: str,
    out: str,
    opponents=DEFAULT_OPPONENTS,
    games: int = 100,
    seed: int = 0,
    workers: int = 1,
    width: int = 8,
    max_actions: int = 20,
) -> dict:
    """Record AZ-style backed-up value targets from vbeam's own searches.

    ``teacher_model`` is a token model .zip (the planner runs on it). Shards
    the per-opponent game count over a process pool like ``collect_value_data``
    and merges into ``out`` (.npz with obs arrays + a ``target`` column that
    ``train_value_head`` picks up automatically). Returns a small manifest.
    """
    opponents = list(opponents)
    workers = max(1, min(workers, games))

    base, extra = divmod(games, workers)
    counts = [base + (1 if w < extra else 0) for w in range(workers)]
    starts = [sum(counts[:w]) for w in range(workers)]
    shard_paths = [f"{out}.shard{w}.npz" for w in range(workers)]

    if workers == 1:
        _collect_backup_shard(
            teacher_model, opponents, games, shard_paths[0], seed, width, max_actions
        )
    else:
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

        from locma.harness.parallel import init_eval_worker  # noqa: PLC0415

        with ProcessPoolExecutor(max_workers=workers, initializer=init_eval_worker) as ex:
            list(
                ex.map(
                    _collect_backup_shard,
                    [teacher_model] * workers,
                    [opponents] * workers,
                    counts,
                    shard_paths,
                    [seed + s for s in starts],
                    [width] * workers,
                    [max_actions] * workers,
                )
            )

    merged: dict[str, list] = {k: [] for k in (*_TOKEN_KEYS, "target")}
    for path in shard_paths:
        with np.load(path) as d:
            for k, chunks in merged.items():
                chunks.append(d[k])
        os.remove(path)
    arrays = {k: np.concatenate(v) for k, v in merged.items()}
    np.savez_compressed(out, **arrays)

    manifest = {
        "kind": "vbeam-backup-targets",
        "teacher_model": teacher_model,
        "opponents": opponents,
        "games": games,
        "seed": seed,
        "width": width,
        "max_actions": max_actions,
        "n_examples": int(len(arrays["target"])),
        "shards": workers,
    }
    with open(_manifest_path(out), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _collect_ensemble_shard(
    model_paths,
    opponents,
    games: int,
    out: str,
    seed: int,
    width: int,
    max_actions: int,
) -> int:
    """Top-level picklable unit of work: one ensemble-distill shard (E9).

    Plays vbeam on the ENSEMBLE evaluator vs each opponent (both seats),
    keeping the ranking-deciding states (root, depth-1 siblings, and every
    stop-eligible state, as in ``_collect_backup_shard``). Each kept state is
    labeled with the ensemble MEAN value (the distillation ``target`` — noise-
    free, sibling-differing, computed by pure forward passes) and the
    cross-member std (``spread`` — the per-state variance the ensemble
    removes; the fidelity gate compares the distilled val RMSE against it).
    Returns the example count.
    """
    from locma.core.engine import run_game  # noqa: PLC0415
    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.envs.encode import encode_battle_tokens  # noqa: PLC0415
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
    me = Composer(vb, BalancedDraftPolicy(), name="vbeam-ens-harvest")

    kept_views: list = []
    for opp_spec in opponents:
        opp = make_policy(opp_spec)
        for g in range(games):
            for my_seat in (0, 1):
                sink.clear()
                p0, p1 = (me, opp) if my_seat == 0 else (opp, me)
                run_game(p0, p1, seed + g)
                for view, _target, depth, stop_ok in sink:
                    if depth <= 1 or stop_ok:
                        kept_views.append(view)

    # Label with each member critic (chunked batched forwards, no games).
    n = len(kept_views)
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
    )
    return n


def collect_ensemble_data(
    model_paths,
    out: str,
    opponents=DEFAULT_OPPONENTS,
    games: int = 50,
    seed: int = 0,
    workers: int = 1,
    width: int = 8,
    max_actions: int = 20,
) -> dict:
    """Record ensemble-mean value targets from vbeam-on-the-ensemble's own play.

    ``model_paths`` are the ensemble members (.zip paths or ``depot:`` refs).
    Shards the per-opponent game count over a process pool like
    ``collect_backup_data`` and merges into ``out`` (.npz with obs arrays, a
    ``target`` column that ``train_value_head`` picks up automatically, and a
    ``spread`` column for the fidelity gate). Returns a manifest including
    ``mean_spread``, the average per-state cross-member std.
    """
    opponents = list(opponents)
    model_paths = list(model_paths)
    workers = max(1, min(workers, games))

    base, extra = divmod(games, workers)
    counts = [base + (1 if w < extra else 0) for w in range(workers)]
    starts = [sum(counts[:w]) for w in range(workers)]
    shard_paths = [f"{out}.shard{w}.npz" for w in range(workers)]

    if workers == 1:
        _collect_ensemble_shard(
            model_paths, opponents, games, shard_paths[0], seed, width, max_actions
        )
    else:
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

        from locma.harness.parallel import init_eval_worker  # noqa: PLC0415

        with ProcessPoolExecutor(max_workers=workers, initializer=init_eval_worker) as ex:
            list(
                ex.map(
                    _collect_ensemble_shard,
                    [model_paths] * workers,
                    [opponents] * workers,
                    counts,
                    shard_paths,
                    [seed + s for s in starts],
                    [width] * workers,
                    [max_actions] * workers,
                )
            )

    merged: dict[str, list] = {k: [] for k in (*_TOKEN_KEYS, "target", "spread")}
    for path in shard_paths:
        with np.load(path) as d:
            for k, chunks in merged.items():
                chunks.append(d[k])
        os.remove(path)
    arrays = {k: np.concatenate(v) for k, v in merged.items()}
    np.savez_compressed(out, **arrays)

    manifest = {
        "kind": "vbeam-ensemble-distill",
        "model_paths": model_paths,
        "opponents": opponents,
        "games": games,
        "seed": seed,
        "width": width,
        "max_actions": max_actions,
        "n_examples": int(len(arrays["target"])),
        "mean_spread": float(arrays["spread"].mean()) if len(arrays["spread"]) else 0.0,
        "shards": workers,
    }
    with open(_manifest_path(out), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def train_value_head(
    base_model: str,
    data: str,
    out: str,
    *,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 3e-4,
    val_frac: float = 0.05,
    seed: int = 0,
) -> dict:
    """Fine-tune ONLY the critic branch of ``base_model`` on ``data``; save ``out``.

    Frozen: the (shared) features extractor and the entire policy path — the
    saved model's action probabilities are byte-identical to ``base_model``'s,
    which vbeam's stop rule depends on. Trained: ``mlp_extractor.value_net``
    (the vf MLP over frozen features) + ``value_net`` (the final linear).

    Features are precomputed once (the extractor is frozen and in eval mode,
    so they are deterministic), which makes the fine-tune itself run in
    seconds. Returns metrics: val MSE and sign accuracy (V > 0 predicts a
    win), before and after.
    """
    import torch  # noqa: PLC0415 — optional [ml] dep
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.depot import resolve_path  # noqa: PLC0415

    model = MaskablePPO.load(resolve_path(base_model))
    policy = model.policy
    policy.set_training_mode(False)  # dropout off: deterministic frozen features
    device = policy.device

    d = np.load(resolve_path(data))
    y_np = _value_targets(d)
    n = int(len(y_np))
    if n == 0:
        raise ValueError(f"no examples in {data}")

    # Precompute frozen-extractor features once, in chunks.
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
    feats = torch.cat(feats_chunks)
    y = torch.from_numpy(y_np)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx = torch.from_numpy(perm[:n_val].copy())
    tr_idx = torch.from_numpy(perm[n_val:].copy())

    for p in policy.parameters():
        p.requires_grad_(False)
    train_params = [
        *policy.mlp_extractor.value_net.parameters(),
        *policy.value_net.parameters(),
    ]
    for p in train_params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(train_params, lr=lr)

    def _v(idx: torch.Tensor) -> torch.Tensor:
        latent_vf = policy.mlp_extractor.forward_critic(feats[idx].to(device))
        return policy.value_net(latent_vf).squeeze(-1)

    def _val_metrics() -> tuple[float, float]:
        with torch.no_grad():
            v = _v(val_idx).cpu()
        t = y[val_idx]
        mse = float(torch.mean((v - t) ** 2))
        sign_acc = float(torch.mean((torch.sign(v) == t).float()))
        return mse, sign_acc

    mse_before, acc_before = _val_metrics()

    gen = torch.Generator().manual_seed(seed)
    for _ep in range(epochs):
        order = torch.randperm(len(tr_idx), generator=gen)
        for i in range(0, len(tr_idx), batch_size):
            bidx = tr_idx[order[i : i + batch_size]]
            loss = torch.nn.functional.mse_loss(_v(bidx), y[bidx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()

    mse_after, acc_after = _val_metrics()
    model.save(out)
    return {
        "n_examples": n,
        "n_val": int(n_val),
        "epochs": epochs,
        "val_mse_before": mse_before,
        "val_mse_after": mse_after,
        "val_sign_acc_before": acc_before,
        "val_sign_acc_after": acc_after,
        "out": out,
    }
