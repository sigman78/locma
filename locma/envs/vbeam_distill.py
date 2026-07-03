"""Distill the vbeam planner back into the reactive net (E4v2 / EXIT).

The vbeam planner (E5 variant 1, 0.863 on the ruler) is the first *fair*
teacher in this repo: unlike the ``mcts``/``dmcts`` cheaters, every planner
decision is a deterministic function of the public ``BattleView`` the student
observes, so imitation has no info ceiling. A practicum recorded with
``teacher="vbeam:..."`` captures the planner's chosen action per non-forced
decision — including its mid-turn Pass choices, which carry the stop
discipline vbeam's own ``would_pass`` rule reads back at plan time.

``train_policy_head`` is the mirror image of ``vbeam_fvi.train_value_head``:
fine-tune ONLY the policy branch of a saved token MaskablePPO
(``mlp_extractor.policy_net`` + ``action_net``) with masked cross-entropy on
frozen precomputed features. The features extractor and the entire critic
path stay byte-identical, so the output drops into ``vbeam:out.zip`` with an
unchanged evaluator and a (hopefully) better stop-eligibility pattern — the
expert-iteration arm.

Imports the ML stack lazily; an ImportError means the ``[ml]`` extra is absent.
"""

from __future__ import annotations

import numpy as np

from locma.envs.distill import split_by_game

# Large negative logit for illegal actions — same masking idea as
# sb3-contrib's MaskableCategorical, applied to our precomputed-feature path.
_MASKED_LOGIT = -1e9


def train_policy_head(
    base_model: str,
    data: str,
    out: str,
    *,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 3e-4,
    val_frac: float = 0.1,
    seed: int = 0,
    verbose: int = 0,
) -> dict:
    """Fine-tune ONLY the policy branch of ``base_model`` on ``data``; save ``out``.

    Frozen: the (shared) features extractor and the entire critic path — the
    saved model's value estimates are byte-identical to ``base_model``'s,
    which vbeam's plan ranking depends on. Trained: ``mlp_extractor.policy_net``
    (the pi MLP over frozen features) + ``action_net`` (the final logits).

    ``data`` is a token practicum (``record_practicum(obs_mode="token")`` /
    ``collect_value_data``): masked CE toward the recorded teacher actions,
    split at game level so no game leaks between train and val. Features are
    precomputed once (frozen extractor in eval mode), so the fine-tune itself
    runs in seconds. Returns metrics: val top-1 agreement and val CE, before
    and after (the "before" agreement = how often the base argmax already
    matches the teacher).
    """
    import torch  # noqa: PLC0415 — optional [ml] dep
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.depot import resolve_path  # noqa: PLC0415

    model = MaskablePPO.load(resolve_path(base_model))
    policy = model.policy
    policy.set_training_mode(False)  # dropout off: deterministic frozen features
    device = policy.device

    d = np.load(resolve_path(data))
    act_np = d["action"].astype(np.int64)
    mask_np = d["mask"].astype(bool)
    n = int(len(act_np))
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
    act = torch.from_numpy(act_np)
    legal = torch.from_numpy(mask_np)

    # Guarantee >= 1 held-out game (tiny practicums round int(n*frac) to 0).
    n_games = len({int(g) for g in d["game_id"]})
    if n_games > 1:
        val_frac = max(val_frac, 1.0 / n_games + 1e-9)
    tr_list, val_list = split_by_game(d["game_id"], val_frac, seed)
    tr_idx = torch.as_tensor(tr_list, dtype=torch.int64)
    val_idx = torch.as_tensor(val_list, dtype=torch.int64)
    if not len(val_idx):
        raise ValueError("empty validation split — need at least 2 games in the practicum")

    for p in policy.parameters():
        p.requires_grad_(False)
    train_params = [
        *policy.mlp_extractor.policy_net.parameters(),
        *policy.action_net.parameters(),
    ]
    for p in train_params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(train_params, lr=lr)

    def _logits(idx: torch.Tensor) -> torch.Tensor:
        latent_pi = policy.mlp_extractor.forward_actor(feats[idx].to(device))
        logits = policy.action_net(latent_pi)
        return logits.masked_fill(~legal[idx].to(device), _MASKED_LOGIT)

    def _val_metrics() -> tuple[float, float]:
        with torch.no_grad():
            logits = _logits(val_idx)
            t = act[val_idx].to(device)
            ce = float(torch.nn.functional.cross_entropy(logits, t))
            agree = float((logits.argmax(dim=1) == t).float().mean())
        return ce, agree

    ce_before, agree_before = _val_metrics()

    gen = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        order = torch.randperm(len(tr_idx), generator=gen)
        total, nb = 0.0, 0
        for i in range(0, len(tr_idx), batch_size):
            bidx = tr_idx[order[i : i + batch_size]]
            loss = torch.nn.functional.cross_entropy(_logits(bidx), act[bidx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        if verbose:
            ce, agree = _val_metrics()
            print(
                f"epoch {ep + 1}/{epochs}  loss={total / max(nb, 1):.4f}  "
                f"val_ce={ce:.4f}  val_agree={agree:.3f}"
            )

    ce_after, agree_after = _val_metrics()
    model.save(out)
    return {
        "n_examples": n,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(val_idx)),
        "epochs": epochs,
        "val_ce_before": ce_before,
        "val_ce_after": ce_after,
        "val_agreement_before": agree_before,
        "val_agreement_after": agree_after,
        "out": out,
    }
