# Flat-obs PPO Autoregressive Action Head — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat PPO policy's single 155-way masked softmax with a true conditional autoregressive head (`type → source|type → target|type,source`), holding the flat-308 observation and the `Discrete(155)` action space fixed, and settle by a symmetric +0.03 paired verdict whether it beats the flat baseline on avg-hard3.

**Architecture:** The 155 action indices factor exactly along `type/source/target`. A custom `MaskableAutoregressivePolicy` (subclass of sb3-contrib's `MaskableActorCriticPolicy`) keeps the action space `Discrete(155)` and the 155-bool mask, but swaps the single action head for three conditional heads scored with sb3's `MaskableCategorical`. All conditional masks are *derived* from the flat 155 mask, so legality can never diverge; non-applicable factors collapse to a single legal cell (log-prob 0, entropy 0), so the three-factor scoring needs no per-type branching.

**Tech Stack:** Python, PyTorch (CPU), stable-baselines3 2.9 + sb3-contrib 2.9 (`MaskablePPO`, `MaskableActorCriticPolicy`, `MaskableCategorical`), gymnasium, numpy, typer, pytest, ruff, uv.

## Global Constraints

- **Single lever:** observation stays `encode_battle` (flat-308 `Box`); action space stays `Discrete(155)` with today's exact mask. Only the action head/distribution changes.
- **Additive, behind a flag:** `head="flat"` must stay byte-identical to today's baseline (default MlpPolicy). The AR head is `head="autoreg"`.
- **Comparability:** actions in the rollout buffer stay flat ints; masks stay 155-bool; MaskablePPO's rollout/GAE/buffer/mask plumbing is untouched. Saved model stays a normal `MaskablePPO.load`-able `.zip`.
- **Baseline recipe (both models identical except head):** LR `3e-4`, `ent_coef 0.02`, `both_seat=True`, zoo curriculum `(greedy, scripted, max-guard, max-attack)` at `200_000` steps each (800k total), `seed 0`.
- **Metric:** `avg-hard3` = mean win-rate vs {scripted, max-guard, max-attack}, deterministic policy, held-out eval seeds (1_000_000+ range), paired (common random numbers) between the two models.
- **Decision rule:** symmetric pre-committed **±0.03** on `avg-hard3(ar) − avg-hard3(flat)`, resolved by paired-difference bootstrap CI. Clear +0.03 with CI excluding 0 → AR helps; whole CI within ±0.03 → no help; else inconclusive (report, no budget-chasing).
- **No new dependencies** (torch/sb3/numpy/typer already present; no TensorBoard).
- **Everything runs via `uv run`.** ML code needs `--extra ml`; tests/lint need `--extra dev`. Torch is CPU-only on this box.
- **CI gate before every commit:** `uv run --extra dev ruff check . && uv run --extra dev ruff format --check . && uv run --extra dev pytest -q`.

## File Structure

| File | Responsibility | ML dep |
|------|----------------|--------|
| `locma/envs/action_factor.py` (new) | torch-free `decode`/`encode` + `factor_masks(flat_mask)` | no |
| `locma/envs/ar_distribution.py` (new) | tensor helpers + `ARHeads` + `ar_sample`/`ar_evaluate` | yes |
| `locma/envs/ar_policy.py` (new) | `MaskableAutoregressivePolicy` (3 heads + critic) | yes |
| `locma/envs/ar_callbacks.py` (new) | periodic avg-hard3 eval + per-head entropy telemetry | yes |
| `locma/envs/training.py` (modify) | `head` param threaded through `_make_model`/`train_agent`/`train_zoo` | yes (lazy) |
| `locma/cli/app.py` (modify) | `--head` on train/train-zoo; new `ar-eval` command | no |
| `locma/harness/ar_study.py` (new) | avg-hard3 + paired bootstrap + verdict + runner | no (numpy) |
| `tests/test_action_factor.py` (new) | round-trip + mask reconstruction | no |
| `tests/test_ar_distribution.py` (new) | tensor helpers + distribution properties | yes |
| `tests/test_ar_policy.py` (new) | policy forward/evaluate/predict + smoke train | yes |
| `tests/test_ar_study.py` (new) | bootstrap + decide | no |

---

### Task 1: Action factorization (torch-free)

**Files:**
- Create: `locma/envs/action_factor.py`
- Test: `tests/test_action_factor.py`

**Interfaces:**
- Consumes: the fixed action layout from `locma/envs/encode.py` (`ACTION_SIZE=155`, index ranges).
- Produces:
  - `PASS, SUMMON, USE, ATTACK = 0, 1, 2, 3`
  - `N_TYPE = 4`, `MAX_SOURCE = 8`, `MAX_TARGET = 13`, `ACTION_SIZE = 155`
  - `SEG` = per-type `(base, n_source, n_target)`: `((0, 1, 1), (1, 8, 1), (9, 8, 13), (113, 6, 7))`
  - `decode(idx: int) -> tuple[int, int, int]` → `(type, source, target)`
  - `encode(t: int, s: int, tgt: int) -> int`
  - `factor_masks(flat_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]` → `(type_mask[4], source_mask[4,8], target_mask[4,8,13])`, all bool

- [ ] **Step 1: Write the failing test**

```python
# tests/test_action_factor.py
import numpy as np
import pytest

from locma.envs.action_factor import (
    ACTION_SIZE, ATTACK, PASS, SUMMON, USE, decode, encode, factor_masks,
)


def test_round_trip_all_indices():
    for idx in range(ACTION_SIZE):
        t, s, tgt = decode(idx)
        assert encode(t, s, tgt) == idx


def test_decode_boundaries():
    assert decode(0) == (PASS, 0, 0)
    assert decode(1) == (SUMMON, 0, 0)
    assert decode(8) == (SUMMON, 7, 0)
    assert decode(9) == (USE, 0, 0)
    assert decode(112) == (USE, 7, 12)
    assert decode(113) == (ATTACK, 0, 0)
    assert decode(154) == (ATTACK, 5, 6)


def test_factor_masks_reconstruct_flat():
    rng = np.random.default_rng(0)
    for _ in range(200):
        flat = rng.random(ACTION_SIZE) < 0.15
        tm, sm, tgtm = factor_masks(flat)
        # every legal flat idx appears in all three conditional masks
        rebuilt = np.zeros(ACTION_SIZE, dtype=bool)
        for idx in range(ACTION_SIZE):
            t, s, tgt = decode(idx)
            if tm[t] and sm[t, s] and tgtm[t, s, tgt]:
                rebuilt[idx] = True
        assert np.array_equal(rebuilt, flat)


def test_factor_masks_nonapplicable_single_cell():
    flat = np.zeros(ACTION_SIZE, dtype=bool)
    flat[0] = True  # Pass legal
    flat[1] = True  # Summon slot 0 legal
    tm, sm, tgtm = factor_masks(flat)
    # Pass: exactly one legal source and one legal target
    assert sm[PASS].sum() == 1 and tgtm[PASS, 0].sum() == 1
    # Summon slot 0: one legal target
    assert tgtm[SUMMON, 0].sum() == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_action_factor.py -q`
Expected: FAIL — `ModuleNotFoundError: locma.envs.action_factor`

- [ ] **Step 3: Write minimal implementation**

```python
# locma/envs/action_factor.py
"""Torch-free factorization of the Discrete(155) action space into
type -> source -> target, plus derivation of the conditional legality masks
from the flat 155-bool mask. See docs/ppo-autoreg-action-design.md."""

from __future__ import annotations

import numpy as np

PASS, SUMMON, USE, ATTACK = 0, 1, 2, 3
N_TYPE = 4
MAX_SOURCE = 8
MAX_TARGET = 13
ACTION_SIZE = 155

# per type: (base flat index, n_source, n_target)
SEG: tuple[tuple[int, int, int], ...] = (
    (0, 1, 1),      # PASS
    (1, 8, 1),      # SUMMON
    (9, 8, 13),     # USE
    (113, 6, 7),    # ATTACK
)


def decode(idx: int) -> tuple[int, int, int]:
    """Map a flat action index to (type, source, target)."""
    if idx < 1:
        return (PASS, 0, 0)
    if idx < 9:
        return (SUMMON, idx - 1, 0)
    if idx < 113:
        off = idx - 9
        return (USE, off // 13, off % 13)
    off = idx - 113
    return (ATTACK, off // 7, off % 7)


def encode(t: int, s: int, tgt: int) -> int:
    """Inverse of decode: (type, source, target) -> flat action index."""
    base, _n_src, n_tgt = SEG[t]
    return base + s * n_tgt + tgt


def factor_masks(flat_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive conditional legality masks from the flat 155-bool mask.

    Returns (type_mask[4], source_mask[4,8], target_mask[4,8,13]), all bool.
    Non-applicable factors (e.g. a Pass has no real source/target) end up with
    exactly one legal cell, so their conditional log-prob and entropy are 0.
    """
    type_mask = np.zeros(N_TYPE, dtype=bool)
    source_mask = np.zeros((N_TYPE, MAX_SOURCE), dtype=bool)
    target_mask = np.zeros((N_TYPE, MAX_SOURCE, MAX_TARGET), dtype=bool)
    for t, (base, n_src, n_tgt) in enumerate(SEG):
        seg = flat_mask[base : base + n_src * n_tgt].reshape(n_src, n_tgt)
        target_mask[t, :n_src, :n_tgt] = seg
        source_mask[t, :n_src] = seg.any(axis=1)
        type_mask[t] = seg.any()
    return type_mask, source_mask, target_mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_action_factor.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add locma/envs/action_factor.py tests/test_action_factor.py
git commit -m "feat(ppo-autoreg): torch-free action factorization + conditional masks"
```

---

### Task 2: Tensor helpers for the AR head

**Files:**
- Create: `locma/envs/ar_distribution.py` (this task adds the tensor helpers only)
- Test: `tests/test_ar_distribution.py` (this task adds the helper tests only)

**Interfaces:**
- Consumes: `SEG`, `decode`, `encode`, `ACTION_SIZE` from `locma.envs.action_factor`.
- Produces:
  - `decode_batch(flat: LongTensor[B]) -> (type[B], source[B], target[B])` (all long)
  - `encode_batch(t: LongTensor[B], s: LongTensor[B], tgt: LongTensor[B]) -> LongTensor[B]`
  - `factor_grids(flat_masks: BoolTensor[B,155]) -> BoolTensor[B,4,8,13]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ar_distribution.py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from locma.envs.action_factor import ACTION_SIZE, decode, factor_masks  # noqa: E402
from locma.envs.ar_distribution import (  # noqa: E402
    decode_batch, encode_batch, factor_grids,
)


def test_decode_batch_matches_scalar():
    flat = torch.arange(ACTION_SIZE, dtype=torch.long)
    t, s, tgt = decode_batch(flat)
    for idx in range(ACTION_SIZE):
        assert (int(t[idx]), int(s[idx]), int(tgt[idx])) == decode(idx)


def test_encode_batch_inverts_decode():
    flat = torch.arange(ACTION_SIZE, dtype=torch.long)
    t, s, tgt = decode_batch(flat)
    assert torch.equal(encode_batch(t, s, tgt), flat)


def test_factor_grids_match_numpy_factor_masks():
    rng = np.random.default_rng(1)
    flat = rng.random((5, ACTION_SIZE)) < 0.2
    grids = factor_grids(torch.as_tensor(flat))
    for b in range(5):
        tm, sm, tgtm = factor_masks(flat[b])
        assert torch.equal(grids[b].any(dim=(1, 2)), torch.as_tensor(tm))
        assert torch.equal(grids[b].any(dim=2), torch.as_tensor(sm))
        assert torch.equal(grids[b], torch.as_tensor(tgtm))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra ml --extra dev pytest tests/test_ar_distribution.py -q`
Expected: FAIL — `ImportError: cannot import name 'decode_batch'` (module/functions absent)

- [ ] **Step 3: Write minimal implementation**

```python
# locma/envs/ar_distribution.py
"""Autoregressive action head: tensor helpers, head modules, and the
sample/evaluate core. See docs/ppo-autoreg-action-design.md.

Requires the [ml] extra (torch). The torch-free factorization lives in
locma.envs.action_factor."""

from __future__ import annotations

import torch
from torch import nn

from locma.envs.action_factor import ACTION_SIZE, MAX_SOURCE, MAX_TARGET, N_TYPE, SEG

EMB_DIM = 8
_NEG = -1e8

# base / n_target per type, as tensors for vectorized encode
_BASE = torch.tensor([s[0] for s in SEG], dtype=torch.long)
_NTGT = torch.tensor([s[2] for s in SEG], dtype=torch.long)


def decode_batch(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized decode: LongTensor[B] -> (type, source, target)."""
    t = torch.zeros_like(flat)
    t = torch.where(flat >= 1, torch.ones_like(flat), t)
    t = torch.where(flat >= 9, torch.full_like(flat, 2), t)
    t = torch.where(flat >= 113, torch.full_like(flat, 3), t)
    src = torch.zeros_like(flat)
    tgt = torch.zeros_like(flat)
    m_sum = t == 1
    src = torch.where(m_sum, flat - 1, src)
    m_use = t == 2
    src = torch.where(m_use, (flat - 9) // 13, src)
    tgt = torch.where(m_use, (flat - 9) % 13, tgt)
    m_att = t == 3
    src = torch.where(m_att, (flat - 113) // 7, src)
    tgt = torch.where(m_att, (flat - 113) % 7, tgt)
    return t, src, tgt


def encode_batch(t: torch.Tensor, s: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Vectorized encode: (type, source, target) -> flat LongTensor[B]."""
    base = _BASE.to(t.device)[t]
    ntgt = _NTGT.to(t.device)[t]
    return base + s * ntgt + tgt


def factor_grids(flat_masks: torch.Tensor) -> torch.Tensor:
    """Bool[B,155] -> Bool[B,4,8,13]: legal (type,source,target) grid per sample."""
    b = flat_masks.shape[0]
    grids = torch.zeros(b, N_TYPE, MAX_SOURCE, MAX_TARGET, dtype=torch.bool, device=flat_masks.device)
    for t, (base, n_src, n_tgt) in enumerate(SEG):
        seg = flat_masks[:, base : base + n_src * n_tgt].reshape(b, n_src, n_tgt)
        grids[:, t, :n_src, :n_tgt] = seg
    return grids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra ml --extra dev pytest tests/test_ar_distribution.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add locma/envs/ar_distribution.py tests/test_ar_distribution.py
git commit -m "feat(ppo-autoreg): vectorized decode/encode/factor-grid tensor helpers"
```

---

### Task 3: AR heads + sample/evaluate core

**Files:**
- Modify: `locma/envs/ar_distribution.py` (append `ARHeads`, `ar_sample`, `ar_evaluate`)
- Test: `tests/test_ar_distribution.py` (append distribution-property tests)

**Interfaces:**
- Consumes: `factor_grids`, `decode_batch`, `encode_batch`, `EMB_DIM` (this module); `MaskableCategorical` from `sb3_contrib.common.maskable.distributions`.
- Produces:
  - `class ARHeads(nn.Module)` with `__init__(self, latent_dim: int, emb_dim: int = EMB_DIM)` and submodules `head_type`, `head_source`, `head_target`, `emb_type`, `emb_source`.
  - `ar_sample(heads: ARHeads, z: Tensor[B,D], flat_masks: BoolTensor[B,155], deterministic: bool) -> (flat_actions[B] long, log_prob[B])`
  - `ar_evaluate(heads: ARHeads, z: Tensor[B,D], flat_masks: BoolTensor[B,155], actions: LongTensor[B]) -> (log_prob[B], entropy[B])`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ar_distribution.py  (append)
from locma.envs.ar_distribution import ARHeads, ar_evaluate, ar_sample  # noqa: E402


def _masks_batch(seed, b=16):
    rng = np.random.default_rng(seed)
    flats = np.zeros((b, ACTION_SIZE), dtype=bool)
    for i in range(b):
        flats[i, 0] = True  # Pass always legal
        # random extra legal actions
        extra = rng.integers(1, ACTION_SIZE, size=rng.integers(1, 6))
        flats[i, extra] = True
    return torch.as_tensor(flats)


def test_sampled_actions_are_always_legal():
    torch.manual_seed(0)
    heads = ARHeads(latent_dim=12)
    flat_masks = _masks_batch(2)
    z = torch.randn(flat_masks.shape[0], 12)
    actions, _ = ar_sample(heads, z, flat_masks, deterministic=False)
    for i, a in enumerate(actions.tolist()):
        assert flat_masks[i, a]


def test_log_prob_equals_sum_of_conditionals_and_finite():
    torch.manual_seed(1)
    heads = ARHeads(latent_dim=12)
    flat_masks = _masks_batch(3)
    z = torch.randn(flat_masks.shape[0], 12)
    actions, lp_sample = ar_sample(heads, z, flat_masks, deterministic=True)
    lp_eval, ent = ar_evaluate(heads, z, flat_masks, actions)
    # teacher-forced eval of the deterministically chosen action matches sampling
    assert torch.allclose(lp_sample, lp_eval, atol=1e-5)
    assert torch.isfinite(lp_eval).all()
    assert torch.isfinite(ent).all()
    assert (ent >= -1e-6).all()  # entropy non-negative


def test_pass_only_has_zero_logprob_and_entropy():
    torch.manual_seed(2)
    heads = ARHeads(latent_dim=12)
    flat = np.zeros((1, ACTION_SIZE), dtype=bool)
    flat[0, 0] = True  # only Pass legal
    flat_masks = torch.as_tensor(flat)
    z = torch.randn(1, 12)
    actions, lp = ar_sample(heads, z, flat_masks, deterministic=True)
    assert int(actions[0]) == 0
    assert torch.allclose(lp, torch.zeros_like(lp), atol=1e-5)  # forced choice
    _, ent = ar_evaluate(heads, z, flat_masks, actions)
    assert torch.allclose(ent, torch.zeros_like(ent), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra ml --extra dev pytest tests/test_ar_distribution.py -q`
Expected: FAIL — `ImportError: cannot import name 'ARHeads'`

- [ ] **Step 3: Write minimal implementation**

Append to `locma/envs/ar_distribution.py`:

```python
from sb3_contrib.common.maskable.distributions import MaskableCategorical  # noqa: E402


class ARHeads(nn.Module):
    """Conditional heads for the autoregressive action distribution.

    type <- z ; source <- (z, emb_type) ; target <- (z, emb_type, emb_source).
    Head widths are the fixed maxima (4 / 8 / 13); masking zeroes out-of-domain
    entries so one source head and one target head serve every action type.
    """

    def __init__(self, latent_dim: int, emb_dim: int = EMB_DIM) -> None:
        super().__init__()
        self.emb_type = nn.Embedding(N_TYPE, emb_dim)
        self.emb_source = nn.Embedding(MAX_SOURCE, emb_dim)
        self.head_type = nn.Linear(latent_dim, N_TYPE)
        self.head_source = nn.Linear(latent_dim + emb_dim, MAX_SOURCE)
        self.head_target = nn.Linear(latent_dim + 2 * emb_dim, MAX_TARGET)


def _arange(b: int, device) -> torch.Tensor:
    return torch.arange(b, device=device)


def ar_sample(heads, z, flat_masks, deterministic):
    """Sample type -> source -> target sequentially under derived masks.

    Returns (flat_actions[B] long, log_prob[B]). log_prob is the sum of the
    three conditional log-probs at the chosen values.
    """
    b = z.shape[0]
    idx = _arange(b, z.device)
    grids = factor_grids(flat_masks)  # [B,4,8,13]

    type_mask = grids.any(dim=(2, 3))  # [B,4]
    type_dist = MaskableCategorical(logits=heads.head_type(z), masks=type_mask)
    types = type_dist.probs.argmax(dim=-1) if deterministic else type_dist.sample()

    src_in = torch.cat([z, heads.emb_type(types)], dim=-1)
    src_mask = grids.any(dim=3)[idx, types]  # [B,8]
    src_dist = MaskableCategorical(logits=heads.head_source(src_in), masks=src_mask)
    sources = src_dist.probs.argmax(dim=-1) if deterministic else src_dist.sample()

    tgt_in = torch.cat([z, heads.emb_type(types), heads.emb_source(sources)], dim=-1)
    tgt_mask = grids[idx, types, sources]  # [B,13]
    tgt_dist = MaskableCategorical(logits=heads.head_target(tgt_in), masks=tgt_mask)
    targets = tgt_dist.probs.argmax(dim=-1) if deterministic else tgt_dist.sample()

    log_prob = type_dist.log_prob(types) + src_dist.log_prob(sources) + tgt_dist.log_prob(targets)
    return encode_batch(types, sources, targets), log_prob


def ar_evaluate(heads, z, flat_masks, actions):
    """Teacher-forced scoring of given flat actions.

    Returns (log_prob[B], entropy[B]) where entropy is the sum of the three
    conditional entropies along the visited prefix.
    """
    b = z.shape[0]
    idx = _arange(b, z.device)
    grids = factor_grids(flat_masks)
    types, sources, targets = decode_batch(actions)

    type_mask = grids.any(dim=(2, 3))
    type_dist = MaskableCategorical(logits=heads.head_type(z), masks=type_mask)

    src_in = torch.cat([z, heads.emb_type(types)], dim=-1)
    src_mask = grids.any(dim=3)[idx, types]
    src_dist = MaskableCategorical(logits=heads.head_source(src_in), masks=src_mask)

    tgt_in = torch.cat([z, heads.emb_type(types), heads.emb_source(sources)], dim=-1)
    tgt_mask = grids[idx, types, sources]
    tgt_dist = MaskableCategorical(logits=heads.head_target(tgt_in), masks=tgt_mask)

    log_prob = type_dist.log_prob(types) + src_dist.log_prob(sources) + tgt_dist.log_prob(targets)
    entropy = type_dist.entropy() + src_dist.entropy() + tgt_dist.entropy()
    return log_prob, entropy
```

**Note on `grids.any(dim=(2, 3))`:** torch supports tuple dims for `any` in 2.x. If the installed torch rejects it, use `grids.any(dim=3).any(dim=2)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra ml --extra dev pytest tests/test_ar_distribution.py -q`
Expected: PASS (6 tests total in the file)

- [ ] **Step 5: Commit**

```bash
git add locma/envs/ar_distribution.py tests/test_ar_distribution.py
git commit -m "feat(ppo-autoreg): ARHeads + masked autoregressive sample/evaluate core"
```

---

### Task 4: `MaskableAutoregressivePolicy`

**Files:**
- Create: `locma/envs/ar_policy.py`
- Test: `tests/test_ar_policy.py` (policy-level tests only; smoke train is Task 6)

**Interfaces:**
- Consumes: `MaskableActorCriticPolicy` (`sb3_contrib.common.maskable.policies`); `ARHeads`, `ar_sample`, `ar_evaluate` (`locma.envs.ar_distribution`).
- Produces: `class MaskableAutoregressivePolicy(MaskableActorCriticPolicy)` overriding `_build`, `forward`, `evaluate_actions`, `_predict`. Registers `self.ar_heads: ARHeads`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ar_policy.py
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")
import torch  # noqa: E402
from gymnasium import spaces  # noqa: E402

from locma.envs.action_factor import ACTION_SIZE  # noqa: E402
from locma.envs.ar_policy import MaskableAutoregressivePolicy  # noqa: E402


def _policy():
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(308,), dtype=np.float32)
    act_space = spaces.Discrete(ACTION_SIZE)
    return MaskableAutoregressivePolicy(obs_space, act_space, lambda _: 3e-4)


def _masks(b):
    m = np.zeros((b, ACTION_SIZE), dtype=bool)
    m[:, 0] = True
    m[:, 1] = True
    m[:, 113] = True
    return m


def test_forward_returns_legal_actions_and_finite():
    torch.manual_seed(0)
    policy = _policy()
    obs = torch.randn(4, 308)
    masks = _masks(4)
    actions, values, log_prob = policy.forward(obs, action_masks=masks)
    assert actions.shape == (4,)
    for i, a in enumerate(actions.tolist()):
        assert masks[i, a]
    assert torch.isfinite(values).all()
    assert torch.isfinite(log_prob).all()


def test_evaluate_actions_grads_flow_to_heads():
    torch.manual_seed(1)
    policy = _policy()
    obs = torch.randn(4, 308)
    masks = _masks(4)
    actions, _, _ = policy.forward(obs, action_masks=masks)
    values, log_prob, entropy = policy.evaluate_actions(obs, actions, action_masks=masks)
    loss = -(log_prob.mean()) + values.mean() - entropy.mean()
    loss.backward()
    assert policy.ar_heads.head_type.weight.grad is not None
    assert torch.isfinite(entropy).all()


def test_predict_is_deterministic_and_legal():
    torch.manual_seed(2)
    policy = _policy()
    obs = torch.randn(3, 308)
    masks = _masks(3)
    a1 = policy._predict(obs, deterministic=True, action_masks=masks)
    a2 = policy._predict(obs, deterministic=True, action_masks=masks)
    assert torch.equal(a1, a2)
    for i, a in enumerate(a1.tolist()):
        assert masks[i, a]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra ml --extra dev pytest tests/test_ar_policy.py -q`
Expected: FAIL — `ModuleNotFoundError: locma.envs.ar_policy`

- [ ] **Step 3: Write minimal implementation**

```python
# locma/envs/ar_policy.py
"""MaskablePPO policy with a conditional autoregressive action head.

Keeps the action space Discrete(155) and the 155-bool mask; swaps only the
single action head for three conditional heads. See
docs/ppo-autoreg-action-design.md. Requires the [ml] extra."""

from __future__ import annotations

import numpy as np
import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from locma.envs.ar_distribution import ARHeads, ar_evaluate, ar_sample


class MaskableAutoregressivePolicy(MaskableActorCriticPolicy):
    """Flat-obs MaskablePPO policy whose action head is autoregressive."""

    def _build(self, lr_schedule) -> None:
        # Build mlp_extractor, value_net, (unused) action_net, and optimizer.
        super()._build(lr_schedule)
        latent_dim = self.mlp_extractor.latent_dim_pi
        self.ar_heads = ARHeads(latent_dim)
        # Re-create the optimizer so it owns the AR head parameters too.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _masks_tensor(self, action_masks, batch: int) -> torch.Tensor:
        m = torch.as_tensor(action_masks, device=self.device)
        return m.reshape(batch, -1).bool()

    def _latents(self, obs):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        return latent_pi, latent_vf

    def forward(self, obs, deterministic=False, action_masks=None):
        latent_pi, latent_vf = self._latents(obs)
        values = self.value_net(latent_vf)
        masks = self._masks_tensor(action_masks, latent_pi.shape[0])
        actions, log_prob = ar_sample(self.ar_heads, latent_pi, masks, deterministic)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions, action_masks=None):
        latent_pi, latent_vf = self._latents(obs)
        values = self.value_net(latent_vf)
        masks = self._masks_tensor(action_masks, latent_pi.shape[0])
        log_prob, entropy = ar_evaluate(self.ar_heads, latent_pi, masks, actions.long().reshape(-1))
        return values, log_prob, entropy

    def _predict(self, observation, deterministic=False, action_masks=None):
        latent_pi, _ = self._latents(observation)
        masks = self._masks_tensor(action_masks, latent_pi.shape[0])
        actions, _ = ar_sample(self.ar_heads, latent_pi, masks, deterministic)
        return actions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra ml --extra dev pytest tests/test_ar_policy.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add locma/envs/ar_policy.py tests/test_ar_policy.py
git commit -m "feat(ppo-autoreg): MaskableAutoregressivePolicy (3 conditional heads)"
```

---

### Task 5: Wire `head` into training + CLI

**Files:**
- Modify: `locma/envs/training.py` (`_make_model`, `train_agent`, `train_zoo`)
- Modify: `locma/cli/app.py` (`train`, `train_zoo_cmd`)
- Test: `tests/test_ar_policy.py` (append a wiring test)

**Interfaces:**
- Consumes: `MaskableAutoregressivePolicy` (`locma.envs.ar_policy`).
- Produces: `_make_model(..., head: str = "flat")`; `train_agent(..., head="flat")`; `train_zoo(..., head="flat")`; `--head` option on both CLI commands (validated to `{"flat", "autoreg"}`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ar_policy.py  (append)
def test_make_model_autoreg_uses_ar_policy():
    pytest.importorskip("stable_baselines3")
    from locma.envs.training import _build_env, _make_model

    env = _build_env("random", seed=0, n_envs=1, both_seat=False, obs_mode="flat")
    model = _make_model(env, obs_mode="flat", seed=0, verbose=0, ent_coef=0.02, head="autoreg")
    assert isinstance(model.policy, MaskableAutoregressivePolicy)
    env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra ml --extra dev pytest tests/test_ar_policy.py::test_make_model_autoreg_uses_ar_policy -q`
Expected: FAIL — `_make_model() got an unexpected keyword argument 'head'`

- [ ] **Step 3: Write minimal implementation**

In `locma/envs/training.py`, change `_make_model` to accept `head` and branch. Replace the current `_make_model` signature and body:

```python
def _make_model(
    env,
    *,
    obs_mode: str,
    seed: int,
    verbose: int,
    ent_coef: float,
    learning_rate: float = 3e-4,
    target_kl: float | None = None,
    head: str = "flat",
):
    """Construct a MaskablePPO model, selecting the policy by obs_mode and head.

    head="flat"    -> the default single-softmax head (unchanged baseline).
    head="autoreg" -> MaskableAutoregressivePolicy (flat obs, factored head).
    """
    from sb3_contrib import MaskablePPO  # noqa: PLC0415 — optional [ml] dep

    if head == "autoreg":
        from locma.envs.ar_policy import MaskableAutoregressivePolicy  # noqa: PLC0415

        return MaskablePPO(
            MaskableAutoregressivePolicy,
            env,
            verbose=verbose,
            seed=seed,
            ent_coef=ent_coef,
            learning_rate=learning_rate,
            target_kl=target_kl,
        )

    if obs_mode == "token":
        from locma.envs.extractor import TokenSetExtractor  # noqa: PLC0415

        return MaskablePPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=dict(features_extractor_class=TokenSetExtractor),
            verbose=verbose,
            seed=seed,
            ent_coef=ent_coef,
            learning_rate=learning_rate,
            target_kl=target_kl,
        )
    return MaskablePPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        seed=seed,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        target_kl=target_kl,
    )
```

Add `head: str = "flat"` to `train_agent` and `train_zoo` signatures and forward it to the `_make_model(...)` call(s). In `train_agent`, the `_make_model` call becomes:

```python
    model = _make_model(
        env,
        obs_mode=obs_mode,
        seed=seed,
        verbose=verbose,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        target_kl=target_kl,
        head=head,
    )
```

In `train_zoo`, the `_make_model(...)` call gains `head=head` the same way.

In `locma/cli/app.py`, add to both `train` and `train_zoo_cmd` an option (place it after `obs_mode`):

```python
    head: str = typer.Option(
        "flat", help="action head: 'flat' (single softmax) or 'autoreg' (factored)"
    ),
```

and a validation guard alongside the existing `obs_mode` check:

```python
    if head not in ("flat", "autoreg"):
        raise typer.BadParameter("head must be 'flat' or 'autoreg'")
```

and pass `head=head` into the `train_agent(...)` / `train_zoo(...)` call.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra ml --extra dev pytest tests/test_ar_policy.py::test_make_model_autoreg_uses_ar_policy -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add locma/envs/training.py locma/cli/app.py tests/test_ar_policy.py
git commit -m "feat(ppo-autoreg): --head flag wires autoreg policy through train/train-zoo"
```

---

### Task 6: Smoke train + save/load + legal game

**Files:**
- Test: `tests/test_ar_policy.py` (append an end-to-end smoke test)

**Interfaces:**
- Consumes: `train_agent` (`locma.envs.training`); `MaskablePPOBattlePolicy` (`locma.policies.ppo`); `run_match` (`locma.harness.match`); `make_policy` (`locma.policies.registry`).
- Produces: none (verification only). Confirms `MaskablePPO.load` round-trips the custom policy and it plays only legal actions.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ar_policy.py  (append)
def test_smoke_train_save_load_play(tmp_path):
    pytest.importorskip("stable_baselines3")
    from locma.envs.training import train_agent
    from locma.harness.match import run_match
    from locma.policies.ppo import MaskablePPOBattlePolicy
    from locma.policies.registry import make_policy

    out = str(tmp_path / "ar.zip")
    saved = train_agent(
        "random", steps=400, out=out, seed=0, verbose=0,
        both_seat=False, obs_mode="flat", head="autoreg",
    )
    ppo = MaskablePPOBattlePolicy(saved, name="ar", deterministic=True)
    res = run_match(ppo, make_policy("random"), games=1, seed=123)
    assert res.games == 2  # mirrored pair completed without illegal-action errors
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra ml --extra dev pytest tests/test_ar_policy.py::test_smoke_train_save_load_play -q`
Expected: FAIL initially only if a wiring bug exists; if Task 5 is correct it may already pass. If it fails on load (custom policy not found), that is the bug this task exists to catch — fix per Step 3.

- [ ] **Step 3: Confirm/repair load path**

`MaskablePPO.load(path)` reconstructs the policy from the class reference pickled at save time; `MaskableAutoregressivePolicy` is importable in-package, so no `custom_objects` are needed. `MaskablePPOBattlePolicy._encode_for` selects `encode_battle` because the AR model's `observation_space` is a `Box` (not `Dict`) — correct, unchanged. If the test fails at `.load`, add the class to `custom_objects`:

```python
# only if load fails — in locma/policies/ppo.py _ensure():
from locma.envs.ar_policy import MaskableAutoregressivePolicy  # noqa: PLC0415
self._model = MaskablePPO.load(
    self.model_path, custom_objects={"policy_class": MaskableAutoregressivePolicy}
)
```

Prefer no change if the test passes as-is.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra ml --extra dev pytest tests/test_ar_policy.py::test_smoke_train_save_load_play -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_ar_policy.py locma/policies/ppo.py
git commit -m "test(ppo-autoreg): smoke train + save/load + legal-game round-trip"
```

---

### Task 7: Verdict stats + `ar-eval` CLI

**Files:**
- Create: `locma/harness/ar_study.py`
- Modify: `locma/cli/app.py` (add `ar-eval` command)
- Test: `tests/test_ar_study.py`

**Interfaces:**
- Consumes: `run_match` (`locma.harness.match`); `make_policy` (`locma.policies.registry`); `MaskablePPOBattlePolicy` (`locma.policies.ppo`).
- Produces:
  - `HARD3: tuple[str, ...] = ("scripted", "max-guard", "max-attack")`
  - `hard3_per_seed(model_path: str, seeds: list[int], games_per_seed: int = 2) -> np.ndarray` (avg-hard3 per seed)
  - `paired_bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]` → `(lo, hi, point)`
  - `decide(lo: float, hi: float, point: float, thresh: float = 0.03) -> str`
  - `run_verdict(flat_path, ar_path, seeds, games_per_seed=2) -> dict`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ar_study.py
import numpy as np

from locma.harness.ar_study import decide, paired_bootstrap_ci


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    diff = rng.normal(0.05, 0.02, size=400)
    lo, hi, point = paired_bootstrap_ci(diff, n_boot=2000, seed=0)
    assert lo < point < hi
    assert abs(point - diff.mean()) < 1e-9


def test_decide_verdicts():
    # clear headroom: point >= +0.03 and CI excludes 0
    assert decide(0.02, 0.08, 0.05) == "ar-helps"
    # tight around zero, within +/-0.03 both sides
    assert decide(-0.01, 0.02, 0.005) == "no-help"
    # wide / straddling the band
    assert decide(-0.05, 0.06, 0.01) == "inconclusive"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_ar_study.py -q`
Expected: FAIL — `ModuleNotFoundError: locma.harness.ar_study`

- [ ] **Step 3: Write minimal implementation**

```python
# locma/harness/ar_study.py
"""avg-hard3 evaluation + symmetric paired-bootstrap verdict for the
autoregressive-head study. See docs/ppo-autoreg-action-design.md."""

from __future__ import annotations

import numpy as np

HARD3: tuple[str, ...] = ("scripted", "max-guard", "max-attack")


def hard3_per_seed(model_path: str, seeds, games_per_seed: int = 2) -> np.ndarray:
    """avg-hard3 for one model, one value per eval seed (paired across models
    when the same seeds are used). Each seed plays `games_per_seed` mirrored
    matches against each of the three hard opponents; the per-seed value is the
    mean win-rate over the three."""
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.ppo import MaskablePPOBattlePolicy  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    ppo = MaskablePPOBattlePolicy(model_path, name="ppo", deterministic=True)
    out = np.zeros(len(seeds), dtype=np.float64)
    for i, s in enumerate(seeds):
        rates = []
        for opp in HARD3:
            res = run_match(ppo, make_policy(opp), games=games_per_seed, seed=int(s))
            rates.append(res.win_rate_a)
        out[i] = float(np.mean(rates))
    return out


def paired_bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of the mean paired difference. Returns (lo, hi, point)."""
    diff = np.asarray(diff, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        means[b] = diff[rng.integers(0, n, size=n)].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi, float(diff.mean())


def decide(lo: float, hi: float, point: float, thresh: float = 0.03) -> str:
    """Symmetric verdict against +/- thresh."""
    if point >= thresh and lo > 0:
        return "ar-helps"
    if -thresh <= lo and hi <= thresh:
        return "no-help"
    return "inconclusive"


def run_verdict(flat_path: str, ar_path: str, seeds, games_per_seed: int = 2) -> dict:
    """Full paired verdict: per-seed avg-hard3 for both models, then bootstrap."""
    flat = hard3_per_seed(flat_path, seeds, games_per_seed)
    ar = hard3_per_seed(ar_path, seeds, games_per_seed)
    diff = ar - flat
    lo, hi, point = paired_bootstrap_ci(diff)
    return {
        "flat_mean": float(flat.mean()),
        "ar_mean": float(ar.mean()),
        "delta": point,
        "ci": (lo, hi),
        "verdict": decide(lo, hi, point),
        "n_seeds": len(seeds),
    }
```

Add the CLI command to `locma/cli/app.py`:

```python
@app.command("ar-eval")
def ar_eval_cmd(
    flat: str = typer.Option(..., help="path to the flat-head baseline model .zip"),
    ar: str = typer.Option(..., help="path to the autoregressive-head model .zip"),
    seeds: int = typer.Option(200, help="number of held-out eval seeds"),
    base_seed: int = typer.Option(1_000_000, help="first eval seed (held-out range)"),
    games_per_seed: int = typer.Option(2, help="mirrored matches per opponent per seed"),
):
    """Paired avg-hard3 verdict: does the autoregressive head beat the flat head?"""
    from locma.harness.ar_study import run_verdict  # noqa: PLC0415

    seed_list = [base_seed + i for i in range(seeds)]
    r = run_verdict(flat, ar, seed_list, games_per_seed)
    table = Table(title="autoregressive-head verdict (avg-hard3)")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("flat mean", f"{r['flat_mean']:.4f}")
    table.add_row("ar mean", f"{r['ar_mean']:.4f}")
    table.add_row("delta (ar - flat)", f"{r['delta']:+.4f}")
    table.add_row("95% CI", f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
    table.add_row("n seeds", str(r["n_seeds"]))
    table.add_row("verdict", r["verdict"])
    console.print(table)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_ar_study.py -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add locma/harness/ar_study.py locma/cli/app.py tests/test_ar_study.py
git commit -m "feat(ppo-autoreg): avg-hard3 paired-bootstrap verdict + ar-eval CLI"
```

---

### Task 8: Telemetry callback (per-head entropy + periodic avg-hard3)

**Files:**
- Create: `locma/envs/ar_callbacks.py`
- Modify: `locma/envs/ar_distribution.py` (record last-batch per-head entropies) and `locma/envs/ar_policy.py` (stash them)
- Test: `tests/test_ar_policy.py` (append a callback smoke test)

**Interfaces:**
- Consumes: `BaseCallback` (`stable_baselines3.common.callbacks`); `hard3_per_seed` (`locma.harness.ar_study`).
- Produces: `class ARTelemetryCallback(BaseCallback)` with `__init__(self, eval_freq: int = 50_000, seeds: int = 40, base_seed: int = 2_000_000, games_per_seed: int = 1)`. Records `eval/avg_hard3` and `ar_entropy/{type,source,target}` to the SB3 logger.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ar_policy.py  (append)
def test_telemetry_callback_runs(tmp_path):
    pytest.importorskip("stable_baselines3")
    from locma.envs.ar_callbacks import ARTelemetryCallback
    from locma.envs.training import train_agent

    out = str(tmp_path / "ar.zip")
    # eval_freq below the step budget so the callback fires at least once
    train_agent(
        "random", steps=600, out=out, seed=0, verbose=0,
        both_seat=False, obs_mode="flat", head="autoreg",
        callback=ARTelemetryCallback(eval_freq=256, seeds=3, games_per_seed=1),
    )
    assert (tmp_path / "ar.zip").exists()
```

Also add a `callback=None` parameter to `train_agent` that is passed to `model.learn(..., callback=callback)` (both the checkpoint and single-run branches).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra ml --extra dev pytest tests/test_ar_policy.py::test_telemetry_callback_runs -q`
Expected: FAIL — `ModuleNotFoundError: locma.envs.ar_callbacks` (and `train_agent` has no `callback` kwarg)

- [ ] **Step 3: Write minimal implementation**

In `locma/envs/ar_distribution.py`, make `ar_evaluate` stash the mean per-head entropies on the heads object for telemetry (append at the end of `ar_evaluate`, before `return`):

```python
    heads.last_head_entropy = (
        float(type_dist.entropy().mean()),
        float(src_dist.entropy().mean()),
        float(tgt_dist.entropy().mean()),
    )
```

and initialize `self.last_head_entropy = (0.0, 0.0, 0.0)` at the end of `ARHeads.__init__`.

Create `locma/envs/ar_callbacks.py`:

```python
# locma/envs/ar_callbacks.py
"""Training telemetry for the autoregressive head: periodic avg-hard3 eval
and per-head entropy logging. Requires the [ml] extra."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class ARTelemetryCallback(BaseCallback):
    """Every `eval_freq` steps: save the model to a temp path, run a quick
    avg-hard3, and record it plus the last per-head entropies to the logger."""

    def __init__(
        self,
        eval_freq: int = 50_000,
        seeds: int = 40,
        base_seed: int = 2_000_000,
        games_per_seed: int = 1,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.seeds = seeds
        self.base_seed = base_seed
        self.games_per_seed = games_per_seed
        self._last = 0

    def _record_entropy(self) -> None:
        heads = getattr(self.model.policy, "ar_heads", None)
        if heads is not None and hasattr(heads, "last_head_entropy"):
            et, es, etg = heads.last_head_entropy
            self.logger.record("ar_entropy/type", et)
            self.logger.record("ar_entropy/source", es)
            self.logger.record("ar_entropy/target", etg)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last < self.eval_freq:
            return True
        self._last = self.num_timesteps
        self._record_entropy()
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        from locma.harness.ar_study import hard3_per_seed  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "snap.zip")
            self.model.save(p)
            seeds = [self.base_seed + i for i in range(self.seeds)]
            avg = float(hard3_per_seed(p, seeds, self.games_per_seed).mean())
        self.logger.record("eval/avg_hard3", avg)
        if self.verbose:
            print(f"[ar-telemetry] step={self.num_timesteps} avg_hard3={avg:.4f}")
        return True
```

In `locma/envs/training.py`, add `callback=None` to `train_agent(...)` and pass it into both `model.learn(...)` calls:

```python
    model.learn(total_timesteps=steps, callback=callback)
```

and in the checkpoint loop:

```python
            model.learn(
                total_timesteps=mark - prev,
                reset_num_timesteps=(i == 0),
                callback=callback,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra ml --extra dev pytest tests/test_ar_policy.py::test_telemetry_callback_runs -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add locma/envs/ar_callbacks.py locma/envs/ar_distribution.py locma/envs/ar_policy.py locma/envs/training.py tests/test_ar_policy.py
git commit -m "feat(ppo-autoreg): ARTelemetryCallback (avg-hard3 curve + per-head entropy)"
```

---

### Task 9: Full-suite green + lint gate

**Files:** none (verification).

- [ ] **Step 1: Run the CI gate**

Run:
```bash
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra ml --extra dev pytest -q
```
Expected: ruff clean; all tests pass (new AR tests + the pre-existing suite).

- [ ] **Step 2: Fix any lint/format issues, re-stage, and commit if anything changed**

```bash
git add -A
git commit -m "chore(ppo-autoreg): lint/format cleanup"
```

(If nothing changed, skip the commit.)

---

## Execution Runbook (inline, CPU)

> These are the training/eval runs, not tooling. They are long on CPU; run them
> after Tasks 1–9 are green. Torch is CPU-only here — the net is tiny, so the
> cost is the Python game-sim per env step (same as the flat baseline). Use
> background runs for the full-budget steps. `avg-hard3` eval seeds live in the
> held-out `1_000_000+` range; telemetry uses a disjoint `2_000_000+` range.

### R0 — Pipeline smoke (both models, tiny budget)

Verifies the end-to-end study wiring and that `ar-eval` produces a verdict object before committing hours of CPU. **Not** a real result.

```bash
uv run --extra ml locma train-zoo --steps-per-opponent 10000 --head flat    --out flat-smoke.zip --seed 0
uv run --extra ml locma train-zoo --steps-per-opponent 10000 --head autoreg --out ar-smoke.zip   --seed 0
uv run --extra ml locma ar-eval --flat flat-smoke.zip --ar ar-smoke.zip --seeds 40 --games-per-seed 1
```
Expected: a verdict table prints (verdict will likely be `inconclusive`/`no-help` at 40k steps — that's fine; this only proves the pipeline).

### R1 — Baseline B_flat (full budget)

```bash
uv run --extra ml locma train-zoo --steps-per-opponent 200000 --head flat --out flat-800k-s0.zip --seed 0
```
Run in the background; it is the canonical flat recipe (LR 3e-4, ent_coef 0.02, both-seat, zoo curriculum, 800k total).

### R2 — Candidate B_ar (full budget, identical recipe)

```bash
uv run --extra ml locma train-zoo --steps-per-opponent 200000 --head autoreg --out ar-800k-s0.zip --seed 0
```
Identical to R1 except `--head autoreg`.

### R3 — Verdict

```bash
uv run --extra ml locma ar-eval --flat flat-800k-s0.zip --ar ar-800k-s0.zip --seeds 300 --games-per-seed 2
```
Read `delta`, `95% CI`, and `verdict`:
- `ar-helps` → the factored head is a real lever past the flat softmax.
- `no-help` → factoring does not move the reactive ceiling (ceiling-confirmed for the head).
- `inconclusive` → report the delta + CI and stop; do not chase budget.

### R4 — Record the result

Append a dated entry to `docs/worklog.md`: the recipe (identical except head), the `delta`/CI/verdict, and the one-line takeaway. Commit:

```bash
git add docs/worklog.md
git commit -m "docs(worklog): autoregressive-head verdict vs flat baseline"
```

---

## Self-Review

**Spec coverage:**
- Design §2 (factorization) → Task 1. Design §3 (masks-from-flat, log-prob = Σ, entropy = Σ, non-applicable→0) → Tasks 1–3. Design §3 three code paths (rollout/update/predict) → Task 4. Design §4 (Discrete(155) retained, custom policy, unchanged critic, `head` selector) → Tasks 4–5. Design §5 (baseline + candidate recipe, `--head`) → Tasks 5, R1/R2. Design §6 (avg-hard3, paired bootstrap, ±0.03 symmetric verdict) → Task 7, R3. Design §7 (`observe effectiveness` telemetry) → Task 8. Design §7 module layout → File Structure table. Design testing list items 1–5 → Tasks 1,2,3,4,6. Design §"not NetOracle" → out of scope, untouched. All covered.

**Placeholder scan:** no TBD/TODO; every code step shows complete code; the only conditional ("only if load fails") in Task 6 gives the exact fallback code.

**Type consistency:** `hard3_per_seed`, `paired_bootstrap_ci(→(lo,hi,point))`, `decide(lo,hi,point)`, `run_verdict` signatures are consistent between Task 7's interface block, implementation, tests, and the `ar-eval` command. `ARHeads`/`ar_sample`/`ar_evaluate` signatures match between Tasks 3, 4, and 8. `_make_model(..., head=...)` matches between Task 5 and the training/CLI edits. `decode`/`encode`/`SEG`/`factor_masks` names match between Tasks 1 and 2. `factor_grids`/`decode_batch`/`encode_batch` match between Tasks 2, 3.

**Known risk flagged in-plan:** `MaskableCategorical` mask convention is True=valid (matches `action_mask`); Task 3's `grids.any(dim=(2,3))` has a torch-version fallback note. Task 6 explicitly exists to catch a custom-policy `load` failure and gives the fix.
