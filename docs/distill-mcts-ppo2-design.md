# Distill mcts plays onto PPO2 (tokenized obs) — PR #18 redo

**Date:** 2026-06-26
**Status:** approved design, entering implementation
**Branch:** `feat/distill-mcts-ppo2` (stacked on `feat/ppo2-tokenized-obs`)

## Motivation

PR #18 behavior-cloned `mcts:100` plays onto the **flat-obs** PPO and plateaued at
**~0.25 top-1 agreement**, inheriting none of the teacher's edge (distilled avg-hard3
≈ a from-scratch-PPO profile, ~0.25 vs the hard baselines; teacher mcts ≈ 0.73). The
conclusion was *"the ceiling is the observation."* That claim tangles two distinct
ceilings:

- **(a) obs ceiling** — the flat 308-vector can't represent the visible board well
  enough for the student to imitate the teacher. *PPO2's tokenized obs could fix this.*
- **(b) info ceiling** — `mcts` is a **perfect-foresight cheater** (it clones the real
  `GameState`, seeing the opponent's hand and both decks' future draws), so it decides
  on hidden information the student can never observe. *No observation fixes this.*

This experiment re-runs the PR #18 distillation with the **richer PPO2 observation +
self-attention net** to test which ceiling was binding: if agreement rises clearly
above 0.25, the flat obs was (part of) the limit; if it stays ~0.25, the binding limit
is the info gap, not the encoding.

## Scope

**In scope:** add a token-observation mode to the practicum recorder and the
distiller, then record an `mcts:100` token practicum and behavior-clone it into a
PPO2 (token) net, comparing top-1 agreement + avg-hard3 to PR #18's flat references.

**Out of scope (YAGNI):** RL fine-tuning after BC; `dmcts`/`azlite` teachers; a
dual-obs matched control (user chose token-only vs PR #18's recorded 0.25); changes to
the action space, the teacher, or the engine.

## Constraints

- **Teacher decisions are observation-independent.** The recorded `action` (semantic
  index) and `mask` are identical to the flat path; only the stored *observation*
  changes. The 155-slot action space and `sem_index`/`action_mask` are untouched.
- **The flat path stays intact** (`obs_mode="flat"` default everywhere): existing
  `record_practicum` / `behavior_clone` behaviour and the flat npz layout are
  unchanged, so prior practicums and PR #18 reproduce.
- **`[ml]` imports stay lazy** (torch/sb3 imported inside functions, as today).
- **Reuse the PPO2 substrate:** `encode_battle_tokens` / `token_obs_space`
  (`encode.py`) and `TokenSetExtractor` (`extractor.py`) from `feat/ppo2-tokenized-obs`.
- **Feature branch only; push over HTTPS** via the gh credential helper.

## Section 1 — Token-mode practicum recording (`locma/envs/practicum.py`)

- `_Collector(teacher_seat, obs_mode="flat")`: in `"token"` mode capture
  `encode_battle_tokens(view)` (a dict) per teacher decision instead of
  `encode_battle(view)`; `sem_index`/`action_mask` capture is unchanged. Keep the same
  drop rules (forced decisions, `sem_index is None`, overflow).
- `record_practicum(..., obs_mode="flat")`: thread `obs_mode` to the collector. In
  `"token"` mode, write the observation as **four arrays** instead of `obs`:
  - `obs_tokens` `(n, 20, 17)` float32
  - `obs_card_ids` `(n, 20)` float32
  - `obs_token_mask` `(n, 20)` float32
  - `obs_scalars` `(n, 13)` float32
  Plus the unchanged `action`/`mask`/`winner`/`seat`/`opponent_id`/`game_id`.
- Manifest: add `obs_mode` (`"flat"`/`"token"`) and, for token, the token dims
  (`max_tokens`, `token_feats`, `n_tactical`) so the distiller can reject a stale set.
  Keep `action_size`. (Flat manifests keep `obs_size`.)
- CLI `record-practicum`: add `--obs-mode flat|token` (default `flat`), validated
  (`BadParameter` otherwise).

## Section 2 — Token-mode distillation (`locma/envs/distill.py`)

- `load_practicum(path)`: read `obs_mode` from the manifest. For `"token"`, load the
  four token arrays and validate the token dims + `action_size` against `encode.py`
  (mirror the existing flat guard — reject a mismatch loudly); for `"flat"`, behave as
  today. Return the arrays dict (`obs` for flat, or the four token keys).
- `behavior_clone(..., obs_mode="flat")`:
  - `"flat"` → unchanged (`MaskablePPO("MlpPolicy", ...)`).
  - `"token"` → build a throwaway token env (`_make_battle_env("random", seed,
    obs_mode="token")`) and `MaskablePPO("MultiInputPolicy", env,
    policy_kwargs=dict(features_extractor_class=TokenSetExtractor), seed=…)`. Feed the
    masked-CE loop a **dict** of batched tensors (`tokens/card_ids/token_mask/scalars`,
    each indexed by the shuffled `sel`) to `model.policy.evaluate_actions(obs_dict, act,
    action_masks=mask)`. Same loss, same game-level train/val split. Top-1
    `val_agreement` via `model.predict(obs_dict_val, action_masks=…, deterministic=True)`.
  - Optional `lr` is already a param; BC is supervised (not the PPO loop) so the
    default 3e-4 is the starting point.
- CLI `distill`: add `--obs-mode flat|token` (default `flat`), validated.

## Section 3 — Experiment protocol (orchestrator, after the code lands)

1. **Record** `runs/practicum-token.npz`: teacher `mcts:100`, default opponents
   (random/scripted/greedy/max-guard/max-attack), ~156 games/opp × 2 seats (~45k
   examples; ~40 min concurrent — cost is the search).
2. **Distill** at PR #18's BC config (10 epochs, lr 3e-4, batch 256, val_frac 0.1) →
   `runs/distilled-token.zip`. (Optionally a 40-epoch run to find the overfit point.)
3. **Measure** vs PR #18's flat references:
   - top-1 `val_agreement` vs **~0.25** (held-out games).
   - distilled net's **avg-hard3** (mean win rate vs scripted/max-guard/max-attack) vs
     PR #18's distilled ~0.25 and the teacher mcts ~0.73.
4. **Interpretation:** agreement clearly **> 0.25** ⇒ flat obs was part of the ceiling
   (richer obs lifts imitation); agreement **still ~0.25** ⇒ the binding ceiling is the
   **info gap** (cheater), not the observation. Write the verdict to
   `docs/baseline.md` + `docs/worklog.md`. Caveat noted: the 0.25 control is from PR
   #18's separate run (token-only choice), so the comparison is cross-run, not a
   same-practicum matched control.

## Section 4 — Testing (cheap; `importorskip`-gated, skipped in dev-only CI)

- `record_practicum(obs_mode="token")` short run: writes the four token arrays with
  correct shapes/dtypes; manifest has `obs_mode="token"` + token dims; `action`/`mask`
  unchanged vs a matching flat run.
- `load_practicum`: accepts a token manifest and returns the four arrays; **rejects** a
  token-dims mismatch loudly (mirrors the flat guard).
- `behavior_clone(obs_mode="token")` smoke: a tiny token practicum → BC builds a
  `MultiInputPolicy`+`TokenSetExtractor`, runs ≥1 epoch, saves a loadable Dict-obs
  model, and returns a finite `val_agreement`.
- CLI: `record-practicum`/`distill` reject a bogus `--obs-mode` (fast-fail, no [ml]).

## Files touched

- `locma/envs/practicum.py` — token recording (collector + npz arrays + manifest).
- `locma/envs/distill.py` — token load guard + `MultiInputPolicy`/`TokenSetExtractor` BC.
- `locma/cli/app.py` — `--obs-mode` on `record-practicum` and `distill`.
- `tests/` — token recording, token load guard, token BC smoke, CLI validation.
- `docs/baseline.md`, `docs/worklog.md` — results after the experiment.

## Success criteria

1. New tests pass; flat practicum/distill path + existing tests reproduce unchanged.
2. The experiment runs end-to-end: a token practicum records, distills, and reports
   `val_agreement` + avg-hard3.
3. A documented verdict on the obs-ceiling-vs-info-ceiling question (lift or null),
   written to `docs/`.
