# Distill mcts → PPO2 (token obs): implementation plan

> Rough task list mapped onto the spec (`docs/distill-mcts-ppo2-design.md`).
> Each task is TDD + a commit. D1–D3 are code (subagent-friendly); D4–D5 are
> compute/analysis runs (orchestrator, background). Branch: `feat/distill-mcts-ppo2`.

**Global constraints (from spec):** teacher decisions are observation-independent
(recorded `action`/`mask` identical to flat — only the stored obs changes; 155-slot
action space + `sem_index`/`action_mask` untouched); the flat practicum/distill path
stays intact (`obs_mode="flat"` default everywhere); `[ml]` imports stay lazy; reuse
`encode_battle_tokens`/`token_obs_space` (encode.py) + `TokenSetExtractor`
(extractor.py); feature branch only, push HTTPS.

---

### Task D1 — Token-mode practicum recording · spec §1
**Files:** modify `locma/envs/practicum.py`; test `tests/test_practicum_token.py`
- `_Collector(teacher_seat, obs_mode="flat")`: in `"token"` mode capture
  `encode_battle_tokens(view)` (dict) per decision instead of `encode_battle(view)`;
  keep the same drop rules + `sem_index`/`action_mask` capture. Store the dicts in a
  list.
- `record_practicum(..., obs_mode="flat")`: thread to the collector. In `"token"`
  mode write **four arrays** to the npz — `obs_tokens (n,20,17)`, `obs_card_ids (n,20)`,
  `obs_token_mask (n,20)`, `obs_scalars (n,13)` (all float32) — instead of `obs`. Keep
  `action`/`mask`/`winner`/`seat`/`opponent_id`/`game_id` unchanged.
- Manifest: add `obs_mode` and, for token, `max_tokens`/`token_feats`/`n_tactical` dims
  (import the constants from encode.py); keep `action_size`. Flat manifest unchanged
  (keeps `obs_size`).
- **Tests:** short token run (or monkeypatched collector like `test_practicum.py`):
  the four arrays have correct shapes/dtypes; manifest `obs_mode=="token"` + dims;
  `action`/`mask` identical to a matching flat run; bogus obs_mode → ValueError.

### Task D2 — Token-mode distillation · spec §2
**Files:** modify `locma/envs/distill.py`; test `tests/test_distill_token.py`
- `load_practicum(path)`: branch on manifest `obs_mode`. Token → load the four arrays
  + validate `max_tokens/token_feats/n_tactical/action_size` against encode.py (mirror
  the flat guard; reject mismatch loudly). Flat → unchanged. Return arrays dict.
- `behavior_clone(..., obs_mode="flat")`: token branch builds the throwaway env via
  `_make_battle_env("random", seed, obs_mode="token")` and
  `MaskablePPO("MultiInputPolicy", env, policy_kwargs=dict(features_extractor_class=
  TokenSetExtractor), seed=…)`; the masked-CE loop indexes a **dict** of tensors
  (`tokens/card_ids/token_mask/scalars`) into `policy.evaluate_actions(obs_dict, act,
  action_masks=mask)`; same loss + game-level split; `val_agreement` via
  `model.predict(obs_dict_val, action_masks=…, deterministic=True)`. Flat branch
  unchanged.
- **Tests** (`importorskip("sb3_contrib")`): tiny token practicum → `behavior_clone(
  obs_mode="token", epochs=1)` builds a `MultiInputPolicy`, saves a loadable Dict-obs
  model, returns finite `val_agreement`; `load_practicum` rejects a token-dims mismatch.

### Task D3 — CLI `--obs-mode` on record-practicum + distill · spec §1/§2
**Files:** modify `locma/cli/app.py`; test extend `tests/test_cli.py`
- Add `obs_mode: str = typer.Option("flat", ...)` to `record-practicum` and `distill`;
  validate `in {"flat","token"}` (BadParameter) BEFORE the lazy import; pass through.
- **Tests:** `record-practicum --obs-mode bogus` and `distill --obs-mode bogus` →
  nonzero exit with `"obs_mode"` in output (cheap; no real run).

### Task D4 — Record + distill + eval experiment · spec §3 [orchestrator, background]
- Record `runs/practicum-token.npz`: `mcts:100`, default 5 opponents, ~156 games/opp ×
  2 seats (~45k ex, ~40 min concurrent). Distill (10 epochs, lr 3e-4, batch 256,
  val_frac 0.1) → `runs/distilled-token.zip`. (Optional 40-epoch overfit check.)
- Eval: `val_agreement` (vs PR#18 ~0.25) + distilled avg-hard3 (vs PR#18 ~0.25 / teacher
  mcts ~0.73, via `tournament ppo:runs/distilled-token.zip scripted max-guard max-attack`).

### Task D5 — Document verdict · spec §3 [orchestrator]
- Write to `docs/baseline.md` + `docs/worklog.md`: agreement + avg-hard3 vs PR#18;
  interpretation (agreement >0.25 ⇒ obs was part of the ceiling; ~0.25 ⇒ binding ceiling
  is the info gap / cheater). Note the cross-run-control caveat (token-only choice).

---

**Spec coverage check:** §1 → D1, D3; §2 → D2, D3; §3 → D4, D5; §4 tests → D1–D3;
constraints (flat intact, decisions obs-independent, lazy ml, reuse substrate, branch)
→ enforced in D1/D2 + global constraints. No spec requirement unmapped.
