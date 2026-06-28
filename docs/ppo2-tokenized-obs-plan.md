# PPO2 — Tokenized obs + self-attention extractor: implementation plan

> Rough task list mapped onto the spec (`docs/ppo2-tokenized-obs-design.md`).
> Each task is TDD + a commit. Tasks 1–6 are code (subagent-friendly); 7–8 are
> compute/analysis runs (orchestrator, background). Branch: `feat/ppo2-tokenized-obs`.

**Global constraints (from spec):** action masking untouched; flat path byte-identical
(stays the A/B control); no hidden info in obs; `[ml]` imports stay lazy so the package
imports without torch; feature branch only, push HTTPS.

---

### Task 1 — Token observation encoder · spec §1
**Files:** modify `locma/envs/encode.py` (additive); test `tests/test_encode_tokens.py`
- Add constants: `MAX_TOKENS=20`, `TOKEN_FEATS=17`, `NUM_CARDS=160`, card-id vocab 161
  (PAD=0), scalar count `N_TACTICAL=13`.
- `encode_battle_tokens(view) -> dict` → `{tokens(20,17), card_ids(20,), token_mask(20,),
  scalars(13,)}`, all float32; slot order hand[0:8]/my_board[8:14]/op_board[14:20];
  pads = zero row, `card_id=0`, `mask=0`. Scalars per spec §1 list.
- `token_obs_space() -> spaces.Dict` — **lazy** `from gymnasium import spaces` inside the
  fn (keep `encode.py` import gym-free, like the rest of the lazy ml stack).
- **Do not touch** `sem_index` / `action_mask` / `index_to_action` / `encode_battle`.
- **Tests:** shapes + dtype; pad rows zero with `card_id==0` & `mask==0`; `token_mask`
  matches real counts; hand-built `BattleView` asserts guard count, `reachable_face_damage==0`
  when a Guard is up, `lethal_available` boundary at `==op_health`.

### Task 2 — Self-attention feature extractor · spec §2
**Files:** create `locma/envs/extractor.py`; test `tests/test_extractor.py`
- `TokenSetExtractor(BaseFeaturesExtractor)` (`stable_baselines3.common.torch_layers`):
  card-id `nn.Embedding(161, id_dim, padding_idx=0)` ⊕ token feats → `Linear→d_model`,
  prepend learned CLS, `TransformerEncoder(n_layers)` with `src_key_padding_mask` from
  `token_mask`, CLS out ⊕ `scalar_mlp(scalars)` → `head→features_dim`. Defaults
  d_model=64, n_heads=4, n_layers=2, id_dim=16, dropout=0.1, features_dim=128, `pool="cls"`.
  Include the C-fallback knob (`n_layers=1` / `pool="attn"`).
- **Tests** (`importorskip("torch")`): forward on a random batch → `(B, features_dim)`,
  finite, grads flow; **permutation/padding invariance** — shuffle real tokens (re-pad) →
  CLS output unchanged.

### Task 3 — BattleEnv obs_mode · spec §1 (env wiring)
**Files:** modify `locma/envs/battle_env.py`; test `tests/test_env_token.py`
- `__init__(..., obs_mode="flat")`; `"flat"` keeps `Box(OBS_SIZE)`+`encode_battle`;
  `"token"` uses `token_obs_space()`+`encode_battle_tokens`. `reset`/`step` pick encoder;
  terminal step → zeroed dict in token mode. `action_masks()` unchanged.
- **Tests** (`importorskip("gymnasium")`): token reset/step obs ∈ `observation_space`;
  terminal returns zero dict; mask still `.any()`; flat default unchanged.

### Task 4 — Training wiring + MultiInputPolicy branch · spec §2
**Files:** modify `locma/envs/training.py`; test `tests/test_training_token.py`
- Thread `obs_mode` through `_make_battle_env` → `_build_env` → `train_agent` / `train_zoo`
  (keyword, default `"flat"`). DRY a `_make_model(env, obs_mode, ...)`: `"flat"`→`"MlpPolicy"`;
  `"token"`→`"MultiInputPolicy"` + `policy_kwargs(features_extractor_class=TokenSetExtractor,
  features_extractor_kwargs=...)`.
- **Spec checkpoint:** the smoke test below verifies `MaskableMultiInputActorCriticPolicy`
  resolves under `"MultiInputPolicy"` for `MaskablePPO`; if not, add a thin maskable Dict
  policy subclass (flag first).
- **Tests** (`importorskip("sb3_contrib")`): `train_agent("random", steps=512,
  obs_mode="token", out=tmp)` runs + saves a file (tiny steps for speed).

### Task 5 — Eval policy auto-detects token models · spec §2/§3
**Files:** modify `locma/policies/ppo.py`; test extend `tests/test_ppo.py`
- In `battle_action`, pick the encoder by the loaded model's obs space: `spaces.Dict` →
  `encode_battle_tokens`, else `encode_battle`. So `ppo:runs/ppo2.zip` evaluates a token
  model with no registry change. Factor an `_encode_for(model, view)` helper.
- **Tests:** `_encode_for` returns a dict for a stub with `Dict` obs space, an ndarray for
  a `Box` stub (no model load needed).

### Task 6 — CLI `--obs-mode` flag · spec §3 (run the experiment)
**Files:** modify `locma/cli/app.py`; test extend `tests/test_cli.py`
- Add `obs_mode: str = "flat"` to `train` and `train-zoo`; validate
  `obs_mode in {"flat","token"}` else `typer.BadParameter`; pass through.
- **Tests:** invalid `--obs-mode bogus` → BadParameter (cheap, no training).

### Task 7 — Pilot (the gate) · spec §3 [orchestrator, background]
- Train flat vs token ~100k steps vs `max-attack`, matched seed/budget, `both_seat`.
  Capture fps/wall-clock; eval both vs scripted/max-guard/max-attack (`locma tournament`).
- **Gate:** pipeline sound AND token not-worse than flat → proceed to Task 8; else stop and
  write the null into `docs/worklog.md`.

### Task 8 — Full A/B + docs · spec §3 [orchestrator, background, conditional]
- Zoo curriculum, both arms, identical budget/seeds, 2 seeds. Headline avg-hard3 +
  token-vs-flat head-to-head. Write results + verdict to `docs/baseline.md` + `docs/worklog.md`.
  Decision: keep PPO2 if it clears flat beyond seed variance; else honest null.

---

**Spec coverage check:** §1 → T1, T3; §2 → T2, T4, T5; §3 tests → T1–T6; §3 pilot/A&B →
T7, T8; constraints (mask untouched, flat byte-identical, lazy ml, branch) → enforced in
T1/T3/T4 + global constraints. No spec requirement is unmapped.
