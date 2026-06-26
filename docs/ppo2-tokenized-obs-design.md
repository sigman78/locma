# PPO2 — Tokenized observation + self-attention feature extractor

**Date:** 2026-06-26
**Status:** approved design, entering implementation
**Branch:** `feat/ppo2-tokenized-obs`

## Motivation

The flat-obs MaskablePPO ("PPO") has hit a ceiling that warm-starting and curriculum
could not move (see `docs/ppo-review.md`, `docs/searchers-fiasco.md`). The two open
levers are (A) a richer board *encoding* and (B) search at play time. This spec
realizes a focused slice of **lever A**: replace the flat 308-d fixed-slot observation
with a **tokenized entity encoding** (per-card tokens + a learned card-id embedding +
computed tactical scalars) consumed by a **self-attention feature extractor**, and
A/B it against the flat-obs baseline. The result is "PPO2".

### Why this is not the §3.4 null

`ppo-review.md` §3.4 showed that adding *more flat features* did not help (rich-308 ≈
lean-146). That A/B only varied the *amount* of the same flat scalar information. PPO2
adds two **different kinds** of information the flat obs cannot express:

1. **Relations** — a self-attention encoder lets cards attend to each other, so the net
   can represent attacker↔blocker/trade structure directly instead of re-inferring it
   from raw slot positions every forward pass.
2. **Card identity** — `CardView.card_id` (1..160), which the flat encoder throws away
   entirely, enters as a learned embedding.
3. **Computed tactical scalars** — cheap 1-ply facts (guard count, reachable face
   damage, lethal-available) baked into the observation.

The explicit attacker×target **relational/trade matrix** (the most ambitious §8.4A item)
is **out of scope** for PPO2 — self-attention provides relations implicitly; the matrix
is a candidate for a later PPO2.1 only if PPO2 shows life.

## Scope

**In scope:** token+scalar observation encoder (additive), a self-attention
`BaseFeaturesExtractor`, `MultiInputPolicy` wiring in training, unit tests, a
pilot-gated A/B vs the flat-obs control.

**Out of scope:** the relational/trade matrix; any change to the 155-slot semantic
action space or action masking; AlphaZero-lite / search-in-the-loop (lever B);
draft-phase encoding.

## Non-negotiable constraints

- **Action masking is untouched.** `sem_index`, `action_mask`, `index_to_action` in
  `encode.py` and `BattleEnv.action_masks()` stay byte-identical. The mask is a
  function of the legal action list, independent of the observation.
- **The flat path stays byte-identical.** `encode_battle` / `OBS_SIZE` and the
  `"MlpPolicy"` training branch are unchanged, so the existing baseline remains
  reproducible and serves as the A/B control.
- **No hidden information enters the obs.** Only public + own-known fields from
  `BattleView` are encoded (both boards, own hand, public counts). `card_id` is public.
- **Feature branch only.** Never commit to `main`; push over HTTPS via the gh
  credential helper.

## Section 1 — Observation encoding (data layer, `locma/envs/encode.py`)

All additions are additive; nothing existing is modified.

### Constants

- `MAX_TOKENS = MAX_HAND + MAX_BOARD + MAX_BOARD = 20` (8 hand + 6 my-board + 6 op-board,
  fixed slots, padded).
- `TOKEN_FEATS = 17` numeric features per token (card identity is a separate int stream):
  - zone one-hot (3): in_hand / my_board / op_board
  - type one-hot (4): creature / green item / red item / blue item
  - cost, attack, defense (3) — raw; the extractor normalizes
  - ability bits (6): B C D G L W
  - readiness (1): `on_board and can_attack and not has_attacked`
- `NUM_CARDS = 160`; card-id embedding vocab `161` with index `0` reserved for PAD
  (real ids are 1..160, confirmed via `load_cards()`).
- `N_TACTICAL` scalars (see below), `S = N_TACTICAL`.

### Observation: a `spaces.Dict`

| key          | shape       | dtype           | meaning                                             |
|--------------|-------------|-----------------|-----------------------------------------------------|
| `tokens`     | `(20, 17)`  | float32         | per-card numeric features; zero rows for pads       |
| `card_ids`   | `(20,)`     | float32→long    | `CardView.card_id` (1..160); `0` for pads           |
| `token_mask` | `(20,)`     | float32         | `1` real / `0` pad → attention `key_padding_mask`   |
| `scalars`    | `(S,)`      | float32         | tactical scalars below                              |

Token slot order is fixed: indices `0..7` hand, `8..13` my board, `14..19` op board.
Padding fills unused slots with a zero feature row, `card_id == 0`, `token_mask == 0`.

### Tactical scalars (`S = 13`)

All public / own-known, 1-ply, computed from the `BattleView`:

1. `turn`
2. `me_health`
3. `op_health`
4. `me_mana`
5. `summonable_count` — hand cards with `cost <= me_mana`
6. `op_hand_count`
7. `my_board_count`
8. `op_board_count`
9. `opp_guard_count` — op-board creatures with the `G` ability
10. `my_total_attack` — Σ attack over my board
11. `my_total_defense` — Σ defense over my board
12. `reachable_face_damage` — `0` if `opp_guard_count > 0` else Σ attack of my ready
    attackers (`can_attack and not has_attacked`)
13. `lethal_available` — `1.0` if `reachable_face_damage >= op_health` else `0.0`

The "own exposed-to-lethal" flag is **deliberately dropped**: it depends on opponent
readiness, which the engine only refreshes on the opponent's `start_turn`, so the
signal is noisy/stale (same caveat the flat encoder notes for op-board readiness).

### New functions / env wiring

- `encode_battle_tokens(view) -> dict[str, np.ndarray]` — pure function of `BattleView`
  (matches the `encode_battle(view)` signature style; does **not** need the legal list,
  since every scalar is computable from the view).
- `token_obs_space() -> spaces.Dict` — the obs space above.
- `BattleEnv.__init__(..., obs_mode: str = "flat")` — `"flat"` (default) keeps today's
  `Box(OBS_SIZE)` + `encode_battle`; `"token"` uses `token_obs_space()` +
  `encode_battle_tokens`. `reset`/`step` select the encoder by mode; the terminal step
  returns a zeroed dict in token mode (zeros for `tokens`/`card_ids`/`token_mask`/`scalars`).
- `obs_mode` threads through `_make_battle_env` → `_build_env` → `train_agent` /
  `train_zoo` (keyword, default `"flat"` everywhere, so existing callers are unaffected).

## Section 2 — Feature extractor + policy wiring

### `locma/envs/extractor.py` (new; torch imported lazily)

```python
class TokenSetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, *,
                 d_model=64, n_heads=4, n_layers=2, id_dim=16,
                 ff_mult=2, dropout=0.1, features_dim=128, pool="cls"):
        ...
```

Forward pass (batch `B`):

1. `ids = obs["card_ids"].long()` → `id_embed = nn.Embedding(161, id_dim, padding_idx=0)(ids)` → `(B,20,id_dim)`
2. `x = proj(cat([obs["tokens"], id_embed], -1))`, `proj = Linear(17+id_dim, d_model)` → `(B,20,d_model)`
3. Prepend a learned **CLS** token → `(B,21,d_model)`; `key_padding_mask` from
   `obs["token_mask"]` (pad = True), with `False` for the CLS slot.
4. `z = TransformerEncoder(TransformerEncoderLayer(d_model, n_heads, ff_mult*d_model,
   dropout, batch_first=True), n_layers)(x, src_key_padding_mask=kpm)`
5. `cls_out = z[:,0]` (when `pool="cls"`); `s = scalar_mlp(obs["scalars"])`
   (`Linear(S, d_model)+ReLU`)
6. `return head(cat([cls_out, s], -1))`, `head = Linear(2*d_model, features_dim)+ReLU`

### Approach-C fallback (one knob set, no rewrite)

The `features_extractor_kwargs` are the only tuning surface. Approach C (single-query
attention pooling, lighter, lower overfit risk) is reachable by `n_layers=1` and/or a
`pool="attn"` switch (learned-query MHA pooling instead of CLS). Approach B (DeepSet,
mean/max pool, no attention) is available only as an ablation, not the headline.

### Policy wiring (`locma/envs/training.py`)

Branch on `obs_mode`:

- `"flat"` (control): unchanged — `MaskablePPO("MlpPolicy", env, ...)`.
- `"token"` (treatment): `MaskablePPO("MultiInputPolicy", env,
  policy_kwargs=dict(features_extractor_class=TokenSetExtractor,
  features_extractor_kwargs={...}), ...)`.

`MultiInputPolicy` is the SB3 policy name for `Dict` obs; sb3-contrib registers the
maskable variant (`MaskableMultiInputActorCriticPolicy`) under it, so
`env.action_masks()` flows exactly as today. The actor/critic MLP heads after the
extractor stay SB3-default.

**Implementation checkpoint (verify, do not assume):** confirm the installed
`sb3-contrib` exposes `MaskableMultiInputActorCriticPolicy` under `"MultiInputPolicy"`
for `MaskablePPO`. If an older pin does not, fall back to a thin custom maskable Dict
policy subclass — flag before writing it.

## Section 3 — Experiment protocol, testing & risks

### Tests (cheap; run in CI without training)

- `encode_battle_tokens`: shapes/dtypes; pad rows are zero with `card_id == 0`;
  `token_mask` matches real counts; hand-built `BattleView` asserts each tactical scalar
  (guard count; `reachable_face_damage == 0` when a Guard is up; `lethal_available`
  boundary at `reachable_face_damage == op_health`).
- `TokenSetExtractor`: random Dict obs → output `(B, features_dim)`, finite, grads flow;
  **permutation/padding invariance** — shuffling real tokens (and re-padding) leaves the
  CLS output unchanged (proves the mask works and the set is order-invariant).
- `BattleEnv(obs_mode="token")`: `reset`/`step` return obs matching `observation_space`;
  terminal step returns the zeroed dict.
- Smoke: `MaskablePPO("MultiInputPolicy", ...)` constructs and `learn(2000)` runs
  without error (this also confirms the §2 policy checkpoint).

### Pilot (the gate)

Train **flat** vs **token** for ~100k steps vs a single hard opponent (`max-attack`),
matched seed + budget, `both_seat=True`. Capture **fps / wall-clock** (attention adds
per-step cost) and eval both vs the three hard baselines. Gate: pipeline is sound *and*
token is at least not-worse than flat. Promising → full A/B; flat-out worse or
pathological → stop and document.

### Full A/B (only if pilot passes)

Same recipe as the current best PPO — **zoo curriculum**
(greedy→scripted→max-guard→max-attack), `both_seat`, identical budget + seeds,
**2 seeds** to gauge variance. Only `obs_mode` + policy differ between arms.
**Headline = avg-hard3** (mean win rate vs scripted / max-guard / max-attack) plus a
**token-vs-flat head-to-head**. Results written to `docs/baseline.md` + `docs/worklog.md`.

**Decision rule:** keep PPO2 if it beats flat on avg-hard3 by a margin that clears seed
variance; otherwise document as an honest null (the project values documented nulls).

### Risks & mitigations

- **Overfit** (transformer on small data) → small dims (d_model 64, 2 layers, id_dim
  16), dropout; one-knob fallback to Approach C.
- **Throughput / longer wall-clock** → measured in the pilot; keep dims small, lean on
  `n_envs` parallelism.
- **Fairness** → flat & token share env / opponent / seed / budget; only obs + policy
  differ. Flat path byte-identical, so the old baseline reproduces.
- **Noisy opp-readiness** → the exposed-to-lethal scalar is dropped (§1).
- **card_id leakage** → `card_id` is public (both boards + own hand visible); no hidden
  info enters the obs.

## Files touched

- `locma/envs/encode.py` — additive: token constants, `encode_battle_tokens`,
  `token_obs_space`.
- `locma/envs/extractor.py` — new: `TokenSetExtractor`.
- `locma/envs/battle_env.py` — `obs_mode` param; encoder/space selection.
- `locma/envs/training.py` — `obs_mode` threading + `MultiInputPolicy` branch.
- `tests/` — new tests for encoder, extractor, env token mode, smoke train.
- `docs/baseline.md`, `docs/worklog.md` — results after the A/B.

## Success criteria

1. All new tests pass; existing tests + flat baseline reproduce unchanged.
2. Pilot runs end-to-end and produces flat-vs-token numbers + fps.
3. If the pilot passes, the full A/B produces an avg-hard3 verdict (win or honest null),
   documented in `docs/`.
