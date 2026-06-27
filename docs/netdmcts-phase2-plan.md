# netdmcts Phase 2 (AlphaZero self-play loop): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> to implement this plan task-by-task. Rough task list mapped onto the spec
> (`docs/netdmcts-phase2-design.md`); each task is TDD + a commit.
> P1–P6 are code (subagent-friendly); P7–P8 are compute/analysis runs (orchestrator,
> background). Branch: `feat/netdmcts-phase2` (stacked on `feat/netdmcts-phase1`).

**Goal:** Close the AlphaZero loop for fair net-guided dmcts — generate `(obs, visit-policy,
outcome)` from netdmcts self-play, train the token net's two heads on them, iterate, and
push fair search past the Phase-1 frozen-oracle 0.817.

**Architecture:** Three new modules in the existing record→train idiom — a self-play
*generator* (`selfplay.py`), an AZ *trainer* (`az_train.py`), an iteration *orchestrator*
(`azloop.py`) — plus one behaviour-preserving extension to the shared `puct_search` (root
Dirichlet noise) and thin CLI wiring. Persisted `.npz` per iteration makes runs recoverable
and generation shardable.

**Tech Stack:** Python, numpy, PyTorch, sb3-contrib `MaskablePPO` (token `MultiInputPolicy`
+ `TokenSetExtractor`), pytest, typer CLI.

## Global Constraints (from spec — every task inherits these)

- **FAIR / human-parity:** no hidden info at inference. Every recorded `obs` is the public
  `BattleView` token obs; the search determinizes the opponent's hidden state
  (`policies.mcts.determinize`), never peeks. The outcome `z` is used **only as a training
  label** (not available to the policy at inference).
- **Behaviour-preserving search:** `puct_search` with `root_noise=None` is byte-identical to
  today; `azlite` and Phase-1 `netdmcts` play paths pass nothing → their existing tests stay
  green (the refactor safety net).
- **No encoding/arch change:** reuse the PPO2 token net and its existing policy + value
  heads unchanged. `obs_mode="token"` only (this is a token-net pipeline).
- **`[ml]` imports lazy:** torch / sb3-contrib imported inside functions, never at module
  top (mirrors `distill.py` / `training.py`).
- **Training knobs (spec defaults):** warm-start from the current best net; soft-CE policy +
  MSE value; `lr=1e-4`, `c_v=0.5`, `epochs=10`, `batch=256`, `window=2`.
- **Process:** feature branch only (never main); push via HTTPS + gh credential helper.

---

### P1 — `puct_search` root Dirichlet noise · spec §1
**Files:** modify `locma/policies/puct.py`; test extend `tests/test_puct.py`

- Add `root_noise=None` kwarg to `puct_search(root_state, oracle, iterations, c_puct, rng,
  root_noise=None)`. When `root_noise=(eps, alpha)`: after building the root priors `P`,
  draw a Dirichlet sample over the legal edges **without numpy** —
  `g = [rng.gammavariate(alpha, 1.0) for _ in P]`, `s = sum(g) or 1.0`,
  `d = [x / s for x in g]` — then `P[i] = (1 - eps) * P[i] + eps * d[i]`. Root node only;
  child priors unchanged. `root_noise=None` skips all of this.
- **Tests:** `root_noise=None` → counts identical to a no-noise call on the same state
  (and `tests/test_azlite.py` still passes — re-run it); `root_noise=(0.25, 0.3)` with a
  stub oracle → root priors stay a valid distribution (sum ≈ 1, all ≥ 0) and visit counts
  still sum to `iterations`; with a fixed-seed `rng` the noised search is reproducible
  (two runs → identical counts); with `eps=1.0` the root priors equal a pure Dirichlet draw
  (oracle priors fully replaced).

### P2 — Self-play recording helpers (pure, `[ml]`-free) · spec §2
**Files:** create `locma/envs/selfplay.py` (helpers only this task); test
`tests/test_selfplay_helpers.py`

- `build_policy_target(view, legal, total) -> tuple[np.ndarray, bool]`: zero `(155,)`
  float32; for each edge `i`, `j = sem_index(view, legal[i])`; if `j is not None and j <
  ACTION_SIZE`: `pi[j] += total[i]`. Normalise to sum 1 over non-zero entries. Return
  `(pi, ok)` where `ok=False` iff every edge mapped to `None` (caller drops the row).
- `outcome_for(winner, seat) -> float`: `+1.0` if `winner == seat`, `-1.0` if
  `winner == 1 - seat`, else `0.0`.
- `select_move_index(total, ply, temp_moves, rng) -> int`: if `ply < temp_moves` sample `i`
  with probability `∝ total[i]` (τ=1, via `rng`); else `argmax(total)`. (Move selection
  only — recording always uses the full `total`.)
- **Tests:** target sums to 1 over non-zero entries, zero on illegal slots, `mask`-aligned;
  an all-`None` legal set → `ok=False`; `outcome_for` truth table (win/loss/draw, both
  seats); `select_move_index` with fixed seed is reproducible and `ply >= temp_moves`
  always returns the argmax; sampling never returns an index with `total[i] == 0`.

### P3 — `record_selfplay` generator (driver + npz/manifest) · spec §2
**Files:** modify `locma/envs/selfplay.py`; test `tests/test_record_selfplay.py`

- `record_selfplay(oracle_path, out="selfplay.npz", self_play_games=240,
  baseline_games=100, baselines=("scripted","max-guard","max-attack"), K=6, I=40,
  c_puct=1.5, eps=0.25, alpha=0.3, temp_moves=10, seed=0) -> dict` (returns manifest).
- Drives games with its own move loop (NOT `battle_action`, which argmaxes / hides visits):
  build one `NetOracle(oracle_path)`; per decision at `gs` with `seat=gs.current`:
  `view`/`legal`; forced (`len<=1`) → play, record nothing; else accumulate `total` across
  `K` worlds (`determinize` → `puct_search(det, oracle, I, c_puct, rng,
  root_noise=(eps,alpha))`), record `(encode_battle_tokens(view), build_policy_target(...),
  action_mask(view,legal), seat, game_id)` when `ok`, then play
  `legal[select_move_index(total, ply, temp_moves, rng)]`. After each game stamp
  `value_target = outcome_for(winner, seat)` on its rows. Self-play = both seats one shared
  oracle, both recorded; baseline = netdmcts seat only, both orientations; failed games
  skipped + counted.
- **Implementation note:** drive games with `run_game` using a small recording battle
  policy that owns the search+sampling and appends to a shared buffer (so both self-play
  seats share one collector), or an explicit game loop — implementer's choice; keep the
  hidden-info invariant (only `BattleView`-derived obs recorded).
- Write `.npz` (`obs_tokens (n,20,17)`, `obs_card_ids (n,20)`, `obs_token_mask (n,20)`,
  `obs_scalars (n,13)`, `policy_target (n,155) f32`, `mask (n,155) bool`, `value_target
  (n,) f32`, `seat (n,) i8`, `game_id (n,) i32`) + manifest (token layout guard fields +
  all generation params + `n_examples`, `failed_games`, `engine_version`). Reuse
  `practicum._manifest_path` / `_engine_version`.
- **Tests** (`importorskip("sb3_contrib")`, tiny model via `distill`/`_make_model` pattern,
  `self_play_games=2, baseline_games=1, K=2, I=4`): produces ≥1 row; every `policy_target`
  row sums to ~1 and is zero off the mask; `value_target ∈ {-1,0,1}`; manifest layout
  matches `encode.py`; a fixed `seed` → identical arrays on a rerun (determinism).

### P4 — `load_selfplay` + `az_train` (warm-start, soft-CE + MSE) · spec §3
**Files:** create `locma/envs/az_train.py`; test `tests/test_az_train.py`

- **FIRST STEP — verify-at-impl probe:** against the installed sb3-contrib, confirm
  `policy.get_distribution(obs, action_masks=mask).distribution.logits` are masked
  **log-probs** (so `loss_pi = -(target * where(target>0, logp, 0)).sum(1).mean()` is a true
  soft cross-entropy) and `policy.predict_values(obs)` returns `[B,1]` critic values with
  the expected sign. Report the confirmed calls; adapt locally if they differ.
- `load_selfplay(paths) -> tuple[dict, dict]`: accept one path or a list; concat arrays
  across files; assert each manifest's `max_tokens/token_feats/n_tactical/action_size`
  match `encode.py` (loud `ValueError`, mirror `distill.load_practicum`). Reuse
  `distill.split_by_game` for the game-level split.
- `az_train(data, warm_start, out="az.zip", epochs=10, batch=256, lr=1e-4, c_v=0.5,
  val_frac=0.1, seed=0, verbose=1) -> dict`: `model = MaskablePPO.load(warm_start)`;
  `set_training_mode(True)`; Adam(`lr`); per batch build dict obs tensors (as `distill`'s
  token path does), `loss = loss_pi + c_v * loss_v`; report val policy-CE + value-MSE on
  held-out games; `model.save(out)`. Lazy `[ml]`.
- **Tests:** `load_selfplay` raises on a layout-mismatched manifest and concats two `.npz`
  correctly (pure, no `[ml]`); `az_train` smoke (`importorskip`) on a tiny P3 dataset + the
  `warm_start` being a freshly-saved small token net → completes, returns finite
  `val_policy_ce`/`val_value_mse`, and the mean combined train loss at the last epoch <
  the first epoch.

### P5 — `az_selfplay` orchestrator + eval helpers + composite gate · spec §4
**Files:** create `locma/envs/azloop.py`; test `tests/test_azloop.py`

- Helpers: `avg_hard3(net_path, games_per_opp, K, I, c_puct, seed) -> float` (mean
  `run_match` win-rate of `make_policy(f"netdmcts:{K},{I},{c_puct},{net_path}")` vs
  `scripted`/`max-guard`/`max-attack`); `h2h_winrate(new_path, best_path, games, K, I,
  c_puct, seed) -> float` (mirrored `run_match`, both netdmcts).
- `az_selfplay(warm_start="runs/selfplay-r2.zip", prefix="runs/az", iterations=4, window=2,
  base_seed=0, …gen…, …train…, K_eval=8, I_eval=40, games_per_opp=20, h2h_games=40,
  h2h_thresh=0.53, hard3_eps=0.02, max_rejects=2) -> dict`: init `best_net=warm_start`,
  `best_score=avg_hard3(best_net,…)`; per iteration generate (oracle=`best_net`) → `az_train`
  (warm_start=`best_net`, last `window` datasets) → eval both axes → **composite adopt iff
  `h2h > h2h_thresh AND score >= best_score - hard3_eps`** (`best_score = max(score,
  best_score)` on adopt) else reject; append to `{prefix}-results.jsonl`; early-stop after
  `max_rejects` consecutive rejects. Final confirm on the survivor (50/opp + 100 h2h vs the
  original `warm_start`). Return `{best_net, best_score, history}`.
- Make the generate/train/eval steps injectable (module-level functions the test can
  monkeypatch) so the **gate logic is unit-tested without `[ml]`**.
- **Tests** (no `[ml]`, stub the three steps): adopt when both conditions hold; reject when
  h2h ≤ thresh; reject when avg-hard3 regresses > eps; `best_score` tracks the high-water
  mark; early-stop fires after 2 consecutive rejects; the next iteration generates/trains
  from the retained `best_net` (not the rejected one).

### P6 — CLI commands · spec §5
**Files:** modify `locma/cli/app.py`; test extend `tests/test_cli.py` (or the CLI test file)

- `record-selfplay` → `record_selfplay`; `az-train` → `az_train` (accept repeated `--data`
  + `--warm-start`); `az-selfplay` → `az_selfplay`. Thin: validate args, lazy `[ml]` import
  with the standard "requires the [ml] extra: uv sync --extra ml" `BadParameter`, print a
  one-line summary (mirror the `distill` / `record-practicum` commands).
- **Tests:** each command parses its options and rejects bad values (e.g. `iterations < 1`,
  `window < 1`) with a friendly error; no model load needed (validate-then-import order, as
  `train`/`distill` do).

### P7 — Calibrate + run the AZ loop · spec §4 [orchestrator, background]
- Re-confirm gen throughput at the chosen `K_gen,I_gen` (P-1 probe already ~13–20 s/game at
  K=6,I=40) and the eval cost; optionally shard generation across processes by seed range
  and merge `.npz`. Run `az_selfplay` for 4 iterations (early-stop honoured). Capture
  per-iteration avg-hard3 + head-to-head and the final confirm.

### P8 — Document the Phase-2 verdict · spec §8 [orchestrator]
- Write results to `docs/baseline.md` ("netdmcts" section), `docs/worklog.md` (dated entry),
  and `docs/ppo-review.md` §8.4B: per-iteration curve (avg-hard3 + h2h), whether the final
  best beat 0.817 **and** progressed head-to-head, and what search-training bought (incl. a
  one-axis-only outcome if that's what happened).

---

**Spec coverage check:** §1 → P1; §2 → P2 (helpers), P3 (generator); §3 → P4
(`load_selfplay` + `az_train`); §4 → P5 (orchestrator + gate), P7 (run); §5 → P6 (CLI);
§6 testing → P1–P6 test sections; the "Hard constraint" (fairness / label-only `z`) →
Global Constraints + enforced in P3 (obs = `BattleView` only); "Success criteria" /
verdict → P7 (measure) + P8 (document). Defaults table → P3/P4/P5 signatures (exact values
copied). No spec requirement unmapped.
