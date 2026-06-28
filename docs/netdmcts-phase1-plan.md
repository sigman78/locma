# netdmcts Phase 1 (fair net-guided determinized PUCT): implementation plan

> Rough task list mapped onto the spec (`docs/netdmcts-phase1-design.md`).
> Each task is TDD + a commit. N1–N5 are code (subagent-friendly); N6–N7 are
> compute/analysis runs (orchestrator, background). Branch: `feat/netdmcts-phase1`.

**Global constraints (from spec):** FAIR / human-parity — no hidden info (net reads only
`BattleView`; search determinizes the hidden state, never peeks); build on dmcts's
determinization, NOT cheating mcts/azlite real-state clone; refactors are
behaviour-preserving (azlite + dmcts existing tests stay green); `[ml]` imports lazy
(torch/sb3 inside functions); feature branch + HTTPS.

---

### N1 — Shared PUCT core + azlite refactor · spec §1
**Files:** create `locma/policies/puct.py`; modify `locma/policies/azlite.py`; test `tests/test_puct.py`
- `puct.py`: lift `_Node` + `_select` (PUCT, opponent-minimises sign) from azlite; add
  `puct_search(root_state, oracle, iterations, c_puct, rng) -> list[int]` returning the
  ROOT edge visit counts. `oracle` exposes `priors(sim, actions, seat) -> list[float]`
  and `value(sim, root_seat) -> float`; the search uses `_clone_battle` + `apply_battle`
  + `battle_legal` to roll out (terminal → `_reward`, leaf → `oracle.value`).
- Refactor `AZLiteBattlePolicy` to call `puct_search` with a heuristic-oracle adapter
  (wrap its existing `_prior`/`_value`). Behaviour byte-identical — its tests must stay
  green.
- **Tests:** stub oracle (uniform priors, fixed value) on a real battle state → visit
  counts sum to `iterations` over the root's legal actions; existing `tests/test_azlite.py`
  still passes (re-run it).

### N2 — Lift `determinize` to module level · spec §1
**Files:** modify `locma/policies/mcts.py`; test extend `tests/test_mcts.py`
- Extract `DMCTSBattlePolicy._determinize` body into a module-level
  `determinize(gs, rng, cards, reshuffle_own=True) -> GameState` (beside `_clone_battle`);
  have the method delegate to it. dmcts behaviour + `tests/test_mcts.py` unchanged.
- **Tests:** `determinize` resamples the opponent's hand+deck (ids change / from pool),
  keeps the agent's own hand+board real, reshuffles own deck order (assert via the
  existing dmcts determinization assertions).

### N3 — `NetOracle` · spec §2
**Files:** create `locma/policies/net_oracle.py`; test `tests/test_net_oracle.py`
- `NetOracle(model_path)`: lazily load the token `MaskablePPO`. `priors(sim, actions,
  seat)`: `view=make_battle_view(sim)`, `obs=encode_battle_tokens(view)`, masked policy
  distribution over 155 → map to `actions` via `sem_index(view,a)` → renormalise (sum 1).
  `value(sim, root_seat)`: net critic on the view (sim.current's perspective), negate if
  `sim.current != root_seat`, clip to [-1,1]. Lazy `[ml]` imports.
- **FIRST STEP — verify-at-impl probe:** confirm the sb3-contrib masked-distribution +
  value API (`policy.get_distribution(obs, action_masks=mask).distribution.probs`,
  `policy.predict_values(obs)`) and value scale/sign against the installed sb3-contrib
  (build a tiny token model via `_make_model`, query one real obs). Adapt `NetOracle` to
  the confirmed API. Report the confirmed calls in the task report.
- **Tests** (`importorskip("sb3_contrib")`): on a real battle state + a small token model,
  `priors` sum to 1 over the node's legal actions; `value` finite ∈ [-1,1]; querying the
  opponent seat negates the value sign vs the same state queried from the other root seat.

### N4 — `NetGuidedDMCTSBattlePolicy` · spec §3
**Files:** modify `locma/policies/net_oracle.py`; test `tests/test_netdmcts.py`
- `NetGuidedDMCTSBattlePolicy(model_path, determinizations=15, iterations=80, c_puct=1.5,
  seed=0, deterministic=False)`: requires forward-model `state` (else ValueError). For
  `k` in `K`: `det = determinize(state, rng, cards)`; `visits = puct_search(det,
  NetOracle, I, c_puct, rng)`; accumulate visits per root action across worlds; return the
  real action with max total visits. `deterministic=True` seeds sampling+search from the
  obs hash (mirror dmcts's distillation path). `[ml]` lazy.
- **Tests** (`importorskip("sb3_contrib")`): 1 legal action → returns it (no search);
  small `K,I` on a real battle state + small model → returns a legal action; raises
  without `state`; `deterministic=True` → identical action on repeat.

### N5 — Registry `netdmcts` spec · spec §3
**Files:** modify `locma/policies/registry.py`; test extend `tests/test_registry.py`
- Add `_netdmcts(params, spec)`: parse `netdmcts:K,I,c_puct,model_path`
  (defaults K=15, I=80, c_puct=1.5, model_path="model.zip"), build
  `NetGuidedDMCTSBattlePolicy` paired with `BalancedDraftPolicy` (lazy import like
  `_ppo`); register in `_FACTORIES`; add `"netdmcts"` to `_HIDDEN` (needs a model
  artifact + [ml], like `ppo`). 
- **Tests:** `make_policy("netdmcts:15,80,1.5,runs/x.zip")` → params parsed (K/I/c_puct/
  model_path), draft is `BalancedDraftPolicy`; `make_policy("netdmcts")` → defaults;
  `"netdmcts" not in policy_names()` (hidden); unknown spec raises.

### N6 — Eval (calibration + headline + iter sweep) · spec §4 [orchestrator, background]
- Calibrate s/move for `netdmcts:15,80,1.5,runs/selfplay-r2.zip`; size eval games.
- avg-hard3 (vs scripted/max-guard/max-attack) of netdmcts vs **dmcts (~0.73)**, **raw net
  selfplay-r2 (0.639)**, **azlite (0.741, cheating ref)**. Iteration sweep (I ∈ a few
  values) netdmcts vs dmcts. Use `locma tournament` / `run_match`.

### N7 — Document Phase-1 verdict · spec §4 [orchestrator]
- Write to `docs/baseline.md` + `docs/worklog.md`: does the net oracle beat the heuristic
  oracle in fair search (netdmcts vs dmcts)? iter-efficiency? Decision on Phase 2.

---

**Spec coverage check:** §1 → N1, N2; §2 → N3; §3 → N4, N5; §4 → N6, N7; §5 tests → N1–N5;
constraints (fair/no-hidden-info, behaviour-preserving refactors, lazy ml) → enforced in
N1/N2/N3/N4 + global constraints. No spec requirement unmapped.
