# Fair net-guided dmcts (AlphaZero-lite) — Phase 1: net-as-oracle determinized PUCT

**Date:** 2026-06-27
**Status:** approved design, entering implementation
**Branch:** `feat/netdmcts-phase1` (stacked on the PPO2 / distill work)

## Motivation

Every reactive lever this project tried — richer encoding (PPO2), BC-distillation,
self-play — plateaus around avg-hard3 0.6–0.64 and never crosses the planning gap to
the search policies (~0.73). `ppo-review.md` §8.4B identifies the one open lever:
**search in the loop** — a policy/value net guiding MCTS, which adds the planning a
reactive net structurally lacks. This spec is **Phase 1**: a *fair* net-guided
determinized search using the existing trained net as a **frozen** `(policy, value)`
oracle — no training loop yet. It answers the make-or-break question before we invest
in the self-play loop: **does fair net-guided search plan better than the heuristic-
guided `dmcts` (~0.73) and the raw net (0.64)?**

## Hard constraint: FAIR / human-parity info (no hidden info)

The deployment goal is a policy that runs on what a human sees — **no hidden info**.
- The **net** reads only the public `BattleView` (our token obs); it cannot see the
  opponent's hand or deck order.
- The **search** handles the opponent's hidden hand/deck by **determinization** —
  sampling plausible hidden state from the public card pool (`dmcts._determinize`),
  exactly as a human reasons under uncertainty. It never peeks at the real hidden state.
- The cheating searchers (`mcts`, `azlite`) clone the **real** `GameState` (opponent
  hand + both decks' future draws) and are **out of scope** — Phase 1 builds on
  `dmcts`'s determinization, not their real-state clone.

## Scope

**In:** a shared PUCT core + a module-level determinizer (refactored from azlite/dmcts),
a `NetOracle` (the net's masked policy-prior + value), a `NetGuidedDMCTSBattlePolicy`
(fair determinized PUCT with the net oracle), registry wiring, and an eval vs
dmcts/raw-net/azlite incl. an iteration-efficiency sweep.

**Out (own spec, later):** Phase 2 — the self-play AlphaZero training loop (generate
`(obs, visit-policy, outcome)` from net-guided-dmcts self-play, train policy→visits +
value→outcome, iterate). Phase 1's result gates whether Phase 2 is worth building.

## Section 1 — Shared PUCT core + determinizer (DRY refactor)

- Create `locma/policies/puct.py` with the PUCT primitives currently inside
  `azlite.py`: `_Node` (seat, actions, P, N, W, children) and a `puct_search(root_state,
  oracle, iterations, c_puct, rng) -> list[int]` that runs the iteration loop
  (select → expand+evaluate → backprop) and returns the **root edge visit counts**.
  The `oracle` is an injected object exposing `priors(sim, actions, seat) -> list[float]`
  and `value(sim, root_seat) -> float` (root-seat perspective). `_select` (PUCT with the
  opponent-minimises sign) lives here.
- Refactor `azlite.AZLiteBattlePolicy` to call `puct_search` with a **heuristic oracle**
  (its existing `_prior`/`_value`, wrapped to the oracle interface) + real-state clone.
  azlite's behaviour and its existing tests must stay green (the safety net for the
  refactor); azlite stays the cheating reference baseline.
- Lift `dmcts`'s determinization to a module-level
  `determinize(gs, rng, cards, reshuffle_own=True) -> GameState` (in `mcts.py`, beside
  `_clone_battle`), and have `DMCTSBattlePolicy._determinize` delegate to it (dmcts
  behaviour + tests unchanged). `netdmcts` reuses the same function.

## Section 2 — `NetOracle` (`locma/policies/net_oracle.py`, new)

Wraps a lazily-loaded token `MaskablePPO` model; reads only the fair `BattleView`.
- `priors(sim, actions, seat)`: `view = make_battle_view(sim)`; `obs =
  encode_battle_tokens(view)`; get the net's **masked policy distribution** over the 155
  semantic actions; map to each action in `actions` via `sem_index(view, a)` and
  renormalise over the legal set → priors (sum to 1). Single forward → all priors.
- `value(sim, root_seat)`: the net's **critic value** for `make_battle_view(sim)` (from
  `sim.current`'s perspective); return it converted to `root_seat` perspective (negate if
  `sim.current != root_seat`), matching azlite's root-seat convention. Clip to [-1, 1].
- `[ml]` imports lazy (torch/sb3 inside methods); the model auto-detects token obs (the
  existing `MaskablePPOBattlePolicy._encode_for` machinery / Dict obs space).
- **Verify-at-impl checkpoint (do NOT assume):** the exact sb3-contrib calls for the
  masked policy probabilities and the value — expected `policy.get_distribution(obs,
  action_masks=mask).distribution.probs` and `policy.predict_values(obs)` — plus the
  value's scale/sign, confirmed in a short probe against the installed sb3-contrib before
  building. If the API differs, adapt (it's a localized change in `NetOracle`).

## Section 3 — `NetGuidedDMCTSBattlePolicy` + registry

- New policy `NetGuidedDMCTSBattlePolicy` in `net_oracle.py` (with `NetOracle`, to keep
  all torch/net-dependent code in the ML module; `mcts.py` stays import-light): like
  `dmcts`, for `k` in
  `K` determinizations build `det = determinize(state, rng, cards)`, run
  `puct_search(det, NetOracle, I, c_puct, rng)` for `I` iterations, **accumulate the root
  edge visit counts across all K worlds**, and return the real action with the most total
  visits. Root legal actions are identical across worlds (own hand/board stay real). A
  `deterministic` flag seeds sampling+search from the observation hash for replay
  (mirrors dmcts).
- Requires the forward-model `state` (raises `ValueError` without it, like dmcts/azlite).
- Registry (`registry.py`): add `netdmcts` to `policy_names()` + parse
  `netdmcts:K,I,c_puct,model_path` (defaults e.g. `K=15,I=80,c_puct=1.5,
  model_path="model.zip"`), paired with the `balanced` draft (like `ppo`). `model_path`
  contains `/` but no `:`/`,`, so the existing split is safe.

## Section 4 — Eval & success criterion (orchestrator)

- **Throughput calibration** first: s/move at default `K,I` (search is K×I net-forwards)
  to size the eval game counts.
- **Headline (avg-hard3 vs scripted/max-guard/max-attack):**
  - **netdmcts vs `dmcts` (~0.73)** — does the net oracle beat the heuristic oracle in
    *fair* search? (the core question)
  - **vs raw net `selfplay-r2` (0.639)** — does search add planning over the bare net?
  - **vs `azlite` (0.741, cheating)** — the cheating-ceiling reference.
- **Iteration-efficiency sweep:** netdmcts avg-hard3 at a few `I` values vs dmcts at the
  same — good priors should reach strength at fewer iterations.
- **Decision:** netdmcts ≥ dmcts (≳0.73) → the net oracle is a good substrate → Phase 2
  (self-play loop) justified. netdmcts < dmcts → the fixed net oracle ≈ heuristic for
  fair search → document and reconsider Phase 2. Either way write the result to
  `docs/baseline.md` + `docs/worklog.md`.

## Section 5 — Testing (`[ml]`-gated by `importorskip` where the net is needed)

- `puct.puct_search`: with a trivial stub oracle (uniform priors, fixed value) on a real
  battle state, returns visit counts summing to `iterations` over the root's legal
  actions; azlite's refactor keeps its existing tests green (no behaviour change).
- `determinize`: opponent hand/deck resampled, own hand/board kept real, own deck order
  reshuffled (assert via the existing dmcts determinization test pattern).
- `NetOracle`: on a real battle state with a small token model, `priors` sum to 1 over
  the node's legal actions and are zero/absent for illegal ones; `value` is finite and in
  [-1, 1].
- `NetGuidedDMCTSBattlePolicy`: 1 legal action → returns it (no search); on a real battle
  state with small `K,I` → returns a legal action; `deterministic=True` → identical action
  on repeat.
- Registry: `make_policy("netdmcts:15,80,1.5,runs/x.zip")` parses params + pairs the
  balanced draft; a bad spec raises `ValueError`.

## Files touched

- `locma/policies/puct.py` — new: shared `_Node`/`_select`/`puct_search`.
- `locma/policies/net_oracle.py` — new: `NetOracle` + `NetGuidedDMCTSBattlePolicy`.
- `locma/policies/mcts.py` — lift `determinize` to module level; dmcts delegates to it.
- `locma/policies/azlite.py` — refactor to use `puct.puct_search` (behaviour unchanged).
- `locma/policies/registry.py` — `netdmcts` spec + `policy_names()`.
- `tests/` — puct, determinize, net_oracle, netdmcts, registry.
- `docs/baseline.md`, `docs/worklog.md` — results after the eval.

## Success criteria

1. New tests pass; azlite + dmcts behaviour and existing tests reproduce unchanged
   (the refactor is behaviour-preserving).
2. `netdmcts` runs end-to-end as a fair policy (no hidden info) and produces avg-hard3 +
   iteration-efficiency numbers vs dmcts / raw-net / azlite.
3. A documented Phase-1 verdict (net oracle helps fair search, or not) that gates Phase 2.
