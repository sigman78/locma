# Fair net-guided dmcts (AlphaZero-lite) — Phase 2: self-play training of the search

**Date:** 2026-06-27
**Status:** approved design, entering implementation
**Branch:** `feat/netdmcts-phase2` (stacked on `feat/netdmcts-phase1`)

## Motivation

Phase 1 built a *fair* net-guided determinized PUCT (`netdmcts`) whose oracle is a
**frozen** token `MaskablePPO` net (`runs/selfplay-r2.zip`). It reached avg-hard3
**0.817** — the strongest, and the only *fair* (no-hidden-info), policy in the project,
beating the heuristic-oracle `dmcts` (0.697), the bare net (0.639), and even the
perfect-foresight cheaters `azlite`/`mcts` (~0.74). The Phase-1 oracle was only
RL-trained; it never learned from the **search's own output**.

Phase 2 closes the AlphaZero loop: generate `(obs, search-visit-policy, game-outcome)`
tuples from net-guided-dmcts self-play, train the net's **two heads** on them (policy →
soft cross-entropy to the visit distribution; value → MSE to the outcome), and **iterate**,
with each new net becoming the next iteration's oracle. The question Phase 2 answers:
**does training the oracle on the search's visits + outcomes push fair net-guided search
beyond the frozen-oracle 0.817 — and does each generation actually beat its parent in fair
self-play?**

## Hard constraint: FAIR / human-parity info (unchanged from Phase 1)

No hidden info, anywhere, at inference:
- Every recorded `obs` is the fair public `BattleView` token observation
  (`encode_battle_tokens(make_battle_view(gs))`). The net never sees the opponent's hand
  or deck order.
- The search handles the opponent's hidden hand/deck by **determinization**
  (`policies.mcts.determinize`), exactly as Phase 1 — sampling plausible hidden state from
  the public card pool, reshuffling the searcher's own deck. It never peeks at real hidden
  state.
- The game **outcome `z`** is used **only as a training label** (it is not available to the
  policy at inference). This is standard and fair — labels may use information the runtime
  policy does not.
- The cheating searchers (`mcts`, `azlite`) and their real-state clone remain out of scope.

## Settled decisions (from brainstorming)

1. **Data source — hybrid:** mostly self-play (netdmcts vs netdmcts, current net on both
   seats, both seats recorded) for state diversity, anchored with a minority of baseline
   games (netdmcts vs scripted/max-guard/max-attack, netdmcts seat recorded) to stay near
   the eval distribution. Default ≈ 70/30 (240 self-play + 100 baseline games per iteration).
2. **Exploration — full AlphaZero, generation-only:** root **Dirichlet noise** mixed into
   the search's root priors, and **temperature** move-sampling from the visit counts
   (τ=1 for the opening plies, then argmax). Eval/play stays pure-argmax (Phase-1 behaviour).
3. **Gating/success — composite (both axes):** a new net is **adopted** as the next oracle
   iff it (a) beats the previous best head-to-head in fair self-play **and** (b) does not
   regress avg-hard3 beyond a small ε. Success = the final best net improves over the
   Phase-1 frozen oracle on **both** avg-hard3 and head-to-head self-play.

## Scope

**In:** an optional root-noise extension to the shared `puct_search`; a self-play generator
that records visit-policy + outcome tuples (`record_selfplay`); an AlphaZero trainer that
warm-starts from a net and trains both heads (`az_train`); an iteration orchestrator with
the composite gate (`az_selfplay`); CLI wiring; and the end-to-end run + documented verdict.

**Out:** changing the observation/action encoding or the net architecture (Phase 2 reuses
the PPO2 token net and its existing policy+value heads unchanged); any non-`netdmcts`
policy; distributed/multi-GPU training. Generation may be sharded across CPU processes by
seed range (an operational convenience), but no new scheduling infrastructure is built.

---

## Section 1 — Search: optional root Dirichlet noise (`puct.py`, modify)

Add an optional keyword arg to `puct_search`:

```python
def puct_search(root_state, oracle, iterations, c_puct, rng, root_noise=None) -> list[int]:
```

- `root_noise=None` (default) → **byte-identical** to today. `azlite` and Phase-1
  `netdmcts` play paths pass nothing and are unchanged (their existing tests are the safety
  net).
- `root_noise=(eps, alpha)` → after building the root priors `P` (from
  `oracle.priors(root_state, legal, root_seat)`), mix in a Dirichlet sample over the legal
  edges: `P[i] ← (1 - eps) * P[i] + eps * d[i]`, where `d ~ Dir(alpha)` is drawn with the
  passed `rng` **without numpy** (keeps `puct.py` import-light): for each legal edge
  `g_i = rng.gammavariate(alpha, 1.0)`, then `d_i = g_i / sum(g)`. Noise is applied to the
  **root only** — child-node priors are untouched.
- The mixed `P` still sums to 1 over the legal edges (convex combination of two
  distributions). The rng, previously "reserved", now drives the Dirichlet draw.

## Section 2 — Self-play generator (`locma/envs/selfplay.py`, new)

`record_selfplay(...)` drives games and records one AlphaZero example per **non-forced**
battle decision, then writes an `.npz` + manifest in the practicum idiom (with a layout
guard the trainer re-checks).

**Driving the games.** The generator owns its move loop (it cannot delegate to
`battle_action`, which always argmaxes and exposes no visit counts). For each decision at a
real `GameState` where `seat = gs.current` is a netdmcts agent:
1. `view = make_battle_view(gs)`; `legal = battle_legal(gs)`. If `len(legal) <= 1`, play the
   forced action; record nothing.
2. Accumulate root visit counts across `K` determinized worlds, exactly as Phase-1 netdmcts
   but with root noise on each search:
   `det = determinize(gs, rng, cards)`;
   `counts = puct_search(det, oracle, I, c_puct, rng, root_noise=(eps, alpha))`;
   sum element-wise into `total` (length `len(legal)`, stable order across worlds).
3. **Record the example:** build the policy target `pi` of length `ACTION_SIZE` (155),
   zero-initialised; for each legal edge `i`, `j = sem_index(view, legal[i])`; if `j is not
   None` and `j < ACTION_SIZE`, `pi[j] += total[i]`. Normalise `pi` to sum 1 over its
   non-zero entries. (If every legal edge maps to `None` — not expected in practice — drop
   the row.) Store `obs = encode_battle_tokens(view)` (four token arrays), `pi`,
   `mask = action_mask(view, legal)`, `seat`, `game_id`.
4. **Select the played move** by the temperature schedule: let `ply` be the per-game count
   of non-forced decisions made so far (shared across both seats in self-play — the opening
   gets exploration, the endgame gets greedy play); if `ply < temp_moves`, sample an index
   `i` with probability `∝ total[i]` (τ=1); else `i = argmax(total)`. Play `legal[i]`.
   Recording (step 3) always uses the full visit distribution regardless of how the move is
   selected — sampling affects which line the game follows, not the target.

After each game, stamp the outcome on that game's recorded rows:
`z = +1.0 if winner == seat else (-1.0 if winner == (1 - seat) else 0.0)` — the moving
seat's perspective, matching the net's value-head perspective. A game that raises is skipped
and counted (`failed_games`), as in `record_practicum`.

**Matchups.** For `self_play_games`: both seats are netdmcts driven by the **same** oracle;
both seats' decisions recorded. For `baseline_games`: netdmcts vs each baseline in
(`scripted`, `max-guard`, `max-attack`) round-robin; only the netdmcts seat recorded; both
seat orientations played for balance. All seeded deterministically from `seed` so runs are
reproducible and shardable by seed range.

**Output arrays** (`.npz`): `obs_tokens (n,20,17)`, `obs_card_ids (n,20)`,
`obs_token_mask (n,20)`, `obs_scalars (n,13)`, `policy_target (n,155) float32`,
`mask (n,155) bool`, `value_target (n,) float32`, `seat (n,) int8`, `game_id (n,) int32`.
**Manifest** records `obs_mode="token"`, `max_tokens/token_feats/n_tactical/action_size`
(the layout guard), plus generation params (`oracle_path`, `K`, `I`, `c_puct`, `eps`,
`alpha`, `temp_moves`, `self_play_games`, `baseline_games`, `seed`, `n_examples`,
`failed_games`, `engine_version`).

**Signature (indicative):**
```python
def record_selfplay(
    oracle_path: str,
    out: str = "selfplay.npz",
    self_play_games: int = 240,
    baseline_games: int = 100,
    baselines=("scripted", "max-guard", "max-attack"),
    K: int = 6, I: int = 40, c_puct: float = 1.5,
    eps: float = 0.25, alpha: float = 0.3, temp_moves: int = 10,
    seed: int = 0,
) -> dict:  # returns the manifest
```

## Section 3 — AlphaZero trainer (`locma/envs/az_train.py`, new)

Loads one or more self-play `.npz` datasets (the sliding window), **warm-starts from an
existing net**, and trains both heads.

- **Dataset loader + guard:** `load_selfplay(paths) -> (arrays, manifest)` concatenates the
  given `.npz` files, asserting each manifest's token layout matches `encode.py`
  (`MAX_TOKENS/TOKEN_FEATS/N_TACTICAL/ACTION_SIZE`) — mirror `distill.load_practicum`'s loud
  `ValueError`. Reuse `distill.split_by_game` for a game-level train/val split (no game in
  both).
- **Warm start:** `model = MaskablePPO.load(warm_start_path)` (same path NetOracle uses —
  the token model reconstructs its `MultiInputPolicy` + `TokenSetExtractor`).
  `model.policy.set_training_mode(True)`; `opt = Adam(model.policy.parameters(), lr)`.
- **Policy loss (soft cross-entropy to visits):** per batch,
  `dist = model.policy.get_distribution(obs_batch, action_masks=mask_batch)`;
  `logp = dist.distribution.logits` (masked log-probs);
  `loss_pi = -(target * torch.where(target > 0, logp, 0)).sum(dim=1).mean()` — the `where`
  guards the `0 · -inf` at masked/zero entries.
- **Value loss (MSE to outcome):** `v = model.policy.predict_values(obs_batch)`;
  `loss_v = F.mse_loss(v.squeeze(-1), z_batch)`. This **retrains the value head into a true
  game-outcome predictor** (it was a leftover PPO/GAE critic) — a direct improvement to the
  oracle's leaf value.
- **Total:** `loss = loss_pi + c_v * loss_v` (`c_v = 0.5`). `epochs=10`, `batch=256`,
  `lr=1e-4` (the token net's stable LR from the PPO2 work), game-level val split.
- **Report (val):** mean policy KL/CE to the visit targets and value MSE on held-out games.
  Save the trained net to `out`.
- `[ml]` imports lazy (torch/sb3 inside the function), like `distill`/`training`.
- **Verify-at-impl checkpoint (do NOT assume):** confirm against the installed sb3-contrib
  that `get_distribution(obs, action_masks=mask).distribution.logits` are masked **log-probs**
  (so the soft-CE is correct) and that `predict_values` returns the critic value with the
  expected `[B,1]` shape/sign. If the API differs, adapt — it is localized to `az_train`.

**Signature (indicative):**
```python
def az_train(
    data,                         # one path or a list of .npz paths (the window)
    warm_start: str,
    out: str = "az.zip",
    epochs: int = 10, batch: int = 256, lr: float = 1e-4,
    c_v: float = 0.5, val_frac: float = 0.1, seed: int = 0, verbose: int = 1,
) -> dict:  # {out, val_policy_ce, val_value_mse, n_train, n_val, epochs}
```

## Section 4 — Orchestrator + composite gate (`locma/envs/azloop.py`, new)

`az_selfplay(...)` runs the loop and persists everything for recovery.

- **Init:** `best_net = warm_start` (default `runs/selfplay-r2.zip`); evaluate its avg-hard3
  once as `best_score`. `datasets = []`.
- **Per iteration `it` in `range(iterations)`:**
  1. **Generate** with `best_net` as oracle:
     `npz = record_selfplay(best_net, out=f"{prefix}-data-{it}.npz", seed=base_seed + it, ...)`;
     append to `datasets`.
  2. **Train:** `new_net = az_train(datasets[-window:], warm_start=best_net,
     out=f"{prefix}-net-{it}.zip", ...)`.
  3. **Eval both axes:**
     - `score = avg_hard3(new_net, games_per_opp, K_eval, I_eval, seed)` — mean win-rate of
       `netdmcts(new_net)` vs (`scripted`, `max-guard`, `max-attack`) via `run_match`.
     - `h2h = winrate(netdmcts(new_net) vs netdmcts(best_net), h2h_games, K_eval, I_eval)` —
       both fair, both searching, mirrored.
  4. **Composite adopt:** if `h2h > h2h_thresh (0.53)` **and** `score >= best_score - eps
     (0.02)` → `best_net, best_score = new_net, max(score, best_score)`; reset the
     reject counter. Else reject (keep `best_net`); increment the reject counter.
  5. Append a row to a results log (`{prefix}-results.jsonl`: it, score, h2h, adopted,
     best_score) and to `docs/worklog.md` at the end. **Early-stop** after 2 consecutive
     rejections.
- **Final confirm** on the surviving `best_net`: avg-hard3 at 50 games/opp + head-to-head
  100 games vs the original frozen-oracle netdmcts (`runs/selfplay-r2.zip`).
- **Keep-best semantics:** generation and training both start from `best_net` each
  iteration; a rejected net is discarded and the next iteration retrains from `best_net` on
  fresh data — preventing drift/collapse. The per-iteration `.npz`, nets, and results log
  let a killed run resume.

`avg_hard3` and the head-to-head win-rate are thin helpers over the existing
`harness.match.run_match` (mirrored play); `netdmcts(net)` is built via
`make_policy(f"netdmcts:{K},{I},{c_puct},{path}")`. Cost note: head-to-head games search on
both sides (~49s/game at K=8,I=40), so per-iteration wall-clock is ~1.5–2 h; `h2h_games`
and the head-to-head K,I are tunable to trade noise for speed.

**Signature (indicative):**
```python
def az_selfplay(
    warm_start: str = "runs/selfplay-r2.zip",
    prefix: str = "runs/az",
    iterations: int = 4, window: int = 2, base_seed: int = 0,
    # generation
    self_play_games: int = 240, baseline_games: int = 100,
    K_gen: int = 6, I_gen: int = 40, c_puct: float = 1.5,
    eps: float = 0.25, alpha: float = 0.3, temp_moves: int = 10,
    # training
    epochs: int = 10, batch: int = 256, lr: float = 1e-4, c_v: float = 0.5,
    # eval / gate
    K_eval: int = 8, I_eval: int = 40, games_per_opp: int = 20, h2h_games: int = 40,
    h2h_thresh: float = 0.53, hard3_eps: float = 0.02, max_rejects: int = 2,
) -> dict:  # {best_net, best_score, history:[...]}
```

## Section 5 — CLI (`locma/cli/app.py`, modify)

Three thin commands delegating to the modules above (validate args, lazy `[ml]` import with
the standard "requires the [ml] extra" message, print a one-line summary):
- `record-selfplay` → `record_selfplay` (manual / sharded generation by seed range).
- `az-train` → `az_train` (accepts one or more `--data` paths + `--warm-start`).
- `az-selfplay` → `az_selfplay` (the full loop).

## Section 6 — Testing (`[ml]`-gated by `importorskip` where the net is needed)

Mirror `distill`'s philosophy: **unit-test the pure helpers; operationally verify the sb3
training loop.**
- **`puct_search` root noise:** with a stub oracle on a real battle state, `root_noise=(eps,
  alpha)` keeps the (internal) root priors a valid distribution and visit counts still sum
  to `iterations`; with a fixed `rng`, the noised search is reproducible; `root_noise=None`
  reproduces today's counts (azlite's existing tests stay green — the refactor safety net).
- **Visit-target construction** (factored into a pure helper): given `total` over a node's
  legal actions, `policy_target` sums to 1 over non-zero entries, is zero on illegal slots,
  drops `sem_index is None` edges, and is `mask`-consistent.
- **Outcome stamping:** `z == +1` when `winner == seat`, `-1` when `winner == 1 - seat`,
  `0` on a draw.
- **Temperature + Dirichlet sampling determinism:** fixed seed → identical played-move
  sequence and identical Dirichlet draws; τ→argmax past `temp_moves`.
- **Self-play dataset layout guard:** `load_selfplay` raises a clear `ValueError` on a
  layout mismatch; concatenation across multiple `.npz` is correct; `split_by_game` keeps
  games disjoint.
- **`az_train` smoke** (`[ml]`): on a tiny generated dataset + a small token net, one run
  completes, losses are finite, and the combined loss decreases over a few epochs.
- **`az_selfplay` wiring:** the composite gate adopts iff `h2h > thresh` and `score >=
  best - eps`, rejects otherwise, and early-stops after 2 rejects (tested with stubbed
  generate/train/eval so it's fast and `[ml]`-free).
- **CLI parse:** `record-selfplay` / `az-train` / `az-selfplay` parse their options; bad
  values raise a friendly error.

## Files touched

- `locma/policies/puct.py` — add optional `root_noise=(eps, alpha)` to `puct_search`
  (Dirichlet via `rng.gammavariate`, root-only; default `None` unchanged).
- `locma/envs/selfplay.py` — new: `record_selfplay` + the visit-target / outcome helpers.
- `locma/envs/az_train.py` — new: `load_selfplay` (+ guard) and `az_train` (soft-CE policy
  + MSE value, warm-started).
- `locma/envs/azloop.py` — new: `az_selfplay` orchestrator + `avg_hard3` / head-to-head
  helpers.
- `locma/cli/app.py` — `record-selfplay`, `az-train`, `az-selfplay` commands.
- `tests/` — puct root-noise, target construction, outcome stamping, sampling determinism,
  dataset guard, az_train smoke, az_selfplay gating wiring, CLI parse.
- `docs/baseline.md`, `docs/worklog.md`, `docs/ppo-review.md` §8.4B — results after the run.

## Defaults (all CLI-tunable)

| knob | default |
|------|---------|
| generation search | K=6, I=40, c_puct=1.5 |
| Dirichlet root noise | α=0.3, ε=0.25 |
| temperature | τ=1 for first 10 plies, then argmax |
| games / iteration | 240 self-play + 100 baseline (≈70/30) |
| training | warm-start, 10 epochs, batch 256, lr 1e-4, c_v=0.5, window=2 |
| gate eval | K=8, I=40, avg-hard3 20 games/opp, head-to-head 40 games |
| composite gate | adopt iff h2h > 0.53 AND avg-hard3 ≥ best − 0.02 |
| iterations | 4 (early-stop after 2 consecutive rejects) |
| final confirm | avg-hard3 50/opp + head-to-head 100 games vs frozen-oracle netdmcts |

## Success criteria

1. New tests pass; `puct_search` with `root_noise=None`, `azlite`, and Phase-1 `netdmcts`
   behaviour reproduce unchanged (the root-noise addition is behaviour-preserving by default).
2. The AZ loop runs end-to-end as a **fair** pipeline (no hidden info at inference) and
   produces, per iteration, an avg-hard3 score and a head-to-head-vs-parent win-rate, with
   the composite gate carrying forward the best net.
3. A documented Phase-2 verdict: **success = the final best net improves over the Phase-1
   frozen oracle on *both* axes** — avg-hard3 **> 0.817** *and* a positive head-to-head
   self-play progression (final best beats the original `selfplay-r2`-oracle netdmcts
   head-to-head). The outcome is documented either way, including a one-axis-only result
   (informative about what search-training buys). Results written to `docs/baseline.md`,
   `docs/worklog.md`, and `docs/ppo-review.md` §8.4B.
