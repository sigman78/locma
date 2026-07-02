# Worklog — policies & engine

Brief running log of experiments and insights. Newest entries at the bottom of
each day. Detail lives in `docs/ppo-review.md` and `docs/baseline.md`; this is the
scannable index. One line per finding.

---

## 2026-06-25 — PPO investigation & draft control

### PPO battle policy
- **Root cause of weak PPO = the action space, not training.** It was *positional*
  (`index_to_action(idx, legal)=legal[idx]`) whose index→action map isn't a
  function of the observation, plus a count-only (length-prefix) action mask.
  Fixed → fixed **semantic 155-action** space + enriched **308-d** obs +
  `ent_coef=0.02`. avg win vs the 4 ground baselines **0.28 → 0.42**. (PR #19,
  `ppo-review.md`.)
- **Observation richness and entropy are NOT levers** (multi-seed 2×2 A/B):
  lean-146 ≈ rich-308, `ent_coef` 0 ≈ 0.02 (all 0.41–0.43). The action
  representation was the lever; the obs is secondary.
- **Ruled out** as causes: training budget (100k→3M flat), opponent diversity
  *under the positional space*, obs normalization (VecNormalize hurt), value
  learnability (`explained_variance` 0.48–0.63 is fine), the 64-action cap (no
  truncation), net size.

### Opponents
- **Opponent diversity now helps** (zoo study, post-fix) — opposite of the
  positional era. `mixed` per-episode pool (0.45) > back-to-back curriculum (0.43)
  > single-`greedy` (0.41). `train-zoo` CLI added (PR #20).
- **Ratings mislead for PPO** (again): openskill/Elo rank `curriculum` #1 while it
  loses head-to-head to both other PPOs. Read the pair-score matrix.

### Draft / decks (the big one)
- **The gap to the ground baselines is mostly the DECK, not the battle policy**
  (`ppo-review.md` §8.1). Pairing the *same* PPO battle net with a `max-guard`
  draft instead of `greedy` lifts it 0.41 → 0.50 vs the hard baselines.
- **The battle net is deck-robust:** training it on a specific draft ≈ just pairing
  the mixed-trained net with that draft (Experiment B ≈ C). The deck at
  *deployment* is the lever, not the deck it trained on.
- **Draft sweep (7 drafts):** `greedy` (the old `ppo:` draft) is the **worst**
  partner (0.39) — even a *random* draft (0.49) beats it; `max-guard` (0.55) and
  the new `balanced` (0.54) make the PPO **beat all three ground baselines**.
  New heuristics added: `max-defense`, `weighted`, `balanced` (PR #21).
- **`ppo:` now pairs with `balanced`** (PR #21) — a no-retrain change from
  baseline-loser to baseline-beater.

### Spell-card valuation
- **Items carry stats applied to the ENEMY:** red/blue removal spells have
  *negative* attack/defense (e.g. *Decimate* def −99 = destroy, *Throwing Axe*
  def −7 = 7 dmg). The stat-summing heuristics scored these by `attack+defense`,
  rating premium removal as the *worst cards in the game*. Fixed: `_card_value`
  values items by effect magnitude (`|atk|+|def|`, capped at 13) + keywords.
- **But correct spell valuation HURT the PPO** (`balanced` 0.544 → 0.487 once it
  drafted removal): the battle net plays creatures far better than spells. Tuned
  `balanced`'s item discount (1.5→6→12 ⇒ 0.47→0.52→0.56) to a strong creature
  bias; shipped `balanced` reaches **0.556**, the best draft in the sweep. (`greedy`
  left deliberately naive as the reference baseline.)
- **Spell-underuse is NOT a training-distribution gap** (hypothesis *refuted*):
  training a PPO battle net *on* spell-heavy hands (≈1.1 removal/deck) did not
  improve spell-deck play (0.493 vs 0.487 creature-trained) and made the net worse
  overall (0.492 vs 0.556 on the good deck). Spell decks are inherently weaker for
  this aggressive tempo style — a creature gives recurring board presence + face
  damage; one-shot removal trades 1-for-1 without advancing your clock.

### vs the "final boss" (`mcts:100`, cheating perfect-info search)
- **The new PPO is competitive with MCTS.** `ppo`(balanced) vs `mcts:100` (as-is,
  greedy deck) = **0.46** (≈ even); `max-guard` vs `mcts:100` = 0.51 (even). MCTS
  crushes `greedy` 0.79 but itself drafts the weak `greedy` deck, so against
  good-deck opponents its lookahead edge is largely cancelled by its deck.
- **Same-deck (both `balanced`), `ppo` vs `mcts` = 0.39** — MCTS's perfect-info
  battle is genuinely stronger; the ~0.07 deck handicap explains the as-is
  near-even. The residual (0.39 → 0.50) is MCTS's *planning* advantage, which is
  what the next lever (reward shaping / battle sharpness) targets. (Old positional
  PPO would have been crushed ~0.21 like `greedy`.)

### Reward shaping (was roadmap #1) — RULED OUT
- Potential-based shaping (PBS, policy-invariant, verified correct to machine
  precision) with `Φ = health-lead + w·board-lead`, trained vs `mixed` 600k, eval
  paired with `balanced`: sparse **0.554** = health-only PBS **0.554** (exactly
  neutral) > health+board coef 0.5 **0.522** > coef 1.0 **0.500**.
- The **board** term hurts (monotonically with coef) — it discourages the favorable
  face-trades that win this aggressive tempo game. Health-only is exactly neutral.
- Conclusion: the sparse ±1 reward is already adequate; shaping doesn't improve
  credit assignment. The residual gap to `mcts:100` is its **lookahead**, needing
  self-play or search — not reward shaping. (`ppo-review.md` §8.2.)

### MCTS rebuilt — fast and much stronger (PR #23)
- Forward model: `copy.deepcopy` → fast battle-only clone (shares immutable cards)
  = **6.7× faster, byte-identical**. + O(1) `has()` / `__slots__` engine micro-opts.
- **Heuristic turn-based rollouts** (`rollout_turns=3` default): random-play a few
  *turn boundaries* (adaptive depth — not a magic ply count), then score the settled
  position with a board/health heuristic. Far **stronger** than random-to-terminal:
  `mcts:100` vs greedy 0.79→**0.91**, and it now beats the whole pool incl the PPO
  (**0.73**). Net `mcts:100` ~7.8 → ~0.2 s/game (**~30×**). Adaptive note: depths
  6/12/24 ≈ turns 2/3 all ~equal — strength is the heuristic leaf value, not the
  depth; pure heuristic (no rollout) is weaker (combat-state misjudged).

### Self-play / league (was the top open lever) — RULED OUT
- Warm-started from zoo-mixed; league pool = past PPO snapshots + ground baselines
  + `mcts:100` (now cheap as a live opponent). Harness adversarially verified first
  (caught + fixed two real bugs: seat-locked training, agent-deck coupled to the
  opponent's draft).
- **480k steps (6 rounds): flat, then slightly DOWN.** vs max-guard 0.59→0.49,
  vs `mcts:100` **0.24→0.16**. No upward trend; it didn't even hold warm-start.
- **Why:** a *reactive policy net cannot match a search policy by playing more
  games* — self-play improves which move the net reflexively picks but adds no
  planning. MCTS wins by lookahead the net structurally lacks. Same wall as every
  training-method lever (budget, opponent diversity, reward shaping — all flat;
  only the **deck** ever moved the PPO). **To beat MCTS you need search in the
  loop (AlphaZero-style: a policy+value net guiding MCTS), not more self-play.**

### Non-cheating MCTS (DMCTS) — the strength is the search, not the peeking
- **DMCTS** = determinized MCTS: don't peek at the opponent's hand; sample K
  plausible hands from the card pool, run the fast heuristic MCTS on each world,
  vote. The move is a function of the **public** obs (info-matched to the student).
- **DMCTS ≈ the cheating `mcts:100` in strength** (K15/I30, 80 games/cell): vs
  greedy **0.96** / max-guard 0.85 / max-attack 0.81 / **ppo 0.76**, head-to-head
  vs cheating mcts **0.463** (even). It beats the pool *harder* than the cheater vs
  greedy/max-guard/ppo. Speed ~0.84 s/game (K10 0.51, K25 2.2) vs the cheater's 0.19.
- **Implication:** the opponent's hidden hand barely changes the best move in this
  board/tempo game — the cheating MCTS's power is its **search**, not the cheating.
  So a strong, *info-matched* (learnable) teacher now exists. It also reframes the
  old distillation cap (0.37 agreement) hypothesis as MCTS **stochasticity**.
- **DMCTS distillation — still ruled out, and it reframes the cap a third time.**
  A *deterministic, info-matched, strong* teacher distilled to only **0.40**
  agreement (vs cheating-mcts 0.37 — a marginal bump) and a **PPO-level** net
  (avg-hard3 0.52, vs mcts **0.21** — the teacher itself gets 0.46). So the cap is
  NOT the info gap (DMCTS proved it small) and NOT mainly stochasticity (determinism
  barely helped): a **search policy's move is the output of lookahead, which has no
  compact reactive (obs→action) form.** Greedy (simple heuristic) clones to 0.95;
  *any* search policy caps ~0.40. Same wall as self-play — reactive nets can't
  absorb planning. The only way to get it is **search at play time** (AlphaZero).
- **Shipped anyway:** `dmcts` is now a registry policy (`dmcts:K,I,seed,turns`,
  default K15/I30) — a strong, *fair* (non-cheating) search policy, ~as strong as
  the cheating `mcts:100`. Replay-deterministic (seeded per game).

### PPO ceiling — *not* a bug (net size × seat 2×2)
- Tested the mundane suspects directly (`baseline.md` "PPO is not under-capacity").
  **MLP size is not the cap — 256×256×256 < 64×64** (under-trained at 400k; default
  size is right). **Both-seat training helps the small net +0.06 (0.49→0.55)** —
  mostly a 2× efficiency win (0.55 ≈ shipped 800k-seat-0 0.556) + correct (eval is
  mirrored). **Action mapping sound** (BC-of-greedy 0.95). **vs mcts flat ~0.2 in
  all four arms** — no mundane fix closes the search gap, so the structural read
  (reactive nets can't plan) is tested, not assumed.
- **Landed:** `train`/`train-zoo` default to both-seat (`--both-seat`,
  `BattleEnv(seat_random=...)`).

### Prior (pre-investigation) context
- Cheating perfect-info `mcts:100` beats `greedy` 0.79 (it *plans*); distilling it
  into a reactive net plateaued (information gap) — `baseline.md`.

---

## Engine / data notes (facts discovered while building experiments)

- **Draft:** 30 rounds; **both seats pick from the SAME triplet each round**
  (`draft.py`). `draft_action(view, legal=[0,1,2]) → index`.
- **Battle start:** `start_battle` deals opening hands (4/5), grants player-1 the
  coin (`bonus_mana=1`), and runs turn 1. `_finish_draft` only *builds* the decks
  and flips phase to BATTLE — any custom draft→battle env MUST call `start_battle`
  (a prototype that skipped it trained on a degenerate turn-0 / empty-hand opening).
- **Card stats sign convention:** green items = positive (buff own minion);
  red/blue items = NEGATIVE attack/defense applied to the *enemy* (removal/damage).
  Charge sets a creature's initial `can_attack`. Ability mask order = `"BCDGLW"`.
- **View limits:** `CardView` exposes type / stats / abilities / readiness but NOT
  `card_draw` / `player_hp` / `enemy_hp` — so pure-utility items are invisible to
  draft heuristics.
- **Reproducibility:** engine is seed-deterministic; eval uses held-out seeds
  (`1_000_000+`) disjoint from training env seeds (`0,1,…`) to avoid leakage.

## Bottom line — the reactive-net wall (every imitation/training-method path is spent)

The whole investigation converges on one structural fact: **a reactive policy net
(obs → action, no lookahead) cannot reach the search policies' strength**, and
*none* of the ways to push it there work — because a search policy's move is the
output of planning, which has no compact reactive form.

| route | result |
|-------|--------|
| RL (budget, opponents, reward, obs, entropy, normalization, **net size**, **seat**) | flat — only the **deck** moved it (+both-seat is a 2× efficiency win, same ceiling) |
| **Self-play / league** (warm-start, both-seat, MCTS in pool) | flat over 480k, then down |
| **Distillation** of MCTS (cheating *or* DMCTS, positional *or* semantic, stochastic *or* deterministic) | caps ~0.40 agreement → PPO-level net |

Greedy (a simple heuristic) clones to 0.95; *any* search policy caps ~0.40. The
deck is the lever for the reactive net (shipped: `ppo+balanced` beats the ground
baselines, ~even with the *old* weak MCTS). The residual gap to the *strong* MCTS is
its **planning**, and the only architecture that gets planning is **search at play
time** — don't re-run the reactive routes.

## Open levers (next) — see `ppo-review.md` §8.4
Two paths remain, and they **compose** — substrate then algorithm:
- **Richer board encoding (substrate)** — the obs is a *flat* 308-d fixed-slot
  vector with no relations/derived tactics. Upgrade: **tokenize** the board (set/
  attention extractor), add **relational objects** (attacker × target legality/
  trade matrix), add **engineered tactical scalars** (opponent Guard count, reachable
  face damage, friendly lethal available, exposed-to-lethal, mana left). This is a
  *different kind* of information than §3.4's flat-scalar A/B (relations + shallow
  lookahead, not more raw scalars), so that null doesn't rule it out. Sharpens the
  reactive net and — the real payoff — yields a much better policy/value net for ↓.
- **SEARCH in the loop (AlphaZero-lite, the real lever)** — a policy+value net
  *guides* MCTS (PUCT priors over the 155 semantic slots + value leaf), trained by
  self-play of the **search** (not the raw net). The only path to MCTS-level
  *planning*. Building blocks already exist: fast `_clone_battle` forward model,
  `dmcts` (a fair determinized search — swap its heuristic leaf for the net value +
  add priors), the semantic action space, and the heuristic leaf to bootstrap.

## 2026-06-26 — full-roster tournament + rating-estimator fix

### Full-roster tournament (`docs/baseline.md`)
- One round-robin over the whole roster — 5 baselines + `mcts:100` (cheating) +
  `azlite:100` + `dmcts` (fair determinized) + `ppo:runs/ppo-shuffled-pool.zip` —
  400 games/pair, `--seed 0`. **First `dmcts` numbers in the docs.**
- **`azlite:100` is the only undefeated policy**: beats `mcts` 0.56, `dmcts` 0.60,
  `ppo` 0.76, and all baselines 0.69–1.00. Head-to-head order:
  `azlite > mcts > dmcts > ppo > {baselines} > random`.
- **`dmcts` debut** — the one *fair* search: beats every baseline and `ppo` (0.74)
  but loses to `mcts` (0.46) and `azlite` (0.40). That gap is partly cheating, not
  skill — `azlite`/`mcts` have perfect foresight (see the searcher audit below).
  `dmcts` vs `ppo` 0.74 is the honest search-vs-reactive number.
- The three searches crush the ground baselines far harder than `ppo` (avg-hard3:
  mcts 0.770, azlite 0.757, dmcts 0.750 vs ppo 0.593).

### Rating-estimator bug — fixed
- **`elo_from_results` / `openskill_from_results` were single-pass and
  order-dependent.** Fed the tournament's pair-grouped game order they **inverted**
  this roster: rated `ppo` #1 (openskill 64.87 / Elo 3177) and the undefeated
  `azlite` #3.
- **Measurement bug, not non-transitivity:** shuffling the game order moved
  `azlite` 3rd→1st and `ppo` 1st→4th, and a convergent Bradley-Terry fit recovered
  the matrix order.
- **Fix:** Elo → order-free **Bradley-Terry MM**; openskill → **seeded
  shuffle-averaging** of the online PlackettLuce (`locma/stats/`). Ratings now
  match the matrix (`azlite` #1). The "ratings mislead, read the matrix" refrain
  across the docs partly misattributed this bug to non-transitivity; genuine
  non-transitivity remains only in the baseline rock-paper-scissors. Rating tables
  in older `baseline.md` sections predate the fix (matrices stand).

## 2026-06-26 — searchers cheat; search-as-training-opponent; curriculum endpoint

### Searcher honesty audit (`docs/searchers-fiasco.md`)
- **`mcts` and `azlite` are perfect-*foresight* cheaters.** They clone the real
  `GameState` and simulate forward, leaking not just the opponent's hidden hand but
  **both decks' shuffled order** — every future draw (decks shuffled once at draft
  `draft.py:50`; draws are deterministic `pop(0)` `battle.py:41`; the clone copies
  both decks in order). They play a fully revealed, deterministic game.
- **`azlite` was mislabeled "non-cheating"** in `baseline.md` and memory; its own
  docstring says `Perfect-information (cheating)`. Corrected both.
- **`dmcts` is the only fair searcher** (resamples the opponent's hidden hand+deck).
  Had a residual self-leak (kept its *own* real deck order → knew its own future
  draws); **fixed** — `_determinize` now reshuffles the own deck (`reshuffle_own`,
  default on). **A/B: the self-leak was worth ~zero** — fair vs leaky head-to-head
  0.51 [0.46–0.56], avg-hard3 0.727 vs 0.724, all per-cell deltas within noise. So
  knowing your *own* future draws barely changes the current move; the damaging
  leak is *opponent* info, which only `mcts`/`azlite` have. `dmcts` now strictly
  leak-free; recorded numbers stand (~0.73 avg-hard3, beats `ppo` ~0.71).
- **Deployment:** against a real `(visible_state, action)` server, reactive policies
  and `dmcts` work (dmcts needs its own engine as a model); `mcts`/`azlite` can't be
  reproduced — the info they need isn't on the wire. The honest ranking is
  `dmcts > ppo/reactive > baselines > random`.

### Search policies as TRAINING opponents (`BattleEnv`)
- `BattleEnv` now passes the forward-model `state` to the opponent
  (`battle_action(view, legal, self.gs)`), mirroring the play harness
  (`engine.py:135`) — heuristics ignore it, search policies need it. Un-defers the
  "search opponents" TODO; enables training a PPO against `mcts`/`azlite`/`dmcts`.

### PPO curriculum endpoint vs `azlite` — flat (prediction held)
- Warm-started the strongest zoo model (`ppo-shuffled-pool.zip`) and continued
  **+200k steps vs `azlite:100`**. Result: **no movement** — avg-hard3 0.594 →
  0.597 (flat); vs `azlite` 0.237 → 0.270 (within ±0.055 noise, still ~73% loss).
- Confirms the structural ceiling: a reactive net can't plan, the training opponent
  doesn't change that — and it especially can't learn to beat a **perfect-foresight
  cheater** reactively. ~95 min at ~35 fps (the search opponent is the cost). Lever
  remains search at play time (`ppo-review.md` §8), not the training opponent.

## 2026-06-26 — PPO2: tokenized observation + self-attention (richer-encoding lever)

Built the `ppo-review.md` §8.4A "richer board encoding" lever as `obs_mode="token"`
(additive; `obs_mode="flat"` stays the default + A/B control): per-card tokens (+ a
learned **card-id** embedding the flat obs discards) + computed tactical scalars + a
self-attention extractor (`TokenSetExtractor` on `MultiInputPolicy`). Two bugs found
along the way, then a fair A/B. Verdict: **parity / a slight lean ahead under the
curriculum, within seed noise — a secondary lever.** Full analysis in §8.4A.

### Two non-obvious bugs
- **Slot-indexed action space ⇒ the encoder must be slot-addressable, not
  permutation-invariant.** v1 pooled tokens to a permutation-invariant CLS vector, but
  Summon/Use/Attack are indexed by hand/board *slot*, so order-invariant features
  can't express slot-specific play (pilot: token 0.46 < flat 0.55). Fix: per-slot
  positional embedding + **flatten** the per-slot transformer outputs (each slot at a
  fixed offset) — attention's relational mixing, slot identity preserved.
- **The bigger net needs gentler PPO.** Default LR 3e-4 drove `approx_kl` to 0.10–0.15
  (clip ~0.4) and the token net *degraded* with training (0.382 → 0.333 @300k). LR 1e-4
  + `target_kl=0.025` tamed it (→0.565). Plumbed `learning_rate`/`target_kl` through
  the trainer + CLI (flat defaults 3e-4/None unchanged).

### Overfitting → the curriculum is the fair test
Trained vs a single opponent (max-attack) the larger token net **overfits it** (strong
vs max-attack, weak vs unseen scripted) while the tiny flat MLP generalizes — so
single-opponent eval is biased against token. The zoo curriculum (4 opponents) is the
fair, overfit-resistant test.

### A/B verdict (curriculum, avg-hard3 = mean win rate vs scripted/max-guard/max-attack)
- Full A/B: 200k×4 = 800k/arm, 2 seeds, 400 games/opp. **token 0.588 vs flat 0.573**
  (+0.015; greedy +0.025) — but seeds disagree (s0 flat 0.592 > 0.562; s1 token
  0.614 > 0.555); per-seed spread ~0.03–0.04 > the gap, so **not significant at n=2**.
  Across 3 independent curriculum runs token won 2/3 (mean ≈ +0.02), strongest vs
  greedy.
- **Bottom line:** the corrected, stably-trained tokenized+attention encoding **reaches
  and marginally exceeds flat under the curriculum, but within 2-seed variance, at ~4×
  train cost.** A real but *secondary* lever — it doesn't break the reactive ceiling;
  its bigger promised value is as a substrate for search (§8.4B, untested). Code is
  additive behind `obs_mode="flat"`; flat baseline untouched. See `baseline.md`
  ("PPO2") and `ppo-review.md` §8.4A.

## 2026-06-27 — Distill search → PPO2 (PR #18 redo): the obs is NOT the BC ceiling

Re-ran the PR #18 distillation (behavior-clone a search teacher into a reactive net)
with the new tokenized obs + a matched flat control + a fair teacher, to find what
actually caps it. Added a token mode to the practicum/distill pipeline (`--obs-mode
token`; teacher decisions are obs-independent, so `action`/`mask` are identical to
flat — verified byte-for-byte).

**All distills land in the same place — agreement ~0.37, avg-hard3 ~0.54:**

| distilled net | teacher | obs | top-1 agreement | avg-hard3 |
|---|---|---|---|---|
| flat (matched) | mcts:100 (cheater) | flat | 0.370 | 0.548 |
| token | mcts:100 (cheater) | token | 0.366 | 0.543 |
| dmcts | dmcts (fair) | token | 0.372 | 0.535 |
| from-scratch PPO2 (RL) | — | token | — | **0.588** |
| teacher strength | mcts / dmcts | — | — | **~0.73** |

- **The observation is not the ceiling.** Flat ≈ token on the *same* 35k mcts games
  (0.370/0.548 vs 0.366/0.543). The earlier "token lifts 0.25→0.37" was a **cross-run
  artifact** vs PR #18's old number; the real gain over PR #18 (0.25/0.29) is the
  **semantic action space + enriched obs (PR #19), which landed *after* PR #18 distilled
  into the old positional space** — not tokenization. (Matched controls earn their keep.)
- **The teacher's cheating is not the ceiling either.** The *fair* dmcts distilled no
  better than the cheating mcts (0.372/0.535 vs 0.366/0.543).
- **Distill does not beat from-scratch.** Every distilled net sits at/just-below
  from-scratch RL (0.588, within seed noise) and inherits **~none** of the teacher's
  ~0.73 edge.
- **Verdict:** the cap is **behavior-cloning a search policy into a reactive net**
  itself — base-, obs-, and teacher-fairness-independent. Strengthens §8.3/PR #18: more
  imitation data won't cross the planning gap; only search-in-the-loop (§8.4B) does.
- *Caveat / only untested lever:* dmcts was recorded **non-deterministically** (samples
  determinizations → label noise that can cap agreement). `DMCTSBattlePolicy.deterministic=True`
  exists for distillation but isn't exposed in the registry spec; a clean deterministic-dmcts
  practicum is unrun.
- *Op note:* `mcts:100` recording is now ~30× faster than PR #18's measurement (the fast
  `_clone_battle` landed after it) — 35k examples in ~75s, not ~40 min. The `random`
  opponent is excluded from the dmcts practicum (degenerate states, noise, not in avg-hard3).
  Pipeline additive behind `--obs-mode flat`; see `baseline.md` ("Distillation").

## 2026-06-27 — Self-play of token PPO2: responds where the flat net decayed (then plateaus)

Quick probe (throwaway scripts, no infra): warm-start the zoo-curriculum PPO2
(`ab-token-s1`) and run self-play rounds — each round trains 200k against a per-episode
mix of a *frozen self* + the ground baselines (conservative `target_kl=0.025` inherited
from the tuned base). Matched eval, 300 games/opp, seed 0:

| stage | avg-hard3 | Δ |
|-------|-----------|---|
| base (`ab-token-s1`) | 0.601 | — |
| self-play r1 | 0.632 | +0.031 |
| self-play r2 (self = r1) | 0.639 | +0.007 |

- **Real but front-loaded + plateauing:** +0.031 (r1, ~2.6σ, consistent across all four
  opponents) then +0.007 (r2, within noise) → converges ~0.64, does **not** compound.
- **Corrects §8.3 for the new architecture:** the flat-net league *decayed* under
  self-play; the slot-addressable token net is *sharpened* by it (capacity + the
  conservative KL cap keeping updates stable). `selfplay-r2` (0.639) is the **strongest
  reactive net** produced — above from-scratch RL (0.588) and the curriculum base (0.601).
- **Ceiling intact:** even 0.64 is well below the search policies (dmcts/azlite ~0.73);
  self-play adds no planning. See `ppo-review.md` §8.3 update.

## 2026-06-27 — netdmcts (AlphaZero-lite Phase 1): a FAIR policy that beats the cheaters

Built the §8.4B lever's Phase 1: fair net-guided determinized PUCT. Shared PUCT core
(`puct.py`, refactored out of azlite), module-level `determinize` (from dmcts), a
`NetOracle` (the token net's masked policy-prior + value, reading only `BattleView`),
and `NetGuidedDMCTSBattlePolicy` = PUCT over K determinized worlds with the net oracle
(`netdmcts:K,I,c_puct,model_path`). Frozen oracle (the self-play net `selfplay-r2`); no
AZ training yet.

**Result (avg-hard3, 2 seeds, 50 games/opp):**

| policy | fair? | mean avg-hard3 |
|--------|-------|----------------|
| **netdmcts:8,40** (net oracle) | ✅ | **0.817** (s0 .820 / s1 .813) |
| dmcts:15,30 (heuristic oracle) | ✅ | 0.697 |
| raw net selfplay-r2 | ✅ | 0.639 |
| azlite / mcts (cheating refs) | ❌ | 0.741 / ~0.74 |

- **Net oracle ≫ heuristic oracle in fair search:** +0.12 avg-hard3 over dmcts on both
  seeds (+0.127, +0.113), at *less* search budget (320 vs 450 sims).
- **Search ≫ raw net:** +0.18 over the same net used bare (0.639 → 0.817) — the planning
  the reactive net structurally lacked.
- **Fair beats cheating:** a no-hidden-info policy (0.817) tops the perfect-foresight
  `azlite`/`mcts` (~0.74). Iteration-efficient (strong at I=20; 0.875 at I=80).
- **Verdict:** §8.4B confirmed — search-in-the-loop with a net oracle is the lever, and
  it's fair/deployable (forward-simulates determinized worlds; never sees the opponent's
  hand). Strongest policy in the project. **Phase 2 justified** (self-play *of the
  search*: AZ targets = visit distribution + outcome, iterated; the oracle here was only
  RL-trained). All behind clean refactors (azlite + dmcts tests unchanged). See
  `ppo-review.md` §8.4B, `baseline.md` ("netdmcts").

## 2026-06-27 — netdmcts Phase 2 (AlphaZero self-play of the search): no gain at this budget

Built the full AZ loop: fair net-guided-dmcts self-play generates `(token-obs, search-visit
policy, game outcome)`; `az_train` warm-starts the net and trains BOTH heads (soft-CE policy →
visit distribution, MSE value → outcome); `az_selfplay` iterates with a **composite gate**
(adopt iff the new net beats its parent head-to-head AND doesn't regress avg-hard3; keep-best,
early-stop). New code: `puct_search(root_noise=)` (Dirichlet, default-off, behaviour-preserving),
`selfplay.record_selfplay`, `az_train.az_train`, `azloop.az_selfplay`, CLI
`record-selfplay`/`az-train`/`az-selfplay`. Also `NetOracle` single combined forward (shares the
self-attention trunk between priors+value → ~2× search speedup, output-identical).

**Run (3 iters; 100 self-play + 40 baseline games/iter; gen K=6,I=40; eval K=8,I=40; 277 min):**

| iter | avg-hard3 (gate, 12/opp) | h2h vs parent (16g) | gate |
|------|--------------------------|---------------------|------|
| 0    | 0.806 | 0.688 | ADOPT (best) |
| 1    | 0.792 | 0.469 | reject |
| 2    | 0.736 | 0.469 | reject → early-stop |

Best = `az-net-0` (one round of AZ training). **Final confirm (larger n): avg-hard3 0.830
(50/opp) vs Phase-1 0.817; h2h 0.54 (net-0 vs frozen `selfplay-r2`, 100 games).**

- **Within noise on both axes.** 0.830 vs 0.817 = +0.013 (50/opp 95% CI ≈ ±0.06); h2h 0.54 over
  100 games (CI ≈ ±0.10, includes 0.50). The gate's 0.688 h2h was 16-game noise; the 100-game
  confirm regressed it to a coin flip.
- **No compounding.** Iters 1–2 didn't beat their parent; iter-2 avg-hard3 drifted to 0.736. The
  composite gate correctly kept the best and early-stopped.
- **Verdict:** one round of fair AZ self-play training neither clearly helps nor hurts at this
  budget — frozen-oracle `netdmcts` (0.817) is already near this kit's ceiling. Infrastructure
  validated (gate disciplined, fairness preserved end-to-end, early-stop fired). Consistent with
  the project pattern: **search-in-the-loop (Phase 1) was the lever; training the search further
  gives diminishing returns at small budget.** A larger run could tighten the CIs but the signal
  is null. See `ppo-review.md` §8.4B, `baseline.md` ("netdmcts Phase 2"),
  `docs/netdmcts-phase2-design.md`.

## 2026-06-28 — draft benchmark (`draft-bench`): isolate deck-building skill; `balanced` is robustly best

New tool `locma draft-bench` (`locma/harness/draft_bench.py`) ranks **draft** policies
in isolation: both seats are piloted by the SAME battle policy and differ ONLY in
their draft. Because the draft deals both seats from the same shared, non-depleting
triplet stream, on a fixed seed both get identical offers — so this is a clean
**paired comparison** and the win-rate edge is pure deck quality. Calibration: a
self-duel is **exactly 0.5000** (mirror cancels seat advantage perfectly; verified
for ground/greedy/azlite/netdmcts pilots). Cleaner than the old "PPO × draft sweep" (which
played a fixed battle net vs the heterogeneous baselines — conflating draft with the
battle matchup). Methodology in `docs/draft-benchmark.md`.

**Pilot choice is load-bearing — weak heuristics disagree.** Ranking by avg win
rate vs field, all 7 built-in drafts:

| draft       | ground (n=600) | ppo deploy (n=160) | dmcts fair (n=50) | netdmcts fair-top (n=50) | azlite (n=120) | greedy-battle (n=600) |
|-------------|:--------------:|:------------------:|:-----------------:|:------------------------:|:--------------:|:---------------------:|
| balanced    | **0.650** | **0.624** | **0.647** | **0.610** | 0.613 | 0.567 |
| random      | 0.471 | 0.476 | 0.643 | 0.543 | **0.674** | 0.080 |
| max-guard   | 0.591 | 0.535 | 0.450 | 0.453 | 0.526 | 0.393 |
| max-attack  | 0.527 | 0.542 | 0.460 | 0.463 | 0.457 | **0.694** |
| max-defense | 0.434 | 0.466 | 0.453 | 0.480 | 0.413 | 0.576 |
| weighted    | 0.414 | 0.407 | 0.420 | 0.467 | 0.428 | 0.574 |
| greedy      | 0.412 | 0.450 | 0.427 | 0.483 | 0.390 | 0.616 |

- **`balanced` is the robustly best draft** — Condorcet winner (beats all six others
  head-to-head) under `ground`, `ppo`, **and `netdmcts`** (the kit's strongest *fair*
  policy; it sweeps all six H2H 0.56–0.64 and this **replicates at seeds 0 and 1** —
  12/12). #1 by avg win rate under the *weaker* `dmcts` (edging `random` 0.647 vs
  0.643, a tie) and close 2nd under `azlite`, though NOT Condorcet there (it loses
  the `random` head-to-head — see below). The shipped `greedy` draft and `weighted`
  rank **≤ a random draft** under every serious pilot — a clean, battle-isolated
  confirmation of "greedy is the worst draft / the deck is the lever."
- **The `greedy`-battle pilot inverts the ranking** (max-attack #1, max-guard
  near-last) — a weak heuristic over-rewards its own style. Lesson: rank drafts under
  a **strong** pilot and report across several; never trust one weak pilot.
- **`random`'s strength is a weak/cheating-searcher artifact.** It carries items the
  searcher uses, so it looks strong under the *weaker* `dmcts` and the *cheating*
  `azlite`, edging `balanced` head-to-head at seed 0 (`balanced` wins only 0.40
  azlite, 0.44 dmcts) — but that does **not** replicate (fresh seed 0.49 / 0.59).
  Under the **strongest fair pilot (`netdmcts`)** the effect collapses: `random` is a
  clear #2 (0.543) and *loses* to `balanced` 0.39 at both seeds. Genuinely strong
  *fair* play prefers `balanced`'s structure over a bag of items.
- **No tuning lever robustly beats `balanced`.** A cheaper curve wins only under
  `ground` (ties under `ppo` — overfit to aggression). A direct item-discount probe
  (12→0, adding premium removal to `balanced`'s structure) **ties** `balanced` under
  both `azlite` (n=160) and `dmcts` (n=80) and is no better under `ppo` — so the
  item-light recipe is robust, not an overfit; those item-variants still beat `random`
  under `dmcts` (0.59–0.62), i.e. structure is what wins. **`balanced` is the best
  drafting strategy under current rules.** See `docs/baseline.md` ("Draft-bench").

## 2026-06-30 — `locma-replay/3`: compact on-disk replay format

New replay format `locma-replay/3` (`locma/harness/replay_codec.py`,
`replay_store.py`) shrinks recorded games while staying **byte-for-byte
lossless** on round-trip (`get_replay(write_replay(rep)) == rep`).

- **Catalog-fill:** card refs compact to `[iid, card_id]` (or `[iid, card_id,
  dev]` when a field deviates from the printed card — buffs, damage, an
  unknown `card_id`, or non-default board readiness); `expand_card` refills
  base stats from the loaded card catalog on read.
- **Carry-forward delta:** each battle step stores only what changed since the
  previous full state (`diff_state`/`apply_delta`) instead of a full snapshot;
  `battle_start` carries one compact keyframe and every later state is
  reconstructed by replaying deltas forward.
- **Draft/battle phase framing:** explicit `draft_start`/`draft_end` and
  `battle_start`/`battle_end` markers replace the old `open`/`close` lines.
- **`get_replay` rehydration:** maintains a `running` full-state dict, applying
  each delta in turn and deep-copying out `opening`/`closing`/per-step
  `state` — so callers see the same shape as `/2` regardless of on-disk
  format.
- **`/2` back-compat:** `_encode_v2`/the `open`/`close`/full-`state` read path
  are untouched; `write_replay` dispatches on `header["format"]`, so existing
  `/2` files and code paths (and the web viewer, which only ever sees
  `get_replay`'s expanded dict) are unaffected.
- **`cardlist_version` guard:** the header records a digest of the printed
  card stats used as the compression dictionary
  (`replay_codec.cardlist_version()`); `get_replay` now `warnings.warn`s if a
  `/3` file's stored digest no longer matches the running catalog (card stats
  were rebalanced since the replay was recorded), flagging that rehydrated
  card stats may not match what was actually played.
- **Size win, measured on the real `replays/` corpus** (28 games recorded
  under `/2`, re-encoded to `/3` via `get_replay`→`write_replay`):
  2.27 MB → 0.56 MB, a **~75% reduction** (range 73–77% per file; the
  brief's ballpark "~86%" was optimistic — delta+catalog-fill compounds less
  on real battles with frequent board churn than on a synthetic small fixture,
  but the win is still large and consistent across every sample game).

Tests: `tests/test_replay_codec.py` (compact/expand round-trip, deviation
fields, unknown-card passthrough, delta apply/diff, `cardlist_version`
stability), `tests/test_replay_store.py` (`/3` phase framing + delta-only
turns, `/2` path unchanged, lossless round-trip incl. buffed/damaged/Guard
cards, `cardlist_version` mismatch warning, `/3` strictly smaller than `/2` on
a card-bearing fixture), `tests/test_replay_stream.py` (built replays are
`/3` with a stamped `cardlist_version` and round-trip).

## 2026-06-30 — PPO ceiling study: Gate 0 (throughput) + kickoff on the RTX 4080 box

Tooling (Tasks 1–10 of `docs/ppo-ceiling-study-plan.md`) confirmed green on the
GPU box: `pytest tests/ -q` with `--extra ml --extra dev --extra sweep` →
**464 passed, 1 skipped** (553s; the ML-gated tests that actually train small
models now run instead of skipping, hence the longer wall time vs. a CPU-only
box).

**R0 — Gate 0 throughput spike** (`scripts/puffer_bench.py`, sb3 arm; PufferLib
not installed so that arm is `(absent)` — deferred, non-blocking per the plan):

| config | SPS |
|---|---:|
| sb3 token n_envs=1 | 6280 |
| sb3 token n_envs=4 | 8695 |
| sb3 token n_envs=8 | 12072 |
| sb3 token n_envs=16 | 14009 |

Env-stepping throughput flattens at `n_envs=16` (this box has 20 logical
cores, so that's already near the practical ceiling — pushing higher would
oversubscribe, not help). A short real-training comparison (16k steps, token
net) gave **2390 fps on `device=cuda`** vs **1638 fps on `device=cpu`** — CUDA
wins but not dramatically, since the token net is small; GPU host↔device
transfer eats into the advantage. Once gradient updates kick in (not just
rollout collection) sustained fps drops further to **~390** on this box —
that's the realistic full-run rate, not the rollout-only 2390 headline.
**Decision: `n_envs=16`, `device=cuda` for the sweep and B0 runs.** Since CPU
is already ~saturated by one run's 16 env workers, B0's 3 seeds run
**sequentially**, not as 3 parallel processes (parallelizing would mostly add
contention here, not speedup).

**PufferLib arm: no native Windows support, confirmed across 4 release
generations** — not a version-picking problem, it just doesn't ship for this
platform:
- `pufferlib==3.0.0` (current): sdist build raises `ValueError: Unsupported
  system: Windows` outright — it compiles CUDA/C++ extensions and vendors
  raylib/Box2D binaries published only for Linux/macOS.
- `pufferlib==2.0.x`: identical Windows block in `setup.py`.
- `pufferlib==1.0.1` / `0.7.3` (older, no explicit OS gate): pin ancient deps
  (`numpy==1.23.3`, legacy `setuptools` build backend) that can't build
  against this box's Python 3.14.
- No Windows wheels published for any release on PyPI.
- WSL2 is enabled on this box but has no Linux distro installed; Docker isn't
  present either — standing one up (distro + CUDA passthrough + build) was
  judged not worth doing just for a throughput spike.

**Decision: defer the PufferLib arm to a future macOS (CPU/MPS) session** —
there's reportedly a newer `4.0` release on GitHub (not yet on PyPI) worth
checking then too. `scripts/puffer_bench.py`'s `puffer_sps()` already
degrades gracefully to `(absent)` when the package isn't importable, so this
doesn't block anything. **The study proceeds on sb3 regardless, per the
plan's Global Constraints — Puffer is informational only.**

**R1 — B0 baseline (×3 seeds, full 800k steps, `n_envs=16`, `device=cuda`,
`lr=1e-4`, `target_kl=0.025`, token obs):**

| run | wall time | avg_hard3 (40 held-out seeds × 25 games/opp) |
|---|---:|---:|
| `b0_s0.zip` | 25.3 min | 0.6447 (range 0.607–0.707) |
| `b0_s1.zip` | 23.8 min | 0.6448 (range 0.573–0.707) |
| `b0_s2.zip` | 24.2 min | 0.6285 (range 0.567–0.700) |

**B0 mean avg_hard3 ≈ 0.639** (3-seed average) — a touch higher than the
~0.60 plateau the design doc's motivating anecdote cited, but in the same
neighborhood; this run's own number is now the actual verdict baseline, not
the anecdote. All three seeds trained clean (no crashes, KL early-stopping
engaged as expected with `target_kl=0.025`).

**Eval-harness speed note (matters for R4/R5):** `avg_hard3_per_seed` /
`run_match` plays games **one at a time, unbatched** — computing this table
(3 models × 40 seeds × 3 opponents × 25 games = 9000 games) took **~40
minutes**, noticeably slower than the 800k-step *training* runs themselves
(~25 min each). R4's rigorous confirm (3–5 survivors + B0, same 40×25
protocol) and R5 (repeat for obs-v1) will each pay this cost again per
candidate set — budget for it, and it's a candidate for a future batching
optimization if the sweep produces many survivors to confirm.

R1 done.

**R2 — Phase 1a Optuna sweep, first attempt (150 trials planned):** started
at `n_envs=16`, `total_steps=320_000`/trial, `n_games=120`. Trial 0 (the
enqueued B0 point at this reduced budget) completed in **~27 min** at
value=0.5611; trial 1 (a TPE sample) completed in **~32 min** at
value=0.5583, closely tracking trial 0's own eval curve — its scary early
0.02 eval was just pre-learning noise, not a bad config, so Hyperband
correctly didn't prune it. **With only 1-2 sibling trials in the bracket,
Hyperband had no basis to prune anything yet — every trial ran to ~full
budget.** Observed pace: **~2.5 trials/hour**, i.e. **~60 hours for the full
150 trials** — not viable for this session. Roughly a third of each trial's
wall time is the unbatched `WinRateEvalCallback` eval games (6 checkpoints ×
360 games each), not training — same cost structure flagged in the R1 eval
note.

**Decision: stopped the 150-trial run after 2 trials (SQLite study
`sqlite:///runs/ceiling.db`, `phase1a`, is resumable — nothing lost) and
restarted at `--n-trials 30`** on the same study — an overnight-scale budget
(~13-16h at the observed per-trial pace) instead of a 2.5-day one. Trial
numbering continues from where it left off (trial 2 was killed mid-run and
is orphaned/incomplete in the DB, harmless). Best-so-far after 2 completed
trials: **trial 0 at 0.5611** (the B0 point) — essentially tied with trial 1,
so no HP lift has separated from baseline yet at this early stage.

**R2 crashed at trial 8/9 (00:XX PDT) — `MemoryError` from a `SubprocVecEnv`
worker subprocess failing to import torch.** Root cause: `train_zoo`'s
opponent-curriculum swap (`model.set_env(_build_env(...))` on each of the 4
phases) never closed the *previous* phase's `VecEnv` — `set_env` only
replaces the reference, so the old `SubprocVecEnv`'s worker processes were
orphaned and relied on Python's GC to eventually reap them. GC-triggered
subprocess cleanup is unreliable on Windows, so within one long-running
`locma sweep` process (many trials × 4 phases × `n_envs=16` workers each),
workers piled up: by the crash, **113 orphaned `python.exe` processes at
~535MB each (full torch import per worker) had consumed ~60GB, leaving only
~8GB free out of 68GB**, and the next subprocess spawn failed outright trying
to import torch. `train_zoo`'s end-of-function path had the same bug for the
*final* opponent phase (its `VecEnv` was never closed at all, not even at GC
time until the caller's `model` reference dropped).

**Fix (`d5b5a18`):** `train_zoo` now explicitly `.close()`s the outgoing
`VecEnv` right after `set_env()` on each curriculum swap, and closes the
final phase's `VecEnv` after `model.save()`. Verified: the 10 tests exercising
`train_zoo`/`model.learn` paths still pass, and no `python.exe` processes
remain after that test run (previously they would have accumulated across
the run in the same way). Stopping the sweep task via `TaskStop` also fully
reclaimed the leaked ~52GB in one shot (kills the whole process tree, not
just the parent) — worth remembering as the recovery move if this class of
leak recurs elsewhere.

Study state at the crash: 8 real trial slots resolved since the restart
(trials 0,1,3,4 complete, 5 & 7 pruned, 6 complete, 8 in-progress when
killed) — **best value still trial 0 (B0 point) at 0.5611**, no HP config has
separated from baseline yet. Nothing is lost; `sqlite:///runs/ceiling.db` has
all completed/pruned trials and is resumable.

**The first fix was incomplete — same leak recurred within ~30 min of
resuming (`8af3383`).** Root cause: the first fix (`d5b5a18`) only closed the
outgoing `VecEnv` on the normal `set_env()` swap path and after
`model.save()`. It missed that **pruning itself works by raising
`optuna.TrialPruned()` from inside `WinRateEvalCallback._on_step()`, which
unwinds straight out of `model.learn()` and out of `train_zoo()` before
either close call is ever reached.** Once pruning started actually working
(most new trials pruned rather than run to completion — 6 of 8 new trials in
this window), that exception path became the *common* case instead of a rare
one, and the same worker pile-up recurred: **115 `python.exe` processes,
~8GB free of 68GB**, caught and stopped before a second crash. Fix: wrap the
opponent-phase loop in `try`/`finally`, closing whatever `VecEnv` is
currently live in the `finally` block so it fires on both the normal-save
path and any mid-phase exception (pruning or otherwise). Verified two ways:
(1) the same 10 `train_zoo`-dependent tests still pass, (2) a forced-exception
repro (raise inside a callback mid-phase, matching the pruning shape) shows
0 net new `python.exe` processes after the exception, vs. leaking under the
first fix.

**Lesson for next time:** a resource-cleanup fix that only covers the happy
path isn't verified until it's checked under the failure path that actually
matters in production — here, pruning-via-exception was the whole point of
Hyperband, so "no leak on a clean run" was never sufficient evidence.

Study state when caught (second time): 17 total trial slots, 6 complete + 8
pruned + 3 orphaned/incomplete (killed) — **best value trial 14 at 0.5694**,
a hair above B0's 0.5611 but not yet clearly separated (single trial, no CI).
Nothing lost; study remains resumable.

**R2 — Phase 1a Optuna sweep, final result.** Ran to completion cleanly with
both leak fixes in place — process count and free memory stayed flat through
24 pruned trials in the second half (vs. crashing after ~8-9 before), and
0 `python.exe` processes remained after the run exited normally.

**Study totals: 33 trial slots — 6 complete, 24 pruned, 3 orphaned/incomplete
(trials 2, 8, 16 — remnants of the two mid-run kills during the leak
incident, harmless).**

| trial | avg_hard3 (320k-step budget) | lr | target_kl | n_steps | batch | n_epochs | ent_coef |
|---|---:|---:|---|---:|---:|---:|---:|
| 14 | **0.5694** (best) | 7.57e-05 | 0.025 | 1024 | 128 | 10 | 0.0165 |
| 0 (B0 point) | 0.5611 | 1.00e-04 | 0.025 | 2048 | 64 | 10 | 0.0200 |
| 1 | 0.5583 | 1.41e-04 | 0.025 | 4096 | 64 | 7 | 0.0061 |
| 3 | 0.5583 | 1.41e-04 (dup of trial 1) | 0.025 | 4096 | 64 | 7 | 0.0061 |
| 4 | 0.5417 | 1.82e-04 | 0.03 | 2048 | 256 | 7 | 0.0017 |
| 6 | 0.5361 | 3.35e-05 | 0.05 | 2048 | 64 | 3 | 0.0011 |

fANOVA param importances weren't computed — `optuna.importance` needs
`sklearn`, which isn't installed, and with only 6 completed trials the
estimate wouldn't have been reliable enough to be worth adding the
dependency for.

**Honest read: no HP config separated meaningfully from B0.** Trial 14 beats
B0 by +0.008 at this reduced budget — nowhere near the design doc's +0.03
symmetric-verdict threshold, and this is a single stochastic eval per trial
(not the 40-seed paired-bootstrap protocol R4 uses for real verdicts), so
+0.008 is well within plausible noise. The completed-trial spread
(0.536–0.569) is narrow and doesn't show a clear HP-driven signal at all —
24 of 30 sampled configs got pruned early for being clearly worse, but among
the survivors nothing stands out. **This is consistent with "ceiling
confirmed, robust to core-PPO HP" rather than "headroom found,"** though a
real verdict requires R4's rigorous full-budget paired comparison, not this
reduced-budget sweep signal alone.

Next: R3 (Phase-1b arch sweep around the 1a winner) and R4 (rigorous confirm
of survivors at full 800k budget vs. B0, with the paired-bootstrap verdict)
are the remaining runbook steps — both need explicit go-ahead. Given how flat
the Phase-1a results look, R4 on trial 14 (the nominal best) might be the
more informative next step over R3 (arch sweep), since core-PPO tuning
hasn't shown a signal worth building an arch sweep around yet.

## R4 — rigorous confirm: trial 14 vs B0, full 800k budget — VERDICT #1

Retrained trial 14's exact HP config (`lr=7.57e-05, target_kl=0.025,
n_steps=1024, batch_size=128, n_epochs=10, gamma=0.99, gae_lambda=0.9,
clip_range=0.3, ent_coef=0.0165, vf_coef=0.5`, token obs V0) at full 800k
steps ×3 seeds — `runs/cand1_s{0,1,2}.zip`. Notably **faster to train than
B0** (~15 min/seed vs. B0's ~24-25 min), consistent with its smaller
`n_steps=1024` rollout. No leak issues — process count and memory stayed
flat across the whole run.

Ran `locma ceiling-eval --candidates runs/cand1_s0.zip,runs/cand1_s1.zip,runs/cand1_s2.zip --baselines runs/b0_s0.zip,runs/b0_s1.zip,runs/b0_s2.zip --seeds 40 --games-per-seed 25 --threshold 0.03`:

```
cand=0.617  B0=0.657  delta=-0.040  95% CI [-0.048, -0.032]
VERDICT: ceiling-confirmed
```

**VERDICT #1: ceiling-confirmed — and not a null result, a confirmed
regression.** The 95% CI is entirely negative and nowhere near crossing zero,
let alone clearing the design doc's +0.03 threshold for "headroom." Trial
14's HP config, which looked marginally better than B0 under the *reduced*
320k-step sweep budget (0.5694 vs 0.5611, a noisy +0.008), **reverses to a
statistically clear -0.040 disadvantage once retrained at the full 800k
budget.** This is a textbook budget-mismatch failure mode for reduced-budget
HP sweeps: a config's relative ranking at a truncated training budget doesn't
reliably predict its ranking at the full budget, and can flip sign entirely.

**Methodology correction to R1's B0 number:** this run's `B0=0.657` (via the
CLI's `_disjoint_eval_seeds` — anchors spaced by `games_per_seed` so each
`run_match` block is independent, the fix landed in `00aaad9` right before
this session) is the *correct* figure. R1's own ad-hoc `avg_hard3` script
(this session, before R4) used `range(1_000_000, 1_000_040)` — seeds spaced
by 1 — which produces **overlapping, autocorrelated `run_match` blocks**
(exactly the anti-conservative-CI bug `00aaad9` fixed in the CLI path, just
not in that one-off script). R1's reported **B0 mean ≈ 0.639 should be
treated as superseded/methodologically flawed; 0.657 (this run, disjoint
seeds, same 3 B0 models) is the trustworthy number.** Worth remembering:
always eval through `locma ceiling-eval` / `_disjoint_eval_seeds`, never
hand-roll the seed list again.

**What this settles:** no evidence of headroom from core-PPO hyperparameter
tuning — Phase 1a's flat sweep result plus R4's confirmed-negative retrain
of its best candidate both point the same way. The ~0.60-0.66ish avg_hard3
plateau looks like a real ceiling for this reactive architecture under
core-PPO tuning, not an under-tuning artifact. The remaining lever per the
design doc is **R5 (Phase 2: obs-encoding variants, e.g. token-v1)** — a
different kind of change (richer observation, not HP tuning) that this
result doesn't speak to either way. R3 (arch sweep around the 1a "winner")
looks like a bad use of compute now that its baseline config underperformed
B0 at full budget.

Next: R5 (Phase 2 obs-v1 vs B0, same +0.03 ruler) is the natural next step —
needs explicit go-ahead, not started.

## E1-E3 fixes + benchmark (2026-07-02, branch `fix/ppo-e1-e3`)

Implemented the top-3 items from `docs/PPO-REVIEW-FB.md` (commit `d60da7f`;
implementation by a subagent, reviewed line-by-line):

- **E1** `TokenSetExtractor` default `dropout` 0.1 -> 0.0 (SB3 collects
  rollouts in eval mode but runs `evaluate_actions` in train mode, so
  dropout>0 contaminates the PPO ratio).
- **E2** token-v1 `op_reachable`/`exposed_to_lethal` now sum the WHOLE
  opponent board (`start_turn` refreshes every creature; readiness-filtering
  excluded exactly last turn's attackers).
- **E3** eval-callback bucket-crossing gating; per-env seed stride (50k);
  `BattleEnv.reset` reseeds the opponent. One regression test per change;
  470 passed.

**Benchmark** (B0 recipe: token V0, lr=1e-4, target_kl=0.025, 800k zoo
curriculum, n_envs=16; 3 seeds/arm; verdicts via `locma ceiling-eval`
--seeds 40 --games-per-seed 25, paired vs `runs/b0_s{0,1,2}.zip`):

| arm | vs | delta | 95% CI | verdict |
|---|---|---:|---|---|
| `e1_s*` dropout-0, token V0 | B0 | **-0.037** | [-0.044, -0.030] | regression |
| `e2_s*` dropout-0, token V1-fixed | e1 arm | **+0.008** | [+0.002, +0.015] | real but sub-threshold |

(cand=0.620 / B0=0.657 for E1; cand=0.629 / base=0.620 for E2. Training was
~11 min/run this session vs R1's ~24 — unexplained, possibly KL-early-stop
frequency shifting without dropout noise; diagnostics in
`runs/e1e3/train_*.log`.)

**Honest read of E1: removing dropout at the tuned recipe made the model
WORSE, clearly (CI entirely negative).** The mechanistic claim stands — the
ratio contamination is real — but its net effect at the *gentle* B0 recipe
(lr=1e-4 + KL cap) was evidently beneficial regularization, not just noise.
Two confounds noted: (a) the e1 arm bundles the E3 seed-stride hygiene (b0
models predate it); (b) without dropout noise `approx_kl` reads lower, so
the `target_kl=0.025` early-stop fires later -> effectively MORE update
epochs per batch -> a more aggressive optimization regime than the recipe
was tuned for. The overnight battery below discriminates.

**E2 read:** the corrected V1 threat scalars give a small, CI-clean lift
(+0.008) over V0 at matched recipe/dropout - the first obs-side change to
produce a nonzero paired delta, but far under the +0.03 headroom bar. Worth
re-testing at whatever recipe wins the dropout question.

**Overnight battery (N-arms, 3 seeds each, all evaluated paired vs B0):**

- **N0** dropout=0.1 explicit + current code (isolates the E3 stride
  confound: if N0 ~= B0, the -0.037 is all dropout).
- **N1** lr=3e-4, target_kl=None, dropout=0 (the config the worklog
  declared broken - retested without the alleged confounder).
- **N2** lr=3e-4, target_kl=0.025, dropout=0 (separates LR from KL cap).

Decision rule: if N0 ~= B0 and N1/N2 < B0, dropout was a *feature* at this
scale - revert the E1 default to 0.1 (keeping the mechanism documented) and
the ceiling verdict hardens. If N1 or N2 >= B0, the "gentle PPO" recipe was
compensating for dropout and the HP story reopens.

## N-battery results + dropout revert (2026-07-02, early morning)

All three arms trained and evaluated clean (paired vs `runs/b0_s{0,1,2}.zip`,
40 seeds x 25 games):

| arm | config | delta vs B0 | 95% CI |
|---|---|---:|---|
| N0 | dropout=0.1 explicit, current code | -0.009 | [-0.016, -0.002] |
| N1 | lr=3e-4, target_kl=None, dropout=0 | **-0.247** | [-0.256, -0.239] |
| N2 | lr=3e-4, target_kl=0.025, dropout=0 | -0.022 | [-0.029, -0.015] |

**Decomposition of E1's -0.037:** N0 shows the bundled E3 hygiene costs at
most ~-0.01 (tiny; possibly seed noise around the stride change), so
**dropout removal accounts for ~-0.028** (N0 0.648 vs e1-arm 0.620 at
identical code). **N1 refutes the "recipe was compensating for dropout"
hypothesis outright**: lr=3e-4 without a KL cap collapses to 0.410 even with
dropout OFF -- the token net's high-LR instability is intrinsic, not
dropout-borne. N2 confirms the KL cap (not low LR alone) does most of the
rescuing (0.635, within ~0.02 of B0).

**Decision (per the pre-committed rule): dropout was a feature.** Reverted
the TokenSetExtractor default to 0.1; the constructor comment + a pinning
test now document both the PPO ratio-noise caveat and the measured
regularization win, so this doesn't get "fixed" again without a paired eval.
The E2 v1-scalar fix and all E3 hygiene fixes stand (correctness, ~free).

**Ceiling verdict hardened:** B0 (token V0, lr=1e-4, target_kl=0.025,
dropout=0.1) = **0.657** remains the best reactive recipe. HP tuning (R2/R4),
dropout removal, and higher-LR regimes all regress or tie. Review-doc H4
("optimization pathology is self-inflicted") is now largely refuted on the
dropout front; H1/H2 (planning gap) stand untouched.

**Next (launched):** R5-proper -- token-v1 (fixed scalars) at the winning
recipe (dropout 0.1) x3 seeds, paired vs B0. The earlier +0.008 v1-vs-v0
delta was measured on the handicapped dropout-0 arms; this is the design
doc's Phase-2 verdict on the correct ruler.
