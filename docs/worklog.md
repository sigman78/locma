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
for ground/greedy/azlite pilots). Cleaner than the old "PPO × draft sweep" (which
played a fixed battle net vs the heterogeneous baselines — conflating draft with the
battle matchup). Methodology in `docs/draft-benchmark.md`.

**Pilot choice is load-bearing — weak heuristics disagree.** Ranking by avg win
rate vs field, all 7 built-in drafts:

| draft       | ground (n=600) | ppo deploy (n=160) | dmcts fair (n=50) | azlite (n=120) | greedy-battle (n=600) |
|-------------|:--------------:|:------------------:|:-----------------:|:--------------:|:---------------------:|
| balanced    | **0.650** | **0.624** | **0.647** | 0.613 | 0.567 |
| random      | 0.471 | 0.476 | 0.643 | **0.674** | 0.080 |
| max-guard   | 0.591 | 0.535 | 0.450 | 0.526 | 0.393 |
| max-attack  | 0.527 | 0.542 | 0.460 | 0.457 | **0.694** |
| max-defense | 0.434 | 0.466 | 0.453 | 0.413 | 0.576 |
| weighted    | 0.414 | 0.407 | 0.420 | 0.428 | 0.574 |
| greedy      | 0.412 | 0.450 | 0.427 | 0.390 | 0.616 |

- **`balanced` is the robustly best draft** — Condorcet winner (beats all six others
  head-to-head) under `ground` and `ppo`; #1 by avg win rate under `dmcts` (edging
  `random` 0.647 vs 0.643, a tie) and close 2nd under `azlite`, though NOT Condorcet
  there (it loses the `random` head-to-head — see below). The shipped `greedy` draft
  and `weighted` rank **≤ a random draft** under every serious pilot — a clean,
  battle-isolated confirmation of "greedy is the worst draft / the deck is the lever."
- **The `greedy`-battle pilot inverts the ranking** (max-attack #1, max-guard
  near-last) — a weak heuristic over-rewards its own style. Lesson: rank drafts under
  a **strong** pilot and report across several; never trust one weak pilot.
- **`random` is competitive only under the search pilots** (it carries items a
  searcher uses) and edges `balanced` head-to-head at seed 0 (`balanced` wins 0.40
  under azlite, 0.44 under dmcts) — but that does **not** replicate: on a fresh seed
  `balanced` vs `random` is 0.49 / 0.59, a seed-noisy near-tie (~0.5, n≈120–160). So
  across seeds they are co-leaders under search; `random`'s seed-0 lead was that seed
  plus it beating the weak heuristics.
- **No tuning lever robustly beats `balanced`.** A cheaper curve wins only under
  `ground` (ties under `ppo` — overfit to aggression). A direct item-discount probe
  (12→0, adding premium removal to `balanced`'s structure) **ties** `balanced` under
  both `azlite` (n=160) and `dmcts` (n=80) and is no better under `ppo` — so the
  item-light recipe is robust, not an overfit; those item-variants still beat `random`
  under `dmcts` (0.59–0.62), i.e. structure is what wins. **`balanced` is the best
  drafting strategy under current rules.** See `docs/baseline.md` ("Draft-bench").
