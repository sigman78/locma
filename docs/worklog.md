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
  old distillation cap (0.37 agreement) as MCTS **stochasticity**, not the info gap
  — the fix is a *deterministic* teacher (obs-seeded DMCTS, verified stable given
  the obs). Distillation re-test in progress.

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

## Open levers (next, ranked) — see `ppo-review.md` §8
- **Spent / ruled out:** action space (fixed), draft (balanced, shipped), reward
  shaping, observation richness, entropy, normalization, opponent-pool diversity,
  **self-play / league** (flat over 480k, even with both-seat training + MCTS in
  the pool), and longer horizon.
- **The remaining lever is SEARCH in the loop** — AlphaZero-style: a policy+value
  net guiding MCTS (priors + leaf value), trained by self-play of the *search*.
  The reactive PPO can't be self-played into planning; this is the one architecture
  that fits everything learned. Bigger build.
- **Distillation of the (now strong + fast) heuristic MCTS** is the cheap thing to
  re-try first — the old practicum/distill plateaued at ~0.25 agreement under the
  *positional* action space; the semantic space lifted greedy-cloning 0.69→0.95, so
  MCTS-cloning is worth re-measuring.
