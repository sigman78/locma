# PPO review — why the learned battle policy is weak

_Date: 2026-06-25_

## TL;DR

The MaskablePPO battle policy is weak for a **structural** reason, and the
structure that matters most is the **action representation**, not the training
budget, the opponent, or (mostly) the observation.

`BattleEnv` exposes a **positional** action space — `index_to_action(idx, legal)
= legal[idx]` — whose index→action mapping is *not even a function of the
observation*, paired with an **information‑free** action mask that encodes only
the *count* of legal actions. We proved this statically (code) and empirically,
then ran the causal test: **swapping only the action space** to a fixed
*semantic* space (stable per‑action indices + a mask that flags *which* concrete
moves are legal), with the same lossy 146‑d observation and everything else
identical, lifts the average win rate vs the four non‑`random` baselines from
**0.28 → 0.45** at 300k steps — turning "crushed by the ground baselines" into
"competitive." Enriching the observation on top adds **nothing**.

So the earlier verdict ("the ceiling is structural") was right, but the lever it
named (observation / reward) was only the *secondary* one. The primary lever is
the action encoding.

This review reproduces the documented ceiling as a control, isolates the cause
with three experiments, and ranks the fixes.

---

## 1. Background — the ceiling this explains

Two prior studies (see `docs/baseline.md`) established a hard ceiling that no
amount of *training effort* moved:

- **From‑scratch PPO** crushes `random` (~0.97), is ~even with `greedy` (~0.48),
  and **loses to `scripted` / `max-guard` / `max-attack` (~0.23–0.33)** — and
  this profile is **flat across 100k → 3M steps** and **identical across all
  five training opponents** (single, mixed pool, or toughest). Budget and
  opponent diversity are ruled out.
- **Distilling the cheating `mcts:100`** into the same net plateaued too — there
  blamed on the teacher–student information gap.

Both pointed at "something structural." This review finds *what*.

---

## 2. The defects (code, adversarially verified)

All four claims below were checked by independent reviewers instructed to
*refute* them against the source; all four survived at high confidence (the only
caveat raised — that D's "secondary to A/B" is a judgment, not a code fact — is
now settled by §3.3). File:line references are to the production tree.

### Defect A — the action space is positional, and the index→action map is not a function of the observation **(primary)**

`index_to_action(idx, legal)` returns `legal[idx]` (`locma/envs/encode.py:69`).
The order and length of `legal` come from `battle_legal`
(`locma/core/battle.py:233-261`), which branches on:

- **card type** — a creature yields one `Summon`; a green item yields one `Use`
  per friendly board unit; red/blue items yield one `Use` per enemy unit (+face
  for blue). So a hand slot contributes a *different number of indices* depending
  on its type.
- **creature readiness** — attacks are emitted only for board creatures with
  `c.can_attack and not c.has_attacked` (`battle.py:255-256`).

The observation (`encode_battle`, `encode.py:44-59`) encodes per hand/board card
only `[present, cost, attack, defense, G, L, W]` and six scalars. It **never
writes card type** (the `CardView.type` field exists at `views.py:10` but is not
read into the vector) and **never writes `can_attack`/`has_attacked`** (those
live on `CardInstance`, `instance.py:15`, and are absent from `CardView`
entirely).

Therefore the enumeration `legal` depends on state the observation does not
carry, so **two distinct game states can produce the identical 146‑d
observation yet different legal lists** — meaning the same action index denotes
different moves. The policy head cannot, even in principle, know what its chosen
index does. For RL this also breaks credit assignment: "index 3 → win" in one
episode does not transfer to another where index 3 is a different action.

### Defect B — the action mask is an information‑free length prefix **(primary, compounds A)**

`action_mask(legal)` (`encode.py:62-66`) sets the first `min(len(legal), 64)`
entries `True` and the rest `False`. It is a pure function of the *count* of
legal actions — every state with the same count has a byte‑identical mask, and
the mask says nothing about *which* actions (or their type/target) are legal.
A correct masked‑RL setup uses the mask to inject exactly that per‑action
legality; here it injects only `log2(#counts)` bits. (Confirmed empirically in
§3.1: 37 distinct masks for 37 distinct counts.)

### Defect C — the observation drops battle‑relevant features **(secondary)**

`_card_feats` (`encode.py:24-41`) checks only `G`, `L`, `W` of the six abilities
in `ABILITY_ORDER = "BCDGLW"` (`cards.py:6`). So **Breakthrough (B), Charge (C),
Drain (D) are dropped**, although the engine fully implements them (Drain
`battle.py:325,346`; Breakthrough `battle.py:348-351`; Charge seeds initial
`can_attack` at `instance.py:26`). The obs also drops **card type**, **card
identity** (`card_id`), and any **readiness / summoning‑sickness** signal. In the
160‑card DB, **46 cards carry ≥1 dropped keyword** (25 B, 18 C, 20 D).

### Defect D — sparse reward, battle‑only, opponent‑drafted decks **(secondary)**

`BattleEnv` gives `reward = 0` every step and only `±1` at terminal
(`battle_env.py:138-140`) — a sparse win/loss signal with no shaping. Training is
battle‑only: the **opponent drafts both decks** (`reset`, `battle_env.py:104-108`),
so PPO never learns or controls the draft and always plays an *opponent‑drafted*
deck. The agent is also always seat 0. (That this is *secondary* to A/B is an
assessment, now backed by §3.3: fixing A alone moves the result the most.)

**Ruled out — the 64‑action cap.** `ACTION_SIZE = 64` never truncates: the most
legal actions seen over 29k decisions was **43**, and `battle_legal` is bounded
well under 64 (Pass + ≤8 summons + item uses + attacker×target attacks, all
throttled by mana). Network size (default `pi=vf=[64,64]`) is also plausibly
adequate for 146 dims and is not the limiter.

---

## 3. Experiments

All experiments use the real engine and the `[ml]` stack; RL arms use
`MaskablePPO("MlpPolicy")`, seed 0, trained vs `greedy`, evaluated mirrored
(both seatings per seed) vs each baseline.

### 3.1 Structure — the mask carries no information and the index is non‑stationary

29,287 multi‑choice decisions from 400 `greedy` self‑play games:

- **Mask = count.** 37 distinct masks for 37 distinct legal‑counts (range 2–43).
  The mask is a length prefix; it never tells the net *which* move is legal.
- **Index is non‑stationary.** Index 0 is always `Pass`. Every other index is a
  mixture: **index 1 = Summon 39% / Attack‑unit 31% / Use 25% / Attack‑face 5%**,
  and similar through index 7. A single output neuron must represent radically
  different actions depending on unobserved card type and readiness.
- **Observation collisions exist.** 67 distinct obs vectors map to ≥2 different
  legal‑kind enumerations (138 decisions sit on such an ambiguous obs). Example:
  one 146‑d vector corresponds to both `(Pass, Attack-unit)` and
  `(Pass, Attack-unit, Attack-unit)` — identical input, different meaning for
  every index. (This undercounts the true ambiguity: exact float‑equality is
  rare, but the *static* argument in Defect A already shows the map is not a
  function of the obs.)

### 3.2 Behavior‑cloning probe — the action representation is the dominant axis

To separate "is it the observation or the action representation," we
behavior‑cloned the **same deterministic expert** (`greedy`, which decides purely
from the `BattleView` the student also sees — so there is **no information gap**,
unlike the MCTS distillation) into the **same target**, varying only the
encoding. 50,667 multi‑choice greedy decisions / 700 games, **game‑level** 15%
val split, a 256×256 MLP with masked cross‑entropy (a standalone probe — *not*
the production distill net, which is 64×64). Val top‑1 agreement with greedy:

| observation | → **semantic** target | → **positional** target |
|-------------|----------------------|-------------------------|
| current (146‑d) | **0.952** | 0.686 |
| rich (308‑d)    | 0.945 | 0.744 |

_(random‑over‑legal floor = 0.237; current‑obs→positional 0.686 is the same
ballpark as the 0.78 `greedy` smoke from the distillation work — different net /
split, same regime.)_

Reading:

- **Action representation dominates.** Holding the observation fixed, going
  positional→semantic adds **+0.26** (0.69→0.95). The semantic target is a stable
  per‑action head; the positional target (`idx = legal.index(action)`) is an
  *unstable enumeration‑order pointer* whose meaning shifts state‑to‑state, so a
  learner must chase a moving target. This label‑stability effect is the primary
  driver; the richer semantic *mask* (which re‑injects type/target legality the
  flat obs drops) is a secondary, complementary contributor.
- **Observation richness barely matters — and only interacts.** Under the
  semantic target, current vs rich obs is within noise (0.952 vs 0.945, *no
  detectable effect*). Under the crippled positional target, richer obs *did*
  help (+0.058). So obs richness is not globally irrelevant, but it is not the
  lever — a good action space largely rescues a lossy observation.

### 3.3 Causal RL — swapping only the action space breaks the ceiling

Same harness, budget (300k steps), seed, and `greedy` opponent; vary only the
encoding. Win rate vs each baseline (400 mirrored games/cell, ~±0.05 CI):

| arm | random | greedy | scripted | max‑guard | max‑attack | **avg non‑random** |
|-----|--------|--------|----------|-----------|------------|--------------------|
| **positional + current obs** (today's design) | 0.963 | 0.450 | 0.228 | 0.220 | 0.220 | **0.279** |
| **semantic + current obs** (only action space changed) | 0.980 | 0.655 | 0.338 | 0.383 | 0.405 | **0.445** |
| **semantic + rich obs** | 0.965 | 0.677 | 0.312 | 0.335 | 0.370 | **0.424** |

- **Control passes.** The positional arm reproduces the documented ceiling
  (greedy 0.45, ground baselines ~0.22), validating the harness.
- **Action space is the primary lever.** Changing *only* positional→semantic,
  with the identical 146‑d obs, lifts avg non‑random **0.279 → 0.445 (+0.17,
  +60% relative)**: vs `greedy` +0.21, vs `max-guard` +0.16, vs `max-attack`
  +0.19. The agent stops getting crushed by the ground baselines.
- **Rich obs adds nothing** (0.445 vs 0.424 — current marginally higher),
  matching §3.2. The observation is secondary.

This is the RL manifestation of the §3.2 BC gap: reward‑driven learning is hit
*harder* by the positional space than supervised cloning is, because credit
assignment must attribute outcomes to indices whose meaning shifts.

### 3.4 Secondary levers — observation normalization and value learnability (both ruled out)

The completeness review nominated two further "high‑impact" suspects: a missing
observation normalization, and an unlearnable value function under the lossy obs.
We tested both in a 2×2 (action space × normalization) at the same 300k budget,
capturing the critic's final `explained_variance`:

| arm | avg vs non‑random | `explained_variance` |
|-----|-------------------|----------------------|
| positional, raw obs | 0.279 | 0.63 |
| positional, +`VecNormalize` | 0.300 | 0.59 |
| semantic, raw obs | 0.445 | 0.58 |
| semantic, +`VecNormalize` | 0.348 | 0.48 |

_(positional‑raw reproduced §3.3 to three decimals — a determinism check.)_

- **Normalization is not the lever.** `VecNormalize` gives the positional arm
  only **+0.02** (within noise) and **hurts** the semantic arm (0.445 → 0.348) —
  at best marginal, and able to *degrade* a working setup. Nowhere near the
  **+0.17** from the action‑space change. (Single seed; read the negative sign on
  the semantic arm as "did not help," not a law.)
- **Value learning is not the bottleneck.** `explained_variance` is **0.48–0.63**
  across all four arms — the critic predicts half to two‑thirds of return
  variance from the obs, so the value head is functional, not collapsed. The
  "lossy obs makes returns unpredictable" hypothesis is unsupported here; the
  stronger semantic policy even has slightly *lower* EV (its richer trajectories
  are harder to predict).

Both nominated secondary levers come back negative, which further isolates the
action representation as the cause.

---

## 4. What is and isn't the problem

| Candidate | Verdict | Evidence |
|-----------|---------|----------|
| **Positional action space** (index not a fn of obs) | **Primary cause** | §2A, §3.1, §3.2 (+0.26 BC), §3.3 (+0.17 RL) |
| **Information‑free mask** (length prefix) | **Primary (compounds A)** | §2B, §3.1 (37 masks = 37 counts) |
| Lossy observation (type, B/C/D, readiness) | Secondary | §2C; §3.2/§3.3 show rich obs ≈ no gain once action space is fixed |
| Sparse reward / battle‑only / opp‑drafted decks / seat 0 | Secondary (untested here) | §2D |
| Observation normalization | **Not the lever (tested)** | §3.4: +0.02 positional, −0.10 semantic |
| Value‑function unlearnable | **Not supported (tested)** | §3.4: EV 0.48–0.63 in all arms |
| Training budget (100k→3M) | **Not the problem** | prior study, flat |
| Opponent (single/mixed/toughest) | **Not the problem** | prior study, identical |
| 64‑action cap truncation | **Not the problem** | max 43 legal, 0 truncations |
| Network size | Unlikely | default 64×64 adequate for 146 dims |

---

## 5. Recommendations (ranked)

1. **Replace the positional action space with a fixed semantic one.** Index by
   *slot‑stable* meaning — e.g. `Pass`; `Summon hand‑slot s`; `Use hand‑slot s →
   target‑code`; `Attack board‑slot a → target‑code` (~155 indices for this
   single‑lane engine). Build the mask from the real legal list so each set bit
   is a concrete legal move. This is the **+0.17 avg** win shown in §3.3 and the
   single highest‑value change. (A verified reference encoder/env — `encode2` /
   `BattleEnv2`, with a passing round‑trip self‑test — was built for this review
   and can be promoted into `locma/envs/`.)
2. **Add the dropped observation features** — card type (one‑hot), B/C/D, and a
   per‑creature `can_attack` flag (and consider `card_id`). Modest on its own
   (§3.3), but it removes the in‑principle ambiguity of Defect A and likely helps
   stronger experts than greedy. Keep it lightweight: the rich 308‑d obs did not
   beat the current 146‑d obs in either the BC probe or RL.
3. **Add an entropy bonus** (`ent_coef` ≈ 0.01–0.03). SB3 defaults to 0, which
   with masking can collapse onto an early local optimum — consistent with the
   flat 100k→3M curve. (Untested here; cheap to A/B.)
4. **Shape the reward and lengthen the horizon** (health/board‑advantage delta;
   `gamma` ≈ 0.997 for multi‑dozen‑decision battles), **train both seats**, and
   either let the agent draft or pin eval to the trained deck/seat distribution.

Do #1 first; it is the cause. #3–#4 close the residual gap to actually *beat* the
ground baselines. **Skip observation normalization** — §3.4 shows `VecNormalize`
does not help (and hurt the semantic arm); and the value function is already
learnable (EV 0.48–0.63), so neither is a productive lever here.

---

## 6. Outcome — fix implemented and verified

Recommendation #1 (+ #2 obs enrichment, #3 entropy) was implemented on
`fix/ppo-semantic-action-space`: the production encoder is now a fixed semantic
155-action space with an enriched 308-d observation, and `train_agent` uses
`ent_coef=0.02`. End-to-end through the **production** path (`train_agent` →
semantic `BattleEnv` → the `ppo:` policy), retrained 300k vs `greedy`, evaluated
mirrored (400 games/cell):

| vs → | random | greedy | scripted | max‑guard | max‑attack | **avg non‑random** |
|------|--------|--------|----------|-----------|------------|--------------------|
| before (positional, §3.3) | 0.963 | 0.450 | 0.228 | 0.220 | 0.220 | **0.279** |
| **after (this fix)** | 0.975 | **0.632** | 0.315 | 0.343 | 0.372 | **0.416** |

**+0.137 avg (+49%)** — from *crushed* by the ground baselines (~0.22) to
*competitive* (0.31–0.37), and even→winning vs `greedy` (0.45→0.63). As predicted,
0.416 sits just under the action‑space‑only experiment (sem+current obs = 0.445):
the bundled observation enrichment is neutral‑to‑slightly‑negative, so essentially
the entire gain is the action representation. A leaner config (semantic action
space + the original 146‑d obs, no entropy) is a reasonable follow‑up if the small
gap matters; closing the rest to actually *beat* the ground baselines is reward
shaping / longer horizon / both‑seat training (§5 #3–#4).

## 7. Reproduce

The production defects are visible directly in `locma/envs/encode.py`
(`index_to_action`, `action_mask`, `encode_battle`) against
`locma/core/battle.py:233-261` (`battle_legal`). The experiments above are
one‑off probes (greedy self‑play data + a slot‑indexed semantic action space +
short MaskablePPO runs); the controlling result is §3.3, reproducible by
training two `MaskablePPO` agents to 300k vs `greedy` that differ only in action
encoding and evaluating both vs the five baselines.

## 8. Future explorations — ranked levers for the "squeeze"

After the fix, semantic-action PPO reaches ~0.42 avg vs the four ground baselines
(competitive, but still losing ~0.31–0.38 to `scripted`/`max-guard`/`max-attack`).
A multi-seed config A/B (2×2 over observation {lean-146, rich-308} × entropy
{0, 0.02}, 250k × 2 seeds) found **all four configs tied at 0.41–0.43** — so
**observation richness and the entropy bonus are not levers** (the single-seed
0.445-vs-0.416 gap was noise). Together with the earlier rule-outs (budget,
opponent diversity *under the positional space*, normalization, value
learnability with `explained_variance` 0.48–0.63, the 64-action cap, net size),
the remaining levers are all **env/training-formulation** changes, not net or
hyperparameter tuning:

| Lever | Status | Note |
|-------|--------|------|
| **Draft control** | ✅ **done (PR #21)** | §8.1: the gap was mostly the deck. `ppo:` now drafts `balanced` → beats the ground baselines (~0.55) and is ~even with `mcts:100`. |
| **Reward shaping** (PBS) | ✗ **ruled out** | §8.2: health-only potential is exactly neutral (= sparse 0.554); adding board advantage *hurts* (→0.50). Sparse ±1 is already adequate. |
| **Self-play / league** | ✗ ruled out (flat net) · ~ partial (token) | §8.3: the **flat**-net league was flat-then-down over 480k. **Update (token PPO2):** the slot-addressable net *responds* — one round of self+baselines = **+0.03 avg-hard3**, plateauing **~0.64** after 2 rounds (front-loaded, diminishing). Real but modest; still far below search (~0.73). |
| **Both-seat training** | ✅ **landed (PR #27)** | Now the default (`--both-seat`); +0.06 / 2× efficiency, **same ceiling** (the 2×2 + 800k retrain confirm it's efficiency+correctness, not a higher ceiling). |
| **Longer horizon** (`gamma` 0.99→0.997) | open — Low | Untested; cheap, but unlikely to matter given the above. |
| Observation richness / entropy / normalization / **net size** | ✗ ruled out | §3.4 multi-seed A/B + the net-size×seat 2×2 (`baseline.md`) — all neutral; bigger net *hurts*. |
| **Richer board encoding** (tokens + self-attention + tactical scalars) | **tested — parity (secondary)** | §8.4A update: a *correctly slot-addressable* token+attention encoding reaches/marginally beats flat under the curriculum (avg-hard3 0.588 vs 0.573, 2 seeds) but within seed noise, at ~4× cost. A real but secondary lever; the relational matrix and its use as a substrate for B remain untested. |
| **Search in the loop** (AlphaZero-lite) | **open — the real lever** | §8.4B: net guides `dmcts`/MCTS (PUCT priors + value leaf), trained by self-play of the *search*. The only path to MCTS-level *planning*; a bigger build. |

**Recommended order (remaining):** every training-method lever is now spent — reward,
obs, opponent diversity, and **self-play** (§8.3) all flat; only the **deck** ever
moved the PPO. The residual gap to `mcts:100` is its **planning**, which a reactive
policy net cannot acquire by playing more games. The real lever is **search in the
loop** (AlphaZero-style: a policy+value net guiding MCTS). The cheap thing to re-try
first is **distilling the now strong + fast heuristic MCTS** — old distillation
plateaued at ~0.25 agreement under the *positional* action space; the semantic space
lifted greedy-cloning 0.69→0.95, so MCTS-cloning is worth re-measuring.

### 8.2 Reward shaping — ruled out

Potential-based shaping (PBS — policy-invariant by Ng et al. 1999, verified
correct here to machine precision) with potential `Φ = health-lead + w·board-lead`,
trained vs `mixed` 600k, evaluated paired with `balanced`:

| reward | avg vs hard baselines |
|--------|-----------------------|
| sparse ±1 (control) | 0.554 |
| PBS, health-only (`w=0`) | 0.554 (exactly neutral) |
| PBS, health + board, coef 0.5 | 0.522 |
| PBS, health + board, coef 1.0 | 0.500 |

A health-only potential neither helps nor hurts; the **board-advantage** term
*hurts* (more shaping → monotonically worse) because it discourages the favorable
face-trades that win this aggressive tempo game. The sparse win/loss signal is
already adequate — densifying it does not improve credit assignment. So the
residual gap to `mcts:100` is its **lookahead** — and §8.3 shows self-play does
not close it either.

### 8.3 Self-play / league — ruled out

A warm-started (from `zoo-mixed`) league: each episode samples an opponent from
past frozen PPO snapshots of the learner (self-play) + the ground baselines
(anti-forget) + `mcts:100` (a strong, now-cheap live teacher). Seat-randomized;
the agent drafts its own `balanced` deck. (Harness adversarially verified first —
the review caught and fixed two real bugs: seat-locked training and the agent's
deck being coupled to the opponent's draft.)

| training | vs max-guard | vs max-attack | vs `mcts:100` |
|----------|--------------|---------------|---------------|
| warm-start (`zoo-mixed`) | 0.59 | 0.58 | 0.24 |
| after 480k self-play (6 league rounds) | 0.49 | 0.59 | **0.16** |

**Flat, then slightly down** — over 480k steps the model only oscillated around
warm-start strength and ended *below* it, and it got *worse* vs `mcts:100`
(0.24→0.16). This is the same wall every training-method lever hit. The reason is
structural: a **reactive policy net cannot match a search policy by playing more
games** — self-play improves which move the net reflexively picks but adds no
*planning*, which is exactly MCTS's edge. The only architecture that closes this is
**search in the loop** (AlphaZero-style: a policy+value net guiding MCTS, trained by
self-play of the search) — not more self-play of the raw net.

**Update (2026-06-27): the *token* net responds where the flat net decayed — a real
but front-loaded, plateauing gain.** Re-ran self-play on the slot-addressable PPO2
(`obs_mode="token"`): warm-start the zoo-curriculum PPO2 (`ab-token-s1`), train one
round (200k) against a per-episode mix of a *frozen self* + the ground baselines,
conservative `target_kl=0.025` (inherited from the tuned base). Matched eval (300
games/opp, seed 0):

| stage | avg-hard3 | Δ |
|-------|-----------|---|
| base (`ab-token-s1`) | 0.601 | — |
| self-play r1 | 0.632 | +0.031 |
| self-play r2 (self = r1) | 0.639 | +0.007 |

The lift is **consistent across all four opponents** but **front-loaded and
diminishing** (r1 +0.031 ≈ 2.6σ real; r2 +0.007 within noise) — it **plateaus ~0.64**,
it does not compound. This *corrects* the flat-net result above for the new
architecture: the richer net can be sharpened by self-play (capacity + the conservative
KL cap keeping updates stable), where the flat net decayed. But the ceiling stands —
even the self-play net (0.64) is well below the search policies (~0.73); self-play
sharpens the reactive policy a bit more than from-scratch RL (0.588), adding no
planning. `selfplay-r2` is the strongest reactive net produced. (Probes were throwaway;
no self-play infra landed.)

### 8.1 Why the heuristics still win — it's mostly the deck, not the battle

A deck-swap probe (zoo `mixed` model, 240 mirrored games/cell) isolates draft from
battle by pairing the **same trained PPO battle net** with different drafts:

| agent (draft + battle) | scripted | greedy | max-guard | max-attack |
|------------------------|----------|--------|-----------|------------|
| PPO + **greedy** draft (shipped `ppo:`) | 0.396 | 0.592 | 0.412 | 0.404 |
| PPO + **max-guard** draft | 0.608 | 0.496 | **0.500** | **0.500** |
| PPO + **max-attack** draft | 0.471 | 0.671 | 0.446 | 0.500 |
| `greedy` (ref) | 0.463 | 0.500 | 0.471 | 0.308 |

Holding the battle net fixed and swapping only the draft to `max-guard`'s lifts PPO
from losing (0.41/0.40) to **even with every ground baseline** (0.50/0.50, and 0.61
vs `scripted`). The battle policy was never the bottleneck — the generic `greedy`
deck was. The strong baselines win largely through **deck construction** (a Guard
wall, or max aggression); even among them, `max-guard`'s wall beats `max-attack`'s
race 0.61. A behavioural probe confirms PPO is *not* a `greedy` clone (it agrees
with `greedy` on only ~36% of decisions and attacks the face far more — 22–31% vs
6–8%), so it learned a real, distinct policy; it is simply playing a deck that was
not built for any coherent plan. **For the residual** (even a good deck only reaches
~0.50, not dominance): the classic limits bite — sparse reward, fixed (not
self-play) opponents, and no search/planning (why cheating `mcts:100` beats `greedy`
0.79: it *plans*; PPO reacts).

**Draft sweep (see `baseline.md`).** A full sweep over seven drafts confirms and
sharpens this: `greedy` (the shipped draft) is the *worst* partner (0.39 avg vs
the hard baselines); `max-guard` (0.55) and the new `balanced` heuristic (0.54) are
best and make the PPO **beat all three ground baselines**, with even a *random*
draft (0.49) beating `greedy`. And **training the battle net on a given deck adds
nothing over simply pairing the mixed-trained net with that deck** — the battle
policy is deck-robust, so the deck at *deployment* is the lever. **Actionable:**
swap the `ppo:` policy's draft from `greedy` to `max-guard`/`balanced` to turn it
from losing to winning vs the ground baselines, no retraining.

### 8.4 Future research paths (the two that are actually open)

Everything in the table above except the last row is spent. Two directions remain,
and they **compose** — one is the substrate, the other the algorithm.

#### A. A better semantic *board* encoding (substrate)

The current obs (`encode.py`) is a **flat 308-d fixed-slot vector**: 8 scalars + 20
card slots × 15 per-card features. It carries no *relations* and no *derived
tactics*, so the net must re-infer the entire combat graph and every tactical fact
from raw positions on every forward pass. Three upgrades, in increasing ambition:

1. **Tokenize the board.** Represent each card as a *token* and use a set/attention
   (transformer) feature extractor instead of flat fixed slots — permutation-
   invariant, naturally variable-length, and able to relate cards directly. (SB3
   takes a custom `features_extractor_class`, so this is a net-side change, not an
   env rewrite.)
2. **Add relational objects.** Feed the explicit **attacker × target legality /
   trade matrix** — for each of my attackers, which enemy targets are legal and what
   the trade outcome is. This is exactly the structure the flat obs *hides* (it is
   the root of Defect A: legality depends on relations the vector never encodes).
3. **Add engineered tactical scalars** — cheap 1-ply facts the net otherwise has to
   learn to compute: **opponent Guard count, reachable face damage this turn,
   friendly lethal available, own exposed-to-lethal flag, mana remaining, on-board
   stat totals**. These are *tactical primitives* — a sliver of shallow lookahead
   baked into the observation.

**Why this isn't contradicted by §3.4** (where rich-308 ≈ lean-146): that A/B only
added *more of the same flat scalars*. Relations, a combat matrix, and computed
lookahead scalars are a **different kind** of information (derived structure, not raw
features), so the null result doesn't cover them. Expect this to sharpen the reactive
net's tactics — and, crucially, to make a far better **policy/value net for path B**.

**Update (2026-06-26): A built and tested — `obs_mode="token"`. Verdict: parity / a
slight lean ahead under the curriculum, within seed noise. A is a *secondary* lever,
as the table now reads.** The encoding is per-card *tokens* (zone/type/cost/attack/
defense/abilities/readiness + a learned **card-id** embedding — `card_id`, which the
flat obs discards), a handful of computed **tactical scalars** (guard count, reachable
face damage, lethal-available, mana, board totals), and a **self-attention** extractor
(`TokenSetExtractor` on `MultiInputPolicy`). Two non-obvious lessons fell out:

1. **The action space is slot-indexed, so the encoder must be slot-*addressable*, not
   permutation-invariant.** The first build pooled the tokens into a permutation-
   invariant CLS vector — elegant, and exactly wrong: Summon/Use/Attack logits are
   indexed by *which hand/board slot* a card occupies, so an order-invariant feature
   collapses states that differ only by slot and the policy cannot learn slot-specific
   play (pilot: token 0.46 < flat 0.55). Fix: a per-slot positional embedding +
   **flatten the per-slot transformer outputs** (each slot at a fixed offset) — keeps
   attention's cross-card relational mixing while preserving slot identity. (The flat
   obs had slot-addressability for free, which is why it was a stronger baseline than
   it looked.)
2. **The bigger net needs gentler PPO.** At the default LR (3e-4) / 10-epoch updates
   the token net's `approx_kl` ran to 0.10–0.15 (clip-fraction ~0.4) and it *degraded
   with training* (0.382 → 0.333 over 300k). Lowering LR to 1e-4 + a `target_kl=0.025`
   early-stop tamed KL to ~0.015 and recovered it. The flat MLP is small enough to be
   stable at 3e-4.

**Results (avg-hard3 = mean win rate vs scripted/max-guard/max-attack):**
- *Single opponent* (300k vs max-attack): flat 0.588, token 0.565 — token slightly
  **behind**, because the larger net **overfits the lone training opponent** (strong
  vs max-attack, weaker vs the unseen scripted). The single-opponent pilot is biased
  against token; the diverse curriculum is the fair test.
- *Zoo curriculum, full A/B* (200k×4 = 800k/arm, 2 seeds, 400 games/opp): token
  **0.588 vs flat 0.573** (+0.015; greedy +0.025) — token wins the mean and is most
  consistently ahead vs greedy, **but the two seeds disagree** (s0 flat 0.592 > 0.562,
  s1 token 0.614 > 0.555) and the per-seed spread (~0.03–0.04) exceeds the gap. Across
  three independent curriculum runs token won 2/3 (mean ≈ +0.02).

**Bottom line:** a *correctly* slot-addressable, stably-trained tokenized+attention
encoding **reaches and marginally exceeds** the flat baseline under the curriculum —
confirming this row's "a different *kind* of obs can help" premise — but the edge
(~+0.015–0.02 avg-hard3) is within 2-seed variance, at ~4× the training cost. So A is
real but **secondary**: on its own it does not lift the reactive net off the ceiling.
Its larger promised value remains as a **better substrate for B** (the policy/value
net for search) — still untested, as is the explicit relational/trade matrix. The
plumbing is additive behind `obs_mode="flat"` (the default), so the flat baseline is
untouched. See `baseline.md` ("PPO2") and the 2026-06-26 worklog entry.

#### B. AlphaZero-lite — search in the loop (the algorithm)

The only route that adds **planning** (the structural gap a reactive net can't
cross). A policy+value net *guides* MCTS — policy → PUCT priors over the 155 semantic
slots, value → the leaf evaluation — trained by self-play of the **search**, not the
raw net. "Lite" = reuse what exists and grow incrementally:

- **Forward model:** the fast `_clone_battle` (≈30× speedup) is the simulator.
- **Search skeleton:** `dmcts` already does determinized MCTS for the imperfect-info
  hand; swap its heuristic leaf for the **net's value** and add **net priors** (PUCT).
- **Targets:** the search's visit distribution + game outcome are the policy/value
  training targets (the AZ loop); iterate.

This is a bigger build, but every piece (fast clone, semantic action space, a fair
determinized search, a heuristic leaf to bootstrap the value net) is already in the
kit. Path A is the substrate that makes B's net stronger and its search cheaper; B is
the planning that A's reactive net can never represent on its own.

**Update (2026-06-26): the skeleton is built and it already wins.** `azlite`
(`locma/policies/azlite.py`) realizes B in its simplest form — PUCT search with a
**heuristic** `(policy, value)` oracle instead of a trained net: prior = 1-ply
heuristic lookahead softmax, value = the board/health leaf. With **no net and no
self-play**, `azlite:100` **beats every baseline** (avg-hard3 0.741), **beats the
strongest PPO head-to-head 0.76**, and is **even-to-ahead of the cheating MCTS**
(0.57) — see `docs/baseline.md` (2026-06-26). This confirms the thesis directly:
**search at play time is the lever**, not more reactive-net training. The remaining
upside of the *full* B (a self-play-trained policy/value net replacing the heuristic
oracle) is to push *past* the cheating MCTS; `azlite` is the working harness it would
drop into — swap `_prior`/`_value` for the net's policy/value heads.
