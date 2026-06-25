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

| # | Lever | Impact | Cost | Why |
|---|-------|--------|------|-----|
| 1 | **Reward shaping** — sparse ±1 → dense potential-based (health / board-advantage delta) | **High** | Small (env reward) | Battles are 30–60 decisions; terminal-only reward makes credit assignment hard. `explained_variance` 0.48–0.63 says the critic is imperfect — dense signal is the most principled fix. |
| 2 | **Opponent strategy** — curriculum / `mixed` / self-play | Medium | Small–Large | Curriculum (`train-zoo`) and `mixed` re-tested under the fixed action space (the positional-era "diversity doesn't help" may not hold). Self-play/league is the high-ceiling version (deferred). |
| 3 | **Both-seat training** | Medium | Small (alternate `agent_seat`) | Agent only trains as seat 0, never learning the second-player coin/bonus-mana mechanic (`battle.py:55-60`), yet eval is mirrored into seat 1 half the time — a real distribution gap. |
| 4 | **Longer horizon** — `gamma` 0.99 → 0.997 | Low–Med | Trivial (1 param) | `0.99^50 ≈ 0.6` heavily discounts the eventual win; compounds with sparse reward. |
| 5 | **Draft control** — agent drafts its own deck | Med–High | Large (draft head/env) | Today the opponent drafts both decks; the agent plays a deck it didn't choose. High ceiling but expands scope to draft+battle. |
| 6 | Net size / `n_steps` / lr sweep | Low | Small | Defaults likely fine for 308 dims; do last. |

**Recommended order:** reward shaping + `gamma=0.997` first (cheap, principled,
targets the long-horizon credit assignment that caps value learning), then
both-seat training; defer draft control and self-play until the cheap structural
levers are spent. The `train-zoo` CLI command (lever #2, curriculum) exists for
quick experimentation; the opponent set is `ZOO_OPPONENTS` in
`locma/envs/training.py`.
