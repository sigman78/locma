# E42 — public-history features: is there anything for MEMORY to remember?

Pre-registered 2026-09-25. Branch `feat/e42-public-history`. Arms `e42h` / `e42c`
via `scripts/e36_pfsp.py --tag` (x86 chain, this box).

## Question

E6 (2026-07-03) added an LSTM to the B0 recipe and landed -0.105. A re-read of that
experiment (rppo.py on `feat/rppo-lstm`, driver, verdict) finds no code bug but
three design choices that made the negative uninformative about memory itself:

1. **No input channel for the opponent's turn.** `BattleEnv._opp_play_until_agent`
   resolves every opponent action inside `step`; the agent sees only the resulting
   board. Cards that leave no board trace (items, removal, burn, a creature that
   traded) are never observable at ANY timestep, so no recurrent state over
   own-decision observations can carry "what the opponent has revealed".
2. **Nothing to remember against that pool.** Training zoo and avg-hard3 ruler are
   greedy heuristics that dump their hand every turn — hand contents are a
   near-deterministic function of `(turn, op_hand_count)`, both already in the obs
   (the PPO review's E6a note predicted this null).
3. **Budget/update confound.** 800k steps from scratch on the old transformer trunk,
   batch 64 -> 2048 at the same lr and step count (~32x fewer gradient updates).

So the memory hypothesis is untested on the current net. The engine keeps NO
history either (`PlayerState` = health/mana/deck/hand/board; `BattleView` exposes
`op_hand_count` only). This experiment asks the cheapest form of the question:

> If the PUBLIC history of the game (cards the opponent has played, their mana
> left at each turn end, your own remaining deck as a multiset) is made explicit
> in the observation, does the reactive net gain against opponents that hold
> cards (self-play, fair search)?

The engineered history features are a SUPERSET of anything an LSTM over
own-decision observations could ever accumulate (the LSTM sees strictly less: it
never sees the opponent's turn). Therefore:

- a **null here kills the LSTM retest** (steps 2-4 of the 2026-09-25 memo) at this
  regime — there is no information to remember, or the net cannot use it;
- a **positive here** bounds what learned memory could add and motivates the
  recurrent retest (does learned memory beat hand features?).

## Fairness (lguard's argument, E38's standard)

Every feature is computable by a human player from public information under LOCM
1.2 rules: the opponent's played cards are revealed as they are played, their mana
is visible, deck counts are visible, and your own 30 drafted cards are known. The
own-deck feature is the remaining deck as an ORDER-DESTROYED multiset (sorted card
ids / aggregate counts) — reading `PlayerState.deck` order would be cheating and is
excluded by construction (test: shuffling `deck` in place changes nothing).

## Plumbing (engine -> view -> obs)

- `PlayerState` gains two public logs, maintained by `core/battle.py`:
  `played: list[int]` (card ids in play order, appended on Summon/Use) and
  `turn_log: list[tuple[int,int]]` (`(mana_left, hand_size)` at each of this
  player's own turn ends, appended at the top of `end_turn`). Gameplay is
  byte-identical; the replay format (`replay_stream._player_dict`) is untouched.
- `BattleView` gains, with defaults so existing constructors keep working:
  `my_deck_count`, `op_deck_count`, `my_deck_cards` (SORTED tuple of remaining own
  card ids), `op_played` (tuple of card ids), `op_turn_log` (tuple of
  `(mana_left, hand_size)`). Own played cards are NOT exposed (not needed).
- New token-obs variant **`fxh`** = `fx` tokens (20 wide) + v0 scalars (13) + a
  NEW obs key **`hist`** of width `N_HIST = 25` (float32, raw counts, the existing
  scalar convention — LayerNorm in the branch normalizes):

  | idx | feature | source |
  |---|---|---|
  | 0 | my_deck_count | view |
  | 1-4 | own remaining deck: count by type (creature, green, red, blue) | `my_deck_cards` + cards DB |
  | 5-8 | own remaining deck: count by cost bucket (0-2, 3-4, 5-6, 7+) | same |
  | 9 | own remaining deck: mean cost (0 if empty) | same |
  | 10 | op_deck_count | view |
  | 11 | n opponent cards played | `op_played` |
  | 12-15 | opponent played: count by type | same |
  | 16-19 | opponent played: count by cost bucket | same |
  | 20 | opponent played: mean cost (0 if none) | same |
  | 21 | opponent mana left at their last turn end (0 if none yet) | `op_turn_log` |
  | 22 | mean opponent mana left over all their turn ends | same |
  | 23 | fraction of opponent turns ended with mana_left >= 2 AND hand_size > 0 ("sandbag rate") | same |
  | 24 | 1.0 if the opponent's last turn was a sandbag turn by that rule | same |

  `obs_mode="token-fxh"` in `BattleEnv`; `token_obs_space("fxh")`;
  `token_variant_for_space` detects `fxh` by the presence of the `hist` key (fx/v0
  paths byte-identical — E28c's rule).
- `SlimTokenExtractor` grows an optional branch when the obs space has `hist`:
  `LayerNorm(N_HIST) -> Linear(N_HIST, d_model) -> ReLU -> Linear(d_model, d_model)`
  with the final Linear **zero-initialized**, its output ADDED to the scalar-branch
  output `s` before `head`. So at step 0 a warm-started fxh net is EXACTLY gen7.
- Warm start across obs spaces (`SB3.load` refuses a changed space): build a fresh
  model at the old model's hyperparameters, copy the old policy `state_dict` with
  `strict=False`, assert the only missing keys are the `hist` branch and there are
  no unexpected keys. **Both arms use this path** (control gets zero missing keys)
  so both start with a fresh optimizer and the ONLY difference is the `hist`
  input + branch. Test: identical action logits and values on matched obs.
- Play time: `ppo:`/`lppo:`/ensemble consumers encode via `token_variant_for_space`
  (variant-agnostic already); a `nohist` flag on `ppo:` zeroes the `hist` vector
  for the ablation instrument.

## Gate 0 — information probe (no training, ~40 min)

`scripts/e42_gate0.py`. Shadow driver: gen7 (`ppo:depot:e36/e36_gen7.zip,ldraft`)
plays 300 games each vs {gen7 mirror, `dmcts:15,60,0,3,ldraft`, scripted,
max-guard, max-attack}; at gen7's FIRST decision of each own turn record
`X0` = v0 scalars (13) + per-zone pooled fx token sums (3 x 20), `H` = the 25
hist features, and hidden-truth targets read from `gs`:

| target | definition |
|---|---|
| T1 | items in the opponent's hand (count) |
| T2 | total mana cost of the opponent's hand |
| T3 | max creature attack in the opponent's hand (0 if none) |
| T4 | red items in the opponent's hand (removal) |
| T5 | face damage I take during the opponent's NEXT turn (realized, from the game continuation) |

Ridge (5-fold CV by game) on `X0` vs `[X0, H]`, per target x opponent; report
R2, delta-R2 with a game-level bootstrap CI, plus the descriptive sandbag rate per
opponent.

| read | rule |
|---|---|
| **KILL** | max delta-R2 over {T1..T5} x {mirror, dmcts} < 0.02 — public history carries no exploitable information about hidden state against the opponents that matter. Close: no training; E6's null generalizes to any memory over these observations. |
| **PASS** | delta-R2 >= 0.05 on at least one hand target (T1-T4) for mirror or dmcts. Sanity: hard3 delta ~ 0 (greedy opponents dump hands). |
| ambiguous (0.02-0.05) | proceed to Gate 1 flagged (asymmetric read, E28c precedent: BC/probe instruments cannot price consequence value). |

## Gate 1 — matched-harness training (E37 protocol)

Both arms warm from `depot:e36/e36_gen7.zip` (x86 parity endpoint on this box),
pool = the x86 terminal `runs/e36/pool.json` copied per tag, `--start-gen 8
--generations 2 --steps 1000000 --n-envs 6` (the x86 chain's regime), seed base
**27_000_000** (clear of every train/eval base in use), 2 arms concurrently
(box limit: 2 trainers).

- **e42h**: `token-fxh`, hist branch (zero-init), widened warm start.
- **e42c**: `token-fx`, same warm-start path (fresh optimizer), byte-identical pool.

Instruments (pre-registered; P1 seed 80M, P3 81M, P2 on the ladder's own hard-rung
base 72M so the gen7 anchor cell is directly comparable to the published re-pin,
P4 on `e36_gate_gen4.py`'s hard3 default, G on the E10 5M base; all CRN where paired):

| id | instrument | n | role |
|---|---|---|---|
| P1 | H2H `e42h_gen9` vs `e42c_gen9`, matched ldraft | 1000 pairs (2000 games) | **primary** |
| P2 | `dmcts:15,150` hard rung, both arms + gen7 anchor (`e36_dmcts_ladder.py --nets`) | 200 pairs each | sensitivity (n too small to decide alone) |
| P3 | H2H `e42h_gen9` with `nohist` vs `e42c_gen9` | 500 pairs | mechanism: does the gain need the features? |
| P4 | avg-hard3 both arms (`e36_gate_gen4.py --hard3-games 150`) | 150/opp | sanity: memory should NOT help vs hand-dumpers |
| G | boardkeep exploit guard (`e36_exploit.py`) on e42h_gen9 | as script | promotion-track guard-rail only |

Verdict words (glossary): positive = P1 CI excludes 0.50 upward; headroom = positive
and mean >= 0.53; null = CI straddles; negative = CI below 0.50.

| outcome | read |
|---|---|
| P1 positive AND P3 loses the gain (nohist arm ~0.50 or below vs control) | **public-history memory IS a lever.** Escalate: 3-seed promotion track with guards; token-level history variant; the LSTM retest (learned memory vs hand features) is justified. |
| P1 positive but P3 keeps the gain | gain is branch capacity / optimizer restart, not information — **null for the hypothesis**; do not promote. |
| P1 null, Gate 0 PASS | information present, not converted in 2M steps — ONE scale check (2 more gens both arms) before closing; if still null, **close: memory over public history is not a reactive-net lever at this regime, and the LSTM retest is dead** (its information set is a subset). |
| P1 negative | branch/shift tax; close. |
| P4 moves as much as P1 | the gain is not memory-specific — report as a generic obs-completion effect, not a memory result. |

## Non-goals

Recurrence (step 3 of the memo — gated on this result); a token-level
"revealed cards" variant (escalation only); any draft-side use of history;
replay-format changes; the web viewer.

## Cost (this box, RTX 4080)

| segment | estimate |
|---|---|
| plumbing + tests | subagent build, ~half day |
| Gate 0 (1500 games, 12 workers) | ~40 min |
| Gate 1 training (2 arms x 2 gens x 1M at ~270 fps, 2 concurrent) | ~2.5-3 h |
| P1 + P3 + P4 (reactive vs reactive) | ~30 min |
| P2 (3 nets x 400 games vs 2250-sim dmcts, 8 workers) | ~1.5-2 h |
| G | ~20 min |

Artifacts: `runs/e42/{gate0.json,gate0.log,p1.json,p2.json,p3.json,p4.json,guard.json}`,
`runs/e36_e42{h,c}_gen{8,9}.zip`, `runs/e36_e42{h,c}/{pool,history_gen8+}.json`
(runs/ gitignored).
