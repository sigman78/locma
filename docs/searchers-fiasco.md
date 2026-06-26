# Searchers fiasco — our "advanced" search policies cheat (perfect foresight)

_Date: 2026-06-26_

## TL;DR

- **`mcts` and `azlite` are perfect-*foresight* cheaters.** They clone the real
  `GameState` and simulate forward. That leaks not just the opponent's hidden
  **hand** but **both players' entire shuffled deck order** — i.e. every future
  draw for both sides. From the searcher's view there is no hidden state and no
  stochasticity left; it plays a fully revealed, deterministic game.
- **`azlite` was mislabeled "non-cheating"** in `docs/baseline.md` and the
  project memory. Its own source docstring says `Perfect-information (cheating)`.
  The "first policy to beat every PPO **without cheating**" claim is wrong.
- **`dmcts` is the only fair searcher** — it resamples the opponent's hidden
  hand+deck (a determinization) instead of peeking. One residual caveat: it keeps
  its **own** real deck order, so it still knows its own future draws.
- **Reactive policies are honest** (`random`, `scripted`, `greedy`, `max-guard`,
  `max-attack`, **`ppo`**): they map a side-visible observation to an action.
- Against a real game server that passes only `(visible_state, actions)`:
  reactive policies and `dmcts` can be deployed (dmcts needs you to bring your own
  engine as a model); **`mcts` and `azlite` cannot be reproduced even in
  principle** — the information they depend on does not exist on the wire.
- The one honest **search-vs-reactive** number in the full-roster matrix is
  **`dmcts` vs `ppo` = 0.74**.

## The trigger

The question that started this: *if we reframe our policies to play a real game
server whose protocol passes only the **side-visible state** and accepts
**actions** — so there is no full forward-model `state` to clone — will the
advanced policies still work? And if they won't, are they cheaters?*

The answer required reading what the search policies actually do with hidden
information, not trusting the labels.

## What "cheating" actually means

Two *different* things get bundled into "needs the full state". Keep them apart:

1. **Needs a forward model (a simulator) to plan.** This is **not cheating.** A
   planning agent builds its own model from the public rules and rolls
   hypotheticals forward. The server not handing you a `state` object is
   irrelevant — you instantiate your own engine, seeded to what you can see.
   Chess engines, AlphaZero, every MCTS need a model; that is legitimate.
2. **Needs information the player cannot legitimately observe** — the opponent's
   hidden hand, deck contents, or the shuffled draw order. **This is cheating.**

So the correct test is **#2, not #1**: *does it use information no real player can
see?* A policy can fail against a `(view, action)` server merely because it needs
a model (reason #1) — that makes it heavier to deploy, not dishonest. The
cheaters are the ones that cannot function without peeking.

## Audit

| policy | needs a forward model? | uses hidden info? | runs vs a `(view, action)` server? | cheater? |
|---|---|---|---|---|
| `random` / `scripted` / `greedy` / `max-guard` / `max-attack` | no | no | yes, directly | **no** |
| **`ppo`** | no | no | yes, directly (obs→action) | **no** |
| **`dmcts`** | yes (bring your own engine) | **no** — samples the opponent's hidden hand+deck | **yes**, with a self-built simulator + a view→state reconstructor | **no — fair** (one caveat below) |
| **`mcts`** | yes | **yes** — peeks at the opponent's real hand *and* both deck orders | no | **yes** |
| **`azlite`** | yes | **yes** — clones the true state; plans against the opponent's real hand *and* both deck orders | no | **yes** |

## The deep leak: perfect foresight of all future draws

The interesting part is that the leak is **bigger than the opponent's hand**. By
simulating from the real cloned state, `mcts`/`azlite` also know **every future
card draw, for both players**. The mechanism, end to end:

1. **The deck is shuffled once, up front, with the seeded RNG** — `draft.py:50`
   `gs.rng.shuffle(deck)`. The draw order is decided at draft end and is
   **hidden**: a fair player knows their deck's *contents* but not its shuffled
   *order* (so not their next draw); the opponent's deck is hidden entirely.
2. **Drawing is deterministic, no RNG** — `battle.py:41`
   `p.hand.append(p.deck.pop(0))`. The shuffled deck order *is* the draw order;
   pop from the front.
3. **The clone copies both decks in order** — `mcts.py` `_clone_player`:
   `q.deck = [_clone_inst(c) for c in p.deck]`, for both players. (`_clone_battle`
   also shares `rng`, but battle never uses it — draws are pure `pop(0)`.)

Put together: the cloned state carries **both players' complete decks in exact
draw order**, and the forward model draws deterministically off the top. So the
search knows, with certainty:

1. the opponent's current hidden **hand**,
2. the opponent's entire deck **contents and future draw sequence**, and
3. its **own** future draw sequence (the shuffle it cannot legitimately know).

That is not "perfect information about the current position" — it is **perfect
foresight of the whole game**. It is the equivalent of playing with both shuffled
decks lying face-up in order. `mcts`/`azlite` are not hand-cheaters; they are
*future*-cheaters, the strongest form of cheating available here.

### Why it's material, not cosmetic

Foresight lets the search line up exact lethal turns, hold removal it "knows" it
will draw next turn, and assume the opponent's precise future board. None of that
is available to a real agent. It inflates the `mcts`/`azlite` numbers by more than
the hand alone, and it explains the full-roster matrix cleanly:

- `azlite` ties cheating `mcts` head-to-head (0.56) **because they both cheat** —
  same information advantage; `azlite` just searches a little smarter.
- The gap from the cheaters to the fair `dmcts` (`azlite` beats `dmcts` 0.60,
  `mcts` beats `dmcts` 0.54) is partly search quality and **partly the foresight
  they have and `dmcts` does not**.

## `dmcts` — the fair searcher, and its one caveat

`dmcts` is built for imperfect information. `mcts.py:266` `_determinize` *"clones
gs but resamples the OPPONENT's hidden hand + deck from the card pool"*, then runs
MCTS on each of `K` sampled worlds and aggregates. Its docstring (`mcts.py:226`)
spells out the contrast: *"Unlike MCTSBattlePolicy (which peeks at the opponent's
real hand), DMCTS samples … a fair [search]."* That closes leaks #1 and #2 — it
never sees the opponent's true hand or deck order.

**Residual caveat (leak #3, self-only):** `_determinize` resamples only the
*opponent's* deck; it keeps the searcher's **own** real deck order from the state.
Since the own shuffle is also hidden in the real game, `dmcts` still knows its own
future draws. A strictly fair version should re-shuffle its *own* unknown deck
too. So `dmcts` is "fair on the opponent, mildly optimistic about itself." This is
worth tightening (reshuffle own deck in `_determinize`) and re-measuring.

## How the `state` reaches the policies

Search policies receive the full state because every caller passes it as the third
positional arg:

- play harness — `engine.py:135` `action = pols[gs.current].battle_action(view, legal, gs)`
- training env — `battle_env.py:89` `action = self.opponent.battle_action(view, legal, self.gs)` (added so search policies can be *training* opponents; see the env change)

The protocol signature is `battle_action(view, legal, state=None)` for **every**
policy (`base.py`); heuristics and `ppo` ignore `state`, searchers require it
(`azlite.py:134` raises if it is `None`). That uniform signature is exactly what
made the cheating invisible at the call site — the env "just passes the state",
and a perfect-foresight searcher silently consumes it.

## Deployment reality (a `(view, action)` server)

- **Reactive + `ppo`:** work as-is. They only ever needed the visible observation.
- **`dmcts`:** works — but you must ship your **own** engine as a model and a
  reconstructor that turns the visible state into sampled full states (sample the
  opponent's hand+deck; ideally reshuffle your own). It needs a *model*, not
  hidden *information*.
- **`mcts`, `azlite`:** do **not** work, and cannot be made to work without
  fabricating information the server does not send. To run them you would have to
  determinize them — at which point they *become* `dmcts` and shed the inflated
  strength. Their measured numbers are not reproducible against a real opponent.

## What this means for the record

1. **`docs/baseline.md` is wrong about `azlite`.** The AlphaZero-lite section
   calls it "the first policy in this kit to [beat every PPO] **without cheating**
   *and* without a trained net". The "without cheating" half is false — `azlite`
   is perfect-foresight, the same class as `mcts` (its own docstring says
   `(cheating)`). The section should be relabeled.
2. **The project memory is wrong** — the note calling `azlite` the "strongest
   **non-cheating** policy" should be corrected to "perfect-foresight (cheating),
   same class as `mcts`; the fair search is `dmcts`."
3. **The honest ranking**, restricted to what survives a real protocol, is
   **`dmcts` (fair search) > `ppo` / reactive > the ground baselines > `random`**.
   `mcts` and `azlite` are out of fair competition. The single honest
   search-vs-reactive comparison on record is **`dmcts` 0.74 vs `ppo`**.
4. The PPO ceiling argument is unaffected in spirit — a fair search (`dmcts`)
   still beats the reactive net (0.74) — but the *magnitude* of "search dominates"
   was overstated by the cheating searchers. Use `dmcts`, not `mcts`/`azlite`, as
   the fair search yardstick.

## Receipts

| claim | source |
|---|---|
| `azlite` self-labels as cheating | `azlite.py:16` — `Perfect-information (cheating) like MCTSBattlePolicy.` |
| `azlite` requires the full state | `azlite.py:134` — raises `ValueError` if `state is None` |
| `azlite` clones the real state and searches the opponent's real moves | `azlite.py:143` `sim = _clone_battle(state)`; `azlite.py:156` expands via `battle_legal(sim)` |
| `mcts` peeks at the opponent's real hand | `mcts.py:226` (DMCTS docstring contrasting it) |
| clone carries both decks in order | `mcts.py` `_clone_player` — `q.deck = [_clone_inst(c) for c in p.deck]` |
| draws are deterministic `pop(0)` | `battle.py:41` |
| decks are shuffled (hidden order) at draft end | `draft.py:50` — `gs.rng.shuffle(deck)` |
| `dmcts` resamples the opponent's hand+deck (fair) | `mcts.py:226`, `mcts.py:266` `_determinize` |
| every caller passes the full state | `engine.py:135`, `battle_env.py:89` |

## Follow-ups

- [ ] Relabel the `azlite` section in `docs/baseline.md` (and the full-roster
  framing) — perfect-foresight cheating, both hand and future draws.
- [ ] Fix the project memory note ("non-cheating" → "perfect-foresight cheating").
- [ ] Tighten `dmcts._determinize` to also reshuffle the searcher's own deck, then
  re-measure `dmcts` vs `ppo` and the baselines (the strictly-fair number).
- [ ] Consider a `cheats: bool` capability flag on policies so the harness can tag
  results and never silently report a cheating searcher beside fair ones.
