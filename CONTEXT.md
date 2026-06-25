# LOCMA

Explore kit for Legends of Code & Magic (LOCM 1.5 battle rules). Ubiquitous
language for the rules engine, the agents that play it, and the experiment
harness around them.

## Agents & decision-making

**Policy**:
The engine-facing decision-maker for one seat: it answers every decision the
engine asks of a player across all phases (`draft_action`, `battle_action`,
`reset`, `name`). This is the contract `run_game` and all consumers depend on.
_Avoid_: Agent, Bot, AI, Player (Player is a seat's game state, not its brain).

**Draft Policy**:
A policy half that makes only draft-phase decisions (which card to pick).
Mode-dependent — absent under Constructed play.
_Avoid_: Drafter, DraftAgent.

**Battle Policy**:
A policy half that makes only battle-phase decisions (the action to take).
The constant half: present in every game mode.
_Avoid_: Battler, BattleAgent, Player.

**Composer**:
Builds a Policy by combining a required Battle Policy with an optional Draft
Policy. With no draft policy the composed Policy plays battle only (Constructed
mode); calling its draft decision then is a hard error.
_Avoid_: CompositePolicy, PolicyPair, CombinedPolicy.

## Game modes

**Draft**:
The deck-building phase where each seat picks 1 of 3 offered cards repeatedly to
build its deck before battle. The default LOCM mode.
_Avoid_: Pick phase, deckbuild.

**Constructed**:
A (future) game mode where decks are predetermined, so there is no Draft phase
at all — the engine goes straight to battle.
_Avoid_: Standard, deck-list mode.

**Battle**:
The phase where the two drafted/constructed decks are played out to a winner.

## Search & learning

**Forward model**:
The live engine `GameState` handed to a battle policy as an optional argument so
it can clone and simulate ahead. A perfect-information view (it exposes the
opponent's hidden cards), so policies that use it are "cheating" — acceptable for
a baseline. Treat read-only; clone before simulating.
_Avoid_: World model, simulator (Simulator is a future wrapper over this).

**Search Policy**:
A battle policy that chooses its action by simulating future states through the
Forward model (e.g. MCTS) rather than by a fixed heuristic.

**Rollout Policy**:
The fast battle policy a Search Policy uses to play a simulated game out to a
result during evaluation.
_Avoid_: Default policy, playout policy.
