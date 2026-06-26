"""Cheating perfect-information MCTS battle policy.

Uses the engine's live ``GameState`` (passed as the optional ``state`` forward
model) as a perfect-information simulator: ``_clone_battle`` makes a fast
battle-only copy (sharing the immutable cards) and drives ``battle_legal`` /
``apply_battle`` to explore. Deterministic from ``seed`` so games remain
byte-identically replayable.

Rollouts are heuristic by default (``rollout_turns=3``): play out a few turns then
score the settled position with a board/health heuristic. This is far stronger AND
~6x faster than random-rollout-to-terminal (set ``rollout_turns<=0`` for the legacy
terminal rollout).
"""

from __future__ import annotations

import math
import random

from locma.core import battle as battlemod
from locma.core.engine import make_battle_view
from locma.core.instance import CardInstance
from locma.core.state import GameState, Phase
from locma.policies.battles import RandomBattlePolicy


def _clone_inst(c: CardInstance) -> CardInstance:
    d = CardInstance.__new__(CardInstance)
    d.card = c.card  # share the immutable (frozen) Card template
    d.instance_id = c.instance_id
    d.attack = c.attack
    d.defense = c.defense
    d.abilities = c.abilities
    d.can_attack = c.can_attack
    d.has_attacked = c.has_attacked
    return d


def _clone_player(p):
    q = type(p).__new__(type(p))
    q.health = p.health
    q.mana = p.mana
    q.max_mana = p.max_mana
    q.bonus_mana = p.bonus_mana
    q.damage_counter = p.damage_counter
    q.bonus_draw = p.bonus_draw
    q.deck = [_clone_inst(c) for c in p.deck]
    q.hand = [_clone_inst(c) for c in p.hand]
    q.board = [_clone_inst(c) for c in p.board]
    return q


def _clone_battle(gs: GameState) -> GameState:
    """Fast battle-only clone of a GameState for MCTS forward simulation.

    Copies only the mutable battle state (players + their card-instance lists) and
    SHARES the immutable frozen ``Card`` templates plus the battle-unused ``rng`` /
    ``draft_pool`` / ``picks``. Equivalent to ``copy.deepcopy`` for battle purposes
    (battle never mutates the shared fields) but ~6.7x faster — deepcopy was ~74%
    of MCTS time. Behaviour is byte-identical, so replay determinism is preserved.
    """
    c = GameState.__new__(GameState)
    c.rng = gs.rng
    c.phase = gs.phase
    c.turn = gs.turn
    c.current = gs.current
    c.draft_pool = gs.draft_pool
    c.draft_round = gs.draft_round
    c.picks = gs.picks
    c.winner = gs.winner
    c.players = (_clone_player(gs.players[0]), _clone_player(gs.players[1]))
    return c


class _Node:
    __slots__ = ("parent", "action", "seat", "children", "untried", "visits", "value")

    def __init__(self, parent, action, seat, untried):
        self.parent = parent
        self.action = action
        self.seat = seat  # seat to move at this node's state
        self.children = []
        self.untried = untried  # list[Action] not yet expanded
        self.visits = 0
        self.value = 0.0  # cumulative reward from the ROOT seat's perspective


def _board_power(p) -> int:
    return sum(c.attack + c.defense for c in p.board)


class MCTSBattlePolicy:
    def __init__(
        self,
        name: str = "mcts",
        iterations: int = 100,
        c: float = math.sqrt(2),
        seed: int = 0,
        rollout=None,
        turn_cap: int = 200,
        rollout_turns: int = 3,
    ):
        self.name = name
        self.iterations = iterations
        self.c = c
        self._seed = seed
        self.turn_cap = turn_cap
        # rollout_turns > 0: play out random until this many turn boundaries pass,
        # then evaluate the (settled) position with a board/health heuristic — far
        # stronger AND faster than random-rollout-to-terminal (a lower-variance leaf
        # estimate). rollout_turns <= 0 restores the legacy terminal rollout.
        self.rollout_turns = rollout_turns
        self.rollout = rollout or RandomBattlePolicy("mcts-rollout", seed=seed)
        # The default random rollout ignores the BattleView, so we skip building it
        # (a big per-ply cost). A custom rollout that reads the view still gets one.
        self._rollout_view = not isinstance(self.rollout, RandomBattlePolicy)
        self._r = random.Random(seed)

    def reset(self, seed=None):
        s = self._seed if seed is None else seed
        self._r = random.Random(s)
        self.rollout.reset(s)

    # -- helpers -------------------------------------------------------------

    def _reward(self, sim, root_seat: int) -> float:
        if sim.winner is not None:
            return 1.0 if sim.winner == root_seat else -1.0
        # turn cap reached with no winner: sign of health differential
        me = sim.players[root_seat].health
        op = sim.players[1 - root_seat].health
        return 1.0 if me > op else (-1.0 if me < op else 0.0)

    def _leaf_value(self, sim, root_seat: int) -> float:
        """Heuristic position value in [-1, 1] from root_seat's perspective:
        health lead + 0.5 * board-power lead, normalized and clipped."""
        me = sim.players[root_seat]
        op = sim.players[1 - root_seat]
        h = (me.health - op.health) / 30.0
        b = (_board_power(me) - _board_power(op)) / 20.0
        v = h + 0.5 * b
        return 1.0 if v > 1.0 else (-1.0 if v < -1.0 else v)

    def _rollout(self, sim, root_seat: int) -> float:
        if self.rollout_turns <= 0:  # legacy: random rollout to terminal
            while sim.phase == Phase.BATTLE and sim.turn <= self.turn_cap:
                legal = battlemod.battle_legal(sim)
                view = make_battle_view(sim) if self._rollout_view else None
                battlemod.apply_battle(sim, self.rollout.battle_action(view, legal))
            return self._reward(sim, root_seat)
        # heuristic rollout: random-play until `rollout_turns` turn boundaries pass
        # (adaptive depth — a turn is variable length), then evaluate the settled
        # position. Resolving the pending combat first keeps the heuristic honest.
        tc = 0
        while sim.phase == Phase.BATTLE and sim.turn <= self.turn_cap and tc < self.rollout_turns:
            owner = sim.current
            legal = battlemod.battle_legal(sim)
            view = make_battle_view(sim) if self._rollout_view else None
            battlemod.apply_battle(sim, self.rollout.battle_action(view, legal))
            if sim.current != owner:
                tc += 1
        if sim.phase != Phase.BATTLE:  # game ended during the rollout
            return self._reward(sim, root_seat)
        return self._leaf_value(sim, root_seat)

    def _ucb_child(self, node: _Node, root_seat: int) -> _Node:
        log_n = math.log(node.visits)
        best, best_score = None, -math.inf
        for ch in node.children:
            exploit = ch.value / ch.visits
            if node.seat != root_seat:  # opponent picks root-pessimal moves
                exploit = -exploit
            score = exploit + self.c * math.sqrt(log_n / ch.visits)
            if score > best_score:
                best, best_score = ch, score
        return best

    # -- policy interface ----------------------------------------------------

    def battle_action(self, view, legal, state=None):
        if state is None:
            raise ValueError("MCTSBattlePolicy requires the forward-model `state` argument")
        if len(legal) == 1:
            return legal[0]

        root_seat = state.current
        root = _Node(None, None, root_seat, list(legal))

        for _ in range(self.iterations):
            sim = _clone_battle(state)
            node = root

            # --- selection: descend fully-expanded nodes by UCB ---
            while not node.untried and node.children and sim.phase == Phase.BATTLE:
                node = self._ucb_child(node, root_seat)
                battlemod.apply_battle(sim, node.action)

            # --- expansion: try one untried action ---
            if node.untried and sim.phase == Phase.BATTLE:
                action = node.untried.pop(self._r.randrange(len(node.untried)))
                battlemod.apply_battle(sim, action)
                child = _Node(node, action, sim.current, list(battlemod.battle_legal(sim)))
                node.children.append(child)
                node = child

            # --- simulation ---
            reward = self._rollout(sim, root_seat)

            # --- backpropagation (root-seat perspective) ---
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent

        return max(root.children, key=lambda ch: ch.visits).action
