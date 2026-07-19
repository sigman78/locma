"""Exhaustive own-turn lethal solver + wrapper policy (E26 micro-guard 1).

Motivation (E14a finding 3a, restated in the E15 closure as a surviving
practical item): shadow-driver diagnostics on the reactive net found it
misses **18.3%** of engine-verified forced wins available on its own turn —
not a training-data or observability gap (E14a/E16a ruled those out), just a
net that does not always find the kill even when the position hands it one.
Rather than spend more training compute chasing a capacity wall (E15 closed
that program), this module bolts on a cheap, exact, zero-training search: at
each decision, ask "is there a sequence of my own actions this turn that wins
right now?" If yes, play it move by move; if no, get out of the way and let
the wrapped policy decide as usual.

Fairness. The search is exhaustive DFS over the SAME forward-model class as
``vbeam`` (``_clone_battle`` battle-only clones, ``battle_legal``/
``apply_battle``): own-turn only, and it never simulates ``Pass`` — passing
ends the turn and triggers the opponent's hidden draw, so simulating it would
leak information the real player does not have at decision time. The engine
is deterministic within a turn (draws only happen at ``start_turn``), so
own-turn lookahead here uses no hidden information, exactly like ``vbeam``'s
beam. This is a *solver*, not a heuristic: every returned line is a real,
engine-verified win, and "no lethal found" is only ever claimed when the
search actually exhausted every reachable non-Pass continuation (a cap hit
is reported separately and never asserts absence).

Negative-cache soundness. When ``find_lethal`` exhausts its search from some
state ``s`` without finding a win, it has, by construction, visited every
state reachable from ``s`` via own-turn action sequences (module ``seen``
dedup collapses order permutations of the same view, so this is a true
closure over the state space, not just the paths tried). Any LATER decision
point in the SAME turn is one of those already-visited states (the engine
only ever adds actions within a turn, never removes the ones already taken),
so its own reachable set is a SUBSET of what the first exhausted search
already covered. It is therefore sound to skip re-searching for the rest of
the turn once one exhaustive (non-cap-hit) search comes back empty — the
``LethalGuardBattlePolicy`` per-turn cache below relies on exactly this.
"""

from __future__ import annotations

from locma.core import battle as battlemod
from locma.core.actions import Attack, Pass, Summon, Use
from locma.core.engine import make_battle_view
from locma.core.state import Phase
from locma.policies.mcts import _clone_battle


def _order_key(action) -> int:
    """Speed-only heuristic ranking for DFS action order (does not affect
    exhaustiveness — every legal non-Pass action is still tried).

    Likely-lethal actions first: face Attack, then face Use, then other
    Attack, then targeted Use, then Summon (develops the board, least likely
    to be the final blow this turn).
    """
    if isinstance(action, Attack) and action.target_id == -1:
        return 0
    if isinstance(action, Use) and action.target_id == -1:
        return 1
    if isinstance(action, Attack):
        return 2
    if isinstance(action, Use):
        return 3
    if isinstance(action, Summon):
        return 4
    return 5


def find_lethal(state, node_cap: int = 3000, stats: dict | None = None) -> tuple[list | None, bool]:
    """Exhaustive DFS for a forced win on ``state.current``'s turn.

    Returns ``(line, exhausted)``:
      - ``(actions, True)`` — ``actions`` is a real action sequence that, when
        replayed via ``apply_battle`` from a clone of ``state``, ends the game
        with ``winner == state.current``.
      - ``(None, True)`` — the search exhausted every own-turn continuation
        (never simulating ``Pass``) without finding a win: no lethal exists
        from this state this turn (see the module docstring for why later
        decisions in the same turn can trust this without re-searching).
      - ``(None, False)`` — the search hit ``node_cap`` before exhausting;
        absence is NOT established (a smaller subtree explored later, e.g.
        after the board has thinned, may still complete).

    Never mutates ``state`` — all simulation runs on ``_clone_battle`` copies.
    Iterative (explicit stack) DFS so ``node_cap`` bounds worst-case work
    without recursion-depth concerns; the stack is LIFO, so each node's legal
    actions are pushed in REVERSE of the desired try order (```_order_key```
    ascending) to explore the most-likely-lethal action first.

    ``stats``, if given, is a mutable dict updated in place with node-count
    bookkeeping (``stats["nodes"] += <count>``) — the one clean extension
    point so ``LethalGuardBattlePolicy`` can report mean nodes/search without
    ``find_lethal`` knowing about the wrapper's counters.
    """
    seat = state.current
    root = _clone_battle(state)
    seen = {make_battle_view(root)}
    stack: list[tuple[object, list]] = [(root, [])]
    nodes = 0

    while stack:
        sim, plan = stack.pop()
        legal = list(battlemod.battle_legal(sim))
        # Try the most-likely-lethal actions first; LIFO stack means we push
        # in the OPPOSITE of desired pop order (least-likely first, so it
        # ends up at the bottom and the most-likely action is popped next).
        legal.sort(key=_order_key, reverse=True)
        for action in legal:
            if isinstance(action, Pass):
                continue  # never simulate Pass — see module docstring
            nodes += 1
            if nodes > node_cap:
                if stats is not None:
                    stats["nodes"] = stats.get("nodes", 0) + nodes
                return None, False
            s2 = _clone_battle(sim)
            battlemod.apply_battle(s2, action)
            plan2 = [*plan, action]
            if s2.phase == Phase.ENDED:
                if s2.winner == seat:
                    if stats is not None:
                        stats["nodes"] = stats.get("nodes", 0) + nodes
                    return plan2, True
                continue  # self-inflicted loss: prune, do not explore further
            v2 = make_battle_view(s2)
            if v2 in seen:
                continue  # order permutation already explored (or queued)
            seen.add(v2)
            stack.append((s2, plan2))

    if stats is not None:
        stats["nodes"] = stats.get("nodes", 0) + nodes
    return None, True


class LethalGuardBattlePolicy:
    """Wraps a battle policy: plays a found lethal line, else delegates.

    Per-turn negative cache: once ``find_lethal`` exhausts without a win at
    some decision point, every later decision in the SAME ``view.turn`` is
    known (see module docstring) to lie in that search's explored closure, so
    the guard skips re-searching for the rest of the turn and delegates
    straight to ``inner``. The cache resets whenever the turn number changes.

    ``stats`` accumulates across games (the E26 mechanism probe reads it
    after many matches) and is NOT reset by ``reset()`` — only ``_plan`` and
    the per-turn cache are.
    """

    def __init__(self, inner, name: str = "lguard", node_cap: int = 3000, probe: bool = False):
        self.inner = inner
        self.name = name
        self.node_cap = node_cap
        self.probe = probe
        self._plan: list = []
        self._no_lethal_turn: int | None = None
        self.stats = {
            "decisions": 0,
            "searches": 0,
            "activations": 0,
            "cap_hits": 0,
            "nodes": 0,
            "guard_changed_move": 0,
        }

    def battle_action(self, view, legal, state=None):
        self.stats["decisions"] += 1
        if self._plan:
            a = self._plan.pop(0)
            if a in legal:
                return a
            self._plan = []  # stale cache (should not happen) — fall through below

        if state is None or len(legal) == 1:
            return self.inner.battle_action(view, legal, state)

        if self._no_lethal_turn == view.turn:
            return self.inner.battle_action(view, legal, state)

        self.stats["searches"] += 1
        line, exhausted = find_lethal(state, node_cap=self.node_cap, stats=self.stats)

        if line is not None:
            self.stats["activations"] += 1
            if self.probe:
                inner_action = self.inner.battle_action(view, legal, state)
                if inner_action != line[0]:
                    self.stats["guard_changed_move"] += 1
            self._plan = line[1:]
            return line[0]

        if exhausted:
            self._no_lethal_turn = view.turn
            return self.inner.battle_action(view, legal, state)

        # cap hit: absence not established — delegate WITHOUT caching, a
        # later (smaller) subtree this same turn may still complete.
        self.stats["cap_hits"] += 1
        return self.inner.battle_action(view, legal, state)

    def reset(self, seed=None) -> None:
        self._plan = []
        self._no_lethal_turn = None
        self.inner.reset(seed)
