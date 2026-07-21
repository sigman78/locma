from __future__ import annotations

import json
import random

_CREATURE = 0  # CardView.type for creatures


class RandomDraftPolicy:
    def __init__(self, name: str = "random-draft", seed: int = 0):
        self.name = name
        self._seed = seed
        self._r = random.Random(seed)

    def draft_action(self, view, legal):
        return self._r.choice(legal)

    def reset(self, seed=None):
        self._r = random.Random(self._seed if seed is None else seed)


def _kw_count(abilities: str) -> int:
    return sum(1 for ch in abilities if ch != "-")


def _score(cv) -> float:
    base = cv.attack + cv.defense + 0.5 * _kw_count(cv.abilities)
    if cv.type != 0:  # items slightly deprioritized in draft
        base -= 1.0
    return base


class GreedyDraftPolicy:
    def __init__(self, name: str = "greedy-draft"):
        self.name = name

    def draft_action(self, view, legal):
        scores = [_score(cv) for cv in view.offered]
        return max(legal, key=lambda i: scores[i])

    def reset(self, seed=None):
        pass


class MaxGuardDraftPolicy:
    """Draft Guard creatures above all else."""

    def __init__(self, name: str = "max-guard-draft"):
        self.name = name

    def draft_action(self, view, legal):
        def key(i):
            cv = view.offered[i]
            is_creature = cv.type == _CREATURE
            has_guard = is_creature and "G" in cv.abilities
            return (has_guard, is_creature, cv.attack + cv.defense)

        return max(legal, key=key)

    def reset(self, seed=None):
        pass


class MaxAttackDraftPolicy:
    """Draft the highest-attack creature."""

    def __init__(self, name: str = "max-attack-draft"):
        self.name = name

    def draft_action(self, view, legal):
        def key(i):
            cv = view.offered[i]
            is_creature = cv.type == _CREATURE
            return (is_creature, cv.attack, cv.defense)

        return max(legal, key=key)

    def reset(self, seed=None):
        pass


class MaxDefenseDraftPolicy:
    """Draft the highest-defense creature (a tanky board). Distinct from
    `max-guard`, which prioritises the Guard keyword over raw defense."""

    def __init__(self, name: str = "max-defense-draft"):
        self.name = name

    def draft_action(self, view, legal):
        def key(i):
            cv = view.offered[i]
            is_creature = cv.type == _CREATURE
            return (is_creature, cv.defense, cv.attack)

        return max(legal, key=key)

    def reset(self, seed=None):
        pass


# Per-keyword draft value (tuned): Guard most, then Lethal/Ward, then the
# tempo/aggression keywords. Used by the weighted heuristic.
_KW_WEIGHT = {"G": 2.0, "L": 1.5, "W": 1.5, "C": 1.0, "B": 1.0, "D": 0.5}
_ABILITY_ORDER = "BCDGLW"


def _kw_value(abilities: str) -> float:
    return sum(_KW_WEIGHT[k] for k in _KW_WEIGHT if abilities[_ABILITY_ORDER.index(k)] != "-")


_STAT_CAP = 13  # clamps removal sentinels (e.g. Decimate's defense -99) to a sane bound


def _card_value(cv) -> float:
    """Draft-time board-impact value, spell-aware.

    Creatures: attack + defense + keyword value (their stats are positive).
    Items (spells): the MAGNITUDE of their effect. Red/blue items carry NEGATIVE
    attack/defense that is applied to the ENEMY minion (removal/damage), so a
    -7-defense removal spell is worth +7, not -7; green items carry positive buffs.
    Either way the magnitude is the value, capped so destroy-sentinels (e.g.
    Decimate, defense -99) don't dominate. (CardView does not expose card-draw or
    hp-swing, so pure utility items are under-valued — a known view limitation.)
    """
    kw = _kw_value(cv.abilities)
    if cv.type == _CREATURE:
        return cv.attack + cv.defense + kw
    return min(abs(cv.attack), _STAT_CAP) + min(abs(cv.defense), _STAT_CAP) + kw


class WeightedDraftPolicy:
    """Stateless improved-greedy: card value = attack + defense + per-keyword
    weights, items lightly penalised. Values keywords properly (vs `greedy`'s flat
    0.5-per-keyword) so Guard/Lethal/Ward creatures are picked over raw stats."""

    def __init__(self, name: str = "weighted-draft"):
        self.name = name

    def _score(self, cv) -> float:
        s = _card_value(cv)  # spell-aware: removal/damage items valued by effect
        if cv.type != _CREATURE:
            s -= 1.0  # mild creature preference (recurring board presence)
        return s

    def draft_action(self, view, legal):
        return max(legal, key=lambda i: self._score(view.offered[i]))

    def reset(self, seed=None):
        pass


class BalancedDraftPolicy:
    """Curve-aware, creature-majority draft (stateful). Tracks its own picks and
    prefers cards that fill an under-target cost bucket, keeping a healthy mana
    curve and a creature majority, with raw stats as a tie-breaker."""

    # Target deck shape over 30 cards (cost bucket -> desired count); cost capped at 7.
    _CURVE_TARGET = {0: 1, 1: 3, 2: 5, 3: 5, 4: 5, 5: 4, 6: 3, 7: 4}
    _CREATURE_TARGET = 24
    _CREATURE_BONUS = 2.0
    # Items are spell-valued correctly (see _card_value) but the learned battle net
    # plays creatures far better than spells: tuning the discount vs the PPO net
    # (1.5→6→12 gave 0.47→0.52→0.56 avg vs the hard baselines) showed a strong
    # creature bias is best, so the deck stays creature-heavy and only takes
    # genuinely premium removal (e.g. Decimate). See docs/baseline.md.
    # That tuning was against the REACTIVE pilot; the vbeam planner converts items
    # ~1.7x better per opportunity (E16a), so ``item_discount`` is a constructor
    # parameter -- E17 sweeps it under the planner. Default preserves the historic
    # reactive-tuned behavior.
    _ITEM_DISCOUNT = 12.0

    def __init__(
        self,
        name: str | None = None,
        item_discount: float = _ITEM_DISCOUNT,
        curve_target: dict[int, int] | None = None,
    ):
        self.item_discount = item_discount
        self.curve_target = curve_target if curve_target is not None else self._CURVE_TARGET
        if name is None:
            suffix = "" if item_discount == self._ITEM_DISCOUNT else f"-d{item_discount:g}"
            suffix += "" if curve_target is None else "-curve"
            name = f"balanced-draft{suffix}"
        self.name = name
        self._picks: list = []

    def reset(self, seed=None):
        self._picks = []

    def _bucket(self, cost: int) -> int:
        return min(cost, 7)

    def _score(self, cv) -> float:
        counts: dict[int, int] = {}
        n_creatures = 0
        for c in self._picks:
            counts[self._bucket(c.cost)] = counts.get(self._bucket(c.cost), 0) + 1
            if c.type == _CREATURE:
                n_creatures += 1
        base = _card_value(cv)  # spell-aware: removal/damage items valued by effect
        bucket = self._bucket(cv.cost)
        need = max(0, self.curve_target.get(bucket, 0) - counts.get(bucket, 0))
        score = base + 3.0 * need
        if cv.type == _CREATURE:
            if n_creatures < self._CREATURE_TARGET:
                score += self._CREATURE_BONUS
        else:
            score -= self.item_discount
        return score

    def note_pick(self, view, idx):
        """Record a pick made on this policy's behalf (e.g. an overridden pick by
        PartialRandomDraftPolicy), keeping the curve/creature tracking accurate."""
        self._picks.append(view.offered[idx])

    def note_cards(self, cards):
        """Seed the tracker with picks made before this policy took over (the web
        Play auto-complete, where the human drafts the first rounds manually).
        Accepts any card-likes exposing ``cost``/``type`` (core Card or CardView)."""
        self._picks.extend(cards)

    def draft_action(self, view, legal):
        idx = max(legal, key=lambda i: self._score(view.offered[i]))
        self.note_pick(view, idx)
        return idx


class DistilledDraftPolicy:
    """Card-priority-table draft (E20): a per-card static value plus
    ``BalancedDraftPolicy``'s curve-need / creature-deficit context terms, no
    neural net at draft time. The value table (and optionally the curve target
    and the two context weights) come from ``scripts/e20_draftdistill.py``,
    either fit by multinomial logistic regression against a learned draft
    net's (E18b) revealed picks, or ELICITED directly from the net via random
    neutral-context comparisons (round 0, empty deck) and spliced onto the
    census-recalibrated curve/creature terms -- see the script for both
    methods. Stateful like ``BalancedDraftPolicy``, whose curve-tracking logic
    it reuses."""

    _CURVE_TARGET = BalancedDraftPolicy._CURVE_TARGET
    _CREATURE_TARGET = BalancedDraftPolicy._CREATURE_TARGET

    def __init__(
        self,
        values: dict[int, float],
        w_need: float,
        w_creature: float,
        curve_target: dict[int, int] | None = None,
        name: str = "distilled-draft",
    ):
        self.values = values
        self.w_need = w_need
        self.w_creature = w_creature
        self.curve_target = curve_target if curve_target is not None else self._CURVE_TARGET
        self.name = name
        self._picks: list = []

    @classmethod
    def load(cls, path: str) -> DistilledDraftPolicy:
        with open(path, encoding="utf-8") as f:
            fit = json.load(f)
        values = {int(k): v for k, v in fit["values"].items()}
        curve_target = fit.get("curve_target")
        if curve_target is not None:
            curve_target = {int(k): v for k, v in curve_target.items()}
        return cls(values, fit["w_need"], fit["w_creature"], curve_target=curve_target)

    def reset(self, seed=None):
        self._picks = []

    def _bucket(self, cost: int) -> int:
        return min(cost, 7)

    def _score(self, cv) -> float:
        counts: dict[int, int] = {}
        n_creatures = 0
        for c in self._picks:
            counts[self._bucket(c.cost)] = counts.get(self._bucket(c.cost), 0) + 1
            if c.type == _CREATURE:
                n_creatures += 1
        bucket = self._bucket(cv.cost)
        need = max(0, self.curve_target.get(bucket, 0) - counts.get(bucket, 0))
        score = self.values.get(cv.card_id, 0.0) + self.w_need * need
        if cv.type == _CREATURE and n_creatures < self._CREATURE_TARGET:
            score += self.w_creature
        return score

    def note_pick(self, view, idx):
        """Record a pick made on this policy's behalf (e.g. an overridden pick by
        PartialRandomDraftPolicy), keeping the curve/creature tracking accurate."""
        self._picks.append(view.offered[idx])

    def note_cards(self, cards):
        """Seed the tracker with picks made before this policy took over (see
        BalancedDraftPolicy.note_cards)."""
        self._picks.extend(cards)

    def draft_action(self, view, legal):
        idx = max(legal, key=lambda i: self._score(view.offered[i]))
        self.note_pick(view, idx)
        return idx


class BothSeatsDraftPolicy:
    """Route BattleEnv's both-seats drafting to two independent child policies.

    In BattleEnv the OPPONENT policy drafts for both seats, alternating within
    each round (seat 0 picks first under the default rule). A stateful draft
    (``balanced``, learned ``ppo-draft``) tracking its own picks would mix the
    two decks into one 60-card history; this wrapper keeps a child per seat:
    the first pick of each round goes to ``first``, the second to ``second``.

    Default draft variant only -- the shared variant alternates WHICH seat
    picks first by round, so call order no longer identifies the seat (raises
    if a shared-draft view is detected). For run_game-style single-seat use,
    pass the underlying policy directly; this wrapper is a training-env aid.
    """

    def __init__(self, first, second, name: str | None = None):
        self.first = first
        self.second = second
        self.name = name if name is not None else f"{first.name}-x2"
        self._round = -1
        self._i = 0

    def reset(self, seed=None):
        self._round = -1
        self._i = 0
        self.first.reset(seed)
        self.second.reset(seed)

    def _child(self, view):
        if view.taken is not None:  # shared draft: first picker alternates
            raise ValueError("BothSeatsDraftPolicy supports the default draft variant only")
        if view.round != self._round:
            self._round = view.round
            self._i = 0
        else:
            self._i += 1
        return self.first if self._i % 2 == 0 else self.second

    def note_pick(self, view, idx):
        """Advance the routing and forward to the child's own tracking (the
        PartialRandomDraftPolicy override hook)."""
        child = self._child(view)
        note = getattr(child, "note_pick", None)
        if note is not None:
            note(view, idx)

    def draft_action(self, view, legal):
        return self._child(view).draft_action(view, legal)


class PartialRandomDraftPolicy:
    """Wrap a base draft policy, overriding exactly ``k`` of the 30 draft rounds
    with a uniformly random pick; every other round delegates to ``base``.

    The random rounds are re-sampled on each ``reset`` (seeded, so an episode is
    reproducible) and keyed on ``view.round`` — not on call count — so the wrapper
    behaves identically whether it drafts one seat (run_game) or both seats
    alternately (BattleEnv, where the opponent drafts for both players): each deck
    gets exactly ``k`` random picks either way.

    If the base policy is stateful (``balanced``), its ``note_pick`` hook is called
    on overridden rounds so its internal deck tracking stays accurate.
    """

    def __init__(self, base, k: int, seed: int = 0, rounds: int = 30, name: str | None = None):
        if not 0 <= k <= rounds:
            raise ValueError(f"k must be in [0, {rounds}], got {k}")
        self.base = base
        self.k = k
        self._rounds = rounds
        self._seed = seed
        self.name = name if name is not None else f"{base.name}+rnd{k}"
        self.reset(seed)

    def reset(self, seed=None):
        eff = self._seed if seed is None else seed
        self._r = random.Random(eff)
        self._random_rounds = frozenset(self._r.sample(range(self._rounds), self.k))
        self.base.reset(seed)

    def draft_action(self, view, legal):
        if view.round in self._random_rounds:
            idx = self._r.choice(legal)
            note = getattr(self.base, "note_pick", None)
            if note is not None:
                note(view, idx)
            return idx
        return self.base.draft_action(view, legal)
