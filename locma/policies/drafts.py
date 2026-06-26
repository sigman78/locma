from __future__ import annotations

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
    _ITEM_DISCOUNT = 12.0

    def __init__(self, name: str = "balanced-draft"):
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
        need = max(0, self._CURVE_TARGET.get(bucket, 0) - counts.get(bucket, 0))
        score = base + 3.0 * need
        if cv.type == _CREATURE:
            if n_creatures < self._CREATURE_TARGET:
                score += self._CREATURE_BONUS
        else:
            score -= self._ITEM_DISCOUNT
        return score

    def draft_action(self, view, legal):
        idx = max(legal, key=lambda i: self._score(view.offered[i]))
        self._picks.append(view.offered[idx])
        return idx
