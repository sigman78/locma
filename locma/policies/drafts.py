from __future__ import annotations

import random

_CREATURE = 0  # CardView.type for creatures
_DRAFT_FACTORIES = {}


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


class WeightedBalancedDraftPolicy(BalancedDraftPolicy):
    """Balanced curve/creature targets with a stronger weighted-card-value core.

    The first `draft-report` pass showed `weighted` has the best raw card value
    but a poor curve, while `balanced` has the best final proxy. This keeps the
    balanced shape and slightly relaxes the creature-only bias to admit premium
    removal.
    """

    _CURVE_WEIGHT = 4.0
    _CREATURE_BONUS = 2.0
    _ITEM_DISCOUNT = 4.0

    def __init__(self, name: str = "weighted-balanced-draft"):
        super().__init__(name=name)

    def _score(self, cv) -> float:
        counts: dict[int, int] = {}
        n_creatures = 0
        for c in self._picks:
            counts[self._bucket(c.cost)] = counts.get(self._bucket(c.cost), 0) + 1
            if c.type == _CREATURE:
                n_creatures += 1
        base = _card_value(cv)
        bucket = self._bucket(cv.cost)
        need = max(0, self._CURVE_TARGET.get(bucket, 0) - counts.get(bucket, 0))
        score = base + self._CURVE_WEIGHT * need
        if cv.type == _CREATURE:
            if n_creatures < self._CREATURE_TARGET:
                score += self._CREATURE_BONUS
        else:
            score -= self._ITEM_DISCOUNT
        return score


_FULL_VALUE_WEIGHTS = {"B": 0.8, "C": 1.2, "D": 1.1, "G": 2.0, "L": 1.8, "W": 1.7}


def _full_card_value(cv) -> float:
    """Full-card draft proxy keyed by CardView.card_id.

    DraftView intentionally exposes a compact card view, but it includes
    `card_id`; for heuristic research policies we can look up static card text
    fields such as HP swing and card draw without using hidden game state.
    """
    from locma.data.cards_db import load_cards  # noqa: PLC0415

    card = _full_card_value._by_id.get(cv.card_id)  # type: ignore[attr-defined]
    if card is None:
        _full_card_value._by_id = {c.id: c for c in load_cards()}  # type: ignore[attr-defined]
        card = _full_card_value._by_id[cv.card_id]  # type: ignore[attr-defined]
    abilities = sum(
        _FULL_VALUE_WEIGHTS[ch] for i, ch in enumerate(_ABILITY_ORDER) if card.abilities[i] != "-"
    )
    hp_swing = 0.8 * card.player_hp - 1.0 * card.enemy_hp
    draw = 2.2 * card.card_draw
    if cv.type == _CREATURE:
        return max(0.0, cv.attack) + max(0.0, cv.defense) + abilities + hp_swing + draw
    stat_effect = min(abs(cv.attack), _STAT_CAP) + min(abs(cv.defense), _STAT_CAP)
    if card.abilities == _ABILITY_ORDER:
        stat_effect += 2.5
    return stat_effect + abilities + hp_swing + draw


_full_card_value._by_id = {}  # type: ignore[attr-defined]


class TrueCostBalancedDraftPolicy(BalancedDraftPolicy):
    """Balanced draft using the full-card value proxy from the cost side quest."""

    _CURVE_WEIGHT = 4.0
    _CREATURE_BONUS = 2.0
    _ITEM_DISCOUNT = 8.0

    def __init__(self, name: str = "truecost-balanced-draft"):
        super().__init__(name=name)

    def _score(self, cv) -> float:
        counts: dict[int, int] = {}
        n_creatures = 0
        for c in self._picks:
            counts[self._bucket(c.cost)] = counts.get(self._bucket(c.cost), 0) + 1
            if c.type == _CREATURE:
                n_creatures += 1
        bucket = self._bucket(cv.cost)
        need = max(0, self._CURVE_TARGET.get(bucket, 0) - counts.get(bucket, 0))
        score = _full_card_value(cv) + self._CURVE_WEIGHT * need
        if cv.type == _CREATURE:
            if n_creatures < self._CREATURE_TARGET:
                score += self._CREATURE_BONUS
        else:
            score -= self._ITEM_DISCOUNT
        return score


class ParametricDraftPolicy(BalancedDraftPolicy):
    """Archetype draft policy with explicit curve/stat/keyword knobs."""

    _PROFILES = {
        "aggro": {
            "curve": {0: 2, 1: 5, 2: 7, 3: 6, 4: 4, 5: 3, 6: 2, 7: 1},
            "creature_target": 26,
            "curve_weight": 4.0,
            "value_weight": 0.7,
            "attack_weight": 1.4,
            "defense_weight": 0.3,
            "over_curve_penalty": 0.8,
            "high_cost_penalty": 1.2,
            "item_discount": 7.0,
            "kw": {"C": 2.0, "B": 1.4, "D": 1.0, "G": 0.5, "L": 0.8, "W": 0.7},
        },
        "midrange": {
            "curve": {0: 1, 1: 3, 2: 5, 3: 6, 4: 6, 5: 4, 6: 3, 7: 2},
            "creature_target": 25,
            "curve_weight": 4.0,
            "value_weight": 1.0,
            "attack_weight": 0.8,
            "defense_weight": 0.8,
            "over_curve_penalty": 0.7,
            "high_cost_penalty": 0.5,
            "item_discount": 8.0,
            "kw": {"C": 1.0, "B": 1.0, "D": 1.0, "G": 1.4, "L": 1.4, "W": 1.3},
        },
        "defense": {
            "curve": {0: 1, 1: 3, 2: 5, 3: 5, 4: 5, 5: 4, 6: 4, 7: 3},
            "creature_target": 26,
            "curve_weight": 4.0,
            "value_weight": 0.8,
            "attack_weight": 0.4,
            "defense_weight": 1.5,
            "over_curve_penalty": 0.6,
            "high_cost_penalty": 0.4,
            "item_discount": 9.0,
            "kw": {"C": 0.4, "B": 0.8, "D": 1.0, "G": 2.5, "L": 1.2, "W": 1.8},
        },
    }

    def __init__(self, archetype: str, name: str | None = None):
        if archetype not in self._PROFILES:
            raise ValueError(f"unknown draft archetype {archetype!r}")
        super().__init__(name=name or f"{archetype}-draft")
        self.archetype = archetype
        self.profile = self._PROFILES[archetype]

    def _score(self, cv) -> float:
        counts: dict[int, int] = {}
        n_creatures = 0
        for c in self._picks:
            counts[self._bucket(c.cost)] = counts.get(self._bucket(c.cost), 0) + 1
            if c.type == _CREATURE:
                n_creatures += 1
        curve = self.profile["curve"]
        bucket = self._bucket(cv.cost)
        target = curve.get(bucket, 0)
        need = max(0, target - counts.get(bucket, 0))
        over = max(0, counts.get(bucket, 0) - target)
        score = self.profile["value_weight"] * _full_card_value(cv)
        score += self.profile["curve_weight"] * need
        score -= self.profile["over_curve_penalty"] * over
        score += self.profile["attack_weight"] * max(0, cv.attack)
        score += self.profile["defense_weight"] * max(0, cv.defense)
        score -= self.profile["high_cost_penalty"] * max(0, cv.cost - 5)
        kw = self.profile["kw"]
        score += sum(kw[ch] for i, ch in enumerate(_ABILITY_ORDER) if cv.abilities[i] != "-")
        if cv.type == _CREATURE:
            if n_creatures < self.profile["creature_target"]:
                score += self._CREATURE_BONUS
        else:
            score -= self.profile["item_discount"]
        return score


class AggroDraftPolicy(ParametricDraftPolicy):
    def __init__(self, name: str = "aggro-draft"):
        super().__init__("aggro", name=name)


class MidrangeDraftPolicy(ParametricDraftPolicy):
    def __init__(self, name: str = "midrange-draft"):
        super().__init__("midrange", name=name)


class DefenseDraftPolicy(ParametricDraftPolicy):
    def __init__(self, name: str = "defense-draft"):
        super().__init__("defense", name=name)


class ImpactWeightsDraftPolicy(BalancedDraftPolicy):
    """Draft from empirical card-impact weights plus curve/creature constraints."""

    def __init__(
        self,
        weights: dict[int, float],
        scale: float = 20.0,
        curve_weight: float = 3.0,
        item_discount: float = 8.0,
        name: str = "impact-weights-draft",
    ):
        super().__init__(name=name)
        self.weights = weights
        self.scale = scale
        self.curve_weight = curve_weight
        self.item_discount = item_discount

    def _score(self, cv) -> float:
        counts: dict[int, int] = {}
        n_creatures = 0
        for c in self._picks:
            counts[self._bucket(c.cost)] = counts.get(self._bucket(c.cost), 0) + 1
            if c.type == _CREATURE:
                n_creatures += 1
        bucket = self._bucket(cv.cost)
        need = max(0, self._CURVE_TARGET.get(bucket, 0) - counts.get(bucket, 0))
        score = self.scale * self.weights.get(cv.card_id, 0.0) + self.curve_weight * need
        if cv.type == _CREATURE:
            if n_creatures < self._CREATURE_TARGET:
                score += self._CREATURE_BONUS
        else:
            score -= self.item_discount
        return score


_DRAFT_FACTORIES = {
    "random": RandomDraftPolicy,
    "greedy": GreedyDraftPolicy,
    "weighted": WeightedDraftPolicy,
    "balanced": BalancedDraftPolicy,
    "weighted-balanced": WeightedBalancedDraftPolicy,
    "truecost-balanced": TrueCostBalancedDraftPolicy,
    "aggro": AggroDraftPolicy,
    "midrange": MidrangeDraftPolicy,
    "defense": DefenseDraftPolicy,
    "max-guard": MaxGuardDraftPolicy,
    "max-attack": MaxAttackDraftPolicy,
    "max-defense": MaxDefenseDraftPolicy,
}


def draft_policy_names() -> list[str]:
    return list(_DRAFT_FACTORIES)


def make_draft_policy(spec: str):
    """Build a draft-only policy from ``draft:<name>`` or a bare draft name."""
    prefix, sep, name = spec.partition(":")
    if sep:
        if prefix != "draft":
            raise ValueError("draft specs must look like draft:<name>")
    else:
        name = spec
    factory = _DRAFT_FACTORIES.get(name)
    if factory is None:
        allowed = ", ".join(sorted(f"draft:{n}" for n in _DRAFT_FACTORIES))
        raise ValueError(f"unknown draft policy {spec!r}; expected one of: {allowed}")
    return factory()
