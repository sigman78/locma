from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from locma.core.engine import run_game
from locma.data.cards_db import load_cards
from locma.harness.match import run_match
from locma.policies.battles import GreedyBattlePolicy, GroundBattlePolicy
from locma.policies.composer import Composer
from locma.policies.drafts import _CREATURE, BalancedDraftPolicy, make_draft_policy
from locma.stats.intervals import binomial_test, wilson_ci


class _OffsetRandomDraftPolicy:
    def __init__(self, offset: int):
        self.name = f"impact-random-{offset}"
        self.offset = offset
        self._rng = random.Random(offset)

    def draft_action(self, view, legal):
        return self._rng.choice(legal)

    def reset(self, seed=None):
        base = 0 if seed is None else seed
        self._rng = random.Random(base + self.offset)


@dataclass(frozen=True)
class CardImpact:
    card_id: int
    name: str
    cost: int
    type: str
    coefficient: float
    effective_cost_delta: float


@dataclass(frozen=True)
class CardImpactReport:
    games: int
    battle: str
    alpha: float
    rows: list[CardImpact]


def write_card_impact(report: CardImpactReport, path: str | Path) -> None:
    payload = {
        "format": 1,
        "games": report.games,
        "battle": report.battle,
        "alpha": report.alpha,
        "weights": {str(row.card_id): row.coefficient for row in report.rows},
        "rows": [row.__dict__ for row in report.rows],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_card_impact_weights(path: str | Path) -> dict[int, float]:
    try:
        payload = json.loads(Path(path).read_text())
    except FileNotFoundError as e:
        raise ValueError(f"card-impact artifact not found: {path}") from e
    if payload.get("format") != 1:
        raise ValueError(f"unsupported card-impact artifact format: {payload.get('format')!r}")
    weights = payload.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("card-impact artifact is missing weights")
    return {int(k): float(v) for k, v in weights.items()}


@dataclass(frozen=True)
class ImpactDraftCandidate:
    name: str
    scale: float
    curve_weight: float
    item_discount: float
    win_rate: float
    ci_low: float
    ci_high: float
    p_value: float
    wins: int
    losses: int
    games: int


@dataclass(frozen=True)
class ImpactDraftSweep:
    fit_report: CardImpactReport
    battle: str
    reference: str
    eval_games: int
    eval_seed: int
    rows: list[ImpactDraftCandidate]


class ImpactDraftPolicy(BalancedDraftPolicy):
    """Draft by fitted empirical card impact plus curve/creature constraints."""

    def __init__(
        self,
        weights: dict[int, float],
        scale: float = 20.0,
        curve_weight: float = 4.0,
        item_discount: float = 8.0,
        name: str = "impact-draft",
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


def _battle_policy(name: str):
    if name == "ground":
        return GroundBattlePolicy()
    if name == "greedy":
        return GreedyBattlePolicy()
    from locma.policies.registry import make_policy  # noqa: PLC0415

    policy = make_policy(name)
    return policy.battle


def _deck_counts(deck_ids: list[int], card_count: int) -> np.ndarray:
    x = np.zeros(card_count, dtype=float)
    for cid in deck_ids:
        x[cid - 1] += 1.0
    return x


def estimate_card_impact(
    games: int = 1000, seed: int = 0, battle: str = "ground", alpha: float = 10.0
):
    """Estimate empirical card impact from random-draft same-battle games.

    Each row is one game: feature = player0 deck card counts minus player1 deck
    card counts; target = +1 when player0 wins, -1 when player1 wins. A ridge fit
    gives a rough card contribution under the selected battle policy. This is not
    causal truth, but it is a useful empirical complement to static card text
    valuation.
    """
    if games < 1:
        raise ValueError("games must be >= 1")
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    cards = load_cards()
    xs: list[np.ndarray] = []
    ys: list[float] = []
    p0 = Composer(_battle_policy(battle), _OffsetRandomDraftPolicy(0), name=f"{battle}+random")
    p1 = Composer(
        _battle_policy(battle), _OffsetRandomDraftPolicy(10_000_000), name=f"{battle}+random"
    )
    for i in range(games):
        decks: dict[int, list[int]] = {}

        def on_snapshot(gs, decks=decks):
            for seat in (0, 1):
                hand = [inst.card.id for inst in gs.players[seat].hand]
                deck = [inst.card.id for inst in gs.players[seat].deck]
                board = [inst.card.id for inst in gs.players[seat].board]
                decks[seat] = hand + deck + board

        result = run_game(p0, p1, seed=seed + i, on_snapshot=on_snapshot)
        x = _deck_counts(decks[0], len(cards)) - _deck_counts(decks[1], len(cards))
        xs.append(x)
        ys.append(1.0 if result.winner == 0 else -1.0)

    xmat = np.vstack(xs)
    yvec = np.asarray(ys, dtype=float)
    reg = alpha * np.eye(xmat.shape[1], dtype=float)
    weights = np.linalg.solve(xmat.T @ xmat + reg, xmat.T @ yvec)
    values = np.asarray([weights[c.id - 1] for c in cards], dtype=float)
    costs = np.asarray([c.cost for c in cards], dtype=float)
    if float(np.ptp(values)) < 1e-12:
        slope, intercept = 0.0, float(np.mean(costs))
    else:
        slope, intercept = np.polyfit(values, costs, deg=1)
    rows = [
        CardImpact(
            card_id=c.id,
            name=c.name,
            cost=c.cost,
            type=c.type.name.lower(),
            coefficient=float(weights[c.id - 1]),
            effective_cost_delta=float((slope * weights[c.id - 1] + intercept) - c.cost),
        )
        for c in cards
    ]
    return CardImpactReport(games=games, battle=battle, alpha=alpha, rows=rows)


def sweep_impact_drafts(
    battle: str,
    fit_games: int = 1000,
    fit_seed: int = 0,
    fit_alpha: float = 20.0,
    eval_games: int = 200,
    eval_seed: int = 1_000_000,
    reference: str = "balanced",
    specs: list[tuple[float, float, float]] | None = None,
) -> ImpactDraftSweep:
    if eval_games < 1:
        raise ValueError("eval_games must be >= 1")
    specs = specs or [
        (10.0, 3.0, 8.0),
        (15.0, 3.0, 8.0),
        (20.0, 3.0, 8.0),
        (15.0, 4.0, 8.0),
        (20.0, 4.0, 8.0),
        (20.0, 4.0, 12.0),
    ]
    fit = estimate_card_impact(games=fit_games, seed=fit_seed, battle=battle, alpha=fit_alpha)
    weights = {row.card_id: row.coefficient for row in fit.rows}
    battle_a = _battle_policy(battle)
    battle_b = _battle_policy(battle)
    ref_draft = make_draft_policy(reference)
    rows: list[ImpactDraftCandidate] = []
    for idx, (scale, curve_weight, item_discount) in enumerate(specs):
        name = f"impact-s{scale:g}-c{curve_weight:g}-i{item_discount:g}"
        cand = Composer(
            battle_a,
            ImpactDraftPolicy(
                weights,
                scale=scale,
                curve_weight=curve_weight,
                item_discount=item_discount,
                name=name,
            ),
            name=name,
        )
        ref = Composer(battle_b, ref_draft, name=f"draft:{reference}")
        result = run_match(cand, ref, games=eval_games, seed=eval_seed + idx * 10_000)
        lo, hi = wilson_ci(result.wins_a, result.games)
        rows.append(
            ImpactDraftCandidate(
                name=name,
                scale=scale,
                curve_weight=curve_weight,
                item_discount=item_discount,
                win_rate=result.win_rate_a,
                ci_low=lo,
                ci_high=hi,
                p_value=binomial_test(result.wins_a, result.games, 0.5),
                wins=result.wins_a,
                losses=result.wins_b,
                games=result.games,
            )
        )
    return ImpactDraftSweep(
        fit_report=fit,
        battle=battle,
        reference=reference,
        eval_games=eval_games,
        eval_seed=eval_seed,
        rows=rows,
    )
