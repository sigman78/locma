"""Draft-duel benchmark — isolate deck-building skill from battle skill.

The draft phase deals both seats from the *same* shared triplet stream each round
(see ``locma/core/draft.py``) and a pick does **not** deplete the offer, so on a
fixed seed both seats are offered identical cards. A head-to-head where both seats
use the **same battle policy** and differ **only** in their draft policy is
therefore a perfectly paired comparison: the battle skill cancels and the win-rate
edge is attributable purely to *which deck was drafted from identical offers*.

This is cleaner than the older "draft sweep" (docs/baseline.md "PPO × draft
sweep"), which paired each draft with a fixed battle net and measured win rate vs
the built-in *baselines* — that conflates draft quality with the battle-policy
matchup (the opponent's battle AND draft both differ). Here only the draft varies.

Calibration guarantee: a *self-duel* (same draft on both seats) is **exactly
0.500**. For each seed the two behaviourally-identical policies produce the same
winner-seat ``w`` in both mirrored games, so seat-0-player A wins exactly one of
the pair — the mirror cancels seat advantage perfectly. Any deviation from 0.5 in
a self-duel signals a benchmark bug.
"""

from __future__ import annotations

from dataclasses import dataclass

from locma.harness.match import run_match
from locma.policies.composer import Composer
from locma.policies.drafts import (
    BalancedDraftPolicy,
    GreedyDraftPolicy,
    MaxAttackDraftPolicy,
    MaxDefenseDraftPolicy,
    MaxGuardDraftPolicy,
    RandomDraftPolicy,
    WeightedDraftPolicy,
)
from locma.policies.registry import make_policy
from locma.stats.intervals import wilson_ci

# Draft policies addressable by name. The policy registry only exposes full
# battle+draft *pairs*, so the draft halves are registered here for the benchmark.
# Registration order drives the default sweep / matrix order.
DRAFTS: dict = {
    "random": lambda: RandomDraftPolicy(seed=0),
    "greedy": GreedyDraftPolicy,
    "weighted": WeightedDraftPolicy,
    "max-attack": MaxAttackDraftPolicy,
    "max-defense": MaxDefenseDraftPolicy,
    "max-guard": MaxGuardDraftPolicy,
    "balanced": BalancedDraftPolicy,
}


def draft_names() -> list[str]:
    """Selectable draft policy names, in registration order."""
    return list(DRAFTS)


def make_draft(name: str):
    """Construct a draft policy by name (a fresh instance per call)."""
    if name not in DRAFTS:
        raise ValueError(f"unknown draft '{name}' (known: {', '.join(DRAFTS)})")
    return DRAFTS[name]()


# Friendly battle-policy aliases. The policy registry has no bare name for the
# `ground` battle (it is the battle half of `max-guard`/`max-attack`), so expose
# it directly; any registry spec (`greedy`, `scripted`, `azlite:…`, `dmcts:…`,
# `ppo:…`) also works unaliased.
_BATTLE_ALIASES = {"ground": "max-guard"}


def make_battle(spec: str):
    """The battle half of a registry policy ``spec`` (a fresh instance per call).

    Lets the benchmark pilot any deck with any battle policy — ``ground``,
    ``greedy``, ``dmcts:…``, ``azlite:…``, ``ppo:…`` — by discarding the draft
    half the registry would otherwise pair it with.
    """
    return make_policy(_BATTLE_ALIASES.get(spec, spec)).battle


def _compose(battle_spec: str, draft_name: str):
    return Composer(make_battle(battle_spec), make_draft(draft_name), name=draft_name)


@dataclass
class DuelResult:
    draft_a: str
    draft_b: str
    battle: str
    n: int  # mirrored game count (2 × games)
    win_rate_a: float
    ci: tuple[float, float]


def duel(
    draft_a: str, draft_b: str, battle: str = "ground", games: int = 100, seed: int = 0
) -> DuelResult:
    """Win rate of ``draft_a`` over ``draft_b`` when both seats are piloted by the
    same ``battle`` policy. ``games`` mirrored pairs → ``2 × games`` total."""
    pa = _compose(battle, draft_a)
    pb = _compose(battle, draft_b)
    res = run_match(pa, pb, games=games, seed=seed)
    return DuelResult(
        draft_a, draft_b, battle, res.games, res.win_rate_a, wilson_ci(res.wins_a, res.games)
    )


@dataclass
class SweepResult:
    drafts: list[str]
    battle: str
    n_per_pair: int  # total mirrored games per pair (2 × games), matching DuelResult.n
    win_matrix: dict  # (row, col) -> row's win rate vs col
    avg_win_rate: dict  # draft -> mean win rate over the rest of the field


def round_robin(
    drafts: list[str], battle: str = "ground", games: int = 100, seed: int = 0
) -> SweepResult:
    """All-pairs draft duel under one battle policy.

    Each unordered pair is played once; the reverse cell is its exact complement
    (``run_match`` already mirrors seats, so ``duel(a,b) == 1 − duel(b,a)``),
    halving the work. The headline ranking is ``avg_win_rate`` — each draft's mean
    win rate over the rest of the field (a Copeland-style score that, unlike Elo,
    is not fooled by lopsided wins vs a single weak draft)."""
    names = list(drafts)
    win_matrix: dict = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            d = duel(a, b, battle=battle, games=games, seed=seed)
            win_matrix[(a, b)] = d.win_rate_a
            win_matrix[(b, a)] = 1.0 - d.win_rate_a
    avg = {}
    for a in names:
        rates = [win_matrix[(a, b)] for b in names if b != a]
        avg[a] = sum(rates) / len(rates) if rates else 0.0
    return SweepResult(names, battle, 2 * games, win_matrix, avg)
