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

The ``shared`` flag switches every duel to the shared draft variant (a pick
removes the card from the other seat's offer; first pick alternates by round).
Offers are then contested, so the duel measures deck-building under competition
rather than from identical offers; the mirror argument above is seed-and-behavior
based, so the 0.500 self-duel calibration still holds exactly.
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
    PartialRandomDraftPolicy,
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
    """Construct a draft policy by name (a fresh instance per call).

    A ``+rndK`` suffix wraps the base draft in PartialRandomDraftPolicy: exactly
    ``K`` of the 30 picks are made uniformly at random, the rest by the base —
    e.g. ``balanced+rnd4``. Run-time reseeding (run_match resets per game) keeps
    the random rounds reproducible.
    """
    base, sep, kstr = name.partition("+rnd")
    if sep:
        if base not in DRAFTS or not kstr.isdigit():
            raise ValueError(f"bad noisy-draft spec '{name}' (want <base>+rnd<K>)")
        return PartialRandomDraftPolicy(DRAFTS[base](), int(kstr), name=name)
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
    draft_a: str,
    draft_b: str,
    battle: str = "ground",
    games: int = 100,
    seed: int = 0,
    shared: bool = False,
) -> DuelResult:
    """Win rate of ``draft_a`` over ``draft_b`` when both seats are piloted by the
    same ``battle`` policy. ``games`` mirrored pairs → ``2 × games`` total.

    ``shared`` runs the shared draft variant (a pick removes the card from the
    other seat's offer; first pick alternates by round). Contested offers make
    the comparison interactive rather than perfectly paired, but the mirror
    still cancels seat advantage exactly — a self-duel remains 0.500."""
    pa = _compose(battle, draft_a)
    pb = _compose(battle, draft_b)
    res = run_match(pa, pb, games=games, seed=seed, shared_draft=shared)
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
    drafts: list[str],
    battle: str = "ground",
    games: int = 100,
    seed: int = 0,
    workers: int = 1,
    shared: bool = False,
) -> SweepResult:
    """All-pairs draft duel under one battle policy.

    Each unordered pair is played once; the reverse cell is its exact complement
    (``run_match`` already mirrors seats, so ``duel(a,b) == 1 − duel(b,a)``),
    halving the work. The headline ranking is ``avg_win_rate`` — each draft's mean
    win rate over the rest of the field (a Copeland-style score that, unlike Elo,
    is not fooled by lopsided wins vs a single weak draft).

    ``workers > 1`` fans the pairs out over a process pool (each duel is an
    independent seeded match, so results are identical to the serial run)."""
    names = list(drafts)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    if workers > 1 and len(pairs) > 1:
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

        from locma.harness.parallel import init_eval_worker  # noqa: PLC0415

        with ProcessPoolExecutor(
            max_workers=min(workers, len(pairs)), initializer=init_eval_worker
        ) as ex:
            duels = list(
                ex.map(
                    duel,
                    [a for a, _ in pairs],
                    [b for _, b in pairs],
                    [battle] * len(pairs),
                    [games] * len(pairs),
                    [seed] * len(pairs),
                    [shared] * len(pairs),
                )
            )
    else:
        duels = [duel(a, b, battle=battle, games=games, seed=seed, shared=shared) for a, b in pairs]
    win_matrix: dict = {}
    for (a, b), d in zip(pairs, duels, strict=True):
        win_matrix[(a, b)] = d.win_rate_a
        win_matrix[(b, a)] = 1.0 - d.win_rate_a
    avg = {}
    for a in names:
        rates = [win_matrix[(a, b)] for b in names if b != a]
        avg[a] = sum(rates) / len(rates) if rates else 0.0
    return SweepResult(names, battle, 2 * games, win_matrix, avg)
