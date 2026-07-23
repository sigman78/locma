"""Cached deck pool — pre-draft decks once, sample them into games, refresh under
a hard generation-cost budget.

Motivation: live drafting is ~60 draft-policy forward passes per game (2 x 30
picks) and is a real self-play throughput cost. A pool of P pre-drafted decks
removes that entirely — a game just samples two decks and calls
``engine.run_battle_from_decks``. The pool is a disk artifact so workers load it
read-only and refresh happens between generations in the main process (no
multi-worker races).

The amortization guard (the whole point): let ``c`` be the cost of one deck and
live drafting cost ``2c``/match. With cumulative ``decks_generated`` decks built
(initial pool + every refresh) over ``matches_served`` matches, the amortized
generation fraction is ``decks_generated / (2 * matches_served)``. A refresh is
ALLOWED only if it keeps that fraction <= ``gen_budget_frac`` (e.g. 0.02). So
cumulative generation can never creep toward the live cost — worst case the pool
goes stale-but-cheap, never per-match redrafting. The refresh-interval floor
``G >= f*P / (2*eps)`` falls straight out of this.

Deck diversity: each cached deck is reshuffled per game seed inside
``run_battle_from_decks``, so the same 30-card list plays different draw orders —
a free anti-overfit multiplier on top of the P^2 matchup combinations.

Mixture: the pool is drafted from a weighted mix of draft policies, e.g.
``[(0.8, "ldraft"), (0.2, "random")]`` — mostly the deploy draft plus a minority
tail-exposing slice (``random`` does not suppress the ~43 under-drafted cards),
so the net gets gradient on the tail without the E28d distribution-shift cliff.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# The default 80/20 tail-enriched mixture: mostly the deploy draft (ldraft) plus
# a random slice that exposes the under-drafted tail (items + rare cards).
DEFAULT_MIXTURE = (("ldraft", 0.8), ("random", 0.2))


def _make_draft(spec: str):
    """Resolve a draft spec to a DraftPolicy (mirrors the ppo draft-slot rules)."""
    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.policies.drafts import (  # noqa: PLC0415
        BalancedDraftPolicy,
        DistilledDraftPolicy,
        GreedyDraftPolicy,
        RandomDraftPolicy,
    )
    from locma.policies.ppo import MaskablePPODraftPolicy  # noqa: PLC0415

    if spec == "random":
        return RandomDraftPolicy(seed=0)
    if spec == "greedy":
        return GreedyDraftPolicy()
    if spec == "balanced":
        return BalancedDraftPolicy()
    if spec == "ldraft":
        return MaskablePPODraftPolicy(model_path=resolve_path("depot:ldraft/ldraft_s0.zip"))
    if spec == "edraft":
        return DistilledDraftPolicy.load(resolve_path("depot:edraft/e20-elicit-fit.json"))
    # a path / depot ref -> learned draft
    return MaskablePPODraftPolicy(model_path=resolve_path(spec))


def _draft_one_deck(pol, seed: int, cards) -> list[int]:
    """Draft a single 30-card deck (seat 0's picks) with ``pol`` at ``seed``."""
    from locma.core import draft as draftmod  # noqa: PLC0415
    from locma.core.engine import make_draft_view  # noqa: PLC0415
    from locma.core.state import GameState, Phase  # noqa: PLC0415

    pol.reset(seed)
    gs = GameState.new(random.Random(seed))
    draftmod.start_draft(gs, cards, shared=False)
    deck: list[int] = []
    while gs.phase == Phase.DRAFT:
        seat = gs.current
        view = make_draft_view(gs)
        idx = pol.draft_action(view, draftmod.draft_legal(gs))
        if seat == 0:
            deck.append(int(view.offered[idx].card_id))
        draftmod.apply_draft_pick(gs, idx)
    return deck


def _allocate(n: int, weights) -> list[int]:
    """Split n items across weights, largest-remainder so the counts sum to n."""
    total = sum(weights)
    raw = [n * w / total for w in weights]
    counts = [int(x) for x in raw]
    rem = n - sum(counts)
    # hand the leftover to the largest fractional parts
    order = sorted(range(len(weights)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in range(rem):
        counts[order[i]] += 1
    return counts


def draft_decks(n: int, mixture, seed: int, cards=None) -> list[list[int]]:
    """Draft ``n`` decks from a weighted ``mixture`` of draft specs.

    ``mixture`` = sequence of ``(spec, weight)``. Deterministic given seed."""
    from locma.data.cards_db import load_cards  # noqa: PLC0415

    cards = cards or load_cards()
    specs = [s for s, _ in mixture]
    counts = _allocate(n, [w for _, w in mixture])
    decks: list[list[int]] = []
    s = seed
    for spec, cnt in zip(specs, counts, strict=True):
        if cnt == 0:
            continue
        pol = _make_draft(spec)
        for _ in range(cnt):
            decks.append(_draft_one_deck(pol, s, cards))
            s += 1
    random.Random(seed ^ 0x9E3779B9).shuffle(decks)  # mix components; deterministic
    return decks


class DeckPool:
    """A cached set of pre-drafted decks with a budget-guarded refresh policy.

    Sampling (``sample_pair``) is pure and cheap — safe to call from read-only
    workers. Budget accounting (``record_matches`` / ``maybe_refresh``) is owned
    by the driver in the main process, between generations."""

    def __init__(
        self,
        decks: list[list[int]],
        mixture=DEFAULT_MIXTURE,
        refresh_fraction: float = 0.05,
        gen_budget_frac: float = 0.02,
        matches_served: int = 0,
        decks_generated: int | None = None,
        gen_seed: int = 0,
    ) -> None:
        if not decks:
            raise ValueError("DeckPool needs at least one deck")
        self.decks = decks
        self.mixture = tuple(tuple(m) for m in mixture)
        self.refresh_fraction = refresh_fraction
        self.gen_budget_frac = gen_budget_frac
        self.matches_served = matches_served
        # cumulative decks ever drafted (initial pool counts — it is a real cost)
        self.decks_generated = decks_generated if decks_generated is not None else len(decks)
        self.gen_seed = gen_seed

    # ---- generation / persistence -------------------------------------------

    @classmethod
    def generate(cls, size: int, mixture=DEFAULT_MIXTURE, seed: int = 0, cards=None, **kw):
        decks = draft_decks(size, mixture, seed, cards)
        return cls(decks, mixture=mixture, decks_generated=size, gen_seed=seed, **kw)

    @classmethod
    def load(cls, path) -> DeckPool:
        d = json.loads(Path(path).read_text())
        return cls(
            decks=d["decks"],
            mixture=d.get("mixture", DEFAULT_MIXTURE),
            refresh_fraction=d.get("refresh_fraction", 0.05),
            gen_budget_frac=d.get("gen_budget_frac", 0.02),
            matches_served=d.get("matches_served", 0),
            decks_generated=d.get("decks_generated", len(d["decks"])),
            gen_seed=d.get("gen_seed", 0),
        )

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(
                {
                    "decks": self.decks,
                    "mixture": [list(m) for m in self.mixture],
                    "refresh_fraction": self.refresh_fraction,
                    "gen_budget_frac": self.gen_budget_frac,
                    "matches_served": self.matches_served,
                    "decks_generated": self.decks_generated,
                    "gen_seed": self.gen_seed,
                }
            )
        )

    # ---- sampling (read-only, worker-safe) ----------------------------------

    def sample_pair(self, rng: random.Random):
        """Two decks, sampled independently (P^2 matchups). Pure — no mutation."""
        n = len(self.decks)
        return self.decks[rng.randrange(n)], self.decks[rng.randrange(n)]

    # ---- budget accounting + refresh (driver-owned) -------------------------

    def record_matches(self, n: int) -> None:
        """Tell the pool how many matches it served (drives the refresh budget)."""
        self.matches_served += int(n)

    def amortized_frac(self) -> float:
        """Cumulative generation cost as a fraction of the live drafting it replaced."""
        if self.matches_served <= 0:
            return float("inf")
        return self.decks_generated / (2 * self.matches_served)

    def refresh_allowed(self) -> bool:
        """True iff refreshing ``refresh_fraction`` of the pool keeps the amortized
        generation fraction within ``gen_budget_frac`` (the hard cost guard)."""
        add = max(1, int(self.refresh_fraction * len(self.decks)))
        if self.matches_served <= 0:
            return False
        return (self.decks_generated + add) / (2 * self.matches_served) <= self.gen_budget_frac

    def maybe_refresh(self, seed: int, cards=None) -> int:
        """Rolling-refresh a fraction of the pool IFF the budget allows; else defer.

        Returns the number of decks replaced (0 if deferred). Replacement is
        uniform-random over the pool so no deck is immortal."""
        if not self.refresh_allowed():
            return 0
        add = max(1, int(self.refresh_fraction * len(self.decks)))
        fresh = draft_decks(add, self.mixture, seed, cards)
        rng = random.Random(seed)
        idxs = rng.sample(range(len(self.decks)), add)
        for j, i in enumerate(idxs):
            self.decks[i] = fresh[j]
        self.decks_generated += add
        return add

    def stats(self) -> dict:
        return {
            "size": len(self.decks),
            "matches_served": self.matches_served,
            "decks_generated": self.decks_generated,
            "amortized_frac": round(self.amortized_frac(), 5) if self.matches_served else None,
            "budget_frac": self.gen_budget_frac,
            "refresh_allowed": self.refresh_allowed(),
        }
