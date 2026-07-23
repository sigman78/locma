"""Tests for the cached deck pool (locma.envs.deckpool) + engine deck injection.

Fast tests (CI): pure budget-guard arithmetic, sampling, save/load — hand-made
decks, no drafting or battles. Slow tests: deck generation (drafts) and
``run_battle_from_decks`` (plays battles)."""

from __future__ import annotations

import random

import pytest

from locma.envs.deckpool import DeckPool, _allocate


def _fake_decks(n, size=30):
    """n distinct 30-card decks of dummy ids — no engine needed."""
    return [[i * 100 + j for j in range(size)] for i in range(n)]


# ---- fast: import + pure logic (CI) -----------------------------------------


def test_module_imports_without_ml():
    import locma.envs.deckpool as m  # noqa: PLC0415

    assert hasattr(m, "DeckPool") and hasattr(m, "draft_decks")


@pytest.mark.parametrize(
    ("n", "weights", "expected"),
    [(10, [0.8, 0.2], [8, 2]), (7, [1.0], [7]), (5, [1, 1, 1], [2, 2, 1]), (0, [0.5, 0.5], [0, 0])],
)
def test_allocate_sums_and_respects_weights(n, weights, expected):
    counts = _allocate(n, weights)
    assert sum(counts) == n
    assert counts == expected


def test_sample_pair_returns_two_decks():
    pool = DeckPool(_fake_decks(8))
    a, b = pool.sample_pair(random.Random(0))
    assert len(a) == 30 and len(b) == 30
    assert a in pool.decks and b in pool.decks


def test_budget_guard_defers_until_amortized():
    # size 20, refresh_fraction 0.05 -> add=1; eps 0.02.
    pool = DeckPool(_fake_decks(20), refresh_fraction=0.05, gen_budget_frac=0.02)
    assert pool.amortized_frac() == float("inf")
    assert pool.refresh_allowed() is False  # 0 matches served
    pool.record_matches(400)  # 20/(2*400)=0.025 > 0.02
    assert pool.refresh_allowed() is False
    pool.record_matches(200)  # total 600 -> (20+1)/1200 = 0.0175 <= 0.02
    assert pool.refresh_allowed() is True


def test_maybe_refresh_deferred_is_noop():
    pool = DeckPool(_fake_decks(20), gen_budget_frac=0.02)
    before = [d[:] for d in pool.decks]
    replaced = pool.maybe_refresh(seed=1)  # no matches served -> deferred
    assert replaced == 0
    assert pool.decks == before
    assert pool.decks_generated == 20  # unchanged


def test_amortized_frac_never_exceeds_budget_after_guarded_refreshes():
    # Simulate many generations: the guard must keep cumulative gen <= budget.
    pool = DeckPool(_fake_decks(100), refresh_fraction=0.05, gen_budget_frac=0.02)
    for _ in range(50):
        pool.record_matches(5000)
        if pool.refresh_allowed():
            # emulate a refresh's accounting without drafting (guard is the point)
            add = max(1, int(pool.refresh_fraction * len(pool.decks)))
            pool.decks_generated += add
    assert pool.amortized_frac() <= pool.gen_budget_frac + 1e-9


def test_save_load_roundtrip(tmp_path):
    pool = DeckPool(_fake_decks(12), gen_budget_frac=0.03, refresh_fraction=0.1)
    pool.record_matches(1234)
    p = tmp_path / "pool.json"
    pool.save(p)
    back = DeckPool.load(p)
    assert back.decks == pool.decks
    assert back.matches_served == pool.matches_served
    assert back.decks_generated == pool.decks_generated
    assert back.gen_budget_frac == pool.gen_budget_frac


def test_empty_pool_rejected():
    with pytest.raises(ValueError, match="at least one deck"):
        DeckPool([])


# ---- slow: drafting + battles ([ml] for ldraft; random mix is [ml]-free) -----


@pytest.mark.slow
def test_generate_random_mixture_produces_valid_decks():
    # random-only mixture avoids the [ml] ldraft dep — pure engine drafting.
    pool = DeckPool.generate(size=6, mixture=(("random", 1.0),), seed=0)
    assert len(pool.decks) == 6
    assert all(len(d) == 30 for d in pool.decks)
    assert pool.decks_generated == 6


@pytest.mark.slow
def test_run_battle_from_decks_deterministic():
    from locma.core.engine import run_battle_from_decks  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    pool = DeckPool.generate(size=4, mixture=(("random", 1.0),), seed=1)
    d0, d1 = pool.sample_pair(random.Random(0))
    a, b = make_policy("greedy"), make_policy("scripted")
    r1 = run_battle_from_decks(d0, d1, a, b, seed=500)
    r2 = run_battle_from_decks(d0, d1, a, b, seed=500)
    assert (r1.winner, r1.turns) == (r2.winner, r2.turns)  # deterministic
    assert r1.turns > 0 and r1.winner in (0, 1)


@pytest.mark.slow
def test_deal_decks_sets_battle_phase():
    import random as _r  # noqa: PLC0415

    from locma.core.engine import deal_decks  # noqa: PLC0415
    from locma.core.state import GameState, Phase  # noqa: PLC0415

    pool = DeckPool.generate(size=2, mixture=(("random", 1.0),), seed=2)
    gs = GameState.new(_r.Random(0))
    deal_decks(gs, pool.decks[0], pool.decks[1])
    assert gs.phase == Phase.BATTLE
    assert len(gs.players[0].deck) == 30 and len(gs.players[1].deck) == 30
