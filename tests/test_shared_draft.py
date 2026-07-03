"""Shared draft variant: a pick removes the card from the other seat's offer,
the second picker chooses from the remaining 2 (the third card is burned), and
the first picker alternates each round for fairness."""

import random

import pytest

from locma.core.draft import apply_draft_pick, draft_legal, start_draft
from locma.core.engine import make_draft_view, run_game
from locma.core.state import GameState, Phase
from locma.data.cards_db import load_cards
from locma.envs.battle_env import BattleEnv
from locma.harness.draft_bench import duel
from locma.policies.registry import make_policy


def _shared_gs(seed: int = 42) -> GameState:
    gs = GameState.new(random.Random(seed))
    start_draft(gs, load_cards(), shared=True)
    return gs


def test_first_picker_alternates_by_round():
    gs = _shared_gs()
    for r in range(30):
        assert gs.draft_round == r
        assert gs.current == r % 2  # first picker alternates
        apply_draft_pick(gs, 0)
        assert gs.current == 1 - (r % 2)  # second picker is the other seat
        apply_draft_pick(gs, 1)
    assert gs.phase == Phase.BATTLE


def test_second_picker_legal_excludes_taken():
    gs = _shared_gs()
    assert draft_legal(gs) == [0, 1, 2]
    apply_draft_pick(gs, 1)
    assert draft_legal(gs) == [0, 2]
    apply_draft_pick(gs, 2)
    # next round: fresh offer, full legal again
    assert draft_legal(gs) == [0, 1, 2]


def test_picking_the_taken_card_raises():
    gs = _shared_gs()
    apply_draft_pick(gs, 1)
    with pytest.raises(ValueError, match="already taken"):
        apply_draft_pick(gs, 1)


def test_round_slots_are_exclusive_and_decks_full():
    gs = _shared_gs()
    while gs.phase == Phase.DRAFT:
        apply_draft_pick(gs, 0)
        assert 0 not in draft_legal(gs)  # the taken slot is gone for the second picker
        apply_draft_pick(gs, draft_legal(gs)[0])
    assert len(gs.picks[0]) == 30 and len(gs.picks[1]) == 30
    assert len(gs.players[0].deck) == 30 and len(gs.players[1].deck) == 30


def test_draft_view_exposes_taken():
    gs = _shared_gs()
    assert make_draft_view(gs).taken is None
    apply_draft_pick(gs, 2)
    assert make_draft_view(gs).taken == 2


def test_default_mode_unchanged():
    gs = GameState.new(random.Random(42))
    start_draft(gs, load_cards())
    assert gs.draft_shared is False
    apply_draft_pick(gs, 0)
    assert gs.draft_taken is None
    assert draft_legal(gs) == [0, 1, 2]
    apply_draft_pick(gs, 0)  # same index legal for the second seat: no depletion
    assert gs.picks[0][0] is gs.picks[1][0]


def test_shared_deterministic_draft_yields_asymmetric_decks():
    # Under the default rule, two seats running the same deterministic draft
    # build identical decks; the shared rule breaks that mirror.
    def deck_ids(shared: bool) -> tuple[list, list]:
        pol0, pol1 = make_policy("max-guard"), make_policy("max-guard")
        gs = GameState.new(random.Random(7))
        start_draft(gs, load_cards(), shared=shared)
        pols = (pol0, pol1)
        while gs.phase == Phase.DRAFT:
            view = make_draft_view(gs)
            pick = pols[gs.current].draft_action(view, draft_legal(gs))
            apply_draft_pick(gs, pick)
        return [c.id for c in gs.picks[0]], [c.id for c in gs.picks[1]]

    d0, d1 = deck_ids(shared=False)
    assert d0 == d1  # the mirror this variant exists to break
    s0, s1 = deck_ids(shared=True)
    assert s0 != s1
    # determinism: same seed reproduces the same shared decks
    assert deck_ids(shared=True) == (s0, s1)


def test_run_game_shared_draft_completes():
    r = run_game(make_policy("max-guard"), make_policy("max-attack"), seed=3, shared_draft=True)
    assert r.winner in (0, 1)


def test_shared_self_duel_is_exactly_half():
    res = duel("balanced", "balanced", battle="ground", games=8, seed=0, shared=True)
    assert res.win_rate_a == 0.5


def test_battle_env_shared_draft():
    env = BattleEnv(make_policy("max-guard"), seed=0, shared_draft=True)
    obs, _ = env.reset()
    assert obs is not None
    # a deterministic draft under the shared rule gives the seats different decks
    assert [c.id for c in env.gs.picks[0]] != [c.id for c in env.gs.picks[1]]
