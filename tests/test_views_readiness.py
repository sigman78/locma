import random

from locma.core import battle as battlemod
from locma.core import draft as draftmod
from locma.core.engine import make_battle_view, make_draft_view
from locma.core.state import GameState
from locma.core.views import CardView
from locma.data.cards_db import load_cards
from locma.policies.registry import make_policy


def test_battle_view_exposes_readiness():
    cards = load_cards()
    gs = GameState.new(random.Random(0))
    draftmod.start_draft(gs, cards)
    g = make_policy("greedy")
    while gs.phase.name == "DRAFT":
        gs_pick = g.draft_action(make_draft_view(gs), [0, 1, 2])
        draftmod.apply_draft_pick(gs, gs_pick)
    battlemod.start_battle(gs)
    view = make_battle_view(gs)
    # CardView carries the new fields with bool type
    for c in view.my_hand:
        assert isinstance(c.can_attack, bool)
        assert isinstance(c.has_attacked, bool)


def test_cardview_defaults_keep_positional_construction():
    c = CardView(1, 2, 0, 3, 4, 5, "------")  # 7 positional args, pre-change call site
    assert c.can_attack is False and c.has_attacked is False
