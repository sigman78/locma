# tests/test_views_readiness.py
from locma.core.engine import make_battle_view
from locma.core.state import GameState
from locma.core import battle as battlemod, draft as draftmod
from locma.data.cards_db import load_cards
import random


def test_battle_view_exposes_readiness():
    cards = load_cards()
    gs = GameState.new(random.Random(0))
    draftmod.start_draft(gs, cards)
    from locma.policies.registry import make_policy
    g = make_policy("greedy")
    while gs.phase.name == "DRAFT":
        from locma.core.engine import make_draft_view
        gs_pick = g.draft_action(make_draft_view(gs), [0, 1, 2])
        draftmod.apply_draft_pick(gs, gs_pick)
    battlemod.start_battle(gs)
    view = make_battle_view(gs)
    # CardView carries the new fields with bool type
    for c in view.my_hand:
        assert isinstance(c.can_attack, bool)
        assert isinstance(c.has_attacked, bool)


def test_cardview_defaults_keep_positional_construction():
    from locma.core.views import CardView
    c = CardView(1, 2, 0, 3, 4, 5, "------")  # 7 positional args, pre-change call site
    assert c.can_attack is False and c.has_attacked is False
