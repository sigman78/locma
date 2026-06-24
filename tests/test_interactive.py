from locma.data.cards_db import load_cards
from locma.harness.interactive import InteractiveGame
from locma.policies.registry import make_policy

CARDS = load_cards()


def make_game(human_seat=0, seed=0):
    return InteractiveGame("g_test", make_policy("random"), seed, human_seat, CARDS).start()


def test_draft_pauses_on_human_seat0():
    g = make_game(human_seat=0)
    p = g.pending()
    assert g.status == "awaiting_human"
    assert p["phase"] == "draft"
    assert p["you"] == 0
    assert p["round"] == 0
    assert p["total"] == 30
    assert len(p["triplet"]) == 3
    assert p["my_picks"] == 0


def test_draft_pauses_on_human_seat1_after_ai_picks():
    # human is seat 1: the AI (seat 0) drafts first, then we pause for the human.
    g = make_game(human_seat=1)
    p = g.pending()
    assert p["phase"] == "draft"
    assert p["you"] == 1
    assert p["round"] == 0  # still round 0 — seat 0 picked, seat 1 (human) to pick


def test_submit_draft_advances_round():
    g = make_game(human_seat=0)
    g.submit_draft(0)
    p = g.pending()
    # human(seat0) picked, AI(seat1) auto-picked, round advanced; human to pick round 1
    assert p["phase"] == "draft"
    assert p["round"] == 1
    assert p["my_picks"] == 1
