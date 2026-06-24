import tempfile

import pytest

from locma.data.cards_db import load_cards
from locma.harness.interactive import IllegalMove, InteractiveGame, WrongPhase
from locma.harness.replay_store import get_replay, write_replay
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


def _legal_pass(g):
    return {"t": "pass"}


def drive_to_end(g, choose=_legal_pass):
    # Drive a whole game: pick draft 0 every time, and in battle apply `choose`
    # (default: always Pass to end the turn quickly).
    steps = 0
    while g.status == "awaiting_human":
        p = g.pending()
        if p["phase"] == "draft":
            g.submit_draft(0)
        else:
            g.submit_action(choose(g))
        steps += 1
        assert steps < 5000, "game did not terminate"
    return g


def test_full_game_finishes_and_produces_replay():
    g = make_game(human_seat=0, seed=3)
    drive_to_end(g)
    assert g.status == "finished"
    r = g.result
    assert isinstance(r["winner_is_human"], bool)
    assert r["turns"] >= 1
    assert r["replay_id"].startswith("r_")
    # the assembled replay round-trips through the store in locma-replay/2
    with tempfile.TemporaryDirectory() as d:
        path = write_replay(d, g._replay)
        assert path.endswith(".jsonl")
        rep = get_replay(d, r["replay_id"])
        assert rep["header"]["format"] == "locma-replay/2"
        assert rep["header"]["policy_a"] == "human"
        assert rep["header"]["a_seat"] == 0
        assert len(rep["battle"]["steps"]) >= 1


def test_battle_pending_is_human_perspective_seat1():
    g = make_game(human_seat=1, seed=3)
    # drive only the draft, stop at the first battle decision
    while g.pending() and g.pending()["phase"] == "draft":
        g.submit_draft(0)
    p = g.pending()
    assert p["phase"] == "battle"
    assert p["you"] == 1
    # "me" reflects the human (physical seat 1), not gs.current bookkeeping
    me_hand = {c["iid"] for c in p["view"]["me"]["hand"]}
    assert me_hand == {c.instance_id for c in g.gs.players[1].hand}


def test_illegal_action_rejected():
    g = make_game(human_seat=0, seed=3)
    while g.pending()["phase"] == "draft":
        g.submit_draft(0)
    # attacking a non-existent unit is never legal
    with pytest.raises(IllegalMove):
        g.submit_action({"t": "attack", "a": 999999, "target": -1})


def test_submit_action_during_draft_is_wrong_phase():
    g = make_game(human_seat=0, seed=3)
    with pytest.raises(WrongPhase):
        g.submit_action({"t": "pass"})


def test_determinism_same_inputs_same_replay():
    g1 = drive_to_end(make_game(human_seat=0, seed=11))
    g2 = drive_to_end(make_game(human_seat=0, seed=11))
    assert g1.result["replay_id"] == g2.result["replay_id"]
    assert g1._replay["header"]["hash"] == g2._replay["header"]["hash"]
