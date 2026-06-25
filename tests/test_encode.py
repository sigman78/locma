from locma.core import battle as battlemod
from locma.core.actions import Pass
from locma.core.engine import make_battle_view, run_game
from locma.data.cards_db import load_cards
from locma.envs.encode import (
    ACTION_SIZE,
    OBS_SIZE,
    action_mask,
    encode_battle,
    index_to_action,
    sem_index,
)
from locma.policies.registry import make_policy


def test_constants():
    assert ACTION_SIZE == 155
    assert OBS_SIZE == 308


def test_encode_length():
    cards = load_cards()
    seen = []

    def cb(seat, action, gs):
        view = make_battle_view(gs)
        seen.append(encode_battle(view).shape[0])

    run_game(make_policy("greedy"), make_policy("greedy"), seed=0, cards=cards, on_pre_step=cb)
    assert seen and all(n == OBS_SIZE for n in seen)


def test_semantic_action_roundtrip_and_mask():
    """Every legal action maps to a unique in-range index that inverts back, and
    the mask flags exactly the legal indices."""
    cards = load_cards()
    stats = {
        "decisions": 0,
        "legal": 0,
        "unmappable": 0,
        "collision": 0,
        "roundtrip_fail": 0,
        "mask_illegal": 0,
        "max_idx": -1,
    }

    def cb(seat, action, gs):
        legal = battlemod.battle_legal(gs)
        view = make_battle_view(gs)
        stats["decisions"] += 1
        stats["legal"] += len(legal)
        seen = {}
        for a in legal:
            idx = sem_index(view, a)
            if idx is None:
                stats["unmappable"] += 1
                continue
            if idx in seen and seen[idx] != a:
                stats["collision"] += 1
            seen[idx] = a
            stats["max_idx"] = max(stats["max_idx"], idx)
            if index_to_action(view, legal, idx) != a:
                stats["roundtrip_fail"] += 1
        mask = action_mask(view, legal)
        for i in range(ACTION_SIZE):
            if mask[i] and index_to_action(view, legal, i) not in legal:
                stats["mask_illegal"] += 1

    for pa, pb in [("greedy", "greedy"), ("random", "max-attack"), ("scripted", "max-guard")]:
        for s in range(40):
            run_game(make_policy(pa), make_policy(pb), seed=s, cards=cards, on_pre_step=cb)

    assert stats["legal"] > 5000
    assert stats["unmappable"] == 0
    assert stats["collision"] == 0
    assert stats["roundtrip_fail"] == 0
    assert stats["mask_illegal"] == 0
    assert stats["max_idx"] < ACTION_SIZE


def test_index_to_action_out_of_range_is_pass():
    cards = load_cards()
    holder = {}

    def cb(seat, action, gs):
        holder["view"] = make_battle_view(gs)
        holder["legal"] = battlemod.battle_legal(gs)

    run_game(make_policy("greedy"), make_policy("greedy"), seed=1, cards=cards, on_pre_step=cb)
    assert index_to_action(holder["view"], holder["legal"], 999) == Pass()
