"""Tests for the E38 afterstate consequence columns (locma/envs/conseq.py).

No [ml] extras required: pure engine + numpy.

The load-bearing properties are (a) non-mutation of the decision state,
(b) fairness — no hidden-information leak, (c) Pass is never simulated, and
(d) the block lines up with the semantic action mask.
"""

import json
import random

import numpy as np

from locma.core import battle as battlemod
from locma.core.actions import Pass
from locma.core.engine import make_battle_view, run_game
from locma.core.instance import CardInstance
from locma.core.state import Phase
from locma.data.cards_db import load_cards
from locma.envs.conseq import CONSEQ_COLUMNS, N_CONSEQ, conseq_features
from locma.envs.encode import ACTION_SIZE, action_mask, sem_index
from locma.envs.practicum import _manifest_path, record_practicum
from locma.policies.mcts import _clone_battle
from locma.policies.registry import make_policy


def _mid_battle_state(seed: int = 3):
    """Play a short scripted game and return a battle state with >=3 legal actions."""
    captured = []

    def hook(seat, action, gs):
        if gs.phase == Phase.BATTLE and len(battlemod.battle_legal(gs)) >= 3:
            captured.append(_clone_battle(gs))

    run_game(make_policy("greedy"), make_policy("scripted"), seed, on_pre_step=hook)
    assert captured, "no multi-action battle state captured"
    return captured[len(captured) // 2]


def test_shape_and_dtype():
    gs = _mid_battle_state()
    block = conseq_features(gs)
    assert block.shape == (ACTION_SIZE, N_CONSEQ)
    assert block.dtype == np.float32
    assert len(CONSEQ_COLUMNS) == N_CONSEQ


def test_does_not_mutate_state():
    """The decision state must be byte-identical after computing the block."""
    gs = _mid_battle_state()
    view_before = make_battle_view(gs)
    before = (
        gs.turn,
        gs.current,
        gs.phase,
        gs.winner,
        tuple((p.health, p.mana, len(p.hand), len(p.deck), len(p.board)) for p in gs.players),
    )
    conseq_features(gs)
    after = (
        gs.turn,
        gs.current,
        gs.phase,
        gs.winner,
        tuple((p.health, p.mana, len(p.hand), len(p.deck), len(p.board)) for p in gs.players),
    )
    assert before == after
    assert make_battle_view(gs) == view_before


def test_nonzero_only_on_legal_non_pass_actions():
    """Rows are zero except at legal, semantically addressable, non-Pass actions."""
    gs = _mid_battle_state()
    legal = list(battlemod.battle_legal(gs))
    view = make_battle_view(gs)
    block = conseq_features(gs, legal)
    mask = action_mask(view, legal)

    allowed = np.zeros(ACTION_SIZE, dtype=bool)
    for a in legal:
        if isinstance(a, Pass):
            continue
        idx = sem_index(view, a)
        if idx is not None and idx < ACTION_SIZE:
            allowed[idx] = True

    nonzero = np.abs(block).sum(axis=1) > 0
    assert not nonzero[~allowed].any(), "consequence values outside legal non-Pass actions"
    # Pass is index 0 and must always be all-zero (apply_battle(Pass) fuses end_turn).
    assert np.all(block[0] == 0.0)
    # Sanity: the mask has at least one legal action the block could have filled.
    assert mask.any()


def test_pass_is_never_simulated():
    """A Pass-only legal list produces an all-zero block and simulates nothing."""
    gs = _mid_battle_state()
    stats: dict = {}
    block = conseq_features(gs, [Pass()], stats=stats)
    assert np.all(block == 0.0)
    assert stats.get("actions", 0) == 0


def test_matches_a_hand_rolled_simulation():
    """dphi / dmg_face / mana_after must equal a direct apply_battle read."""
    gs = _mid_battle_state()
    seat = gs.current
    legal = [a for a in battlemod.battle_legal(gs) if not isinstance(a, Pass)]
    assert legal, "expected a non-Pass legal action"
    view = make_battle_view(gs)
    block = conseq_features(gs, [*legal, Pass()])

    def power(board):
        return sum(c.attack + c.defense for c in board)

    checked = 0
    for a in legal:
        idx = sem_index(view, a)
        if idx is None or idx >= ACTION_SIZE:
            continue
        sim = _clone_battle(gs)
        phi0 = power(gs.players[seat].board) - power(gs.players[1 - seat].board)
        op_hp0 = gs.players[1 - seat].health
        battlemod.apply_battle(sim, a)
        phi1 = power(sim.players[seat].board) - power(sim.players[1 - seat].board)
        row = block[idx]
        assert row[0] == float(phi1 - phi0), f"dphi mismatch at {a}"
        assert row[1] == float(op_hp0 - sim.players[1 - seat].health), f"dmg_face mismatch at {a}"
        assert row[6] == float(sim.players[seat].mana), f"mana_after mismatch at {a}"
        checked += 1
    assert checked, "no actions checked"


def test_no_hidden_information_leak():
    """Fairness: the block must be invariant to the OPPONENT's hidden hand/deck.

    Replacing the opponent's hand and deck with completely different cards (their
    board, health and mana untouched) must not change a single column — that is
    exactly the property that makes E38 fair rather than "cheating".
    """
    gs = _mid_battle_state()
    base = conseq_features(gs)

    cards = load_cards()
    alt = _clone_battle(gs)
    opp = alt.players[1 - alt.current]
    opp.hand = [CardInstance.from_card(cards[(i * 7) % len(cards)], 9000 + i) for i in range(4)]
    opp.deck = [CardInstance.from_card(cards[(i * 13) % len(cards)], 9500 + i) for i in range(10)]
    swapped = conseq_features(alt)

    assert np.array_equal(base, swapped), "consequence block depends on opponent hidden cards"


def test_own_deck_order_does_not_leak():
    """Self-leak guard: our own deck ORDER must not change any column.

    A player knows their deck's contents but not its shuffled order, so a column
    that moved when the deck is reshuffled would leak our own future draws.
    """
    gs = _mid_battle_state()
    base = conseq_features(gs)
    alt = _clone_battle(gs)
    random.Random(11).shuffle(alt.players[alt.current].deck)
    assert np.array_equal(base, conseq_features(alt)), "own deck order leaks into the block"


def test_practicum_records_conseq(tmp_path):
    out = str(tmp_path / "p.npz")
    manifest = record_practicum(
        teacher="greedy",
        opponents=("random",),
        games=1,
        out=out,
        seed=0,
        obs_mode="token",
        conseq=True,
    )
    n = manifest["n_examples"]
    assert n > 0
    with np.load(out) as d:
        assert "obs_conseq" in d
        assert d["obs_conseq"].shape == (n, ACTION_SIZE, N_CONSEQ)
        assert d["obs_conseq"].dtype == np.float32
        # every record must have at least one filled action row (>=2 legal by
        # construction, and a Pass-only decision is never recorded)
        assert (np.abs(d["obs_conseq"]).sum(axis=(1, 2)) > 0).all()
    with open(_manifest_path(out)) as f:
        m = json.load(f)
    assert m["conseq_columns"] == list(CONSEQ_COLUMNS)
    assert m["conseq_stats"]["actions"] > 0


def test_practicum_default_has_no_conseq(tmp_path):
    """Default path stays byte-compatible: no obs_conseq key, no manifest entries."""
    out = str(tmp_path / "p.npz")
    record_practicum(
        teacher="greedy", opponents=("random",), games=1, out=out, seed=0, obs_mode="token"
    )
    with np.load(out) as d:
        assert "obs_conseq" not in d
    with open(_manifest_path(out)) as f:
        m = json.load(f)
    assert "conseq_columns" not in m
    assert "conseq_stats" not in m
