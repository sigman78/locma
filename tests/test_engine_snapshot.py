from __future__ import annotations

from locma.core.engine import run_game
from locma.core.state import Phase
from locma.harness.trace import record_game, serialize_trace
from locma.policies.battles import RandomBattlePolicy
from locma.policies.composer import Composer
from locma.policies.drafts import RandomDraftPolicy


def _random(name):
    return Composer(RandomBattlePolicy(seed=0), RandomDraftPolicy(seed=0), name=name)


def test_on_snapshot_fires_once_at_battle_start():
    phases = []
    run_game(
        _random("r0"),
        _random("r1"),
        seed=1,
        on_snapshot=lambda gs: phases.append(gs.phase),
    )
    assert phases == [Phase.BATTLE]


def test_on_snapshot_does_not_change_trace():
    _, baseline = record_game(_random("a"), _random("b"), seed=3)
    steps = []
    run_game(
        _random("a"),
        _random("b"),
        seed=3,
        on_step=lambda seat, action, gs: steps.append((seat, action)),
        on_snapshot=lambda gs: None,
    )
    assert serialize_trace(steps) == serialize_trace(baseline)
