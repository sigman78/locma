import pytest

from locma.policies.composer import Composer
from locma.policies.drafts import GreedyDraftPolicy, MaxGuardDraftPolicy
from locma.policies.mcts import MCTSBattlePolicy
from locma.policies.registry import make_policy, policy_names


def test_policy_names_lists_mcts_not_ppo():
    names = policy_names()
    assert "mcts" in names
    assert "ppo" not in names
    assert names[:5] == ["random", "scripted", "greedy", "max-guard", "max-attack"]


def test_legacy_preset_is_composer_with_spec_name():
    p = make_policy("max-guard")
    assert isinstance(p, Composer)
    assert p.name == "max-guard"
    assert isinstance(p.draft, MaxGuardDraftPolicy)


def test_greedy_preset_pairs_greedy_draft():
    p = make_policy("greedy")
    assert isinstance(p.draft, GreedyDraftPolicy)


def test_mcts_params_positional_with_defaults():
    p = make_policy("mcts:200,1.4,7")
    assert isinstance(p.battle, MCTSBattlePolicy)
    assert p.battle.iterations == 200
    assert p.battle.c == 1.4
    assert p.battle._seed == 7
    assert p.name == "mcts:200,1.4,7"
    d = make_policy("mcts")  # all defaults
    assert d.battle.iterations == 100


def test_mcts_partial_params():
    p = make_policy("mcts:50")
    assert p.battle.iterations == 50
    assert p.battle.iterations != 100


def test_ppo_default_and_path():
    assert make_policy("ppo").battle.model_path == "model.zip"
    assert make_policy("ppo:runs/exp1.zip").battle.model_path == "runs/exp1.zip"


def test_ppo_pairs_balanced_draft():
    # The draft sweep found `greedy` is the worst partner; `ppo:` now pairs the
    # learned battle net with `balanced` (docs/baseline.md "PPO × draft sweep").
    from locma.policies.drafts import BalancedDraftPolicy  # noqa: PLC0415

    assert isinstance(make_policy("ppo:runs/exp1.zip").draft, BalancedDraftPolicy)


def test_unknown_spec_raises():
    with pytest.raises(ValueError):
        make_policy("nope")
