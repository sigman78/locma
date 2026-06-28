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


def test_mcts_rollout_turns_param():
    assert make_policy("mcts:100").battle.rollout_turns == 3  # heuristic default
    assert make_policy("mcts:50,1.41,0,5").battle.rollout_turns == 5
    assert make_policy("mcts:50,1.41,0,0").battle.rollout_turns == 0  # legacy terminal


def test_dmcts_spec_and_selectable():
    assert "dmcts" in policy_names()  # selectable like mcts
    p = make_policy("dmcts:20,40,7")
    assert p.battle.K == 20 and p.battle.I == 40 and p.battle._seed == 7
    d = make_policy("dmcts")  # defaults
    assert d.battle.K == 15 and d.battle.I == 30


def test_dmcts_deterministic_param():
    p = make_policy("dmcts:2,3,4,5,1")
    assert p.battle.K == 2
    assert p.battle.I == 3
    assert p.battle.rollout_turns == 5
    assert p.battle.deterministic is True


def test_puct_ppo_spec_pairs_balanced_draft():
    from locma.policies.azlite import PUCTPPOBattlePolicy  # noqa: PLC0415
    from locma.policies.drafts import BalancedDraftPolicy  # noqa: PLC0415

    assert "puct-ppo" not in policy_names()
    p = make_policy("puct-ppo:25,runs/ppo-shuffled-pool.zip,1.25,7,2,tactical")
    assert isinstance(p.battle, PUCTPPOBattlePolicy)
    assert p.battle.iterations == 25
    assert p.battle.model_path == "runs/ppo-shuffled-pool.zip"
    assert p.battle.c_puct == 1.25
    assert p.battle._seed == 7
    assert p.battle.rollout_turns == 2
    assert p.battle.obs_mode == "tactical"
    assert isinstance(p.draft, BalancedDraftPolicy)


def test_dpuct_ppo_spec_pairs_balanced_draft():
    from locma.policies.azlite import DeterminizedPUCTPPOBattlePolicy  # noqa: PLC0415
    from locma.policies.drafts import BalancedDraftPolicy  # noqa: PLC0415

    assert "dpuct-ppo" not in policy_names()
    p = make_policy("dpuct-ppo:3,8,runs/ppo-shuffled-pool.zip,1.25,7,2,tactical")
    assert isinstance(p.battle, DeterminizedPUCTPPOBattlePolicy)
    assert p.battle.K == 3
    assert p.battle.I == 8
    assert p.battle.model_path == "runs/ppo-shuffled-pool.zip"
    assert p.battle.c_puct == 1.25
    assert p.battle._seed == 7
    assert p.battle.rollout_turns == 2
    assert p.battle.obs_mode == "tactical"
    assert isinstance(p.draft, BalancedDraftPolicy)


def test_ppo_default_and_path():
    assert make_policy("ppo").battle.model_path == "model.zip"
    assert make_policy("ppo:runs/exp1.zip").battle.model_path == "runs/exp1.zip"


def test_ppo_pairs_balanced_draft():
    # The draft sweep found `greedy` is the worst partner; `ppo:` now pairs the
    # learned battle net with `balanced` (docs/baseline.md "PPO × draft sweep").
    from locma.policies.drafts import BalancedDraftPolicy  # noqa: PLC0415

    assert isinstance(make_policy("ppo:runs/exp1.zip").draft, BalancedDraftPolicy)


def test_ppo_tactical_uses_tactical_obs_and_balanced_draft():
    from locma.policies.drafts import BalancedDraftPolicy  # noqa: PLC0415

    p = make_policy("ppo-tactical:runs/tactical.zip")
    assert p.battle.model_path == "runs/tactical.zip"
    assert p.battle.obs_mode == "tactical"
    assert isinstance(p.draft, BalancedDraftPolicy)
    assert "ppo-tactical" not in policy_names()


def test_unknown_spec_raises():
    with pytest.raises(ValueError):
        make_policy("nope")


def test_rich_mixed_specs_are_hidden_training_opponents():
    from locma.policies.mixed import MixedOpponentPolicy  # noqa: PLC0415

    rich = make_policy("mixed-rich:7")
    assert isinstance(rich, MixedOpponentPolicy)
    assert rich._seed == 7
    assert len(rich.pool) == 6
    assert "mixed-rich" not in policy_names()
