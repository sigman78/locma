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


def test_azlite_default_and_draft_override():
    from locma.policies.drafts import BalancedDraftPolicy  # noqa: PLC0415
    from locma.policies.ppo import MaskablePPODraftPolicy  # noqa: PLC0415

    d = make_policy("azlite:100")  # bare spec: balanced draft (byte-compatible)
    assert isinstance(d.draft, BalancedDraftPolicy)
    assert d.battle.iterations == 100
    # 5th param overrides the draft, like ppo/vbeam/netdmcts (E25 matched-draft);
    # a model path loads a learned draft lazily (nothing read here).
    o = make_policy("azlite:100,1.5,0,0,runs/ldraft_s0.zip")
    assert isinstance(o.draft, MaskablePPODraftPolicy)


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
