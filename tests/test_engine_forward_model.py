from locma.core.engine import run_game
from locma.policies.registry import make_policy


def test_mcts_plays_a_full_game_via_run_game():
    # If run_game did not pass `state`, MCTSBattlePolicy.battle_action would
    # raise ValueError("requires the forward-model `state`").
    a = make_policy("mcts:8,1.4,0")
    b = make_policy("greedy")
    result = run_game(a, b, seed=0)
    assert result.winner in (0, 1)
