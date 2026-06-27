from locma.harness.action_stats import policy_action_stats


def test_policy_action_stats_smoke():
    stats = policy_action_stats("greedy", "random", games=2, seed=0)
    rates = stats.as_rates()
    assert stats.decisions > 0
    assert rates["decisions"] == float(stats.decisions)
    assert 0.0 <= rates["attack"] <= 1.0
    assert 0.0 <= rates["lethal_take"] <= 1.0
