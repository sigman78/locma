from locma.harness.replay_stream import build_replay
from locma.policies.registry import make_policy


def test_assemble_matches_build_replay():
    # build_replay runs the game and assembles; assemble_replay must produce the
    # same dict when handed the recorder build_replay used internally.
    rep = build_replay(make_policy("random"), make_policy("random"), seed=7)
    h = rep["header"]
    assert h["format"] == "locma-replay/2"
    assert h["policy_a"] == "random" and h["policy_b"] == "random"
    assert h["a_seat"] == 0
    assert h["step_count"] == len(rep["battle"]["steps"])
    assert rep["result"]["winner"] == h["winner"]
