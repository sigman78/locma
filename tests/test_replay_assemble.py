from locma.core.engine import run_game
from locma.harness.replay_stream import StreamRecorder, assemble_replay, build_replay
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


def test_assemble_replay_directly_matches_build_replay():
    rec = StreamRecorder()
    result = run_game(
        make_policy("random"),
        make_policy("random"),
        7,
        on_step=rec.on_step,
        on_snapshot=rec.on_snapshot,
        on_pre_step=rec.on_pre_step,
        on_event=rec.on_event,
    )
    rep = assemble_replay(
        rec,
        winner=result.winner,
        turns=result.turns,
        policy_a="random",
        policy_b="random",
        seed=7,
        a_seat=0,
        source="ad-hoc",
        created_at="2020-01-01T00:00:00+00:00",
    )
    expected = build_replay(
        make_policy("random"),
        make_policy("random"),
        7,
        created_at="2020-01-01T00:00:00+00:00",
    )
    assert rep == expected
