from locma.harness import ppo_scorecard as S
from locma.harness.ppo_scorecard import HARD_BASELINES, PolicyScore


def test_policy_score_avg_hard3():
    row = PolicyScore(
        spec="x",
        label="x",
        scores={"greedy": 0.6, "max-guard": 0.5, "max-attack": 0.7, "dmcts": 0.25},
        games={},
    )
    assert HARD_BASELINES == ("greedy", "max-guard", "max-attack")
    assert row.avg_hard3 == 0.6
    assert row.dmcts == 0.25


def test_score_policies_parallel_assembles_rows(monkeypatch):
    def fake_score_cell(spec, label, opponent, games, seed):
        rate = {"greedy": 0.6, "max-guard": 0.5, "dmcts": 0.25}[opponent]
        return label, opponent, rate, games * 2

    monkeypatch.setattr(S, "_score_cell", fake_score_cell)
    rows = S.score_policies(
        [("spec-a", "a")],
        opponents=("greedy", "max-guard", "dmcts"),
        games=10,
        dmcts_games=3,
        seed=7,
        workers=2,
    )
    assert rows[0].spec == "spec-a"
    assert rows[0].scores == {"greedy": 0.6, "max-guard": 0.5, "dmcts": 0.25}
    assert rows[0].games == {"greedy": 20, "max-guard": 20, "dmcts": 6}


def test_score_policies_can_skip_dmcts(monkeypatch):
    calls = []

    def fake_score_cell(spec, label, opponent, games, seed):
        calls.append(opponent)
        return label, opponent, 0.5, games * 2

    monkeypatch.setattr(S, "_score_cell", fake_score_cell)
    rows = S.score_policies(
        [("spec-a", "a")],
        opponents=("greedy", "dmcts"),
        games=10,
        dmcts_games=0,
        seed=7,
    )
    assert calls == ["greedy"]
    assert rows[0].scores == {"greedy": 0.5}
