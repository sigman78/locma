from __future__ import annotations

import pytest
from typer.testing import CliRunner

from locma.cli.app import app
from locma.harness.draft_bench import (
    DRAFTS,
    draft_names,
    duel,
    make_battle,
    make_draft,
    round_robin,
)
from locma.policies.drafts import PartialRandomDraftPolicy


def test_draft_names_nonempty_and_known():
    names = draft_names()
    assert "balanced" in names and "greedy" in names
    assert set(names) == set(DRAFTS)


def test_make_draft_constructs_and_rejects_unknown():
    assert make_draft("balanced").name
    assert make_draft("random").name
    with pytest.raises(ValueError):
        make_draft("nope")


def test_make_battle_extracts_battle_half():
    # `ground` is the battle half shared by max-guard/max-attack; it must exist.
    b = make_battle("ground")
    assert hasattr(b, "battle_action")


def test_duel_is_deterministic():
    r1 = duel("greedy", "random", battle="ground", games=20, seed=0)
    r2 = duel("greedy", "random", battle="ground", games=20, seed=0)
    assert r1.win_rate_a == r2.win_rate_a


@pytest.mark.parametrize("draft", ["greedy", "random", "balanced"])
def test_self_duel_is_exactly_half(draft):
    # Same draft on both seats + same battle, mirrored: for each seed the two
    # identical policies yield the same winner-seat, so A wins exactly one of the
    # mirrored pair -> exactly 0.5. This is the benchmark's calibration check, and
    # it must hold for stateless ("greedy"), seeded ("random"), AND stateful
    # ("balanced", which tracks its own picks) drafts — i.e. per-game reset works.
    r = duel(draft, draft, battle="ground", games=64, seed=0)
    assert r.win_rate_a == 0.5


def test_duel_mirror_is_antisymmetric():
    ab = duel("balanced", "greedy", battle="ground", games=25, seed=3)
    ba = duel("greedy", "balanced", battle="ground", games=25, seed=3)
    assert abs(ab.win_rate_a - (1.0 - ba.win_rate_a)) < 1e-9


def test_round_robin_matrix_is_antisymmetric_and_scored():
    drafts = ["greedy", "balanced", "random"]
    s = round_robin(drafts, battle="ground", games=12, seed=0)
    for a in s.drafts:
        for b in s.drafts:
            if a != b:
                assert abs(s.win_matrix[(a, b)] + s.win_matrix[(b, a)] - 1.0) < 1e-9
    assert set(s.avg_win_rate) == set(s.drafts)
    # avg win rate over the field is bounded in [0, 1]
    for v in s.avg_win_rate.values():
        assert 0.0 <= v <= 1.0
    # n_per_pair is the doubled (total mirrored) game count, matching DuelResult.n
    assert s.n_per_pair == 24


def test_round_robin_cells_match_independent_duel():
    # Non-tautological: a matrix cell must equal an independent duel() of the same pair.
    drafts = ["greedy", "balanced", "random"]
    s = round_robin(drafts, battle="ground", games=12, seed=0)
    d = duel("greedy", "balanced", battle="ground", games=12, seed=0)
    assert s.win_matrix[("greedy", "balanced")] == d.win_rate_a
    # avg_win_rate is the row mean over the rest of the field
    for a in drafts:
        row = [s.win_matrix[(a, b)] for b in drafts if b != a]
        assert abs(s.avg_win_rate[a] - sum(row) / len(row)) < 1e-12


def test_cli_draft_bench_smoke():
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["draft-bench", "greedy", "balanced", "--battle", "ground", "--games", "8", "--seed", "0"],
    )
    assert res.exit_code == 0
    assert "ranking" in res.stdout.lower()


def test_cli_draft_bench_rejects_unknown_draft():
    runner = CliRunner()
    res = runner.invoke(app, ["draft-bench", "greedy", "nope", "--games", "4"])
    assert res.exit_code != 0


def test_cli_draft_bench_rejects_unknown_battle():
    runner = CliRunner()
    res = runner.invoke(
        app, ["draft-bench", "greedy", "balanced", "--battle", "bogus", "--games", "4"]
    )
    assert res.exit_code != 0


def test_cli_draft_bench_rejects_zero_games():
    runner = CliRunner()
    res = runner.invoke(app, ["draft-bench", "greedy", "balanced", "--games", "0"])
    assert res.exit_code != 0


def test_make_draft_parses_rnd_suffix():
    p = make_draft("balanced+rnd4")
    assert isinstance(p, PartialRandomDraftPolicy)
    assert p.k == 4 and p.name == "balanced+rnd4"
    with pytest.raises(ValueError):
        make_draft("nope+rnd4")  # unknown base
    with pytest.raises(ValueError):
        make_draft("balanced+rndx")  # non-integer K
    with pytest.raises(ValueError):
        make_draft("balanced+rnd31")  # K out of [0, 30]


def test_noisy_self_duel_is_exactly_half():
    # The calibration guarantee must survive the noise wrapper: run_match resets
    # both policies to the game seed, so two identical `+rndK` instances pick the
    # same random rounds AND the same random cards -> mirror still cancels exactly.
    r = duel("balanced+rnd4", "balanced+rnd4", battle="ground", games=32, seed=0)
    assert r.win_rate_a == 0.5


def test_noisy_draft_loses_to_clean_base():
    # Sanity direction check: replacing 8 of 30 balanced picks with uniform noise
    # must not IMPROVE the deck (win rate <= 0.5 vs the clean draft, same pilot).
    r = duel("balanced+rnd8", "balanced", battle="ground", games=48, seed=0)
    assert r.win_rate_a <= 0.5


def test_round_robin_parallel_matches_serial():
    drafts = ["greedy", "balanced", "random"]
    serial = round_robin(drafts, battle="ground", games=8, seed=0, workers=1)
    par = round_robin(drafts, battle="ground", games=8, seed=0, workers=2)
    assert serial.win_matrix == par.win_matrix
    assert serial.avg_win_rate == par.avg_win_rate
