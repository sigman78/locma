"""Pure plumbing tests for the E25 strong-league driver (no matches run)."""

from locma.policies.registry import make_policy
from scripts.e25_strong_league import CANDIDATE, LDRAFT, OPPONENTS


def test_e25_candidate_is_rbeam_ror_at_matched_ldraft():
    assert CANDIDATE.startswith("rbeam:depot:shared/shared_s0.zip|")
    assert ",8,20,4,4," in CANDIDATE  # width, max_actions, n_plans, n_worlds (4x4)
    assert CANDIDATE.endswith(LDRAFT)


def test_e25_every_opponent_uses_matched_ldraft_and_parses(monkeypatch):
    # make_policy eagerly resolves depot: refs to on-disk blobs; CI runs without
    # pulled artifacts, so stub the resolver. Models still load lazily, so this
    # exercises spec parsing / policy construction without needing the weights.
    monkeypatch.setattr("locma.policies.registry.resolve_path", lambda p: p)
    labels = [label for label, _spec, _fair in OPPONENTS]
    assert {"vbeam", "dmcts:15,100", "azlite:100"} <= set(labels)
    for _label, spec, _fair in OPPONENTS:
        assert spec.endswith(LDRAFT), spec  # matched draft isolates battle search
        make_policy(spec)  # constructs (lazily) without error


def test_e25_only_azlite_is_flagged_cheating():
    cheats = {label for label, _spec, fair in OPPONENTS if not fair}
    assert cheats == {"azlite:100"}  # perfect foresight; excluded from fair mean
