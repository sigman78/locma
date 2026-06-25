import json

import numpy as np

from locma.envs.encode import ACTION_SIZE, OBS_SIZE
from locma.envs.practicum import _Collector, _manifest_path, record_practicum


def test_manifest_path_derivation():
    assert _manifest_path("runs/p.npz") == "runs/p.manifest.json"
    assert _manifest_path("p") == "p.manifest.json"


def test_collector_records_only_teacher_seat_and_skips_forced(monkeypatch):
    import locma.envs.practicum as P  # noqa: PLC0415
    from locma.core.actions import Attack, Pass  # noqa: PLC0415

    # Fake out the engine/encoding lookups the collector makes. monkeypatch
    # auto-restores them so later tests run against the real functions.
    state = {"legal": [Attack(1, -1), Pass()]}
    monkeypatch.setattr(P, "battle_legal", lambda gs: state["legal"])
    monkeypatch.setattr(P, "make_battle_view", lambda gs: object())
    monkeypatch.setattr(P, "encode_battle", lambda view: np.zeros(OBS_SIZE, dtype=np.float32))
    monkeypatch.setattr(P, "action_mask", lambda view, legal: np.zeros(ACTION_SIZE, dtype=bool))
    monkeypatch.setattr(P, "sem_index", lambda view, a: 0)

    c = _Collector(teacher_seat=0)
    c(1, Pass(), object())  # wrong seat -> ignored
    assert c.action == []
    c(0, Attack(1, -1), object())  # teacher seat, 2 legal -> recorded at index 0
    assert c.action == [0]
    state["legal"] = [Pass()]
    c(0, Pass(), object())  # forced (1 legal) -> skipped
    assert c.action == [0]


def test_collector_drops_unmappable_action(monkeypatch):
    import locma.envs.practicum as P  # noqa: PLC0415
    from locma.core.actions import Attack  # noqa: PLC0415

    monkeypatch.setattr(P, "battle_legal", lambda gs: [Attack(1, -1), Attack(2, -1)])
    monkeypatch.setattr(P, "make_battle_view", lambda gs: object())
    monkeypatch.setattr(P, "encode_battle", lambda view: np.zeros(OBS_SIZE, dtype=np.float32))
    monkeypatch.setattr(P, "action_mask", lambda view, legal: np.zeros(ACTION_SIZE, dtype=bool))
    monkeypatch.setattr(P, "sem_index", lambda view, a: None)  # unmappable

    c = _Collector(teacher_seat=0)
    c(0, Attack(1, -1), object())
    assert c.action == []
    assert c.dropped == 1


def test_collector_skips_illegal_action(monkeypatch):
    import locma.envs.practicum as P  # noqa: PLC0415
    from locma.core.actions import Attack, Pass  # noqa: PLC0415

    monkeypatch.setattr(P, "battle_legal", lambda gs: [Attack(1, -1), Pass()])
    monkeypatch.setattr(P, "make_battle_view", lambda gs: object())
    monkeypatch.setattr(P, "encode_battle", lambda view: np.zeros(OBS_SIZE, dtype=np.float32))
    monkeypatch.setattr(P, "action_mask", lambda view, legal: np.zeros(ACTION_SIZE, dtype=bool))
    monkeypatch.setattr(P, "sem_index", lambda view, a: 0)

    c = _Collector(teacher_seat=0)
    # Pass an action not in the legal list — should return silently, not raise.
    c(0, Attack(99, -1), object())
    assert c.action == []
    assert c.dropped == 0


def test_record_practicum_writes_npz_and_manifest(tmp_path):
    out = str(tmp_path / "practicum.npz")
    # Cheap teacher (no sb3, no search): greedy vs random, both seats.
    manifest = record_practicum(teacher="greedy", opponents=("random",), games=3, out=out, seed=0)
    data = np.load(out)
    n = manifest["n_examples"]
    assert n > 0
    assert data["obs"].shape == (n, OBS_SIZE)
    assert data["action"].shape == (n,)
    assert data["mask"].shape == (n, ACTION_SIZE)
    assert data["game_id"].shape == (n,)
    # both seat orientations captured, and multiple games recorded
    assert set(np.unique(data["seat"]).tolist()) == {0, 1}
    assert len(np.unique(data["game_id"])) >= 2
    # every recorded action index is legal-representable
    assert data["action"].max() < ACTION_SIZE
    # manifest round-trips and carries the layout guard
    with open(_manifest_path(out)) as f:
        m = json.load(f)
    assert m["obs_size"] == OBS_SIZE and m["action_size"] == ACTION_SIZE
    assert m["teacher"] == "greedy" and m["opponents"] == ["random"]
