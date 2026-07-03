"""Tests for the artifact depot (locma/depot): publish/resolve/pin/verify/gc,
the dir remote roundtrip, and the gh remote's command construction (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from locma.depot import core, remote
from locma.depot.core import (
    DepotError,
    gc,
    load_record,
    parse_ref,
    pin,
    publish,
    resolve,
    resolve_path,
    verify,
)


@pytest.fixture()
def root(tmp_path, monkeypatch):
    """A fresh depot root, also set as the process default via LOCMA_DEPOT."""
    d = tmp_path / "depot"
    monkeypatch.setenv("LOCMA_DEPOT", str(d))
    return d


def _mkfile(tmp_path, name: str, content: bytes = b"payload") -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


# ---------------------------------------------------------------------------
# publish + index + store
# ---------------------------------------------------------------------------


def test_publish_creates_index_and_blobs(root, tmp_path):
    f0 = _mkfile(tmp_path, "m_s0.zip", b"aaa")
    f1 = _mkfile(tmp_path, "m_s1.zip", b"bbb")
    vrec = publish("m", [f0, f1], kind="model", note="first", root=root)

    assert vrec["version"] == 1
    rec = load_record("m", root)
    assert rec["pin"] == 1 and rec["kind"] == "model"
    assert set(vrec["files"]) == {"m_s0.zip", "m_s1.zip"}
    for fname, entry in vrec["files"].items():
        blob = core.blob_path(root, entry["sha256"], fname)
        assert blob.is_file()
        assert core.sha256_file(blob) == entry["sha256"]


def test_publish_advances_pin_and_keep_pin(root, tmp_path):
    f = _mkfile(tmp_path, "a.zip", b"v1")
    publish("m", [f], root=root)
    f.write_bytes(b"v2")
    publish("m", [f], root=root)
    assert load_record("m", root)["pin"] == 2
    f.write_bytes(b"v3")
    publish("m", [f], keep_pin=True, root=root)
    rec = load_record("m", root)
    assert rec["pin"] == 2 and len(rec["versions"]) == 3


def test_publish_absorbs_sidecar_manifest(root, tmp_path):
    f = _mkfile(tmp_path, "data.npz", b"blob")
    (tmp_path / "data.manifest.json").write_text(json.dumps({"games": 400, "seed": 7}))
    vrec = publish("d", [f], kind="dataset", root=root)
    assert vrec["meta"]["manifests"]["data.npz"] == {"games": 400, "seed": 7}


def test_publish_rejects_bad_names_and_kind_change(root, tmp_path):
    f = _mkfile(tmp_path, "a.zip")
    with pytest.raises(DepotError, match="invalid artifact name"):
        publish("Bad/Name", [f], root=root)
    publish("m", [f], kind="model", root=root)
    with pytest.raises(DepotError, match="kind"):
        publish("m", [f], kind="dataset", root=root)


def test_publish_records_provenance(root, tmp_path):
    f = _mkfile(tmp_path, "a.zip")
    vrec = publish("m", [f], parents=["zoo@1"], meta={"avg_hard3": 0.657}, root=root)
    assert vrec["parents"] == ["zoo@1"]
    assert vrec["meta"]["avg_hard3"] == 0.657
    assert "command" in vrec and "created" in vrec
    # runs inside the repo, so the commit should be captured
    assert vrec["git_commit"] is None or len(vrec["git_commit"]) >= 7


# ---------------------------------------------------------------------------
# refs + resolve
# ---------------------------------------------------------------------------


def test_parse_ref_grammar():
    assert parse_ref("depot:b0/b0_s0.zip") == ("b0", "", "b0_s0.zip")
    assert parse_ref("depot:b0@3/b0_s0.zip") == ("b0", "3", "b0_s0.zip")
    assert parse_ref("depot:b0@latest/b0_s0.zip") == ("b0", "latest", "b0_s0.zip")
    assert parse_ref("depot:b0") == ("b0", "", None)
    with pytest.raises(DepotError):
        parse_ref("depot:b0@nope/x")
    with pytest.raises(DepotError):
        parse_ref("runs/b0_s0.zip")


def test_resolve_pin_explicit_latest(root, tmp_path):
    f = _mkfile(tmp_path, "a.zip", b"v1")
    publish("m", [f], root=root)
    f.write_bytes(b"v2")
    publish("m", [f], root=root)
    pin("m", 1, root=root)

    assert resolve("depot:m/a.zip", root).read_bytes() == b"v1"
    assert resolve("depot:m@2/a.zip", root).read_bytes() == b"v2"
    assert resolve("depot:m@latest/a.zip", root).read_bytes() == b"v2"


def test_resolve_single_file_shorthand_and_ambiguity(root, tmp_path):
    f0 = _mkfile(tmp_path, "s0.zip", b"0")
    f1 = _mkfile(tmp_path, "s1.zip", b"1")
    publish("solo", [f0], root=root)
    publish("pair", [f0, f1], root=root)
    assert resolve("depot:solo", root).name == "s0.zip"
    with pytest.raises(DepotError, match="ambiguous"):
        resolve("depot:pair", root)
    with pytest.raises(DepotError, match="no file"):
        resolve("depot:pair/nope.zip", root)


def test_resolve_missing_blob_says_pull(root, tmp_path):
    f = _mkfile(tmp_path, "a.zip")
    vrec = publish("m", [f], root=root)
    core.blob_path(root, vrec["files"]["a.zip"]["sha256"], "a.zip").unlink()
    with pytest.raises(DepotError, match="locma depot pull m@1"):
        resolve("depot:m/a.zip", root)


def test_resolve_path_passthrough_and_scheme(root, tmp_path):
    assert resolve_path("runs/b0_s0.zip") == "runs/b0_s0.zip"
    assert resolve_path(r"F:\abs\model.zip") == r"F:\abs\model.zip"
    f = _mkfile(tmp_path, "a.zip", b"x")
    publish("m", [f], root=root)  # LOCMA_DEPOT env makes this the default root
    assert Path(resolve_path("depot:m/a.zip")).read_bytes() == b"x"


def test_unknown_artifact_errors(root):
    with pytest.raises(DepotError, match="no artifact named"):
        resolve("depot:nope/x.zip", root)


# ---------------------------------------------------------------------------
# verify + gc
# ---------------------------------------------------------------------------


def test_verify_detects_corruption(root, tmp_path):
    f = _mkfile(tmp_path, "a.zip", b"good")
    vrec = publish("m", [f], root=root)
    assert verify(root) == []
    blob = core.blob_path(root, vrec["files"]["a.zip"]["sha256"], "a.zip")
    blob.write_bytes(b"evil")
    problems = verify(root)
    assert len(problems) == 1 and "hash mismatch" in problems[0]


def test_gc_keeps_pinned_drops_rest(root, tmp_path):
    f = _mkfile(tmp_path, "a.zip", b"v1")
    publish("m", [f], root=root)
    f.write_bytes(b"v2")
    publish("m", [f], root=root)  # pin -> v2; v1's blob now unreachable

    removed, freed = gc(root, dry_run=True)
    assert len(removed) == 1 and freed == 2
    assert resolve("depot:m@1/a.zip", root).is_file()  # dry run deletes nothing

    removed, _ = gc(root, dry_run=False)
    assert len(removed) == 1
    assert resolve("depot:m@2/a.zip", root).is_file()
    with pytest.raises(DepotError, match="pull"):
        resolve("depot:m@1/a.zip", root)


# ---------------------------------------------------------------------------
# remotes
# ---------------------------------------------------------------------------


def test_dir_remote_roundtrip_between_machines(root, tmp_path, monkeypatch):
    """Machine A publishes and pushes; machine B has the index (via git) but no
    blobs, pulls, and resolves — with hash verification on the way in."""
    share = tmp_path / "share"
    monkeypatch.setenv("LOCMA_DEPOT_REMOTE", f"dir:{share}")

    f = _mkfile(tmp_path, "a.zip", b"model-bytes")
    publish("m", [f], note="rr", root=root)
    locator = remote.push("m", root=root)
    assert locator.startswith("dir:")
    assert load_record("m", root)["versions"][0]["published"] == locator

    # "machine B": copy only the committed index, then pull
    root_b = tmp_path / "depot-b"
    (root_b / "index").mkdir(parents=True)
    (root_b / "index" / "m.json").write_text(
        (root / "index" / "m.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    fetched = remote.pull("m", root=root_b)
    assert fetched == ["a.zip"]
    assert resolve("depot:m/a.zip", root_b).read_bytes() == b"model-bytes"
    assert remote.pull("m", root=root_b) == []  # already complete


def test_pull_rejects_tampered_remote(root, tmp_path, monkeypatch):
    share = tmp_path / "share"
    monkeypatch.setenv("LOCMA_DEPOT_REMOTE", f"dir:{share}")
    f = _mkfile(tmp_path, "a.zip", b"good")
    vrec = publish("m", [f], root=root)
    remote.push("m", root=root)
    (share / "m" / "v1" / "a.zip").write_bytes(b"tampered")

    core.blob_path(root, vrec["files"]["a.zip"]["sha256"], "a.zip").unlink()
    with pytest.raises(DepotError, match="hash mismatch"):
        remote.pull("m", root=root)
    # nothing tampered was stored
    assert not core.blob_path(root, vrec["files"]["a.zip"]["sha256"], "a.zip").is_file()


def test_remote_spec_selection(root, tmp_path, monkeypatch):
    monkeypatch.delenv("LOCMA_DEPOT_REMOTE", raising=False)
    assert remote.remote_spec(root) == "gh"  # no config -> default
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"remote": "dir:X"}))
    assert remote.remote_spec(root) == "dir:X"
    monkeypatch.setenv("LOCMA_DEPOT_REMOTE", "dir:Y")
    assert remote.remote_spec(root) == "dir:Y"
    assert isinstance(remote.make_remote("gh"), remote.GhReleasesRemote)
    assert isinstance(remote.make_remote("dir:Z"), remote.DirRemote)
    with pytest.raises(DepotError):
        remote.make_remote("s3://nope")


def test_gh_remote_command_construction(root, tmp_path, monkeypatch):
    """No network: capture the gh invocations for create-new and update paths."""
    calls: list[list[str]] = []

    class FakeProc:
        def __init__(self, rc):
            self.returncode = rc
            self.stderr = ""

    gh = remote.GhReleasesRemote()

    def fake_run(args, **kw):
        calls.append(args)
        # first probe: release does not exist -> create; later probes: exists
        if args[:2] == ["release", "view"]:
            return FakeProc(0 if any(a[:2] == ["release", "create"] for a in calls) else 1)
        return FakeProc(0)

    monkeypatch.setattr(gh, "_run", fake_run)
    f = _mkfile(tmp_path, "a.zip")
    gh.push_files("b0", 1, {"a.zip": f}, "notes")
    create = next(a for a in calls if a[:2] == ["release", "create"])
    assert create[2] == "depot/b0-v1" and str(f) in create and "--notes" in create

    gh.push_files("b0", 1, {"a.zip": f}, "notes")  # now exists -> upload --clobber
    upload = next(a for a in calls if a[:2] == ["release", "upload"])
    assert upload[2] == "depot/b0-v1" and "--clobber" in upload

    calls.clear()
    gh.pull_files("b0", 1, ["a.zip"], tmp_path)
    dl = calls[0]
    assert dl[:3] == ["release", "download", "depot/b0-v1"]
    assert "--pattern" in dl and "a.zip" in dl


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_publish_list_resolve_show(root, tmp_path):
    from typer.testing import CliRunner  # noqa: PLC0415

    from locma.cli.app import app  # noqa: PLC0415

    runner = CliRunner()
    f = _mkfile(tmp_path, "a.zip", b"x")
    res = runner.invoke(
        app,
        ["depot", "publish", "m", str(f), "--note", "n", "--meta", '{"avg_hard3": 0.5}'],
    )
    assert res.exit_code == 0, res.output
    assert "published m v1" in res.output

    res = runner.invoke(app, ["depot", "list"])
    assert res.exit_code == 0 and "m" in res.output

    res = runner.invoke(app, ["depot", "resolve", "depot:m/a.zip"])
    assert res.exit_code == 0
    assert Path(res.output.strip()).read_bytes() == b"x"

    res = runner.invoke(app, ["depot", "show", "m"])
    assert res.exit_code == 0 and "avg_hard3" in res.output

    res = runner.invoke(app, ["depot", "verify"])
    assert res.exit_code == 0

    res = runner.invoke(app, ["depot", "resolve", "depot:nope/x"])
    assert res.exit_code == 1
