# tests/test_fetch_art.py
from __future__ import annotations

import json
import os

from locma.data import fetch


def _fake_download_factory(calls):
    def fake(url, path):
        calls.append((url, path))
        with open(path, "wb") as f:
            f.write(b"PNG")
        return True
    return fake


def test_url_and_filename_zero_padded(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fetch, "_download", _fake_download_factory(calls))
    monkeypatch.setattr(fetch, "_REQUEST_DELAY", 0)
    n = fetch.fetch_art(dest=str(tmp_path))
    assert n == 160
    first_url, first_path = calls[0]
    assert first_url == "https://legendsofcodeandmagic.com/portraits/001.png"
    assert first_path.endswith("001.png")
    assert (tmp_path / "001.png").exists()
    assert (tmp_path / "160.png").exists()


def test_manifest_written(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "_download", _fake_download_factory([]))
    monkeypatch.setattr(fetch, "_REQUEST_DELAY", 0)
    fetch.fetch_art(dest=str(tmp_path))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["1"] == {
        "file": "001.png",
        "url": "https://legendsofcodeandmagic.com/portraits/001.png",
    }


def test_corrupt_manifest_tolerated(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(fetch, "_download", _fake_download_factory([]))
    monkeypatch.setattr(fetch, "_REQUEST_DELAY", 0)
    n = fetch.fetch_art(dest=str(tmp_path))  # must not raise
    assert n == 160


def test_never_raises_on_download_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "_download", lambda url, path: False)
    monkeypatch.setattr(fetch, "_REQUEST_DELAY", 0)
    n = fetch.fetch_art(dest=str(tmp_path))  # all fail, no exception
    assert n == 0


def test_skip_existing(tmp_path, monkeypatch):
    (tmp_path / "001.png").write_bytes(b"old")
    ids = []
    def fake(url, path):
        ids.append(os.path.basename(path))
        with open(path, "wb") as f:
            f.write(b"PNG")
        return True
    import os
    monkeypatch.setattr(fetch, "_download", fake)
    monkeypatch.setattr(fetch, "_REQUEST_DELAY", 0)
    n = fetch.fetch_art(dest=str(tmp_path))
    assert "001.png" not in ids       # skipped
    assert n == 159                   # 160 - 1 already present


def test_force_redownloads(tmp_path, monkeypatch):
    (tmp_path / "001.png").write_bytes(b"old")
    ids = []
    def fake(url, path):
        ids.append(os.path.basename(path))
        with open(path, "wb") as f:
            f.write(b"PNG")
        return True
    import os
    monkeypatch.setattr(fetch, "_download", fake)
    monkeypatch.setattr(fetch, "_REQUEST_DELAY", 0)
    n = fetch.fetch_art(dest=str(tmp_path), force=True)
    assert "001.png" in ids           # re-downloaded
    assert n == 160
