"""Remote transports for the depot — share published versions across machines.

A remote is deliberately dumb storage: it holds the files of (artifact, version)
under a stable address, nothing else. Provenance, hashes and pins travel in the
git-committed index, and every pull is verified against those hashes — so the
transport needs no trust and no smarts, and swapping backends means
implementing the two-method ``Remote`` protocol (~30 lines).

Backends:
  ``gh``          GitHub Releases via the ``gh`` CLI (default) — one release per
                  published version, tag ``depot/<name>-v<N>``, files as assets.
                  Zero infra beyond the repo itself; 2 GB/file limit.
  ``dir:<path>``  A plain directory tree ``<path>/<name>/v<N>/<file>`` — for a
                  NAS/network share, and the template for an S3/rclone backend
                  (same layout, swap copy for put/get).

Selection: ``LOCMA_DEPOT_REMOTE`` env var, else ``depot/config.json``
``{"remote": "gh"}``, else ``gh``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

from locma.depot.core import (
    DepotError,
    blob_path,
    depot_root,
    load_record,
    select_version,
    sha256_file,
)


class Remote(Protocol):
    """The whole backend contract: upload/download one version's files."""

    def push_files(self, name: str, version: int, files: dict[str, Path], notes: str) -> str:
        """Upload ``{filename: local_path}``; return a locator string for the index."""
        ...

    def pull_files(self, name: str, version: int, filenames: list[str], dest: Path) -> None:
        """Download the named files into ``dest`` (flat)."""
        ...


class DirRemote:
    """Dumb directory remote: ``<base>/<name>/v<N>/<file>``."""

    def __init__(self, base: str | Path):
        self.base = Path(base)

    def push_files(self, name: str, version: int, files: dict[str, Path], notes: str) -> str:
        dest = self.base / name / f"v{version}"
        dest.mkdir(parents=True, exist_ok=True)
        for fname, src in files.items():
            shutil.copy2(src, dest / fname)
        (dest / "provenance.json").write_text(notes, encoding="utf-8")
        return f"dir:{dest}"

    def pull_files(self, name: str, version: int, filenames: list[str], dest: Path) -> None:
        src = self.base / name / f"v{version}"
        for fname in filenames:
            path = src / fname
            if not path.is_file():
                raise DepotError(f"remote is missing {name}@{version}/{fname} ({path})")
            shutil.copy2(path, dest / fname)


class GhReleasesRemote:
    """GitHub Releases via the ``gh`` CLI (auth and repo come from the checkout)."""

    @staticmethod
    def _tag(name: str, version: int) -> str:
        return f"depot/{name}-v{version}"

    def _run(self, args: list[str], **kw) -> subprocess.CompletedProcess:
        return subprocess.run(["gh", *args], capture_output=True, text=True, **kw)

    def push_files(self, name: str, version: int, files: dict[str, Path], notes: str) -> str:
        tag = self._tag(name, version)
        paths = [str(p) for p in files.values()]
        if self._run(["release", "view", tag]).returncode == 0:
            res = self._run(["release", "upload", tag, *paths, "--clobber"])
        else:
            res = self._run(
                [
                    "release",
                    "create",
                    tag,
                    *paths,
                    "--title",
                    f"depot: {name} v{version}",
                    "--notes",
                    notes,
                ]
            )
        if res.returncode != 0:
            raise DepotError(f"gh push of {tag} failed: {res.stderr.strip()}")
        return f"gh:{tag}"

    def pull_files(self, name: str, version: int, filenames: list[str], dest: Path) -> None:
        tag = self._tag(name, version)
        args = ["release", "download", tag, "--dir", str(dest)]
        for fname in filenames:
            args += ["--pattern", fname]
        res = self._run(args)
        if res.returncode != 0:
            raise DepotError(f"gh pull of {tag} failed: {res.stderr.strip()}")


def remote_spec(root: Path | None = None) -> str:
    env = os.environ.get("LOCMA_DEPOT_REMOTE")
    if env:
        return env
    config = (root or depot_root()) / "config.json"
    if config.is_file():
        with open(config, encoding="utf-8") as f:
            return json.load(f).get("remote", "gh")
    return "gh"


def make_remote(spec: str) -> Remote:
    if spec == "gh":
        return GhReleasesRemote()
    if spec.startswith("dir:"):
        return DirRemote(spec[4:])
    raise DepotError(f"unknown remote spec '{spec}' (want 'gh' or 'dir:<path>')")


def _get_remote(root: Path | None) -> Remote:
    return make_remote(remote_spec(root))


def _provenance_notes(rec: dict, vrec: dict) -> str:
    body = {k: vrec[k] for k in vrec if k != "files"}
    body["files"] = {f: e["sha256"] for f, e in vrec["files"].items()}
    return (
        f"Depot artifact `{rec['name']}` v{vrec['version']} ({rec['kind']}).\n\n"
        "```json\n" + json.dumps(body, indent=2) + "\n```\n"
    )


def push(
    name: str, selector: str = "", root: Path | None = None, remote: Remote | None = None
) -> str:
    """Upload one version's blobs (default: the pin) and mark it published."""
    root = root or depot_root()
    rec = load_record(name, root)
    vrec = select_version(rec, selector)
    files: dict[str, Path] = {}
    for fname, entry in vrec["files"].items():
        blob = blob_path(root, entry["sha256"], fname)
        if not blob.is_file():
            raise DepotError(f"cannot push {name}@{vrec['version']}: blob missing for {fname}")
        files[fname] = blob
    remote = remote or _get_remote(root)
    locator = remote.push_files(rec["name"], vrec["version"], files, _provenance_notes(rec, vrec))
    vrec["published"] = locator
    from locma.depot.core import _save_record  # noqa: PLC0415 — avoid exporting it

    _save_record(rec, root)
    return locator


def pull(
    name: str, selector: str = "", root: Path | None = None, remote: Remote | None = None
) -> list[str]:
    """Download one version's blobs (default: the pin), verify hashes, store them.

    Returns the filenames fetched (already-present blobs are skipped)."""
    root = root or depot_root()
    rec = load_record(name, root)
    vrec = select_version(rec, selector)
    missing = {
        fname: entry
        for fname, entry in vrec["files"].items()
        if not blob_path(root, entry["sha256"], fname).is_file()
    }
    if not missing:
        return []
    remote = remote or _get_remote(root)
    tmp_base = root / "tmp"
    tmp_base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_base) as tmp:
        tmp_dir = Path(tmp)
        remote.pull_files(rec["name"], vrec["version"], sorted(missing), tmp_dir)
        for fname, entry in missing.items():
            got = tmp_dir / fname
            if not got.is_file():
                raise DepotError(f"remote did not deliver {fname}")
            digest = sha256_file(got)
            if digest != entry["sha256"]:
                raise DepotError(
                    f"hash mismatch for {name}@{vrec['version']}/{fname}: "
                    f"index {entry['sha256'][:12]}.., got {digest[:12]}.. — refusing to store"
                )
            dest = blob_path(root, entry["sha256"], fname)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(got), dest)
    return sorted(missing)
