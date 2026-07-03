"""Versioned artifact depot — provenance-tracked storage for checkpoints and data.

Two-tier storage split: ``runs/`` stays a disposable scratch directory; anything
load-bearing is *published* into the depot, which separates small committed
metadata from large gitignored blobs:

    depot/
      config.json        committed   {"remote": "gh"}  (see locma/depot/remote.py)
      index/<name>.json  committed   versions + provenance + pin, one file per artifact
      store/sha256/<h2>/<hash>/<filename>   gitignored content-addressed blobs

Because the index lives in git, every clone knows every artifact, its versions,
provenance and hashes before downloading a byte — and a checkpoint cited in the
worklog is verifiable (hash mismatch = not what was measured).

An *artifact* is a named, versioned bundle of files (the natural unit here is
the seed-triple, e.g. b0_s{0,1,2}.zip). References:

    depot:<name>/<file>          pinned version (the default — reproducible)
    depot:<name>@3/<file>        explicit version
    depot:<name>@latest/<file>   deliberately floating
    depot:<name>                 whole-bundle shorthand, valid iff one file

``resolve_path()`` is the consumer choke point: raw paths pass through
untouched, ``depot:`` refs resolve against the local store. The depot root is
``./depot`` (cwd-relative, like ``runs/``), overridable via ``LOCMA_DEPOT``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

SCHEME = "depot:"

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

KINDS = ("model", "dataset", "eval", "other")


class DepotError(Exception):
    """Raised for all user-facing depot failures (bad ref, missing blob, ...)."""


def depot_root() -> Path:
    return Path(os.environ.get("LOCMA_DEPOT", "depot"))


def _index_dir(root: Path) -> Path:
    return root / "index"


def _store_dir(root: Path) -> Path:
    return root / "store"


def _index_path(root: Path, name: str) -> Path:
    return _index_dir(root) / f"{name}.json"


def blob_path(root: Path, digest: str, filename: str) -> Path:
    return _store_dir(root) / "sha256" / digest[:2] / digest / filename


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# index records
# ---------------------------------------------------------------------------


def artifact_names(root: Path | None = None) -> list[str]:
    root = root or depot_root()
    if not _index_dir(root).is_dir():
        return []
    return sorted(p.stem for p in _index_dir(root).glob("*.json"))


def load_record(name: str, root: Path | None = None) -> dict:
    root = root or depot_root()
    path = _index_path(root, name)
    if not path.is_file():
        raise DepotError(f"no artifact named '{name}' in {_index_dir(root)}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_record(rec: dict, root: Path) -> None:
    _index_dir(root).mkdir(parents=True, exist_ok=True)
    path = _index_path(root, rec["name"])
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=2, sort_keys=False)
        f.write("\n")


def find_version(rec: dict, version: int) -> dict:
    for v in rec["versions"]:
        if v["version"] == version:
            return v
    raise DepotError(f"artifact '{rec['name']}' has no version {version}")


def _git_provenance() -> tuple[str | None, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "-uno"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return commit, bool(status)
    except (OSError, subprocess.CalledProcessError):
        return None, False


def _sidecar_manifest(path: Path) -> dict | None:
    """Fold in an existing ``<stem>.manifest.json`` sidecar (the convention the
    npz collectors already write) so provenance is absorbed, not duplicated."""
    side = path.with_name(path.stem + ".manifest.json")
    if side.is_file():
        with open(side, encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# publish / pin
# ---------------------------------------------------------------------------


def publish(
    name: str,
    files: list[str | Path],
    *,
    kind: str = "model",
    note: str = "",
    parents: list[str] | tuple[str, ...] = (),
    meta: dict | None = None,
    command: str | None = None,
    keep_pin: bool = False,
    root: Path | None = None,
) -> dict:
    """Copy ``files`` into the content-addressed store and append a new version.

    Captures provenance automatically: git commit + dirty flag, the invoking
    command line, and any ``<stem>.manifest.json`` sidecars. The new version
    becomes the pin unless ``keep_pin`` (old refs keep meaning either way —
    a bare ref follows the pin, ``@N`` refs are immutable).
    """
    root = root or depot_root()
    if not _NAME_RE.match(name):
        raise DepotError(f"invalid artifact name '{name}' (want kebab-case: [a-z0-9._-])")
    if kind not in KINDS:
        raise DepotError(f"invalid kind '{kind}' (one of {KINDS})")
    paths = [Path(f) for f in files]
    if not paths:
        raise DepotError("publish needs at least one file")
    for p in paths:
        if not p.is_file():
            raise DepotError(f"file not found: {p}")
    filenames = [p.name for p in paths]
    if len(set(filenames)) != len(filenames):
        raise DepotError(f"duplicate filenames in bundle: {filenames}")

    try:
        rec = load_record(name, root)
        if rec.get("kind") != kind:
            raise DepotError(
                f"artifact '{name}' already exists with kind '{rec.get('kind')}', not '{kind}'"
            )
    except DepotError:
        if _index_path(root, name).is_file():
            raise
        rec = {"name": name, "kind": kind, "pin": 0, "versions": []}

    version = max((v["version"] for v in rec["versions"]), default=0) + 1
    files_entry: dict[str, dict] = {}
    manifests: dict[str, dict] = {}
    for p in paths:
        digest = sha256_file(p)
        dest = blob_path(root, digest, p.name)
        if not dest.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        files_entry[p.name] = {"sha256": digest, "size": p.stat().st_size}
        side = _sidecar_manifest(p)
        if side is not None:
            manifests[p.name] = side

    commit, dirty = _git_provenance()
    vmeta = dict(meta or {})
    if manifests:
        vmeta.setdefault("manifests", manifests)
    vrec = {
        "version": version,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "command": command if command is not None else " ".join(sys.argv),
        "parents": list(parents),
        "note": note,
        "meta": vmeta,
        "files": files_entry,
    }
    rec["versions"].append(vrec)
    if not (keep_pin and rec["pin"]):
        rec["pin"] = version
    _save_record(rec, root)
    return vrec


def pin(name: str, version: int, root: Path | None = None) -> None:
    root = root or depot_root()
    rec = load_record(name, root)
    find_version(rec, version)  # validates
    rec["pin"] = version
    _save_record(rec, root)


# ---------------------------------------------------------------------------
# reference resolution
# ---------------------------------------------------------------------------


def parse_ref(ref: str) -> tuple[str, str, str | None]:
    """``depot:name[@ver]/file`` -> (name, ver_selector, filename|None).

    ``ver_selector`` is ``""`` (pin), ``"latest"``, or a decimal string.
    """
    if not ref.startswith(SCHEME):
        raise DepotError(f"not a depot ref: '{ref}'")
    body = ref[len(SCHEME) :]
    namever, sep, filename = body.partition("/")
    name, _, ver = namever.partition("@")
    if not name:
        raise DepotError(f"malformed depot ref: '{ref}'")
    if ver and ver != "latest" and not ver.isdigit():
        raise DepotError(f"bad version selector '@{ver}' in '{ref}' (want @N or @latest)")
    return name, ver, filename if sep else None


def select_version(rec: dict, selector: str) -> dict:
    """Pick a version record by selector: '' = pin, 'latest', or 'N'."""
    if not rec["versions"]:
        raise DepotError(f"artifact '{rec['name']}' has no versions")
    if selector == "latest":
        return max(rec["versions"], key=lambda v: v["version"])
    if selector == "":
        if not rec["pin"]:
            raise DepotError(f"artifact '{rec['name']}' has no pinned version")
        return find_version(rec, rec["pin"])
    return find_version(rec, int(selector))


def resolve(ref: str, root: Path | None = None) -> Path:
    """Resolve a ``depot:`` ref to a local blob path, verifying it exists."""
    root = root or depot_root()
    name, selector, filename = parse_ref(ref)
    rec = load_record(name, root)
    vrec = select_version(rec, selector)
    if filename is None:
        if len(vrec["files"]) != 1:
            raise DepotError(
                f"'{ref}' is ambiguous — version {vrec['version']} bundles "
                f"{sorted(vrec['files'])}; append /<file>"
            )
        filename = next(iter(vrec["files"]))
    entry = vrec["files"].get(filename)
    if entry is None:
        raise DepotError(
            f"no file '{filename}' in {name}@{vrec['version']} (has {sorted(vrec['files'])})"
        )
    blob = blob_path(root, entry["sha256"], filename)
    if not blob.is_file():
        raise DepotError(
            f"blob for '{ref}' is not in the local store — run: "
            f"locma depot pull {name}@{vrec['version']}"
        )
    return blob


def resolve_path(path_or_ref: str) -> str:
    """Consumer choke point: ``depot:`` refs resolve, anything else passes through."""
    if isinstance(path_or_ref, str) and path_or_ref.startswith(SCHEME):
        return str(resolve(path_or_ref))
    return path_or_ref


# ---------------------------------------------------------------------------
# maintenance: status / verify / gc
# ---------------------------------------------------------------------------


def version_status(rec: dict, vrec: dict, root: Path | None = None) -> str:
    """'local', 'partial' or 'missing' — presence of the version's blobs."""
    root = root or depot_root()
    present = [blob_path(root, e["sha256"], f).is_file() for f, e in vrec["files"].items()]
    if all(present):
        return "local"
    return "partial" if any(present) else "missing"


def verify(root: Path | None = None) -> list[str]:
    """Re-hash every locally present blob against the index; return problems."""
    root = root or depot_root()
    problems = []
    for name in artifact_names(root):
        rec = load_record(name, root)
        for vrec in rec["versions"]:
            for fname, entry in vrec["files"].items():
                blob = blob_path(root, entry["sha256"], fname)
                if not blob.is_file():
                    continue
                if sha256_file(blob) != entry["sha256"]:
                    problems.append(f"{name}@{vrec['version']}/{fname}: hash mismatch ({blob})")
    return problems


def gc(root: Path | None = None, *, dry_run: bool = True) -> tuple[list[str], int]:
    """Drop local blobs not reachable from any pinned version.

    Unpinned versions stay in the index (and on the remote if pushed) — gc only
    frees local disk. Returns (removed blob dirs, bytes freed).
    """
    root = root or depot_root()
    keep: set[str] = set()
    for name in artifact_names(root):
        rec = load_record(name, root)
        if not rec["pin"]:
            continue
        vrec = find_version(rec, rec["pin"])
        keep.update(e["sha256"] for e in vrec["files"].values())

    removed, freed = [], 0
    sha_root = _store_dir(root) / "sha256"
    if not sha_root.is_dir():
        return [], 0
    for digest_dir in sorted(sha_root.glob("*/*")):
        if digest_dir.name in keep:
            continue
        size = sum(p.stat().st_size for p in digest_dir.iterdir() if p.is_file())
        removed.append(digest_dir.name)
        freed += size
        if not dry_run:
            shutil.rmtree(digest_dir)
    return removed, freed
