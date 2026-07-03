"""Versioned artifact depot: provenance-tracked checkpoints/datasets, shareable
across machines. See locma/depot/core.py for the design and docs/depot.md for
the workflow."""

from locma.depot.core import (
    SCHEME,
    DepotError,
    artifact_names,
    depot_root,
    gc,
    load_record,
    parse_ref,
    pin,
    publish,
    resolve,
    resolve_path,
    verify,
    version_status,
)
from locma.depot.remote import make_remote, pull, push, remote_spec

__all__ = [
    "SCHEME",
    "DepotError",
    "artifact_names",
    "depot_root",
    "gc",
    "load_record",
    "make_remote",
    "parse_ref",
    "pin",
    "publish",
    "pull",
    "push",
    "remote_spec",
    "resolve",
    "resolve_path",
    "verify",
    "version_status",
]
