# Artifact depot

`runs/` is scratch: disposable, gitignored, no guarantees. Anything
load-bearing — a checkpoint cited in the worklog, a dataset another experiment
builds on — gets *published* into the depot, which adds provenance,
content-addressed storage, versioned references, and cross-machine sharing.

## Layout

```
depot/
  config.json        committed   {"remote": "gh"}
  index/<name>.json  committed   versions + provenance + pin (one file per artifact)
  store/sha256/...   gitignored  content-addressed blobs
```

The split is the point: small JSON metadata travels in git (every clone knows
every artifact, its lineage and hashes, before downloading a byte); large blobs
travel through the remote and are verified against the index hashes on pull.

An **artifact** is a named, versioned bundle of files — the natural unit is the
seed-triple, e.g. `b0` = `b0_s{0,1,2}.zip`. Each version records: created
timestamp, git commit + dirty flag, the producing command line, parent refs
(lineage), free-form metadata, a note, and per-file sha256 + size. Publishing a
file that has a `<stem>.manifest.json` sidecar (the npz collector convention)
absorbs it into the version's metadata automatically.

## References

```
depot:b0/b0_s0.zip           pinned version (the default -- reproducible)
depot:b0@3/b0_s0.zip         explicit version (immutable)
depot:b0@latest/b0_s0.zip    deliberately floating
depot:b0                     whole-bundle shorthand, valid iff one file
```

Refs work anywhere a model path does: policy specs
(`vbeam:depot:b0/b0_s0.zip`, `ppo:depot:b0/b0_s0.zip`), `ceiling-eval
--candidates`, and `train_value_head` inputs. Resolution happens in
`locma.depot.resolve_path()` — raw paths pass through untouched, so nothing
existing changes behavior. A bare name follows the **pin**; `@latest` is
opt-in because silently floating refs are how eval numbers stop being
reproducible.

`publish` moves the pin to the new version by default (`--keep-pin` to hold).
`@N` refs never change meaning.

## CLI

```
locma depot publish b0 runs/b0_s0.zip runs/b0_s1.zip runs/b0_s2.zip \
    --kind model --note "recipe of record" --meta '{"avg_hard3": 0.657}' \
    --parent zoo-mix@1
locma depot list                  # pin, versions, local status, published
locma depot show b0               # full provenance record
locma depot pin b0 2
locma depot push b0               # upload pin's blobs (or b0@N)
locma depot pull b0               # fetch + hash-verify (or b0@N)
locma depot resolve depot:b0/b0_s0.zip
locma depot verify                # re-hash all local blobs
locma depot gc [--yes]            # drop local blobs not reachable from any pin
```

Second-machine bootstrap: `git pull && locma depot pull b0`.

## Remotes

The transport is dumb storage — provenance and hashes live in the index, every
pull is verified — so a backend only implements two methods
(`locma/depot/remote.py`, `Remote` protocol):

- `gh` (default): GitHub Releases via the `gh` CLI. One release per published
  version, tag `depot/<name>-vN`, files as assets, provenance JSON in the
  release notes. Zero infra; 2 GB/file limit.
- `dir:<path>`: plain directory tree `<path>/<name>/vN/<file>` — for a NAS or
  network share, and the template for an S3/rclone backend (same layout, swap
  `copy` for `put`/`get`).

Selection: `LOCMA_DEPOT_REMOTE` env var > `depot/config.json` > `gh`.
The depot root itself is `./depot`, overridable via `LOCMA_DEPOT`.
