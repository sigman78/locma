"""`locma depot` — CLI over the artifact depot (locma/depot)."""

from __future__ import annotations

import json

import typer
from rich.console import Console

from locma import depot as dep

depot_app = typer.Typer(
    help="Versioned artifact depot: publish/pin/push/pull checkpoints and datasets "
    "with provenance. Refs: depot:<name>[@N|@latest][/<file>]"
)
console = Console()


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n}B"


def _ascii_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    """Plain space-padded columns with a dashed header rule — no box drawing."""
    widths = [len(h) for h in headers]
    for r in rows:
        widths = [max(w, len(c)) for w, c in zip(widths, r, strict=True)]
    lines = [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        "  ".join("-" * w for w in widths),
    ]
    lines += ["  ".join(r[i].ljust(widths[i]) for i in range(len(headers))) for r in rows]
    return "\n".join(line.rstrip() for line in lines)


def _split_selector(name: str) -> tuple[str, str]:
    """'b0' -> ('b0', ''); 'b0@3' -> ('b0', '3'); 'b0@latest' -> ('b0', 'latest')."""
    base, _, ver = name.partition("@")
    return base, ver


def _fail(e: Exception) -> None:
    console.print(f"[red]error:[/] {e}")
    raise typer.Exit(1)


@depot_app.command()
def publish(
    name: str,
    files: list[str],
    kind: str = typer.Option("model", help="one of: model, dataset, eval, other"),
    note: str = typer.Option("", help="what this artifact is / why it matters"),
    parent: list[str] = typer.Option(  # noqa: B008
        None, help="lineage ref(s), e.g. zoo-mix@2 (repeatable)"
    ),
    meta: str = typer.Option(None, help="extra metadata as JSON, e.g. '{\"avg_hard3\": 0.657}'"),
    keep_pin: bool = typer.Option(False, help="do not move the pin to this new version"),
):
    """Publish FILES as a new version of artifact NAME (store + index + provenance)."""
    try:
        extra = json.loads(meta) if meta else None
        vrec = dep.publish(
            name,
            files,
            kind=kind,
            note=note,
            parents=list(parent or ()),
            meta=extra,
            keep_pin=keep_pin,
        )
    except (dep.DepotError, json.JSONDecodeError) as e:
        _fail(e)
    rec = dep.load_record(name)
    console.print(
        f"published [bold]{name}[/] v{vrec['version']} ({len(vrec['files'])} files), "
        f"pin -> v{rec['pin']}"
    )


@depot_app.command("list")
def list_(kind: str = typer.Option(None, help="filter by kind")):
    """All artifacts: pin, versions, local blob status, published state."""
    rows = []
    for name in dep.artifact_names():
        rec = dep.load_record(name)
        if kind and rec["kind"] != kind:
            continue
        vrec = next((v for v in rec["versions"] if v["version"] == rec["pin"]), None)
        size = _human(sum(e["size"] for e in vrec["files"].values())) if vrec else "-"
        status = dep.version_status(rec, vrec) if vrec else "-"
        published = "yes" if vrec and vrec.get("published") else "no"
        rows.append(
            (
                name,
                rec["kind"],
                f"v{rec['pin']}" if rec["pin"] else "-",
                str(len(rec["versions"])),
                size,
                status,
                published,
            )
        )
    print(f"depot: {dep.depot_root()}")
    print(_ascii_table(("name", "kind", "pin", "versions", "size@pin", "local", "published"), rows))


@depot_app.command()
def show(name: str):
    """Full index record (all versions, provenance, hashes) as JSON."""
    try:
        rec = dep.load_record(name)
    except dep.DepotError as e:
        _fail(e)
    console.print_json(json.dumps(rec))


@depot_app.command("pin")
def pin_(name: str, version: int):
    """Move NAME's pin — what bare depot:NAME/... refs resolve to."""
    try:
        dep.pin(name, version)
    except dep.DepotError as e:
        _fail(e)
    console.print(f"pinned [bold]{name}[/] -> v{version}")


@depot_app.command()
def push(name: str):
    """Upload a version's blobs to the remote (NAME or NAME@N; default = pin)."""
    base, ver = _split_selector(name)
    try:
        locator = dep.push(base, ver)
    except dep.DepotError as e:
        _fail(e)
    console.print(f"pushed [bold]{name}[/] -> {locator}")


@depot_app.command()
def pull(name: str):
    """Fetch a version's blobs from the remote (NAME or NAME@N; default = pin)."""
    base, ver = _split_selector(name)
    try:
        fetched = dep.pull(base, ver)
    except dep.DepotError as e:
        _fail(e)
    if fetched:
        console.print(f"pulled [bold]{name}[/]: {', '.join(fetched)}")
    else:
        console.print(f"[bold]{name}[/] already complete locally")


@depot_app.command()
def resolve(ref: str):
    """Print the local path a depot: ref resolves to (shell interop)."""
    try:
        print(dep.resolve_path(ref))
    except dep.DepotError as e:
        _fail(e)


@depot_app.command()
def verify():
    """Re-hash all local blobs against the index."""
    problems = dep.verify()
    if problems:
        for p in problems:
            console.print(f"[red]BAD[/] {p}")
        raise typer.Exit(1)
    console.print("all local blobs verified OK")


@depot_app.command()
def gc(yes: bool = typer.Option(False, "--yes", help="actually delete (default: dry run)")):
    """Drop local blobs not reachable from any pin (they stay on the remote)."""
    removed, freed = dep.gc(dry_run=not yes)
    verb = "removed" if yes else "would remove"
    console.print(f"{verb} {len(removed)} blob(s), {_human(freed)}")
    if removed and not yes:
        console.print("re-run with --yes to delete")
