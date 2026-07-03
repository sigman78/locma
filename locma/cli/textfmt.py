"""Plain-text CLI output helpers — space-padded ASCII tables, no rich markup.

CLI reporting deliberately avoids rich rendering (colors, box drawing, width-
dependent wrapping): output stays grep-able, diff-able, and identical when
piped to a file. Only the interactive game renderer (locma/cli/render.py)
keeps rich.
"""

from __future__ import annotations


def ascii_table(
    headers: tuple[str, ...] | list[str],
    rows: list[tuple[str, ...]] | list[list[str]],
    align: str = "",
    title: str | None = None,
) -> str:
    """Space-padded columns with a dashed header rule.

    ``align`` is one char per column, 'l' or 'r' (default all-left); shorter
    strings leave the remaining columns left-aligned.
    """
    widths = [len(h) for h in headers]
    for r in rows:
        widths = [max(w, len(c)) for w, c in zip(widths, r, strict=True)]

    def fmt(cells):
        out = []
        for i, c in enumerate(cells):
            just = align[i] if i < len(align) else "l"
            out.append(c.rjust(widths[i]) if just == "r" else c.ljust(widths[i]))
        return "  ".join(out).rstrip()

    lines = [] if title is None else [title]
    lines.append(fmt(headers))
    lines.append("  ".join("-" * w for w in widths))
    lines.extend(fmt(r) for r in rows)
    return "\n".join(lines)


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n}B"
