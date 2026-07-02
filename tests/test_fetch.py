"""Tests for data fetch helpers (cards refresh + best-effort art)."""

from __future__ import annotations

import pytest

from locma.data.fetch import fetch_art


@pytest.mark.slow  # ~30s of best-effort art downloads
def test_fetch_art_never_raises(monkeypatch):
    """fetch_art must never raise, even when _download fails."""
    import locma.data.fetch as F  # noqa: PLC0415

    # Force all downloads to fail
    monkeypatch.setattr(F, "_download", lambda url, path: False)

    # Must return an int without raising
    result = fetch_art()
    assert isinstance(result, int)
