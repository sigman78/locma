"""Data fetch helpers for LOCM 1.2 (cards refresh + best-effort art).

This module provides:
- fetch_cards: download and verify cardlist, atomically replace vendored file
- fetch_art: best-effort download of card images (returns int count, never raises)
"""

from __future__ import annotations

import json
import os
import urllib.request
from importlib import resources

from locma.data.cards_db import load_cards, parse_cardlist

CARDLIST_URL = "https://raw.githubusercontent.com/ronaldosvieira/gym-locm/master/gym_locm/engine/resources/cardlist.txt"
ART_URL_TEMPLATE = ""  # empty string disables art fetch


def _download(url: str, path: str) -> bool:
    """Download a URL to a file path.

    Returns True on success, False on any exception (never raises).
    """
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


def _data_dir() -> str:
    """Return the absolute path to locma.data package directory."""
    return str(resources.files("locma.data"))


def fetch_cards(dest=None) -> str:
    """Download cardlist, verify it parses to 160 cards, atomically replace vendored file.

    Args:
        dest: optional destination path; defaults to locma/data/cardlist.txt

    Returns:
        path to the cardlist file (newly downloaded or existing)

    Guarantees:
        - on network failure or parse failure, returns existing vendored path unchanged
        - on success, atomically replaces the target path
        - never corrupts the vendored file
    """
    path = dest or os.path.join(_data_dir(), "cardlist.txt")
    tmp = path + ".tmp"

    # Download to temporary file
    if _download(CARDLIST_URL, tmp):
        try:
            # Read and parse the temp file
            with open(tmp, encoding="utf-8") as f:
                text = f.read()

            cards = parse_cardlist(text)

            # Verify we have exactly 160 cards
            if len(cards) == 160:
                # Atomic replace: on Windows replace() overwrites atomically.
                if os.path.exists(path):
                    os.replace(tmp, path)
                else:
                    os.replace(tmp, path)
            else:
                # Wrong card count, discard temp file
                if os.path.exists(tmp):
                    os.remove(tmp)
        except Exception:
            # Parse or file I/O error, discard temp file
            if os.path.exists(tmp):
                os.remove(tmp)
    else:
        # Download failed, clean up temp if it exists
        if os.path.exists(tmp):
            os.remove(tmp)

    return path


def fetch_art(dest=None) -> int:
    """Best-effort download of card images from ART_URL_TEMPLATE.

    Iterates loaded cards, skips existing files, downloads per card,
    updates manifest.json with successful downloads.

    Args:
        dest: optional destination directory; defaults to locma/data/assets

    Returns:
        count of successfully downloaded images (0 if ART_URL_TEMPLATE is empty or disabled)

    Guarantees:
        - NEVER raises an exception
        - returns an int (count of successes) always
        - if ART_URL_TEMPLATE is empty string, returns 0 immediately
    """
    # Empty template means art fetch is disabled
    if not ART_URL_TEMPLATE:
        return 0

    try:
        art_dir = dest or os.path.join(_data_dir(), "assets")
        os.makedirs(art_dir, exist_ok=True)

        manifest_path = os.path.join(art_dir, "manifest.json")
        manifest = {}

        # Load existing manifest if it exists
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                manifest = {}

        count = 0

        # Iterate over all loaded cards
        for card in load_cards():
            fname = f"{card.id}.png"
            fpath = os.path.join(art_dir, fname)

            # Skip if already downloaded
            if os.path.exists(fpath):
                continue

            # Attempt download
            url = ART_URL_TEMPLATE.format(id=card.id)
            if _download(url, fpath):
                manifest[str(card.id)] = {"file": fname, "url": url}
                count += 1

        # Update manifest (best-effort; ignore write errors)
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception:
            pass

        return count

    except Exception:
        # Any other exception, still return an int (0)
        return 0
