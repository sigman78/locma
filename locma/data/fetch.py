"""Data fetch helpers for LOCM 1.2 (cards refresh + opt-in portrait art).

This module provides:
- fetch_cards: download and verify cardlist, atomically replace vendored file
- fetch_art: opt-in download of card portrait images (returns int count, never raises)
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from importlib import resources

from locma.data.cards_db import load_cards, parse_cardlist

CARDLIST_URL = "https://raw.githubusercontent.com/ronaldosvieira/gym-locm/master/gym_locm/engine/resources/cardlist.txt"
ART_URL_TEMPLATE = "https://legendsofcodeandmagic.com/portraits/{id:03d}.png"
USER_AGENT = "locma-fetch-art/1.0 (local research tool)"
_REQUEST_DELAY = 0.2  # seconds between requests; be polite to the host


def _download(url: str, path: str) -> bool:
    """Download a URL to a file path.

    Returns True on success, False on any exception (never raises).
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as r:
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

    Guarantees:
        - on network failure or parse failure, returns existing vendored path unchanged
        - on success, atomically replaces the target path
        - never corrupts the vendored file
    """
    path = dest or os.path.join(_data_dir(), "cardlist.txt")
    tmp = path + ".tmp"

    if _download(CARDLIST_URL, tmp):
        try:
            with open(tmp, encoding="utf-8") as f:
                text = f.read()
            cards = parse_cardlist(text)
            if len(cards) == 160:
                os.replace(tmp, path)
            else:
                if os.path.exists(tmp):
                    os.remove(tmp)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
    else:
        if os.path.exists(tmp):
            os.remove(tmp)

    return path


def _load_manifest(manifest_path: str) -> dict:
    """Load the manifest, tolerating a missing or corrupt file (returns {})."""
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def fetch_art(dest=None, force: bool = False) -> int:
    """Opt-in download of card portrait art from ART_URL_TEMPLATE.

    Iterates loaded cards, skips existing files (unless force), downloads each
    portrait as a zero-padded {id:03d}.png, updates manifest.json.

    Guarantees:
        - NEVER raises an exception
        - returns an int (count of successful downloads) always
    """
    try:
        art_dir = dest or os.path.join(_data_dir(), "assets")
        os.makedirs(art_dir, exist_ok=True)

        manifest_path = os.path.join(art_dir, "manifest.json")
        manifest = _load_manifest(manifest_path)

        count = 0
        for card in load_cards():
            fname = f"{card.id:03d}.png"
            fpath = os.path.join(art_dir, fname)

            if os.path.exists(fpath) and not force:
                continue

            url = ART_URL_TEMPLATE.format(id=card.id)
            ok = _download(url, fpath)
            if _REQUEST_DELAY:
                time.sleep(_REQUEST_DELAY)
            if ok:
                manifest[str(card.id)] = {"file": fname, "url": url}
                count += 1

        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception:
            pass

        return count
    except Exception:
        return 0
