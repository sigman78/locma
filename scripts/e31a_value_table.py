"""E31a: corrected per-card draft value table (data-only) + deck census.

The E29 pre-read data caveat (worklog 2026-07-19) established two draft-side
holes: GreedyDraftPolicy is sign-blind on items (blues score -1..-5, zero
blue items in 63k+ practicum decisions), and CardView hides the play-effect
fields (player_hp/enemy_hp/card_draw), so even the magnitude-correct
_card_value prices 4 of 8 blues at ~0 and encode_draft makes 153/154/160
exact input aliases. See docs/e31-draft-valuation-plan.md.

This script needs NO engine/policy code change: it reads the cards DB
directly (which has all three hidden fields) and emits a
DistilledDraftPolicy values JSON — the registry and the training
draft_override already load "values"-keyed JSON tables.

Value formula (pre-registered in the plan): per card,
    min(|atk|, 13) + min(|def|, 13) + player_hp + |enemy_hp|
    + 2*card_draw + kw_value
(13 = drafts._STAT_CAP, clamps removal sentinels like Decimate's -99;
player_hp signed so self-damage costs subtract; enemy damage is value;
card advantage weighted 2x a stat point; keyword value = the tuned
_KW_WEIGHT table). Context terms mirror BalancedDraftPolicy: w_need=3.0,
w_creature=2.0, default curve target — the SAME constants the E20
elicited fit calibrated to. Deliberately NO item discount: the table is a
correct-value reference / item-rich deck source, not a play-strength
recipe (E17 closed draft-side enrichment for win rate).

Census acceptance: 200 seeded self-drafts -> items/deck by color; blues
must appear at a sane nonzero rate (they are 8/160 of the pool offered at
most twice each, so a few per deck at most).

Outputs: runs/e31a_values.json (the table), runs/e31a_diet.json (the
training-diet variant: same values +4 on every item — the mirror image of
BalancedDraftPolicy's item_discount, the knob E17 calibrated; used ONLY as
a training-deck source where exposure dose, not deck realism, is the
goal), runs/e31a-values-summary.json (censuses + top-value spot check).
"""

from __future__ import annotations

import json
import random

from locma.core import draft as draftmod
from locma.core.engine import make_draft_view
from locma.core.state import GameState, Phase
from locma.data.cards_db import load_cards
from locma.policies.drafts import _KW_WEIGHT, _STAT_CAP, DistilledDraftPolicy, _kw_value

CENSUS_N = 200
TABLE_PATH = "runs/e31a_values.json"
DIET_PATH = "runs/e31a_diet.json"
DIET_ITEM_BONUS = 4.0
SUMMARY_PATH = "runs/e31a-values-summary.json"

TYPE_NAMES = {0: "creature", 1: "green", 2: "red", 3: "blue"}


def card_value(c) -> float:
    return (
        min(abs(c.attack), _STAT_CAP)
        + min(abs(c.defense), _STAT_CAP)
        + c.player_hp
        + abs(c.enemy_hp)
        + 2 * c.card_draw
        + _kw_value(c.abilities)
    )


def main() -> None:
    cards = load_cards()
    values = {c.id: round(card_value(c), 2) for c in cards}
    table = {
        "values": values,
        "w_need": 3.0,
        "w_creature": 2.0,
        "note": "E31a hand-computed true values incl. hidden play effects; "
        "formula min(|atk|,13)+min(|def|,13)+player_hp+|enemy_hp|+2*draw+kw; "
        "no item discount by design (item-rich deck source, not a strength recipe)",
    }
    with open(TABLE_PATH, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=1)
    print(f"wrote {TABLE_PATH} ({len(values)} cards, kw weights {_KW_WEIGHT})")

    by_id = {c.id: c for c in cards}
    top_items = sorted((cid for cid in values if by_id[cid].type != 0), key=lambda i: -values[i])[
        :10
    ]
    spot = [
        {
            "id": cid,
            "name": by_id[cid].name,
            "type": TYPE_NAMES[by_id[cid].type],
            "value": values[cid],
        }
        for cid in top_items
    ]
    blues = {cid: values[cid] for cid in sorted(values) if by_id[cid].type == 3}
    print("top-10 items:", json.dumps(spot, indent=1))
    print("blue values:", blues)

    diet = {
        "values": {
            cid: round(v + (DIET_ITEM_BONUS if by_id[cid].type != 0 else 0.0), 2)
            for cid, v in values.items()
        },
        "w_need": 3.0,
        "w_creature": 2.0,
        "note": f"E31a training-diet variant: e31a_values +{DIET_ITEM_BONUS:g} on every "
        "item (mirror of balanced's item_discount, E17's knob) — exposure-dose "
        "deck source for training only, NOT a reference valuation",
    }
    with open(DIET_PATH, "w", encoding="utf-8") as f:
        json.dump(diet, f, indent=1)
    print(f"wrote {DIET_PATH}")

    censuses = {}
    for tag, path in (("reference", TABLE_PATH), ("diet", DIET_PATH)):
        counts = {"creature": 0, "green": 0, "red": 0, "blue": 0}
        costs = 0.0
        blue_decks = 0
        for seed in range(CENSUS_N):
            gs = GameState.new(random.Random(seed))
            draftmod.start_draft(gs, cards)
            pols = (DistilledDraftPolicy.load(path), DistilledDraftPolicy.load(path))
            while gs.phase == Phase.DRAFT:
                pick = pols[gs.current].draft_action(make_draft_view(gs), draftmod.draft_legal(gs))
                draftmod.apply_draft_pick(gs, pick)
            deck = gs.picks[0]
            for c in deck:
                counts[TYPE_NAMES[c.type]] += 1
            costs += sum(c.cost for c in deck)
            if any(c.type == 3 for c in deck):
                blue_decks += 1
        censuses[tag] = {
            "per_deck": {k: round(v / CENSUS_N, 2) for k, v in counts.items()},
            "items_per_deck": round(
                sum(v for k, v in counts.items() if k != "creature") / CENSUS_N, 2
            ),
            "decks_with_blue": f"{blue_decks}/{CENSUS_N}",
            "mean_card_cost": round(costs / CENSUS_N / 30, 2),
        }
        print(f"census[{tag}]:", json.dumps(censuses[tag], indent=1))

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump({"top_items": spot, "blue_values": blues, "censuses": censuses}, f, indent=1)
    print(f"wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
