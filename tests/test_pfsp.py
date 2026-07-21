"""PFSP mixture opponent (E36) — pure-engine test, no [ml] (scripted pool only)."""

from __future__ import annotations

import json

from locma.policies.pfsp import PFSPBattleMixture
from locma.policies.registry import make_policy


def _pool(tmp_path):
    p = tmp_path / "pool.json"
    p.write_text(
        json.dumps(
            [
                {"spec": "scripted", "weight": 3.0},
                {"spec": "greedy", "weight": 1.0},
            ]
        )
    )
    return str(p)


def test_mixture_constructs_and_samples(tmp_path):
    mix = PFSPBattleMixture(_pool(tmp_path), seed=0)
    assert len(mix._battles) == 2  # noqa: SLF001
    # weighted-normalised
    assert abs(sum(mix._weights) - 1.0) < 1e-9  # noqa: SLF001
    # reset resamples the active member deterministically from the seed
    picks = set()
    for s in range(30):
        mix.reset(s)
        picks.add(id(mix._active))  # noqa: SLF001
    assert len(picks) == 2, "both pool members should be reachable across seeds"


def test_mixture_plays_a_game(tmp_path):
    from locma.core.engine import run_game  # noqa: PLC0415
    from locma.policies.composer import Composer  # noqa: PLC0415
    from locma.policies.drafts import GreedyDraftPolicy  # noqa: PLC0415

    opp = Composer(PFSPBattleMixture(_pool(tmp_path)), GreedyDraftPolicy())
    res = run_game(make_policy("scripted"), opp, seed=42)
    assert res.winner in (0, 1)


def test_registry_pfsp_spec(tmp_path):
    p = _pool(tmp_path)
    pol = make_policy(f"pfsp:{p}")
    assert hasattr(pol.battle, "battle_action")
