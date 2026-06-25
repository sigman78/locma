from rich.console import Console

from locma.cli.render import GameRenderer
from locma.core.engine import run_game
from locma.policies.battles import GreedyBattlePolicy
from locma.policies.composer import Composer
from locma.policies.drafts import GreedyDraftPolicy


def _greedy():
    return Composer(GreedyBattlePolicy(), GreedyDraftPolicy(), name="greedy")


def test_renderer_prints_without_error():
    console = Console(record=True, width=100)
    r = GameRenderer(console=console)
    run_game(_greedy(), _greedy(), seed=2, on_step=r.on_step)
    text = console.export_text()
    assert "turn" in text.lower() or "draft" in text.lower()
