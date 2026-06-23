from __future__ import annotations

from rich.console import Console

from locma.core.actions import Attack, Pass, Summon, Use


def _action_summary(action) -> str:
    if isinstance(action, int):
        return f"draft pick {action}"
    if isinstance(action, Summon):
        return f"summon #{action.card_instance_id}"
    if isinstance(action, Attack):
        tgt = "face" if action.target_id == -1 else f"#{action.target_id}"
        return f"attack #{action.attacker_id} -> {tgt}"
    if isinstance(action, Use):
        tgt = "face/none" if action.target_id == -1 else f"#{action.target_id}"
        return f"use #{action.item_instance_id} -> {tgt}"
    if isinstance(action, Pass):
        return "pass"
    return repr(action)


class GameRenderer:
    """Prints a compact line per applied step, driven by run_game's on_step."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def on_step(self, seat: int, action, gs) -> None:
        if isinstance(action, int):
            self.console.print(f"[dim]draft[/] P{seat}: {_action_summary(action)}")
            return
        h0 = gs.players[0].health
        h1 = gs.players[1].health
        self.console.print(
            f"[bold]turn {gs.turn}[/] P{seat}: {_action_summary(action)}  "
            f"[green]hp {h0}[/] / [red]hp {h1}[/]"
        )
