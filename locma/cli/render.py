from __future__ import annotations

from rich.console import Console

from locma.core.actions import Attack, Pass, Summon, Use


def _action_summary(action) -> str:
    match action:
        case int():
            return f"draft pick {action}"
        case Summon(card_instance_id=iid):
            return f"summon #{iid}"
        case Attack(attacker_id=aid, target_id=tid):
            tgt = "face" if tid == -1 else f"#{tid}"
            return f"attack #{aid} -> {tgt}"
        case Use(item_instance_id=iid, target_id=tid):
            tgt = "face/none" if tid == -1 else f"#{tid}"
            return f"use #{iid} -> {tgt}"
        case Pass():
            return "pass"
        case _:
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
