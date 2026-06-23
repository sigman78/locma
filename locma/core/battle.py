from __future__ import annotations
from locma.core.state import GameState, Phase

MAX_MANA = 12

def draw(gs: GameState, player: int, n: int) -> None:
    p = gs.players[player]
    for _ in range(n):
        if p.deck:
            p.hand.append(p.deck.pop(0))
            if len(p.hand) > 8:
                p.hand.pop()  # overdraw burns the card (1.2 hand cap 8)
        else:
            # deck-out rune penalty: escalating self damage
            p.health -= 1

def start_turn(gs: GameState) -> None:
    p = gs.players[gs.current]
    p.max_mana = min(MAX_MANA, p.max_mana + 1)
    p.mana = p.max_mana
    for c in p.board:
        c.can_attack = True
        c.has_attacked = False
    draw(gs, gs.current, 1 + p.bonus_draw)
    p.bonus_draw = 0

def start_battle(gs: GameState) -> None:
    gs.phase = Phase.BATTLE
    gs.turn = 1
    gs.current = 0
    draw(gs, 0, 4)
    draw(gs, 1, 5)
    p = gs.players[0]
    p.max_mana = 1
    p.mana = 1

def end_turn(gs: GameState) -> None:
    gs.current = gs.opponent(gs.current)
    gs.turn += 1
    start_turn(gs)

def check_winner(gs: GameState) -> None:
    for idx in (0, 1):
        if gs.players[idx].health <= 0:
            gs.winner = gs.opponent(idx)
            gs.phase = Phase.ENDED
            return
