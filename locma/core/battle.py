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


# ---------------------------------------------------------------------------
# Task 8: legal-action generation + apply (summon / use / pass)
# ---------------------------------------------------------------------------

from locma.core.actions import Summon, Attack, Use, Pass, Action
from locma.core.cards import CardType, ABILITY_ORDER


def _find_in_hand(p, iid):
    for c in p.hand:
        if c.instance_id == iid:
            return c
    return None


def _find_on_board(p, iid):
    for c in p.board:
        if c.instance_id == iid:
            return c
    return None


def _enemy_guards(opp):
    return [c for c in opp.board if c.has("G")]


def _trigger_summon_effects(gs, player, c):
    p = gs.players[player]
    opp = gs.players[gs.opponent(player)]
    p.health += c.card.player_hp
    opp.health += c.card.enemy_hp
    p.bonus_draw += c.card.card_draw


def _merge_abilities(base, mod, add):
    out = []
    for i, ch in enumerate(ABILITY_ORDER):
        present = base[i] != "-"
        changed = mod[i] != "-"
        if add and changed:
            present = True
        if (not add) and changed:
            present = False
        out.append(ch if present else "-")
    return "".join(out)


def _apply_item(gs, item, target_id):
    p = gs.players[gs.current]
    opp = gs.players[gs.opponent(gs.current)]
    t = item.card.type
    if t == CardType.GREEN_ITEM:
        tgt = _find_on_board(p, target_id)
        if tgt:
            tgt.attack = max(0, tgt.attack + item.card.attack)
            tgt.defense += item.card.defense
            tgt.abilities = _merge_abilities(tgt.abilities, item.card.abilities, add=True)
        p.health += item.card.player_hp
        opp.health += item.card.enemy_hp
        p.bonus_draw += item.card.card_draw
    elif t == CardType.RED_ITEM:
        tgt = _find_on_board(opp, target_id)
        if tgt:
            tgt.attack = max(0, tgt.attack + item.card.attack)
            tgt.abilities = _merge_abilities(tgt.abilities, item.card.abilities, add=False)
            tgt.defense += item.card.defense
            if tgt.defense <= 0:
                opp.board.remove(tgt)
        p.health += item.card.player_hp
        opp.health += item.card.enemy_hp
    else:  # BLUE_ITEM
        if target_id == -1:
            # Blue items targeting face reuse card.defense as face damage (it is
            # always negative in the real card data).  Adding both item.card.defense
            # (via opp.health += below) and item.card.enemy_hp is correct: only
            # card 155 "Scroll of Firebolt" has both fields non-zero, and its
            # intended total effect is the union of the two (face damage + enemy
            # health modifier), not a double-count.
            opp.health += item.card.defense  # blue items carry negative defense as damage
        else:
            tgt = _find_on_board(opp, target_id)
            if tgt:
                tgt.defense += item.card.defense
                if tgt.defense <= 0:
                    opp.board.remove(tgt)
        p.health += item.card.player_hp
        opp.health += item.card.enemy_hp


def battle_legal(gs: GameState) -> list[Action]:
    p = gs.players[gs.current]
    opp = gs.players[gs.opponent(gs.current)]
    actions: list[Action] = [Pass()]
    for c in p.hand:
        if c.card.cost > p.mana:
            continue
        if c.card.type == CardType.CREATURE:
            if len(p.board) < 6:
                actions.append(Summon(c.instance_id))
        elif c.card.type == CardType.GREEN_ITEM:
            for t in p.board:
                actions.append(Use(c.instance_id, t.instance_id))
        elif c.card.type == CardType.RED_ITEM:
            for t in opp.board:
                actions.append(Use(c.instance_id, t.instance_id))
        else:  # BLUE_ITEM
            for t in opp.board:
                actions.append(Use(c.instance_id, t.instance_id))
            actions.append(Use(c.instance_id, -1))
    guards = _enemy_guards(opp)
    targets = guards if guards else opp.board
    for c in p.board:
        if c.can_attack and not c.has_attacked:
            for t in targets:
                actions.append(Attack(c.instance_id, t.instance_id))
            if not guards:
                actions.append(Attack(c.instance_id, -1))
    return actions


def apply_battle(gs: GameState, action: Action) -> None:
    p = gs.players[gs.current]
    if isinstance(action, Pass):
        end_turn(gs)
        return
    if isinstance(action, Summon):
        c = _find_in_hand(p, action.card_instance_id)
        p.hand.remove(c)
        p.mana -= c.card.cost
        p.board.append(c)
        _trigger_summon_effects(gs, gs.current, c)
    elif isinstance(action, Use):
        c = _find_in_hand(p, action.item_instance_id)
        p.hand.remove(c)
        p.mana -= c.card.cost
        _apply_item(gs, c, action.target_id)
    elif isinstance(action, Attack):
        _resolve_attack(gs, action.attacker_id, action.target_id)  # Task 9
    check_winner(gs)


# ---------------------------------------------------------------------------
# Task 9: combat resolution with B/C/D/G/L/W keywords
# ---------------------------------------------------------------------------

def _clear_ward(unit) -> None:
    i = ABILITY_ORDER.index("W")
    unit.abilities = unit.abilities[:i] + "-" + unit.abilities[i + 1:]


def _deal_to_unit(unit, amount: int, lethal: bool) -> int:
    """Returns damage actually applied to defense (0 if warded)."""
    if amount <= 0:
        return 0
    if unit.has("W"):
        _clear_ward(unit)
        return 0
    if lethal:
        unit.defense = 0
        return amount
    unit.defense -= amount
    return amount


def _resolve_attack(gs: GameState, attacker_id: int, target_id: int) -> None:
    p = gs.players[gs.current]; opp = gs.players[gs.opponent(gs.current)]
    atk = _find_on_board(p, attacker_id)
    if atk is None:
        return
    atk.has_attacked = True; atk.can_attack = False
    if target_id == -1:
        dmg = atk.attack
        opp.health -= dmg
        if atk.has("D") and dmg > 0:
            p.health += dmg
        check_winner(gs); return
    dfn = _find_on_board(opp, target_id)
    if dfn is None:
        return
    warded = dfn.has("W")
    def_before = dfn.defense
    applied = _deal_to_unit(dfn, atk.attack, atk.has("L"))  # consumes ward if present
    if atk.has("D") and applied > 0:
        p.health += applied
    if atk.has("B") and not warded:
        overflow = atk.attack - max(0, def_before)
        if overflow > 0:
            opp.health -= overflow
    _deal_to_unit(atk, dfn.attack, dfn.has("L"))
    if dfn.defense <= 0 and dfn in opp.board:
        opp.board.remove(dfn)
    if atk.defense <= 0 and atk in p.board:
        p.board.remove(atk)
    check_winner(gs)
