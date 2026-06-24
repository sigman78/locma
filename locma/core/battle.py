from __future__ import annotations

from locma.core.actions import Action, Attack, Pass, Summon, Use
from locma.core.cards import ABILITY_ORDER, CardType
from locma.core.state import GameState, Phase

MAX_MANA = 12


def _change_health(p, damage: int, *, from_opponent: bool = False) -> None:
    """Apply ``damage`` to a player (negative = healing).

    LOCM 1.5 draw rule (no runes): for every 5 health a player loses to
    *opponent* damage during a round, they draw one extra card at the start of
    their next turn.  ``damage_counter`` accumulates the running remainder and
    is reset each turn in :func:`start_turn`.  Self-inflicted damage (a card's
    own ``player_hp``) and game damage (deck-out, the 50-turn penalty) do not
    count toward the bonus, so only opponent-sourced damage passes
    ``from_opponent=True``.
    """
    p.health -= damage
    if from_opponent and damage > 0:
        p.damage_counter += damage
        while p.damage_counter >= 5:
            p.damage_counter -= 5
            p.bonus_draw += 1


def draw(gs: GameState, player: int, n: int) -> None:
    p = gs.players[player]
    for _ in range(n):
        if len(p.hand) >= 8:
            break  # hand full (cap 8): the card stays in the deck, not drawn/burned
        if p.deck:
            p.hand.append(p.deck.pop(0))
        else:
            # deck-out: 10 self-damage per missed draw (LOCM 1.5, no runes)
            _change_health(p, 10)


def start_turn(gs: GameState) -> None:
    p = gs.players[gs.current]
    # Second-player bonus mana is lost the turn after it is fully spent: if the
    # player ended their previous turn with 0 mana (and had ramped at all),
    # drop the bonus before refilling.
    if p.max_mana > 0 and p.mana == 0:
        p.bonus_mana = 0
    if p.max_mana < MAX_MANA:
        p.max_mana += 1
    p.mana = p.max_mana + p.bonus_mana
    p.damage_counter = 0  # fresh accumulator for opponent damage taken this round
    for c in p.board:
        c.can_attack = True
        c.has_attacked = False
    # 50-turn rule (LOCM 1.5): once a player has played over 50 turns they take
    # 10 damage at the start of every turn (gs.turn counts plies, so >100).
    if gs.turn > 100:
        _change_health(p, 10)
    draw(gs, gs.current, 1 + p.bonus_draw)
    p.bonus_draw = 0
    # Deck-out / 50-turn damage at turn start can drop HP to 0; the Pass path
    # returns before apply_battle's check_winner, so settle the result here.
    check_winner(gs)


def start_battle(gs: GameState) -> None:
    gs.phase = Phase.BATTLE
    gs.turn = 1
    gs.current = 0
    draw(gs, 0, 4)
    draw(gs, 1, 5)
    gs.players[1].bonus_mana = 1  # second-player compensation ("the coin")
    # Player 0's first turn: ramp to 1 mana and draw a card (so both players
    # reach 5 cards at the start of their first turn).
    start_turn(gs)


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
    _change_health(p, -c.card.player_hp)
    _change_health(opp, -c.card.enemy_hp, from_opponent=True)
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
        _change_health(p, -item.card.player_hp)
        _change_health(opp, -item.card.enemy_hp, from_opponent=True)
        p.bonus_draw += item.card.card_draw
    elif t == CardType.RED_ITEM:
        tgt = _find_on_board(opp, target_id)
        if tgt:
            tgt.attack = max(0, tgt.attack + item.card.attack)
            tgt.abilities = _merge_abilities(tgt.abilities, item.card.abilities, add=False)
            tgt.defense += item.card.defense
            if tgt.defense <= 0:
                opp.board.remove(tgt)
        _change_health(p, -item.card.player_hp)
        _change_health(opp, -item.card.enemy_hp, from_opponent=True)
    else:  # BLUE_ITEM
        if target_id == -1:
            # Blue items targeting face reuse card.defense as face damage (it is
            # always negative in the real card data).  Applying both card.defense
            # (the _change_health below) and card.enemy_hp (the trailing block) is
            # correct: only card 155 "Scroll of Firebolt" has both fields non-zero,
            # and its intended total effect is the union of the two (face damage +
            # enemy health modifier), not a double-count.
            # blue carries negative defense as face damage (opponent-sourced)
            _change_health(opp, -item.card.defense, from_opponent=True)
        else:
            tgt = _find_on_board(opp, target_id)
            if tgt:
                tgt.defense += item.card.defense
                if tgt.defense <= 0:
                    opp.board.remove(tgt)
        _change_health(p, -item.card.player_hp)
        _change_health(opp, -item.card.enemy_hp, from_opponent=True)


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
    unit.abilities = unit.abilities[:i] + "-" + unit.abilities[i + 1 :]


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
    p = gs.players[gs.current]
    opp = gs.players[gs.opponent(gs.current)]
    atk = _find_on_board(p, attacker_id)
    if atk is None:
        return
    atk.has_attacked = True
    atk.can_attack = False
    if target_id == -1:
        dmg = atk.attack
        _change_health(opp, dmg, from_opponent=True)
        if atk.has("D") and dmg > 0:
            _change_health(p, -dmg)
        check_winner(gs)
        return
    dfn = _find_on_board(opp, target_id)
    if dfn is None:
        return
    warded = dfn.has("W")
    def_before = dfn.defense
    applied = _deal_to_unit(dfn, atk.attack, atk.has("L"))  # consumes ward if present
    if atk.has("D") and applied > 0:
        _change_health(p, -applied)
    if atk.has("B") and not warded:
        overflow = atk.attack - max(0, def_before)
        if overflow > 0:
            _change_health(opp, overflow, from_opponent=True)
    _deal_to_unit(atk, dfn.attack, dfn.has("L"))
    if dfn.defense <= 0 and dfn in opp.board:
        opp.board.remove(dfn)
    if atk.defense <= 0 and atk in p.board:
        p.board.remove(atk)
    check_winner(gs)
