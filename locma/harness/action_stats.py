from __future__ import annotations

from dataclasses import dataclass

from locma.core.actions import Attack, Pass, Summon, Use
from locma.core.battle import battle_legal
from locma.core.engine import make_battle_view, run_game
from locma.policies.registry import make_policy


@dataclass
class ActionStats:
    decisions: int = 0
    passes: int = 0
    summons: int = 0
    uses: int = 0
    attacks: int = 0
    face_attacks: int = 0
    unit_attacks: int = 0
    lethal_available: int = 0
    lethal_taken: int = 0
    guard_present: int = 0
    guard_attacks: int = 0

    def as_rates(self) -> dict[str, float]:
        n = max(self.decisions, 1)
        lethal = max(self.lethal_available, 1)
        return {
            "decisions": float(self.decisions),
            "pass": self.passes / n,
            "summon": self.summons / n,
            "use": self.uses / n,
            "attack": self.attacks / n,
            "face_attack": self.face_attacks / n,
            "unit_attack": self.unit_attacks / n,
            "lethal_available": self.lethal_available / n,
            "lethal_take": self.lethal_taken / lethal,
            "guard_present": self.guard_present / n,
            "guard_attack": self.guard_attacks / n,
        }


def _attack_power(view, attack: Attack) -> int:
    for c in view.my_board:
        if c.instance_id == attack.attacker_id:
            return c.attack
    return 0


def _has_guard(view, target_id: int) -> bool:
    for c in view.op_board:
        if c.instance_id == target_id:
            return "G" in c.abilities
    return False


def _face_damage_available(view, legal) -> int:
    total = 0
    for action in legal:
        if isinstance(action, Attack) and action.target_id == -1:
            total += _attack_power(view, action)
    for action in legal:
        if isinstance(action, Use) and action.target_id == -1:
            for c in view.my_hand:
                if c.instance_id == action.item_instance_id:
                    total += max(0, -c.defense)
    return total


def policy_action_stats(policy: str, opponent: str, games: int = 100, seed: int = 0) -> ActionStats:
    """Collect simple tactical action histograms for policy A in mirrored matches."""
    stats = ActionStats()

    for g in range(games):
        s = seed + g
        for a_seat in (0, 1):
            pa, pb = make_policy(policy), make_policy(opponent)
            p0, p1 = (pa, pb) if a_seat == 0 else (pb, pa)
            target_seat = a_seat

            def cb(seat, action, gs):
                if seat != target_seat:
                    return
                legal = battle_legal(gs)
                view = make_battle_view(gs)
                stats.decisions += 1
                stats.passes += int(isinstance(action, Pass))
                stats.summons += int(isinstance(action, Summon))
                stats.uses += int(isinstance(action, Use))
                is_attack = isinstance(action, Attack)
                stats.attacks += int(is_attack)
                stats.face_attacks += int(is_attack and action.target_id == -1)
                stats.unit_attacks += int(is_attack and action.target_id != -1)
                guard_targets = [c for c in view.op_board if "G" in c.abilities]
                stats.guard_present += int(bool(guard_targets))
                stats.guard_attacks += int(
                    is_attack and action.target_id != -1 and _has_guard(view, action.target_id)
                )
                lethal = _face_damage_available(view, legal) >= view.op_health
                chosen_face_damage = (
                    (is_attack and action.target_id == -1)
                    or (isinstance(action, Use) and action.target_id == -1)
                )
                stats.lethal_available += int(lethal)
                stats.lethal_taken += int(lethal and chosen_face_damage)

            run_game(p0, p1, seed=s, on_pre_step=cb)

    return stats
