from __future__ import annotations
from dataclasses import dataclass, field
from locma.core.engine import run_game
from locma.harness.records import write_records


@dataclass
class MatchResult:
    name_a: str
    name_b: str
    games: int
    wins_a: int
    wins_b: int
    win_rate_a: float
    records: list = field(default_factory=list)


def run_match(policy_a, policy_b, games: int, seed: int = 0, jsonl_path=None) -> MatchResult:
    wins_a = wins_b = 0
    records = []
    for k in range(games):
        s = seed + k
        # game 1: A is player0 — A wins iff winner == 0
        r1 = run_game(policy_a, policy_b, seed=s)
        a_won_1 = (r1.winner == 0)
        # game 2: B is player0 (mirror) — A is player1, wins iff winner == 1
        r2 = run_game(policy_b, policy_a, seed=s)
        a_won_2 = (r2.winner == 1)
        for won, gr, a_seat in ((a_won_1, r1, 0), (a_won_2, r2, 1)):
            if won:
                wins_a += 1
            else:
                wins_b += 1
            records.append({
                "seed": gr.seed,
                "a_seat": a_seat,
                "turns": gr.turns,
                "winner_is_a": bool(won),
            })
    total = games * 2
    res = MatchResult(
        policy_a.name, policy_b.name, total, wins_a, wins_b,
        wins_a / total if total else 0.0, records,
    )
    if jsonl_path:
        write_records(jsonl_path, records)
    return res
