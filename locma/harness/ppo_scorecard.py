from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass

from locma.harness.match import run_match
from locma.policies.registry import make_policy

HARD_BASELINES: tuple[str, ...] = ("greedy", "max-guard", "max-attack")
DEFAULT_OPPONENTS: tuple[str, ...] = ("scripted", *HARD_BASELINES, "dmcts")


@dataclass(frozen=True)
class PolicyScore:
    spec: str
    label: str
    scores: dict[str, float]
    games: dict[str, int]

    @property
    def avg_hard3(self) -> float:
        return sum(self.scores[o] for o in HARD_BASELINES) / len(HARD_BASELINES)

    @property
    def dmcts(self) -> float:
        return self.scores.get("dmcts", float("nan"))


def score_policies(
    policies: list[tuple[str, str]],
    opponents: tuple[str, ...] = DEFAULT_OPPONENTS,
    games: int = 100,
    dmcts_games: int | None = None,
    seed: int = 0,
    workers: int = 1,
) -> list[PolicyScore]:
    """Score policies against a fixed opponent panel.

    ``games`` follows ``locma play`` semantics: it means that many mirrored seed
    pairs, so every non-dmcts cell has ``2 * games`` actual games. ``dmcts_games``
    defaults to ``games`` but can be lowered for faster exploratory probes.
    """
    jobs = []
    for spec, label in policies:
        for opp in opponents:
            n_games = dmcts_games if opp == "dmcts" and dmcts_games is not None else games
            if n_games == 0:
                continue
            jobs.append((spec, label, opp, n_games, seed))

    if workers == 1:
        results = [_score_cell(*job) for job in jobs]
    else:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda job: _score_cell(*job), jobs))

    by_label: dict[str, PolicyScore] = {}
    rows: dict[str, dict[str, float]] = {label: {} for _, label in policies}
    ns: dict[str, dict[str, int]] = {label: {} for _, label in policies}
    specs: dict[str, str] = {label: spec for spec, label in policies}
    for label, opp, rate, n in results:
        rows[label][opp] = rate
        ns[label][opp] = n
    for _, label in policies:
        by_label[label] = PolicyScore(
            spec=specs[label], label=label, scores=rows[label], games=ns[label]
        )
    return [by_label[label] for _, label in policies]


def _score_cell(spec: str, label: str, opponent: str, games: int, seed: int):
    result = run_match(make_policy(spec), make_policy(opponent), games=games, seed=seed)
    return label, opponent, result.win_rate_a, result.games
