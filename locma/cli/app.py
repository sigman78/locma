from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from locma.policies.random_policy import RandomPolicy
from locma.policies.scripted import ScriptedPolicy
from locma.policies.greedy import GreedyPolicy
from locma.harness.match import run_match
from locma.harness.tournament import run_tournament
from locma.stats.intervals import wilson_ci, binomial_test
from locma.stats.sprt import sprt

app = typer.Typer(help="Legends of Code & Magic 1.2 explore kit")
console = Console()


def make_policy(spec: str):
    table = {"random": RandomPolicy, "scripted": ScriptedPolicy, "greedy": GreedyPolicy}
    if spec in table:
        return table[spec](spec)
    raise typer.BadParameter(f"unknown policy '{spec}'")


@app.command()
def play(a: str, b: str, games: int = 100, seed: int = 0):
    res = run_match(make_policy(a), make_policy(b), games=games, seed=seed)
    lo, hi = wilson_ci(res.wins_a, res.games)
    p = binomial_test(res.wins_a, res.games, 0.5)
    console.print(
        f"[bold]{a}[/] vs [bold]{b}[/]  win rate A = {res.win_rate_a:.3f} "
        f"(95% CI {lo:.3f}-{hi:.3f}), p={p:.4g}, n={res.games}"
    )


@app.command()
def tournament(
    names: list[str],
    games: int = 50,
    seed: int = 0,
    reference: str = "random",
):
    pols = [make_policy(n) for n in names]
    res = run_tournament(pols, games=games, seed=seed, reference=reference)
    t = Table(title="Ratings")
    t.add_column("policy")
    t.add_column("elo", justify="right")
    t.add_column("p vs ref", justify="right")
    for n in sorted(res.ratings, key=lambda k: -res.ratings[k]):
        t.add_row(n, f"{res.ratings[n]:.0f}", f"{res.p_vs_reference.get(n, float('nan')):.4g}")
    console.print(t)


@app.command()
def eval(
    x: str,
    vs: str = "random",
    p0: float = 0.5,
    p1: float = 0.6,
    max_games: int = 1000,
    batch: int = 20,
    seed: int = 0,
):
    px, py = make_policy(x), make_policy(vs)
    wins = n = 0
    k = 0
    r = None
    while n < max_games:
        res = run_match(px, py, games=batch, seed=seed + k)
        k += batch
        wins += res.wins_a
        n += res.games
        r = sprt(wins, n, p0, p1)
        if r.decision != "continue":
            break
    lo, hi = wilson_ci(wins, n)
    console.print(
        f"verdict: [bold]{r.decision}[/]  winrate={wins/n:.3f} "
        f"(CI {lo:.3f}-{hi:.3f}), n={n}"
    )


@app.command("fetch-cards")
def fetch_cards_cmd():
    from locma.data.fetch import fetch_cards  # noqa: PLC0415 — lazy import
    path = fetch_cards()
    console.print(f"cards at {path}")


@app.command("fetch-art")
def fetch_art_cmd():
    from locma.data.fetch import fetch_art  # noqa: PLC0415 — lazy import
    n = fetch_art()
    console.print(f"fetched {n} art assets (best-effort)")
