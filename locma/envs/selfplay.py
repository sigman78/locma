"""Pure self-play recording helpers for the AlphaZero pipeline.

These three functions convert PUCT search output into training targets
and steer the game during self-play generation.  They are deliberately
ml-free (numpy only) so they can be exercised in the test suite without
any torch/sb3 dependency.

``record_selfplay`` (added in P3) drives the full self-play data-collection
loop; it imports NetOracle lazily inside the function so this module stays
importable without the ``[ml]`` extra.
"""

from __future__ import annotations

import random

import numpy as np

from locma.envs.encode import ACTION_SIZE, sem_index


def build_policy_target(view, legal: list, total) -> tuple[np.ndarray, bool]:
    """Build the AlphaZero policy target from per-edge visit counts.

    Parameters
    ----------
    view:
        A ``BattleView`` for the current position.
    legal:
        The list of legal actions in ``battle_legal(state)`` order.
    total:
        A sequence of non-negative visit counts, one per entry of ``legal``
        (same order).

    Returns
    -------
    (pi, ok)
        ``pi`` is a float32 array of shape ``(ACTION_SIZE,)``.
        ``ok`` is ``False`` when the caller should drop this row (all
        legal edges mapped to ``None``, or all mapped edges had zero
        visits); ``True`` when ``pi`` is a valid normalised distribution.
    """
    pi = np.zeros(ACTION_SIZE, dtype=np.float32)
    for i, action in enumerate(legal):
        j = sem_index(view, action)
        if j is not None and j < ACTION_SIZE:
            pi[j] += total[i]
    s = float(pi.sum())
    if s <= 0.0:
        return (pi, False)
    pi /= s
    return (pi, True)


def outcome_for(winner, seat: int) -> float:
    """Value-head training target from the moving seat's perspective.

    Returns ``+1.0`` if *winner* == *seat*, ``-1.0`` if *winner* is the
    opponent, and ``0.0`` for a draw or when *winner* is ``None``.
    """
    if winner == seat:
        return 1.0
    if winner == 1 - seat:
        return -1.0
    return 0.0


def select_move_index(total, ply: int, temp_moves: int, rng: random.Random) -> int:
    """Pick the legal index to play during self-play generation.

    Before *temp_moves* plies, samples proportionally to visit counts
    (temperature τ=1).  Afterwards returns the argmax (ties broken by
    lowest index, matching Python ``max``).

    Parameters
    ----------
    total:
        Sequence of non-negative visit counts (one per legal action).
    ply:
        Current half-move number (0-indexed).
    temp_moves:
        Number of plies over which to sample stochastically.
    rng:
        A ``random.Random`` instance — makes sampling reproducible.
    """
    if ply < temp_moves and sum(total) > 0:
        return rng.choices(range(len(total)), weights=list(total), k=1)[0]
    return int(max(range(len(total)), key=lambda i: total[i]))


# ---------------------------------------------------------------------------
# P3 — recording battle policy + record_selfplay generator
# ---------------------------------------------------------------------------


class _SelfPlayContext:
    """Shared mutable state threaded through all recording battle policies in one game."""

    __slots__ = ("buffer", "ply", "game_id", "rng")

    def __init__(self) -> None:
        self.buffer: list = []
        self.ply: int = 0
        self.game_id: int = 0
        self.rng: random.Random = random.Random()


class _RecordingBattlePolicy:
    """Net-guided determinized PUCT battle policy that records AZ training tuples.

    Each non-forced battle decision appends a row dict to ``ctx.buffer``.
    The value_target is left as 0.0 and stamped by the caller after the game
    ends (once the outcome is known).

    Import-safe: imports of ``determinize`` and ``puct_search`` stay at the
    call-site (both live in pure-Python modules with no torch/sb3 dependency).
    """

    name = "recording-netdmcts"

    def __init__(
        self,
        oracle,
        ctx: _SelfPlayContext,
        cards,
        K: int,
        I: int,  # noqa: E741
        c_puct: float,
        eps: float,
        alpha: float,
        temp_moves: int,
    ) -> None:
        self._oracle = oracle
        self._ctx = ctx
        self._cards = cards
        self.K = K
        self.I = I
        self.c_puct = c_puct
        self.eps = eps
        self.alpha = alpha
        self.temp_moves = temp_moves

    def reset(self, seed=None) -> None:  # noqa: ARG002
        """No-op: ctx seeding is managed by record_selfplay before each game."""

    def battle_action(self, view, legal, state=None):
        """Return a legal action, recording one AZ tuple when the decision is non-forced.

        Parameters
        ----------
        view:
            Public ``BattleView`` for the current player (no hidden information).
        legal:
            Legal actions at this state.
        state:
            The live ``GameState`` (required for determinization).
        """
        # Inline imports keep this module importable without [ml]; these modules
        # are pure-Python so they add no heavy dependency at import time.
        from locma.envs.encode import action_mask, encode_battle_tokens  # noqa: PLC0415
        from locma.policies.mcts import determinize  # noqa: PLC0415
        from locma.policies.puct import puct_search  # noqa: PLC0415

        ctx = self._ctx

        if len(legal) <= 1:
            return legal[0]

        seat = state.current

        # Accumulate root-edge visit counts over K determinized worlds.
        total = [0] * len(legal)
        for _ in range(self.K):
            det = determinize(state, ctx.rng, self._cards)
            counts = puct_search(
                det,
                self._oracle,
                self.I,
                self.c_puct,
                ctx.rng,
                root_noise=(self.eps, self.alpha),
            )
            assert len(counts) == len(total), (
                f"visit-count length mismatch: {len(counts)} vs {len(total)}"
            )
            for i in range(len(total)):
                total[i] += counts[i]

        pi, ok = build_policy_target(view, legal, total)
        if ok:
            ctx.buffer.append(
                {
                    **encode_battle_tokens(view),
                    "policy_target": pi,
                    "mask": action_mask(view, legal),
                    "seat": seat,
                    "game_id": ctx.game_id,
                    "value_target": 0.0,
                }
            )

        i = select_move_index(total, ctx.ply, self.temp_moves, ctx.rng)
        ctx.ply += 1
        return legal[i]


def record_selfplay(
    oracle_path: str,
    out: str = "selfplay.npz",
    self_play_games: int = 240,
    baseline_games: int = 100,
    baselines=("scripted", "max-guard", "max-attack"),
    K: int = 6,
    I: int = 40,  # noqa: E741
    c_puct: float = 1.5,
    eps: float = 0.25,
    alpha: float = 0.3,
    temp_moves: int = 10,
    seed: int = 0,
) -> dict:
    """Generate AlphaZero self-play data and write an .npz + manifest.

    Plays ``self_play_games`` fully-recorded self-play games (both seats use
    the same net-guided PUCT policy and shared context) plus ``baseline_games``
    games against scripted opponents (played in both seat orientations for
    balance; only the netdmcts seat is recorded).

    All observations use the public ``BattleView`` only (fair / no hidden
    information).  The net is loaded once and reused across all games.

    Parameters
    ----------
    oracle_path:
        Path to a saved token ``MaskablePPO`` ``.zip`` model.
    out:
        Output ``.npz`` path; a ``.manifest.json`` is written alongside it.
    self_play_games:
        Number of self-play games (both seats recorded).
    baseline_games:
        Total baseline games, distributed round-robin over ``baselines``
        and played in both seat orientations.
    baselines:
        Scripted opponent specs for the baseline phase.
    K:
        Determinizations per decision (worlds sampled per PUCT call).
    I:
        PUCT iterations per world.
    c_puct:
        PUCT exploration constant.
    eps:
        Dirichlet noise mixing coefficient (root noise for exploration).
    alpha:
        Dirichlet concentration parameter.
    temp_moves:
        Plies over which to sample stochastically (τ=1); afterwards argmax.
    seed:
        Base RNG seed — ensures reruns are byte-identical.

    Returns
    -------
    dict
        The manifest dictionary written to disk.
    """
    import json  # noqa: PLC0415 — stdlib, but keep inside function for style consistency

    from locma.core.engine import run_game  # noqa: PLC0415
    from locma.data.cards_db import load_cards  # noqa: PLC0415
    from locma.envs.encode import (  # noqa: PLC0415
        ACTION_SIZE,
        MAX_TOKENS,
        N_TACTICAL,
        TOKEN_FEATS,
    )
    from locma.envs.practicum import _engine_version, _manifest_path  # noqa: PLC0415
    from locma.policies.composer import Composer  # noqa: PLC0415
    from locma.policies.drafts import BalancedDraftPolicy  # noqa: PLC0415
    from locma.policies.net_oracle import NetOracle  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    baselines_list = list(baselines)
    oracle = NetOracle(oracle_path)
    cards = load_cards()
    ctx = _SelfPlayContext()

    failed_games = 0
    gid = 0

    def _make_rec_pol() -> Composer:
        """Return a new Composer wrapping a fresh recording battle policy + balanced draft."""
        return Composer(
            _RecordingBattlePolicy(oracle, ctx, cards, K, I, c_puct, eps, alpha, temp_moves),
            BalancedDraftPolicy(),
            name="selfplay-netdmcts",
        )

    # ------------------------------------------------------------------
    # Self-play phase: both seats share ctx + oracle
    # ------------------------------------------------------------------
    for _g in range(self_play_games):
        ctx.ply = 0
        ctx.game_id = gid
        ctx.rng = random.Random(seed + gid)

        pol0 = _make_rec_pol()
        pol1 = _make_rec_pol()

        start = len(ctx.buffer)
        try:
            result = run_game(pol0, pol1, seed + gid)
        except Exception:  # noqa: BLE001
            # Discard the partial game's rows (they carry a default z=0 label).
            del ctx.buffer[start:]
            failed_games += 1
            gid += 1
            continue

        for row in ctx.buffer[start:]:
            row["value_target"] = outcome_for(result.winner, row["seat"])

        gid += 1

    # ------------------------------------------------------------------
    # Baseline phase: only the netdmcts seat records; both orientations
    # ------------------------------------------------------------------
    for i_b in range(baseline_games):
        baseline_spec = baselines_list[i_b % len(baselines_list)]
        for netdmcts_seat in (0, 1):
            ctx.ply = 0
            ctx.game_id = gid
            ctx.rng = random.Random(seed + gid)

            net_pol = _make_rec_pol()
            opp_pol = make_policy(baseline_spec)

            p0, p1 = (net_pol, opp_pol) if netdmcts_seat == 0 else (opp_pol, net_pol)

            start = len(ctx.buffer)
            try:
                result = run_game(p0, p1, seed + gid)
            except Exception:  # noqa: BLE001
                # Discard the partial game's rows (they carry a default z=0 label).
                del ctx.buffer[start:]
                failed_games += 1
                gid += 1
                continue

            for row in ctx.buffer[start:]:
                row["value_target"] = outcome_for(result.winner, row["seat"])

            gid += 1

    # ------------------------------------------------------------------
    # Stack rows into arrays and write .npz
    # ------------------------------------------------------------------
    buf = ctx.buffer
    n = len(buf)

    np.savez(
        out,
        obs_tokens=np.asarray([d["tokens"] for d in buf], dtype=np.float32).reshape(
            n, MAX_TOKENS, TOKEN_FEATS
        ),
        obs_card_ids=np.asarray([d["card_ids"] for d in buf], dtype=np.float32).reshape(
            n, MAX_TOKENS
        ),
        obs_token_mask=np.asarray([d["token_mask"] for d in buf], dtype=np.float32).reshape(
            n, MAX_TOKENS
        ),
        obs_scalars=np.asarray([d["scalars"] for d in buf], dtype=np.float32).reshape(
            n, N_TACTICAL
        ),
        policy_target=np.asarray([d["policy_target"] for d in buf], dtype=np.float32).reshape(
            n, ACTION_SIZE
        ),
        mask=np.asarray([d["mask"] for d in buf], dtype=bool).reshape(n, ACTION_SIZE),
        value_target=np.asarray([d["value_target"] for d in buf], dtype=np.float32),
        seat=np.asarray([d["seat"] for d in buf], dtype=np.int8),
        game_id=np.asarray([d["game_id"] for d in buf], dtype=np.int32),
    )

    # ------------------------------------------------------------------
    # Write manifest
    # ------------------------------------------------------------------
    manifest: dict = {
        "obs_mode": "token",
        "max_tokens": MAX_TOKENS,
        "token_feats": TOKEN_FEATS,
        "n_tactical": N_TACTICAL,
        "action_size": ACTION_SIZE,
        "oracle_path": oracle_path,
        "K": K,
        "I": I,
        "c_puct": c_puct,
        "eps": eps,
        "alpha": alpha,
        "temp_moves": temp_moves,
        "self_play_games": self_play_games,
        "baseline_games": baseline_games,
        "baselines": baselines_list,
        "seed": seed,
        "n_examples": n,
        "failed_games": failed_games,
        "engine_version": _engine_version(),
    }

    with open(_manifest_path(out), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
