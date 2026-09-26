"""E42 Gate 0: is there any exploitable information in public-history features?

Pre-registered 2026-09-25, docs/e42-public-history-plan.md ("Gate 0 — information
probe"). E6 (2026-07-03) found a recurrent-net arm -0.105 against the B0 recipe,
but the LSTM there could only ever remember its OWN past decision observations —
it never saw the opponent's turn (``BattleEnv._opp_play_until_agent`` resolves the
whole opponent half-turn inside ``step``), so a card that leaves no board trace
(an item, removal, a creature that traded) is invisible at every timestep. The
engineered public-history features (``locma.envs.encode.hist_features``, N_HIST=25
-- own remaining deck as an order-destroyed multiset, the opponent's played cards,
their turn-end mana/hand log) are a SUPERSET of anything that LSTM could ever have
accumulated. So before spending a training run (Gate 1) on a ``hist`` input
branch, this gate asks the cheap question directly: does the information in those
25 features predict anything about the HIDDEN opponent hand / incoming damage that
the reactive net's current input (v0 scalars + fx token pools) does not already
predict?

Shadow driver: the subject (gen7, ``ppo:depot:e36/e36_gen7.zip,ldraft``) plays full
games against 5 opponents (mirror / dmcts / scripted / max-guard / max-attack), in
BOTH seats. At the subject's FIRST decision of each of its own turns we record:

  X0 (73-d) = the 13 v0 tactical scalars + per-zone pooled sums of the "fx" token
              block (sum over the 8 hand slots, 6 my-board slots, 6 op-board slots
              of the 20-wide fx token row -> 3*20 = 60 values). This is what the
              current reactive net effectively has access to (fx tokens + v0
              scalars are exactly gen7's obs).
  H  (25-d) = ``hist_features(view)`` -- the new public-history vector.
  targets   = read from the TRUE hidden state (``gs.players[1-seat].hand`` and the
              game continuation) -- allowed ONLY inside this probe, never in an
              observation:
                T1 = opponent hand item count (card.type != CREATURE)
                T2 = opponent hand total mana cost
                T3 = opponent hand max creature attack (0 if none)
                T4 = opponent hand red-item count (card.type == RED_ITEM)
                T5 = face damage the subject takes before its NEXT first-decision
                     (health_now - health_next, clipped at 0; missing for a game's
                     last recorded row -- see ``_finalize_t5``)

Per opponent we also record the descriptive "sandbag rate": the fraction of the
OPPONENT's ``turn_log`` entries with ``mana_left >= 2 and hand_size > 0`` (read
from the final game state), i.e. how often that opponent archetype holds mana and
cards instead of dumping its hand -- the confound this whole gate exists to rule
in or out (E6a's prediction: greedy hard3 opponents dump every turn, so history
should tell us nothing new about THEM even if it does about mirror/dmcts).

Analysis (numpy only, no sklearn): per (opponent, target) we fit ridge regression
(closed-form, alpha=1.0 on standardized columns, intercept via centering) on X0
alone and on [X0, H], with 5-fold cross-validation GROUPED by game id (all rows of
one game land in one fold, so no within-game leakage), and report out-of-fold R2
for each plus delta-R2 = R2([X0,H]) - R2(X0). The bootstrap CI for delta-R2 does
NOT refit ridge 200 times (5 opponents x 5 targets x 2 designs x 200 resamples x
5 folds was measured to be far slower than the game generation itself for a
40-minute gate); instead, per the plan's explicit fallback, it resamples the
ALREADY-COMPUTED out-of-fold predictions at the game level (200 resamples, with
replacement over unique game ids, rows carried along with their game) and
recomputes R2 on each resample. This slightly understates within-fold refitting
variance but is the correct order of magnitude for a KILL/PASS/AMBIGUOUS read.

Usage:
    .venv/Scripts/python scripts/e42_gate0.py --smoke
    .venv/Scripts/python scripts/e42_gate0.py --games 150 --workers 12
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

LDRAFT = "depot:ldraft/ldraft_s0.zip"
SUBJECT = f"ppo:depot:e36/e36_gen7.zip,{LDRAFT}"
DMCTS = f"dmcts:15,60,0,3,{LDRAFT}"

# label -> opponent spec. Order fixes the seed-base offset below.
OPPONENTS: dict[str, str] = {
    "mirror": SUBJECT,  # built as a SECOND policy instance -- see _get_policy
    "dmcts": DMCTS,
    "scripted": "scripted",
    "max-guard": "max-guard",
    "max-attack": "max-attack",
}
HARD3 = ("scripted", "max-guard", "max-attack")

SEED0 = 82_000_000
SEED_STRIDE = 100_000  # per opponent index, clear of every base in project_*.md

N_FOLDS = 5
RIDGE_ALPHA = 1.0
N_BOOT = 200
TARGETS = ("T1", "T2", "T3", "T4", "T5")

LOG_PATH = Path("runs/e42/gate0.log")
_CACHE: dict = {}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _get_policy(cache_key: str, spec: str):
    """Per-process policy cache. ``cache_key`` (not ``spec``) identifies the slot,
    so ``mirror``'s opponent gets its OWN instance even though its spec is
    byte-identical to the subject's -- sharing one object across both seats of a
    single ``run_game`` call would alias their internal RNG state (the mirror
    game would no longer be two independent policies, just one policy replying to
    itself with one shared random stream)."""
    if cache_key not in _CACHE:
        from locma.policies.registry import make_policy  # noqa: PLC0415

        _CACHE[cache_key] = make_policy(spec)
    return _CACHE[cache_key]


def _init_worker() -> None:
    try:
        from locma.harness.parallel import init_eval_worker  # noqa: PLC0415

        init_eval_worker()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Shadow driver
# ---------------------------------------------------------------------------


def _zone_pool(tokens: np.ndarray) -> np.ndarray:
    """Per-zone pooled sums of the (20,20) fx token block -> (60,).

    Slot order (see ``encode_battle_tokens``): 0..7 my_hand, 8..13 my_board,
    14..19 op_board."""
    return np.concatenate(
        [tokens[0:8].sum(axis=0), tokens[8:14].sum(axis=0), tokens[14:20].sum(axis=0)]
    )


def _hand_targets(hand) -> tuple[int, int, int, int]:
    """T1..T4 from the opponent's TRUE hand (list of CardInstance). Hidden truth
    -- allowed only inside this probe, never fed to a policy observation."""
    t1 = sum(1 for c in hand if int(c.card.type) != 0)  # 0 = CREATURE
    t2 = sum(c.card.cost for c in hand)
    creature_atk = [c.attack for c in hand if int(c.card.type) == 0]
    t3 = max(creature_atk) if creature_atk else 0
    t4 = sum(1 for c in hand if int(c.card.type) == 2)  # 2 = RED_ITEM
    return t1, t2, t3, t4


def _sandbag_rate(turn_log) -> float | None:
    if not turn_log:
        return None
    sandbags = [1.0 if (mana >= 2 and hand > 0) else 0.0 for mana, hand in turn_log]
    return sum(sandbags) / len(sandbags)


def _shadow_game(subject, opp, seed: int, seat: int, opp_label: str, game_id: str) -> dict:
    """Play one shadow game; return {"rows": [...], "sandbag_rate": float|None}.

    ``rows`` are per-own-turn records (X0, H, T1..T4, health, missing T5). T5 is
    filled in by ``_finalize_t5`` once the whole game's rows are known (it needs
    the NEXT row's health)."""
    from locma.core.engine import make_battle_view, run_game  # noqa: PLC0415
    from locma.core.state import Phase  # noqa: PLC0415
    from locma.envs.encode import encode_battle_tokens, hist_features  # noqa: PLC0415

    st = {"turn": -1}
    rows: list[dict] = []
    last_gs_holder: list = [None]

    def hook(s: int, action, gs) -> None:  # noqa: ARG001 -- action unused, matches on_pre_step sig
        last_gs_holder[0] = gs
        if s != seat or gs.phase != Phase.BATTLE:
            return
        if gs.turn == st["turn"]:
            return  # not the first decision of this turn
        st["turn"] = gs.turn

        view = make_battle_view(gs)
        enc = encode_battle_tokens(view, "fx")
        x0 = np.concatenate([enc["scalars"], _zone_pool(enc["tokens"])]).astype(np.float32)
        h = hist_features(view)
        t1, t2, t3, t4 = _hand_targets(gs.players[1 - seat].hand)
        rows.append(
            {
                "game_id": game_id,
                "opp": opp_label,
                "turn": gs.turn,
                "x0": x0,
                "h": h,
                "t1": float(t1),
                "t2": float(t2),
                "t3": float(t3),
                "t4": float(t4),
                "health": float(view.me_health),
            }
        )

    if seat == 0:
        run_game(subject, opp, seed, on_pre_step=hook)
    else:
        run_game(opp, subject, seed, on_pre_step=hook)

    gs = last_gs_holder[0]
    sandbag = _sandbag_rate(gs.players[1 - seat].turn_log) if gs is not None else None
    return {"rows": rows, "sandbag_rate": sandbag}


def _finalize_t5(rows: list[dict]) -> None:
    """T5[i] = max(0, health[i] - health[i+1]) for consecutive rows of the SAME
    game (rows are appended in turn order within a game); the last row of each
    game has no "next" decision to diff against, so T5 stays missing (None) --
    the plan's explicit simplest choice over reading a final/continuation health."""
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        if a["game_id"] == b["game_id"]:
            a["t5"] = max(0.0, a["health"] - b["health"])
        else:
            a["t5"] = None
    if rows:
        rows[-1]["t5"] = None


def _run_block(opp_label: str, subject_spec: str, opp_spec: str, seed0: int, n_pairs: int) -> dict:
    """One block of seed-pairs for one opponent: n_pairs seeds x 2 seats = 2*n_pairs
    games. Returns rows (T5 unfinalized within this block -- finalized after all
    blocks for an opponent are merged, in game-id turn order) + sandbag rates +
    timing, for both the aggregate JSON and the per-opponent s/game budget."""
    subject = _get_policy("subject", subject_spec)
    opp = _get_policy(opp_label, opp_spec)

    t0 = time.perf_counter()
    rows: list[dict] = []
    sandbags: list[float] = []
    n_games = 0
    for i in range(n_pairs):
        seed = seed0 + i
        for seat in (0, 1):
            game_id = f"{opp_label}:{seed}:{seat}"
            res = _shadow_game(subject, opp, seed, seat, opp_label, game_id)
            rows.extend(res["rows"])
            if res["sandbag_rate"] is not None:
                sandbags.append(res["sandbag_rate"])
            n_games += 1
    return {
        "opp_label": opp_label,
        "rows": rows,
        "sandbags": sandbags,
        "seconds": time.perf_counter() - t0,
        "n_games": n_games,
    }


def run_shadow(games: int, block_pairs: int, workers: int) -> dict[str, dict]:
    """games = seed-pairs PER OPPONENT (2 games/pair, subject in both seats)."""
    units = []
    for idx, (label, spec) in enumerate(OPPONENTS.items()):
        seed0 = SEED0 + idx * SEED_STRIDE
        off = 0
        while off < games:
            n = min(block_pairs, games - off)
            units.append((label, spec, seed0 + off, n))
            off += n
    logging.info("%d blocks on %d workers (%d seed-pairs/opponent)", len(units), workers, games)

    per_opp: dict[str, dict] = {
        label: {"rows": [], "sandbags": [], "seconds": 0.0, "n_games": 0} for label in OPPONENTS
    }
    if workers > 1 and len(units) > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
            futs = {
                ex.submit(_run_block, label, SUBJECT, spec, seed0, n): (label, seed0, n)
                for label, spec, seed0, n in units
            }
            for f in as_completed(futs):
                label, seed0, n = futs[f]
                res = f.result()
                per_opp[label]["rows"].extend(res["rows"])
                per_opp[label]["sandbags"].extend(res["sandbags"])
                per_opp[label]["seconds"] += res["seconds"]
                per_opp[label]["n_games"] += res["n_games"]
                logging.info(
                    "  [%-11s] seed %d (%d pairs): %d rows, %.0fs",
                    label,
                    seed0,
                    n,
                    len(res["rows"]),
                    res["seconds"],
                )
    else:
        for label, spec, seed0, n in units:
            res = _run_block(label, SUBJECT, spec, seed0, n)
            per_opp[label]["rows"].extend(res["rows"])
            per_opp[label]["sandbags"].extend(res["sandbags"])
            per_opp[label]["seconds"] += res["seconds"]
            per_opp[label]["n_games"] += res["n_games"]
            logging.info(
                "  [%-11s] seed %d (%d pairs): %d rows, %.0fs",
                label,
                seed0,
                n,
                len(res["rows"]),
                res["seconds"],
            )

    for agg in per_opp.values():
        _finalize_t5(agg["rows"])
    return per_opp


# ---------------------------------------------------------------------------
# Ridge regression (closed-form, numpy only) + grouped CV + bootstrap
# ---------------------------------------------------------------------------


def _ridge_fit_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float = RIDGE_ALPHA
) -> np.ndarray:
    """Ridge with an unpenalized intercept: standardize X columns on the TRAIN
    fold only, center y on the train fold, solve the penalized normal equations,
    predict on x_test (standardized with the SAME train-fold mean/std)."""
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd < 1e-8] = 1.0
    xtr = (x_train - mu) / sd
    xte = (x_test - mu) / sd
    y_mean = y_train.mean()
    ytr = y_train - y_mean
    n_feat = xtr.shape[1]
    a = xtr.T @ xtr + alpha * np.eye(n_feat)
    b = xtr.T @ ytr
    beta = np.linalg.solve(a, b)
    return xte @ beta + y_mean


def _oof_predict(
    x: np.ndarray, y: np.ndarray, game_ids: list[str], n_folds: int = N_FOLDS, seed: int = 0
) -> np.ndarray:
    """Out-of-fold predictions, folds grouped by game id (every row of one game
    lands in exactly one fold -- no within-game leakage across turns)."""
    uniq = sorted(set(game_ids))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    fold_of = {uniq[order[i]]: i % n_folds for i in range(len(uniq))}
    row_fold = np.array([fold_of[g] for g in game_ids])
    oof = np.full(len(y), np.nan)
    for k in range(n_folds):
        test_mask = row_fold == k
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        oof[test_mask] = _ridge_fit_predict(x[train_mask], y[train_mask], x[test_mask])
    return oof


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _bootstrap_delta_ci(
    y: np.ndarray,
    oof_x0: np.ndarray,
    oof_x0h: np.ndarray,
    game_ids: list[str],
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap CI for delta-R2, resampling GAMES from the already-computed
    out-of-fold predictions (see module docstring for why this replaces refitting
    ridge 200x per cell). Rows are grouped by game id; a resample draws len(uniq)
    game ids WITH replacement and pools every row belonging to each draw."""
    game_ids_arr = np.asarray(game_ids)
    uniq = sorted(set(game_ids))
    rows_by_game = {g: np.where(game_ids_arr == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by_game[g] for g in draw])
        d = _r2(y[idx], oof_x0h[idx]) - _r2(y[idx], oof_x0[idx])
        deltas.append(d)
    deltas = np.sort(np.asarray(deltas))
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))
    return lo, hi


def analyze_opponent(rows: list[dict]) -> dict:
    """Ridge(X0) vs ridge([X0,H]) for every target, over one opponent's rows."""
    out: dict = {}
    for tkey, field in (("T1", "t1"), ("T2", "t2"), ("T3", "t3"), ("T4", "t4"), ("T5", "t5")):
        valid = [r for r in rows if r.get(field) is not None]
        n = len(valid)
        if n < 2 * N_FOLDS:  # too few rows/games to CV meaningfully (smoke runs)
            out[tkey] = {"n": n, "r2_x0": None, "r2_x0h": None, "delta_r2": None, "ci": None}
            continue
        x0 = np.stack([r["x0"] for r in valid])
        h = np.stack([r["h"] for r in valid])
        x0h = np.concatenate([x0, h], axis=1)
        y = np.array([r[field] for r in valid], dtype=np.float64)
        game_ids = [r["game_id"] for r in valid]

        oof_x0 = _oof_predict(x0, y, game_ids)
        oof_x0h = _oof_predict(x0h, y, game_ids)
        r2_x0 = _r2(y, oof_x0)
        r2_x0h = _r2(y, oof_x0h)
        lo, hi = _bootstrap_delta_ci(y, oof_x0, oof_x0h, game_ids)
        out[tkey] = {
            "n": n,
            "r2_x0": round(r2_x0, 4),
            "r2_x0h": round(r2_x0h, 4),
            "delta_r2": round(r2_x0h - r2_x0, 4),
            "delta_r2_ci": [round(lo, 4), round(hi, 4)],
        }
    return out


# ---------------------------------------------------------------------------
# Verdict (plan's literal rules)
# ---------------------------------------------------------------------------


def verdict(analysis: dict[str, dict]) -> dict:
    def deltas_for(labels) -> list[float]:
        return [
            analysis[lbl][t]["delta_r2"]
            for lbl in labels
            for t in TARGETS
            if analysis.get(lbl, {}).get(t, {}).get("delta_r2") is not None
        ]

    key_deltas = deltas_for(("mirror", "dmcts"))
    hand_key_deltas = [
        analysis[lbl][t]["delta_r2"]
        for lbl in ("mirror", "dmcts")
        for t in ("T1", "T2", "T3", "T4")
        if analysis.get(lbl, {}).get(t, {}).get("delta_r2") is not None
    ]
    hard3_deltas = deltas_for(HARD3)

    max_key = max(key_deltas) if key_deltas else None
    max_hand_key = max(hand_key_deltas) if hand_key_deltas else None
    max_hard3 = max(hard3_deltas) if hard3_deltas else None

    if max_key is None:
        read = "AMBIGUOUS (insufficient data -- likely a --smoke run)"
    elif max_key < 0.02:
        read = "KILL: public history carries no exploitable information vs {mirror, dmcts}"
    elif max_hand_key is not None and max_hand_key >= 0.05:
        read = "PASS: at least one hand target (T1-T4) clears 0.05 delta-R2 vs mirror/dmcts"
    else:
        read = "AMBIGUOUS (0.02-0.05): proceed to Gate 1 flagged"

    return {
        "max_delta_r2_key_all_targets": max_key,
        "max_delta_r2_key_hand_targets_T1_T4": max_hand_key,
        "max_delta_r2_hard3_sanity": max_hard3,
        "read": read,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=150, help="seed-pairs PER OPPONENT (2 games/pair)")
    ap.add_argument("--block-pairs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="3 pairs/opponent, serial")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    games = 3 if args.smoke else args.games
    workers = 1 if args.smoke else args.workers
    out = args.out or ("runs/e42/gate0_smoke.json" if args.smoke else "runs/e42/gate0.json")

    Path("runs/e42").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
    )

    logging.info(
        "E42 Gate 0 -- %s -- %d opponents x %d seed-pairs (%d games/opponent)",
        utc_now(),
        len(OPPONENTS),
        games,
        2 * games,
    )
    t0 = time.perf_counter()
    per_opp = run_shadow(games, args.block_pairs, workers)

    analysis: dict[str, dict] = {}
    descriptive: dict[str, dict] = {}
    for label, agg in per_opp.items():
        rows = agg["rows"]
        analysis[label] = analyze_opponent(rows)
        sandbags = agg["sandbags"]
        descriptive[label] = {
            "n_rows": len(rows),
            "n_games": agg["n_games"],
            "seconds": round(agg["seconds"], 1),
            "s_per_game": round(agg["seconds"] / agg["n_games"], 3) if agg["n_games"] else None,
            "mean_sandbag_rate": round(sum(sandbags) / len(sandbags), 4) if sandbags else None,
        }

    verd = verdict(analysis)

    payload = {
        "generated": utc_now(),
        "question": (
            "does public-history information predict hidden opponent-hand / "
            "incoming-damage targets beyond the current fx/v0 observation?"
        ),
        "seed_base": SEED0,
        "seed_stride": SEED_STRIDE,
        "games_per_opponent": 2 * games,
        "opponents": list(OPPONENTS.keys()),
        "descriptive": descriptive,
        "analysis": analysis,
        "verdict": verd,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    Path(out).write_text(json.dumps(payload, indent=2))

    print("\n============ E42 GATE 0 ============")
    for label in OPPONENTS:
        d = descriptive[label]
        print(
            f"{label:11s} n_rows={d['n_rows']:5d} n_games={d['n_games']:4d} "
            f"{d['s_per_game']}s/game sandbag={d['mean_sandbag_rate']}"
        )
        for t in TARGETS:
            a = analysis[label][t]
            if a["delta_r2"] is None:
                print(f"  {t}: n={a['n']} (too few rows for CV)")
            else:
                print(
                    f"  {t}: R2(X0)={a['r2_x0']:.4f} R2([X0,H])={a['r2_x0h']:.4f} "
                    f"delta={a['delta_r2']:+.4f} CI={a['delta_r2_ci']}"
                )
    print("\nVERDICT:", verd["read"])
    print(f"  max delta-R2 {{T1..T5}} x {{mirror,dmcts}} = {verd['max_delta_r2_key_all_targets']}")
    k4 = verd["max_delta_r2_key_hand_targets_T1_T4"]
    print(f"  max delta-R2 {{T1..T4}} x {{mirror,dmcts}} = {k4}")
    print(f"  hard3 sanity max delta-R2                  = {verd['max_delta_r2_hard3_sanity']}")
    print(f"\nwrote {out}  ({payload['seconds']}s)")


if __name__ == "__main__":
    main()
