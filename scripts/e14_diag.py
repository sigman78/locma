"""E14a reactive-failure diagnostic: WHERE does the PPO net misplay within a turn?

User hypothesis: the reactive net, invoked several times per turn, lacks
intra-turn context ("which minions already attacked, what happened so far")
and its decisions should degrade as the turn progresses. Before building a
token-v2 obs variant (Stage B), this shadow study maps where the net actually
fails, using the vbeam planner ON THE SAME NET as the oracle: any systematic
disagreement is pure search benefit — same parameters, same critic, the
planner just sees consequences before choosing.

Setup: depot:b0k s0 (reactive recipe of record) plays full games vs the ruler
opponents (HARD3) + boardkeep (the known exploiter, E10). At every b0k
decision point with >= 2 legal actions we compute the vbeam plan (width 8,
the RoR planner config) from the identical state and record:

Per-decision probes
  P1 disagreement: net action vs plan[0] (semantic-index compare), plus an
     action-type confusion matrix (pass/summon/use/attack).
  P2 first-deviation hazard: P(first disagree at index i | agreed before i)
     — the clean within-turn curve (later raw indices are off-plan-path
     conditioned on earlier deviation; the hazard is not).
  P3 context conditioning: disagreement split by #my-board minions that
     already attacked (the user's exact variable) and by mana already spent
     this turn (hidden from the net — read from the full state).
  P4 choice complexity: disagreement by legal-action count.
  P5 critic-value trajectory: value at successive own decision points; a
     drop means the net took an action its own critic scores as harmful.
  P6 calibration: critic value vs final outcome — reliability curve + ECE,
     and squared error by within-turn index (per turn-phase bucket).

Per-turn probes
  P7 premature pass / overextension: net passes while the plan continues
     (and the reverse: plan stops first).
  P8 per-turn value regret: critic value of the planner's end-of-turn state
     minus the net's actual end-of-turn state (same critic, same scale) —
     how much win-probability the reactive policy leaves on the table per
     turn, split by turn phase and by whether the first action already
     disagreed.
  P9 missed lethal: the plan wins the game outright but the net's actual
     turn does not.
  P10 harmless permutations: first actions can disagree yet the turn can
     still commute to the planner's exact end state (view equality at the
     pre-pass state) — "same_end" separates order-of-play noise from real
     deviation, so per-decision disagreement is not over-read.

Pre-registered reads (diagnostic, not promotion gates) -> lever they select:
  H1 intra-turn context (-> token-v2 obs): hazard at index >= 3 is >= 1.5x
     hazard at index 0, or disagreement at 2+ attacked minions >= 1.3x the
     0-attacked rate (both computed on index >= 1 to dodge the trivially
     fresh board at index 0).
  H2 premature pass (-> stop-value / continuation lever): net-pass-while-
     plan-continues in > 2% of turns AND those turns carry mean regret
     > +0.02.
  H3 missed lethal (-> search deficit, known planning gap): > 1% of turns
     where the plan finds a forced win.
  H4 critic mid-turn degradation (-> critic-side v2 features): (v - z)^2 at
     index >= 3 exceeds index 0 by >= 10% relative within a majority of
     turn-phase buckets.
  H5 complexity (-> capacity, not observability): disagreement in the top
     legal-count bucket >= 1.5x the bottom bucket.

Seed range: 9_000_000+ (disjoint from 1M-8M eval/confirm and 5M match seeds).
Raw per-decision/per-turn records go to runs/e14a-raw-<opp>.jsonl.gz for
offline re-slicing; aggregates + H-verdicts to runs/e14a-summary.json.
Smoke mode: E14_SMOKE=1 -> tiny grid, separate paths.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import time
import traceback

SMOKE = os.environ.get("E14_SMOKE") == "1"
WORKERS = 4 if SMOKE else 19
SEEDS_PER_OPP = 2 if SMOKE else 250  # x2 seats = games per opponent
SEED0 = 9_000_000
SUMMARY_PATH = "runs/e14a-smoke.json" if SMOKE else "runs/e14a-summary.json"
LOG_PATH = "runs/e14a-smoke.log" if SMOKE else "runs/e14a.log"
RAW_TMPL = "runs/e14a-smoke-raw-{opp}.jsonl.gz" if SMOKE else "runs/e14a-raw-{opp}.jsonl.gz"

NET_REF = "depot:b0k/b0k_s0.zip"  # reactive recipe of record, single net
BEAM_WIDTH = 8  # the RoR planner config
OPPONENTS = ("scripted", "max-guard", "max-attack", "boardkeep")

summary: dict = {}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def record(key: str, value) -> None:
    summary[key] = value
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"{key}: {json.dumps(value)}")


# ---------------------------------------------------------------------------
# Worker: one instrumented game
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _atype(action) -> str:
    from locma.core.actions import Attack, Summon, Use  # noqa: PLC0415

    if isinstance(action, Summon):
        return "S"
    if isinstance(action, Use):
        return "U"
    if isinstance(action, Attack):
        return "A"
    return "P"


def _shadow_game(opp_spec: str, seed: int, seat: int) -> tuple[list, list, dict]:
    """Play one game (net at `seat`) with per-decision shadow planning.

    Returns (decision_records, turn_records, game_meta).
    """
    from locma.core import battle as battlemod  # noqa: PLC0415
    from locma.core.actions import Pass  # noqa: PLC0415
    from locma.core.engine import make_battle_view, run_game  # noqa: PLC0415
    from locma.core.state import Phase  # noqa: PLC0415
    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.envs.encode import action_mask, sem_index  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415
    from locma.policies.vbeam import NetValueEvaluator, _clone_battle, plan_turn  # noqa: PLC0415

    if "net" not in _CACHE:
        _CACHE["net"] = make_policy(f"ppo:{NET_REF}")
        _CACHE["eval"] = NetValueEvaluator(resolve_path(NET_REF))
    if opp_spec not in _CACHE:
        _CACHE[opp_spec] = make_policy(opp_spec)
    net, ev, opp = _CACHE["net"], _CACHE["eval"], _CACHE[opp_spec]

    recs: list[dict] = []
    turns: list[dict] = []
    st = {"turn": -1, "idx": -1, "agreed": True, "prev_v": None, "pending": None, "pend_view": None}

    def close_turn(v_b0_end, how: str, end_view=None) -> None:
        p, pv = st["pending"], st["pend_view"]
        st["pending"] = st["pend_view"] = None
        if p is None:
            return
        p["v_b0_end"] = v_b0_end
        p["b0_len"] = st["idx"] if how == "pass" else st["idx"] + 1
        p["closed"] = how
        # P10: did the actual turn commute to the planner's exact end state?
        # ("end"-closed turns are outcome-compared post-game; "cap" stays None.)
        p["same_end"] = (pv is not None and end_view == pv) if how == "pass" else None
        turns.append(p)

    def plan_end_value(gs, plan) -> tuple[float, bool, object]:
        sim = _clone_battle(gs)
        for a in plan:
            if isinstance(a, Pass):
                break
            battlemod.apply_battle(sim, a)
            if sim.phase == Phase.ENDED:
                return (1.0 if sim.winner == seat else -1.0), sim.winner == seat, None
        end_view = make_battle_view(sim)
        return ev.values([end_view])[0], False, end_view

    def hook(s: int, action, gs) -> None:
        if s != seat or gs.phase != Phase.BATTLE:
            return
        if gs.turn != st["turn"]:
            close_turn(None, "cap")  # previous turn never saw its Pass (rare)
            st.update(turn=gs.turn, idx=0, agreed=True, prev_v=None)
        else:
            st["idx"] += 1
        idx = st["idx"]

        view = make_battle_view(gs)
        legal = list(battlemod.battle_legal(gs))
        nleg = len(legal)
        v = ev.evaluate([view], [action_mask(view, legal)])[0][0]

        plan = plan_turn(gs, ev, width=BEAM_WIDTH) if nleg >= 2 else [legal[0]]
        pfirst = plan[0]
        agree = sem_index(view, action) == sem_index(view, pfirst)
        hazard_ok = st["agreed"] and nleg >= 2
        if not agree:
            st["agreed"] = False

        if idx == 0:
            v_pe, pwin, st["pend_view"] = plan_end_value(gs, plan)
            st["pending"] = {
                "opp": opp_spec,
                "seed": seed,
                "seat": seat,
                "turn": gs.turn,
                "v_root": round(float(v), 4),
                "v_plan_end": round(float(v_pe), 4),
                "plan_win": pwin,
                "plan_len": sum(1 for a in plan if not isinstance(a, Pass)),
                "root_agree": agree,
            }

        me = gs.players[seat]
        recs.append(
            {
                "opp": opp_spec,
                "seed": seed,
                "seat": seat,
                "turn": gs.turn,
                "idx": idx,
                "v": round(float(v), 4),
                "dv": None if st["prev_v"] is None else round(float(v - st["prev_v"]), 4),
                "agree": agree,
                "hz": hazard_ok,
                "at": _atype(action),
                "pt": _atype(pfirst),
                "nleg": nleg,
                "natk": sum(1 for c in view.my_board if c.has_attacked),
                "nboard": len(view.my_board),
                "spent": me.max_mana + me.bonus_mana - me.mana,
            }
        )
        st["prev_v"] = float(v)
        if isinstance(action, Pass):
            close_turn(float(v), "pass", end_view=view)

    if seat == 0:
        res = run_game(net, opp, seed, on_pre_step=hook)
    else:
        res = run_game(opp, net, seed, on_pre_step=hook)

    z = 1.0 if res.winner == seat else -1.0
    close_turn(z, "end")  # a pending turn at game end = the game ended mid-turn
    for r in recs:
        r["z"] = z
    for t in turns:
        t["z"] = z
        t["b0_win_turn"] = t["closed"] == "end" and z > 0
        if t["closed"] == "end":
            # Game ended mid-turn: outcome-equivalent iff both plan and play win.
            t["same_end"] = bool(t["plan_win"]) and z > 0
    meta = {"opp": opp_spec, "seed": seed, "seat": seat, "win": z > 0, "turns": res.turns}
    return recs, turns, meta


# ---------------------------------------------------------------------------
# Stage 1: play the instrumented games (per opponent, resumable)
# ---------------------------------------------------------------------------


def run_games(ex, opp: str) -> None:
    tag = f"games_{opp}"
    raw_path = RAW_TMPL.format(opp=opp)
    if tag in summary and os.path.exists(raw_path):
        log(f"{tag}: exists, skip")
        return
    t0 = time.time()
    specs = [(opp, SEED0 + i, seat) for i in range(SEEDS_PER_OPP) for seat in (0, 1)]
    wins = n_dec = n_turn = 0
    with gzip.open(raw_path, "wt", encoding="utf-8") as f:
        results = ex.map(_shadow_game, *zip(*specs, strict=True))
        for k, (recs, turns, meta) in enumerate(results):
            for r in recs:
                f.write(json.dumps({"k": "d", **r}) + "\n")
            for t in turns:
                f.write(json.dumps({"k": "t", **t}) + "\n")
            wins += meta["win"]
            n_dec += len(recs)
            n_turn += len(turns)
            if (k + 1) % 50 == 0:
                log(f"{tag}: {k + 1}/{len(specs)} games, {(time.time() - t0) / 60:.1f} min")
    record(
        tag,
        {
            "games": len(specs),
            "wr": round(wins / len(specs), 4),
            "decisions": n_dec,
            "turns": n_turn,
            "minutes": round((time.time() - t0) / 60, 1),
        },
    )


# ---------------------------------------------------------------------------
# Stage 2: aggregate probes from the raw records
# ---------------------------------------------------------------------------


def _load_raw() -> tuple[list[dict], list[dict]]:
    decs, turns = [], []
    for opp in OPPONENTS:
        with gzip.open(RAW_TMPL.format(opp=opp), "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                (decs if r.pop("k") == "d" else turns).append(r)
    return decs, turns


def _idx_bucket(i: int) -> str:
    return str(i) if i < 5 else "5+"


def _turn_bucket(ply: int) -> str:
    return "early" if ply <= 12 else ("mid" if ply <= 28 else "late")


def _legal_bucket(n: int) -> str:
    if n <= 4:
        return "2-4"
    if n <= 8:
        return "5-8"
    if n <= 14:
        return "9-14"
    return "15+"


def _rate(rows, pred) -> tuple[float, int]:
    n = len(rows)
    return (round(sum(1 for r in rows if pred(r)) / n, 4) if n else 0.0, n)


def _group(rows, key):
    out: dict = {}
    for r in rows:
        out.setdefault(key(r), []).append(r)
    return out


def aggregate() -> None:
    if "calibration" in summary:
        log("aggregate: exists, skip")
        return
    decs, turns = _load_raw()
    contested = [d for d in decs if d["nleg"] >= 2]
    log(f"aggregate: {len(decs)} decisions ({len(contested)} contested), {len(turns)} turns")

    # P1/P2: disagreement + first-deviation hazard by within-turn index.
    by_idx = {}
    for b, rows in sorted(_group(contested, lambda d: _idx_bucket(d["idx"])).items()):
        dis, n = _rate(rows, lambda d: not d["agree"])
        hz_rows = [d for d in rows if d["hz"]]
        hz, hn = _rate(hz_rows, lambda d: not d["agree"])
        by_idx[b] = {"n": n, "disagree": dis, "hazard": hz, "hazard_n": hn}
    record("by_index", by_idx)

    # P1: action-type confusion (net rows x planner cols), contested only.
    conf: dict = {}
    for d in contested:
        conf[d["at"] + d["pt"]] = conf.get(d["at"] + d["pt"], 0) + 1
    record("confusion", dict(sorted(conf.items())))

    # P3: context conditioning (index >= 1 so a fresh board can't hide the effect).
    inner = [d for d in contested if d["idx"] >= 1]
    by_atk = {}
    for b, rows in sorted(
        _group(inner, lambda d: "2+" if d["natk"] >= 2 else str(d["natk"])).items()
    ):
        dis, n = _rate(rows, lambda d: not d["agree"])
        by_atk[b] = {"n": n, "disagree": dis}
    record("by_attacked", by_atk)
    by_spent = {}
    for b, rows in sorted(_group(inner, lambda d: min(d["spent"], 8)).items()):
        dis, n = _rate(rows, lambda d: not d["agree"])
        by_spent[str(b)] = {"n": n, "disagree": dis}
    record("by_mana_spent", by_spent)

    # P4: choice complexity.
    by_leg = {}
    for b, rows in sorted(_group(contested, lambda d: _legal_bucket(d["nleg"])).items()):
        dis, n = _rate(rows, lambda d: not d["agree"])
        by_leg[b] = {"n": n, "disagree": dis}
    record("by_legal", by_leg)

    # P5: value drops along the net's own trajectory (dv < -0.02).
    dvs = [d for d in decs if d["dv"] is not None]
    mono = {}
    for b, rows in sorted(_group(dvs, lambda d: _idx_bucket(d["idx"])).items()):
        drop, n = _rate(rows, lambda d: d["dv"] < -0.02)
        mono[b] = {"n": n, "drop_rate": drop, "mean_dv": round(sum(r["dv"] for r in rows) / n, 4)}
    record("value_drops", mono)

    # P6: calibration (reliability + ECE) and squared error by index x turn phase.
    bins: dict = {}
    for d in decs:
        b = min(int((d["v"] + 1) / 2 * 10), 9)
        bins.setdefault(b, [0, 0])
        bins[b][0] += d["z"] > 0
        bins[b][1] += 1
    reliability = {
        f"{b / 10:.1f}-{(b + 1) / 10:.1f}": {"n": n, "win_rate": round(w / n, 4)}
        for b, (w, n) in sorted(bins.items())
    }
    ece = sum(abs(w / n - (b + 0.5) / 10) * n for b, (w, n) in bins.items()) / max(1, len(decs))
    record("calibration", {"ece": round(ece, 4), "reliability": reliability})

    sq_err = {}
    for tb, rows in sorted(_group(decs, lambda d: _turn_bucket(d["turn"])).items()):
        sq_err[tb] = {
            ib: round(sum((r["v"] - r["z"]) ** 2 for r in g) / len(g), 4)
            for ib, g in sorted(_group(rows, lambda d: _idx_bucket(d["idx"])).items())
        }
    record("value_sq_err", sq_err)

    # P7: premature pass / overextension (contested decisions).
    pp, pp_n = _rate(contested, lambda d: d["at"] == "P" and d["pt"] != "P")
    oe, _ = _rate(contested, lambda d: d["at"] != "P" and d["pt"] == "P")
    n_turns = max(1, len(turns))
    pp_turns = sum(1 for d in contested if d["at"] == "P" and d["pt"] != "P")
    record(
        "pass_probes",
        {
            "premature_pass_rate_dec": pp,
            "overextend_rate_dec": oe,
            "contested_n": pp_n,
            "premature_pass_per_turn": round(pp_turns / n_turns, 4),
        },
    )

    # P8: per-turn regret (planner end value minus actual end value, same critic).
    scored = [t for t in turns if t["v_b0_end"] is not None and t["closed"] != "cap"]

    def _reg(rows):
        if not rows:
            return {"n": 0}
        vals = [t["v_plan_end"] - t["v_b0_end"] for t in rows]
        return {"n": len(rows), "mean": round(sum(vals) / len(vals), 4)}

    regret = {"overall": _reg(scored)}
    for tb, rows in sorted(_group(scored, lambda t: _turn_bucket(t["turn"])).items()):
        regret[tb] = _reg(rows)
    regret["root_agree"] = _reg([t for t in scored if t["root_agree"]])
    regret["root_disagree"] = _reg([t for t in scored if not t["root_agree"]])
    regret["disagree_same_end"] = _reg([t for t in scored if not t["root_agree"] and t["same_end"]])
    regret["disagree_diff_end"] = _reg(
        [t for t in scored if not t["root_agree"] and t["same_end"] is False]
    )
    record("regret", regret)

    # P10: how much first-action disagreement is order-of-play noise?
    judged = [t for t in turns if t["same_end"] is not None]
    dis_turns = [t for t in judged if not t["root_agree"]]
    harmless, hn = _rate(dis_turns, lambda t: t["same_end"])
    real_dev, jn = _rate(judged, lambda t: not t["root_agree"] and not t["same_end"])
    record(
        "turn_equivalence",
        {
            "judged_turns": jn,
            "root_disagree_turns": hn,
            "harmless_perm_share": harmless,
            "real_deviation_per_turn": real_dev,
        },
    )

    # P9: missed lethal.
    lethal = [t for t in turns if t["plan_win"]]
    missed, ln = _rate(lethal, lambda t: not t["b0_win_turn"])
    record("lethal", {"plan_win_turns": ln, "missed_rate": missed})


def findings() -> None:
    if "e14a_findings" in summary:
        log("e14a_findings: exists, skip")
        return
    g: dict = {}
    bi = summary["by_index"]
    hz0 = bi.get("0", {}).get("hazard", 0.0)
    hz3 = [bi[b]["hazard"] for b in ("3", "4", "5+") if b in bi and bi[b]["hazard_n"] >= 50]
    hz_ratio = (sum(hz3) / len(hz3) / hz0) if (hz3 and hz0 > 0) else None
    ba = summary["by_attacked"]
    atk_ratio = None
    if "0" in ba and "2+" in ba and ba["0"]["disagree"] > 0 and ba["2+"]["n"] >= 50:
        atk_ratio = ba["2+"]["disagree"] / ba["0"]["disagree"]
    h1 = (hz_ratio is not None and hz_ratio >= 1.5) or (atk_ratio is not None and atk_ratio >= 1.3)
    g["H1_intra_turn_context"] = {
        "supported": h1,
        "hazard_ratio_idx3_vs_0": round(hz_ratio, 3) if hz_ratio else None,
        "attacked_2plus_vs_0_ratio": round(atk_ratio, 3) if atk_ratio else None,
    }
    pp = summary["pass_probes"]["premature_pass_per_turn"]
    g["H2_premature_pass"] = {"supported": bool(pp > 0.02), "rate_per_turn": pp}
    ml = summary["lethal"]
    g["H3_missed_lethal"] = {"supported": bool(ml["missed_rate"] > 0.01), **ml}
    se = summary["value_sq_err"]
    worse = sum(
        1
        for tb in se
        if "0" in se[tb]
        and any(b in se[tb] for b in ("3", "4", "5+"))
        and max(se[tb].get(b, 0) for b in ("3", "4", "5+")) > 1.1 * se[tb]["0"]
    )
    g["H4_critic_mid_turn"] = {"supported": worse > len(se) / 2, "buckets_worse": worse}
    bl = summary["by_legal"]
    h5 = None
    if "2-4" in bl and "15+" in bl and bl["2-4"]["disagree"] > 0:
        h5 = bl["15+"]["disagree"] / bl["2-4"]["disagree"]
    g["H5_complexity"] = {
        "supported": bool(h5 and h5 >= 1.5),
        "top_vs_bottom_ratio": round(h5, 3) if h5 else None,
    }
    record("e14a_findings", g)


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary.update(json.load(f))
    log(f"=== E14a diagnostic start (net={NET_REF}, opponents={OPPONENTS}) ===")
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for opp in OPPONENTS:
            run_games(ex, opp)
    aggregate()
    findings()
    log("=== E14a diagnostic DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
