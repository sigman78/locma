"""E36 prototype: batched-opponent self-play rollout driver.

Today's stack embeds the opponent forward INSIDE each env step (SubprocVecEnv,
one-sample forwards). The profile showed those forwards are ~92% of env time, and
the batch benchmark showed one forward of size B is ~B times cheaper than B
single forwards. This driver realises that: it runs N games in ONE process and
resolves every pending opponent (and agent, and draft) decision with a single
batched forward, bucketed by pool net (PFSP samples a different net per game).

Two modes share the exact same engine stepping so we can prove equivalence:
  - "sequential": resolve each pending opponent inline, one at a time (lean argmax
    -- mirrors today's per-env inline loop, minus the SubprocVec IPC).
  - "batched": bucket all pending opponents by net, one forward per bucket.
With deterministic agent + opponents and identical per-env seed schedules, each
env's action stream must be identical across modes -- the correctness gate. Then
we time both for the throughput delta.

This measures ROLLOUT COLLECTION throughput only (no gradient update -- that cost
is identical for both architectures). The agent uses deterministic argmax here to
keep its forward cost equal across modes and isolate the opponent-batching effect;
real training samples + updates, so absolute FPS is higher than a live trainer but
the sequential-vs-batched RATIO is the number we care about.

CAVEAT (see docs/worklog 2026-07): the ~3x here is vs a single-process SEQUENTIAL
control. Integrated end-to-end (locma/envs/batched_selfplay.py) the batched VecEnv
is SLOWER than the process-parallel SubprocVecEnv baseline, because this control
does not model SubprocVec's parallel game-engine stepping. This script stands as
the batch-ceiling / decision-equivalence probe, not evidence the design wins.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from locma.core import battle as battlemod
from locma.core import draft as draftmod
from locma.core.engine import make_battle_view, make_draft_view
from locma.core.state import GameState, Phase
from locma.data.cards_db import load_cards
from locma.depot import resolve_path
from locma.envs.encode import (
    action_mask,
    draft_action_mask,
    encode_battle_tokens,
    encode_draft,
    index_to_action,
    token_variant_for_space,
)
from locma.policies.ppo import _lean_masked_argmax
from locma.policies.registry import make_policy

AGENT = "runs/e36_gen4.zip"
LDRAFT = "depot:ldraft/ldraft_s0.zip"
POOL = "runs/e36/pool.json"


def _batched_argmax(policy, obs_list, mask_arr):
    """One masked-argmax forward over a batch of dict obs (byte-identical to a
    per-row lean argmax; softmax is monotone so argmax(masked logits) is stable)."""
    if isinstance(obs_list[0], dict):
        batch = {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
    else:
        batch = np.stack(obs_list)  # flat-array obs (draft)
    obs_t, _ = policy.obs_to_tensor(batch)
    with torch.no_grad():
        feats = policy.extract_features(obs_t, policy.pi_features_extractor)
        latent = policy.mlp_extractor.forward_actor(feats)
        logits = policy.action_net(latent)
        mask_t = torch.as_tensor(mask_arr, dtype=torch.bool, device=logits.device)
        logits = torch.where(mask_t, logits, torch.full_like(logits, float("-inf")))
        return logits.argmax(dim=1).cpu().numpy()


class Driver:
    def __init__(self, n_envs: int, seed: int = 0):
        from sb3_contrib import MaskablePPO  # noqa: PLC0415 — lazy [ml] dep

        self.n = n_envs
        self.cards = load_cards()

        # shared nets, loaded ONCE (vs one copy per SubprocVec worker today)
        self.agent = MaskablePPO.load(resolve_path(AGENT))
        self.agent.policy.set_training_mode(False)
        self.variant = token_variant_for_space(self.agent.observation_space)
        self.ldraft = MaskablePPO.load(resolve_path(LDRAFT))
        self.ldraft.policy.set_training_mode(False)

        # pool: each net loaded once; scripted members kept as heuristics
        entries = json.loads(Path(POOL).read_text())
        self.weights = [float(e.get("weight", 1.0)) for e in entries]
        self.members = []  # ("net", MaskablePPO) | ("script", policy)
        for e in entries:
            spec = e["spec"]
            if spec.startswith("ppo:"):
                path = spec[4:].split(",")[0].split("|")[0]
                m = MaskablePPO.load(resolve_path(path))
                m.policy.set_training_mode(False)
                self.members.append(("net", m))
            else:
                self.members.append(("script", make_policy(spec).battle))

        # per-env state
        self.gs: list = [None] * n_envs
        self.seat = [0] * n_envs
        self.member = [0] * n_envs  # active pool-member index per game
        self.picks: list = [None] * n_envs
        self._ep = [0] * n_envs
        self._base = [seed + i * 100_000 for i in range(n_envs)]
        self.winners: list = []

    # -- episode seed schedule (deterministic, identical across modes) --
    def _seed(self, i: int) -> int:
        return self._base[i] + self._ep[i]

    def _sample_member(self, s: int) -> int:
        return random.Random(s).choices(range(len(self.members)), weights=self.weights)[0]

    # -- reset a wave of envs together; draft is driven in lockstep so the
    #    ldraft picks batch across the whole wave --
    def _reset_wave(self, idxs: list[int], mode: str) -> None:
        if not idxs:
            return
        for i in idxs:
            s = self._seed(i)
            self.seat[i] = random.Random(s ^ 0x5EA7).randint(0, 1)
            self.member[i] = self._sample_member(s)
            gs = GameState.new(random.Random(s))
            draftmod.start_draft(gs, self.cards, shared=False)
            self.gs[i] = gs
            self.picks[i] = []
        # lockstep draft: every env in the wave is at DRAFT; step pick-by-pick
        while any(self.gs[i].phase == Phase.DRAFT for i in idxs):
            pend = [i for i in idxs if self.gs[i].phase == Phase.DRAFT]
            obs_l, mask_l = [], []
            for i in pend:
                dv = make_draft_view(self.gs[i])
                obs_l.append(encode_draft(dv, self.picks[i]))
                mask_l.append(draft_action_mask(draftmod.draft_legal(self.gs[i])))
            if mode == "batched":
                picks = _batched_argmax(self.ldraft.policy, obs_l, np.stack(mask_l))
            else:
                picks = [
                    _lean_masked_argmax(self.ldraft, o, m)
                    for o, m in zip(obs_l, mask_l, strict=False)
                ]
            for i, p in zip(pend, picks, strict=False):
                dv = make_draft_view(self.gs[i])
                self.picks[i].append(dv.offered[int(p)])
                draftmod.apply_draft_pick(self.gs[i], int(p))
        for i in idxs:
            battlemod.start_battle(self.gs[i])

    # -- resolve all pending opponent decisions until every env is at the agent's
    #    turn or ended (ragged: variable-length opponent turns) --
    def _resolve_opponents(self, mode: str, trace) -> None:
        while True:
            pend = [
                i
                for i in range(self.n)
                if self.gs[i].phase == Phase.BATTLE and self.gs[i].current != self.seat[i]
            ]
            if not pend:
                return
            if mode == "batched":
                # bucket by net member; scripted resolved inline
                buckets: dict[int, list[int]] = {}
                for i in pend:
                    kind, _ = self.members[self.member[i]]
                    if kind == "net":
                        buckets.setdefault(self.member[i], []).append(i)
                    else:
                        self._apply_scripted(i, trace)
                for mem_idx, group in buckets.items():
                    _, net = self.members[mem_idx]
                    obs_l, mask_l, actions = [], [], []
                    for i in group:
                        view = make_battle_view(self.gs[i])
                        legal = battlemod.battle_legal(self.gs[i])
                        obs_l.append(encode_battle_tokens(view, self.variant))
                        mask_l.append(action_mask(view, legal))
                        actions.append((view, legal))
                    idxs = _batched_argmax(net.policy, obs_l, np.stack(mask_l))
                    for i, (view, legal), a in zip(group, actions, idxs, strict=False):
                        act = index_to_action(view, legal, int(a))
                        battlemod.apply_battle(self.gs[i], act)
                        if trace is not None:
                            trace[i].append(int(a))
            else:
                for i in pend:
                    kind, obj = self.members[self.member[i]]
                    if kind == "net":
                        view = make_battle_view(self.gs[i])
                        legal = battlemod.battle_legal(self.gs[i])
                        obs = encode_battle_tokens(view, self.variant)
                        a = _lean_masked_argmax(obj, obs, action_mask(view, legal))
                        battlemod.apply_battle(self.gs[i], index_to_action(view, legal, int(a)))
                        if trace is not None:
                            trace[i].append(int(a))
                    else:
                        self._apply_scripted(i, trace)

    def _apply_scripted(self, i: int, trace) -> None:
        _, heur = self.members[self.member[i]]
        view = make_battle_view(self.gs[i])
        legal = battlemod.battle_legal(self.gs[i])
        act = heur.battle_action(view, legal, self.gs[i])
        battlemod.apply_battle(self.gs[i], act)
        if trace is not None:
            trace[i].append(-1)  # scripted marker (action objects aren't ints)

    def _agent_step(self, mode: str, trace) -> None:
        # every env is at its agent decision; batch the agent forward across all N
        obs_l, mask_l, meta = [], [], []
        for i in range(self.n):
            view = make_battle_view(self.gs[i])
            legal = battlemod.battle_legal(self.gs[i])
            obs_l.append(encode_battle_tokens(view, self.variant))
            mask_l.append(action_mask(view, legal))
            meta.append((view, legal))
        if mode == "batched":
            idxs = _batched_argmax(self.agent.policy, obs_l, np.stack(mask_l))
        else:
            idxs = [
                _lean_masked_argmax(self.agent, o, m) for o, m in zip(obs_l, mask_l, strict=False)
            ]
        for i, (view, legal), a in zip(range(self.n), meta, idxs, strict=False):
            battlemod.apply_battle(self.gs[i], index_to_action(view, legal, int(a)))
            if trace is not None:
                trace[i].append(int(a))

    def run(self, total_steps: int, mode: str, record: bool = False):
        trace = [[] for _ in range(self.n)] if record else None
        self._reset_wave(list(range(self.n)), mode)
        self._resolve_opponents(mode, trace)
        steps = 0
        while steps < total_steps:
            self._agent_step(mode, trace)
            steps += self.n
            # terminal handling + reset wave
            ended = [i for i in range(self.n) if self.gs[i].phase == Phase.ENDED]
            for i in ended:
                self.winners.append(1 if self.gs[i].winner == self.seat[i] else 0)
                self._ep[i] += 1
            self._reset_wave(ended, mode)
            self._resolve_opponents(mode, trace)
        return steps, trace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-envs", type=int, default=12)
    ap.add_argument("--check-steps", type=int, default=3000)
    ap.add_argument("--time-steps", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # ---- correctness gate: identical per-env action streams across modes ----
    d1 = Driver(args.n_envs, seed=args.seed)
    _, t_seq = d1.run(args.check_steps, "sequential", record=True)
    d2 = Driver(args.n_envs, seed=args.seed)
    _, t_bat = d2.run(args.check_steps, "batched", record=True)
    mism = sum(1 for a, b in zip(t_seq, t_bat, strict=False) if a != b)
    lens = [len(x) for x in t_seq]
    print(
        f"[check] {args.n_envs} envs, {sum(lens)} actions traced; "
        f"per-env stream mismatches: {mism}/{args.n_envs}"
    )
    if mism:
        for i, (a, b) in enumerate(zip(t_seq, t_bat, strict=False)):
            if a != b:
                # first divergence
                j = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), None)
                print(f"  env {i}: len {len(a)} vs {len(b)}, first diff at {j}")
        print("  ABORT: batched decisions diverge from sequential")
        return
    print("  OK: batched == sequential (decision-preserving)\n")

    # ---- throughput ----
    for mode in ("sequential", "batched"):
        d = Driver(args.n_envs, seed=args.seed + 1)
        d.run(2000, mode)  # warmup
        t0 = time.perf_counter()
        steps, _ = d.run(args.time_steps, mode)
        dt = time.perf_counter() - t0
        print(f"[{mode:10s}] {steps} agent-steps in {dt:6.2f}s = {steps / dt:7.1f} steps/s")


if __name__ == "__main__":
    main()
