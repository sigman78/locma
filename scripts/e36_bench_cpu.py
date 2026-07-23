"""E36 bench addendum: where does the CPU actually go, and what parallelizes?

Decomposes one PPO iteration (rollout collection vs gradient update) on the
pfsp pool env and sweeps the three levers that matter on a small net:
  - main-process torch threads (update GEMMs are tiny at batch 64)
  - worker OMP_NUM_THREADS (each SubprocVecEnv worker loads torch for the
    ppo pool opponent; 12 workers x 8 default OpenMP threads oversubscribes)
  - PPO minibatch size (throughput only — changing it for real training is a
    separate optimization-dynamics question)

target_kl is disabled so every update runs the full n_epochs — deterministic
cost, comparable across configs (production runs early-stop at 3-6 epochs, so
absolute update times here are inflated; ratios are what matter).

Run: .venv/bin/python scripts/e36_bench_cpu.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

E29 = "depot:e29slim/e29slim_s0.zip"
LDRAFT = "depot:ldraft/ldraft_s0.zip"
BENCH_POOL = "runs/e36/bench_pool.json"


class CpuSampler(threading.Thread):
    """Samples %cpu of the main process and its children via ps."""

    def __init__(self, pid: int, interval: float = 0.5):
        super().__init__(daemon=True)
        self.pid, self.interval = pid, interval
        self.samples: list[tuple[float, float, float]] = []  # (t, main%, workers%)
        self.stop_flag = False

    def run(self) -> None:
        while not self.stop_flag:
            try:
                main = float(
                    subprocess.check_output(["ps", "-o", "%cpu=", "-p", str(self.pid)], text=True)
                )
                # descendants, transitively: SubprocVecEnv may use forkserver,
                # which makes the workers grandchildren of the main process
                kids: list[str] = []
                frontier = [str(self.pid)]
                while frontier:
                    out = subprocess.run(
                        ["pgrep", "-P", ",".join(frontier)], capture_output=True, text=True
                    ).stdout.split()
                    kids.extend(out)
                    frontier = out
                workers = 0.0
                if kids:
                    out = subprocess.check_output(
                        ["ps", "-o", "%cpu=", "-p", ",".join(kids)], text=True
                    )
                    workers = sum(float(x) for x in out.split())
                self.samples.append((time.time(), main, workers))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self.interval)

    def mean_in(self, t0: float, t1: float) -> tuple[float, float]:
        rows = [(m, w) for t, m, w in self.samples if t0 <= t <= t1]
        if not rows:
            return (0.0, 0.0)
        return (
            sum(m for m, _ in rows) / len(rows),
            sum(w for _, w in rows) / len(rows),
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-envs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=14_910_000)
    args = ap.parse_args()

    import torch  # noqa: PLC0415
    from sb3_contrib import MaskablePPO  # noqa: PLC0415
    from stable_baselines3.common.callbacks import BaseCallback  # noqa: PLC0415

    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.envs.training import _build_env  # noqa: PLC0415

    class PhaseTimer(BaseCallback):
        def __init__(self):
            super().__init__()
            self.marks: list[tuple[str, float, float]] = []  # (tag, wall, main proc cpu)

        def _on_step(self) -> bool:
            return True

        def _on_rollout_start(self) -> None:
            self.marks.append(("rs", time.time(), time.process_time()))

        def _on_rollout_end(self) -> None:
            self.marks.append(("re", time.time(), time.process_time()))

    sampler = CpuSampler(os.getpid())
    sampler.start()
    results: list[dict] = []

    def build_env():
        return _build_env(
            f"pfsp:{BENCH_POOL}",
            args.seed,
            args.n_envs,
            both_seat=True,
            obs_mode="token-fx",
            draft_override=LDRAFT,
        )

    def load(env):
        model = MaskablePPO.load(resolve_path(E29), env=env, device="cpu")
        model.target_kl = None  # full n_epochs: deterministic update cost
        return model

    def cycle(model, env, tag: str, discard: bool = False) -> dict:
        cb = PhaseTimer()
        t0 = time.time()
        model.learn(total_timesteps=1, reset_num_timesteps=True, callback=cb)
        t1, p1 = time.time(), time.process_time()
        (_, rs_w, _), (_, re_w, re_p) = cb.marks[0], cb.marks[1]
        roll_w, upd_w = re_w - rs_w, t1 - re_w
        upd_cores = (p1 - re_p) / upd_w if upd_w > 0 else 0.0
        steps = model.n_steps * env.num_envs
        roll_main, roll_workers = sampler.mean_in(rs_w, re_w)
        upd_main, _ = sampler.mean_in(re_w, t1)
        row = {
            "tag": tag,
            "rollout_s": round(roll_w, 1),
            "update_s": round(upd_w, 1),
            "steps_per_sec": round(steps / (t1 - t0), 1),
            "rollout_cpu_main_pct": round(roll_main, 0),
            "rollout_cpu_workers_pct": round(roll_workers, 0),
            "update_cores_main": round(upd_cores, 2),
            "update_cpu_main_pct": round(upd_main, 0),
        }
        if not discard:
            results.append(row)
        print(
            f"{tag:<38} rollout {roll_w:>5.1f}s "
            f"(main {roll_main:>4.0f}% + workers {roll_workers:>4.0f}%)  "
            f"update {upd_w:>5.1f}s ({upd_cores:.2f} cores)  "
            f"-> {steps / (t1 - t0):>5.0f} steps/s{'  [warm-up, discarded]' if discard else ''}",
            flush=True,
        )
        return row

    # --- phase 1: default worker threads, sweep main-process torch threads ---
    print(f"== workers at default OMP threads, n_envs={args.n_envs}, full {10} epochs ==")
    env = build_env()
    model = load(env)
    cycle(model, env, "warm-up", discard=True)
    by_threads = {}
    for nt in (8, 4, 1):
        torch.set_num_threads(nt)
        by_threads[nt] = cycle(model, env, f"main-threads={nt} batch=64")

    # --- phase 2: sweep batch size at the best main-thread count so far ---
    best_nt = min(by_threads, key=lambda k: by_threads[k]["update_s"])
    torch.set_num_threads(best_nt)
    print(f"\n== batch-size sweep at main-threads={best_nt} ==")
    by_batch = {64: by_threads[best_nt]}
    for bs in (512, 2048):
        model.batch_size = bs
        by_batch[bs] = cycle(model, env, f"main-threads={best_nt} batch={bs}")
    model.batch_size = 64
    env.close()

    # --- phase 3: workers pinned to OMP_NUM_THREADS=1 ---
    print("\n== workers at OMP_NUM_THREADS=1 ==")
    os.environ["OMP_NUM_THREADS"] = "1"
    env = build_env()
    model = load(env)
    torch.set_num_threads(best_nt)
    cycle(model, env, f"omp1-workers main-threads={best_nt} batch=64")
    best_bs = min(by_batch, key=lambda k: by_batch[k]["update_s"])
    if best_bs != 64:
        model.batch_size = best_bs
        cycle(model, env, f"omp1-workers main-threads={best_nt} batch={best_bs}")
    env.close()

    sampler.stop_flag = True
    Path("runs/e36/bench_cpu.json").write_text(json.dumps(results, indent=2))
    print("\nwrote runs/e36/bench_cpu.json")


if __name__ == "__main__":
    main()
