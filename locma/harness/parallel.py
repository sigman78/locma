"""Shared helpers for CPU-parallel eval harnesses (ceiling-eval, draft-bench).

Eval work parallelises over *picklable spec strings / paths*, never live policy
objects: each worker process rebuilds its policies (Windows spawn-safe) and
caches heavyweight ones (PPO nets) per process.
"""

from __future__ import annotations

import os


def resolve_workers(workers: int) -> int:
    """Resolve a --workers value: 0 (auto) = all logical CPUs minus one."""
    if workers <= 0:
        return max(1, (os.cpu_count() or 2) - 1)
    return workers


def init_eval_worker() -> None:
    """Process-pool initializer: pin BLAS/torch to one thread per worker so N
    workers don't oversubscribe the box with N x default-thread-count.

    Env vars (not ``torch.set_num_threads``) so torch is not eagerly imported in
    workers that never load a net; torch reads OMP_NUM_THREADS at first import,
    which is guaranteed to happen after this initializer."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
