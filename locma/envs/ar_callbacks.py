"""Training telemetry for the autoregressive head: periodic avg-hard3 eval
and per-head entropy logging. Requires the [ml] extra."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class ARTelemetryCallback(BaseCallback):
    """Every ``eval_freq`` steps: save the model to a temp path, run a quick
    avg-hard3, and record it plus the last per-head entropies to the logger."""

    def __init__(
        self,
        eval_freq: int = 50_000,
        seeds: int = 40,
        base_seed: int = 2_000_000,
        games_per_seed: int = 1,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.seeds = seeds
        self.base_seed = base_seed
        self.games_per_seed = games_per_seed
        self._last = 0

    def _record_entropy(self) -> None:
        heads = getattr(self.model.policy, "ar_heads", None)
        if heads is not None and hasattr(heads, "last_head_entropy"):
            et, es, etg = heads.last_head_entropy
            self.logger.record("ar_entropy/type", et)
            self.logger.record("ar_entropy/source", es)
            self.logger.record("ar_entropy/target", etg)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last < self.eval_freq:
            return True
        self._last = self.num_timesteps
        self._record_entropy()
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        from locma.harness.ar_study import hard3_per_seed  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "snap.zip")
            self.model.save(p)
            seeds = [self.base_seed + i for i in range(self.seeds)]
            avg = float(hard3_per_seed(p, seeds, self.games_per_seed).mean())
        self.logger.record("eval/avg_hard3", avg)
        if self.verbose:
            print(f"[ar-telemetry] step={self.num_timesteps} avg_hard3={avg:.4f}")
        return True
