"""In-training win-rate telemetry for the PPO ceiling study (requires [ml]).

Every ``eval_freq`` steps, plays paired games with the *current* policy vs a fixed
set of scripted opponents, logs avg-hard3 + per-opponent rates to TensorBoard, and
(optionally) reports the score to an Optuna trial so hopeless runs prune early.
Reuses the registry + run_match path for eval fidelity; the PPO net is piloted with
the BalancedDraftPolicy (the deployment pairing).
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class WinRateEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_opponents: tuple[str, ...] = ("scripted", "max-guard", "max-attack"),
        eval_freq: int = 50_000,
        n_games: int = 120,
        eval_seed: int = 1_000_000,
        trial=None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.eval_opponents = eval_opponents
        self.eval_freq = eval_freq
        self.n_games = n_games
        self.eval_seed = eval_seed
        self.trial = trial
        self.last_avg_hard3: float | None = None
        self.logged_keys: set[str] = set()

    def _eval_policy(self):
        from locma.policies.composer import Composer  # noqa: PLC0415
        from locma.policies.drafts import BalancedDraftPolicy  # noqa: PLC0415
        from locma.policies.ppo import MaskablePPOBattlePolicy  # noqa: PLC0415

        battle = MaskablePPOBattlePolicy(model=self.model, deterministic=True)
        return Composer(battle, BalancedDraftPolicy(), name="ppo")

    def _evaluate(self) -> float:
        from locma.harness.match import run_match  # noqa: PLC0415
        from locma.policies.registry import make_policy  # noqa: PLC0415

        me = self._eval_policy()
        rates = []
        for opp in self.eval_opponents:
            res = run_match(me, make_policy(opp), games=self.n_games, seed=self.eval_seed)
            wr = res.win_rate_a
            rates.append(wr)
            key = f"eval/vs_{opp.replace('-', '_')}"
            self.logger.record(key, wr)
            self.logged_keys.add(key)
        avg = sum(rates) / len(rates)
        self.logger.record("eval/avg_hard3", avg)
        self.logged_keys.add("eval/avg_hard3")
        return avg

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True
        avg = self._evaluate()
        self.last_avg_hard3 = avg
        self.logger.dump(self.num_timesteps)
        if self.trial is not None:
            import optuna  # noqa: PLC0415

            self.trial.report(avg, self.num_timesteps)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        return True

    def _on_training_end(self) -> None:
        # Guarantee at least one eval even if no step hit the modulus boundary.
        if self.last_avg_hard3 is None:
            self.last_avg_hard3 = self._evaluate()
