"""MaskablePPO training entrypoint (requires the [ml] extra).

Drives the `locma train` CLI command. The training loop lives here so it has a
single home. Supports parallel envs (CPU speedup), a seeded trainer
(reproducibility), and intermediate checkpoints saved along one trajectory.
"""

from __future__ import annotations

import functools


def _make_battle_env(
    opponent_spec: str,
    seed: int,
    agent_seat: int = 0,
    seat_random: bool = False,
    obs_mode: str = "flat",
    draft_noise: int = 0,
    shared_draft: bool = False,
):
    """Top-level env factory (picklable for SubprocVecEnv spawn on Windows).

    Rebuilds the opponent from its spec string inside each subprocess, so the
    (possibly stateful) opponent never has to be pickled and crosses no process
    boundary. ``seat_random`` trains the agent as both first and second player.
    ``obs_mode`` selects the observation encoding: "flat" (default) or "token".
    ``draft_noise`` (k > 0) wraps the opponent's draft half so exactly k of each
    deck's 30 picks are uniformly random — the opponent drafts BOTH seats in the
    battle env, so this diversifies the decks the agent trains on.
    ``shared_draft`` runs the shared draft variant (picks deplete the offer,
    first pick alternates by round), giving the two seats asymmetric decks.
    """
    from locma.envs.battle_env import BattleEnv  # noqa: PLC0415 — optional [ml] dep
    from locma.policies.composer import Composer  # noqa: PLC0415
    from locma.policies.drafts import PartialRandomDraftPolicy  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    opponent = make_policy(opponent_spec)
    if draft_noise:
        if getattr(opponent, "draft", None) is None:
            raise ValueError(f"draft_noise needs an opponent with a draft half: '{opponent_spec}'")
        opponent = Composer(
            opponent.battle,
            PartialRandomDraftPolicy(opponent.draft, draft_noise, seed=seed),
            name=f"{opponent_spec}+rnd{draft_noise}",
        )
    return BattleEnv(
        opponent=opponent,
        seed=seed,
        agent_seat=agent_seat,
        seat_random=seat_random,
        obs_mode=obs_mode,
        shared_draft=shared_draft,
    )


def _ckpt_path(out: str, steps: int) -> str:
    """Derive a step-suffixed checkpoint path: model.zip + 1000 -> model-1000.zip."""
    base = out[:-4] if out.endswith(".zip") else out
    return f"{base}-{steps}.zip"


def _build_env(
    opponent_spec: str,
    seed: int,
    n_envs: int,
    both_seat: bool = True,
    obs_mode: str = "flat",
    draft_noise: int = 0,
    shared_draft: bool = False,
):
    """Build a (vectorised) training env. n_envs>1 runs each env in its own
    process for true CPU parallelism; each env gets a distinct seed. ``both_seat``
    randomizes the agent's seat per episode (the +0.06-and-2x-efficiency fix).
    ``obs_mode`` selects the observation encoding: "flat" (default) or "token".
    ``draft_noise`` (k) makes k of each deck's 30 draft picks uniformly random.
    ``shared_draft`` runs the shared draft variant (asymmetric decks)."""
    from stable_baselines3.common.vec_env import (  # noqa: PLC0415
        DummyVecEnv,
        SubprocVecEnv,
        VecMonitor,
    )

    # Stride per-env seeds: each BattleEnv draws episode seeds base+ep, so a
    # plain seed+i overlaps worker i's episode k with worker j's episode
    # k+(i-j). The stride must exceed episodes-per-env for any realistic run
    # (800k steps / n_envs at ~40 steps/episode is ~1-3k episodes) and the
    # whole block must stay below the 1_000_000+ held-out eval-seed range
    # (16 envs * 50_000 = 800k max offset).
    fns = [
        functools.partial(
            _make_battle_env,
            opponent_spec,
            seed + i * 50_000,
            0,
            both_seat,
            obs_mode,
            draft_noise,
            shared_draft,
        )
        for i in range(n_envs)
    ]
    # VecMonitor fills ep_info_buffer (ep_rew_mean for logs and the web panel's
    # live reward curve); it does not touch rewards, obs, or RNG.
    return VecMonitor(DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns))


def _make_model(
    env,
    *,
    obs_mode: str,
    seed: int,
    verbose: int,
    ent_coef: float,
    learning_rate: float = 3e-4,
    target_kl: float | None = None,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    device: str = "auto",
    extractor_kwargs: dict | None = None,
    tensorboard_log: str | None = None,
):
    """Construct a MaskablePPO model, selecting the policy class by obs_mode.

    All PPO knobs are explicit so a sweep can set them; defaults match SB3's own
    defaults, so an unset knob reproduces the pre-sweep behavior byte-for-byte.
    """
    from sb3_contrib import MaskablePPO  # noqa: PLC0415 — optional [ml] dep

    common = dict(
        verbose=verbose,
        seed=seed,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        target_kl=target_kl,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        device=device,
        tensorboard_log=tensorboard_log,
    )

    if obs_mode.startswith("token"):
        from locma.envs.extractor import TokenSetExtractor  # noqa: PLC0415

        pk = dict(features_extractor_class=TokenSetExtractor)
        if extractor_kwargs:
            pk["features_extractor_kwargs"] = dict(extractor_kwargs)
        return MaskablePPO("MultiInputPolicy", env, policy_kwargs=pk, **common)

    # Default: flat obs → MlpPolicy (byte-identical to the pre-PPO2 baseline).
    return MaskablePPO("MlpPolicy", env, **common)


def train_agent(
    opponent_spec: str,
    steps: int = 50_000,
    out: str = "model.zip",
    seed: int = 0,
    verbose: int = 1,
    n_envs: int = 1,
    checkpoints=None,
    ent_coef: float = 0.02,
    both_seat: bool = True,
    obs_mode: str = "flat",
    learning_rate: float = 3e-4,
    target_kl: float | None = None,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    device: str = "auto",
    extractor_kwargs: dict | None = None,
    callback=None,
    tensorboard_log: str | None = None,
    draft_noise: int = 0,
    shared_draft: bool = False,
):
    """Train a seeded MaskablePPO agent against `opponent_spec` and save it.

    Parameters
    ----------
    opponent_spec: registry spec string for the opponent (rebuilt per env).
    steps: total env timesteps (ignored when `checkpoints` is given).
    out: output model path; checkpoints derive step-suffixed siblings.
    n_envs: number of parallel envs (CPU speedup).
    both_seat: train as both first and second player (default True; eval is
        mirrored, so seat-0-only training is a coverage gap — see docs/baseline.md).
    obs_mode: ``"flat"`` (default) for MlpPolicy + flat Box obs; ``"token"``
        for MultiInputPolicy + TokenSetExtractor + tokenized Dict obs.
    checkpoints: optional iterable of step marks. When given, training runs as
        one continuous trajectory, saving a step-suffixed model at each mark, and
        returns the list of saved paths. Otherwise trains `steps` and returns
        the single `out` path.
    learning_rate: PPO learning rate (default 3e-4, matching SB3's own default).
    target_kl: PPO target KL divergence for early stopping (None = off, the default).
    n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, vf_coef,
        max_grad_norm: PPO hyperparameters, forwarded to `_make_model` (defaults
        match SB3's own defaults).
    device: torch device passed to SB3 (default "auto").
    extractor_kwargs: optional kwargs for TokenSetExtractor (token obs_mode only).
    callback: optional SB3 callback (or CallbackList), passed to every `model.learn`.
    tensorboard_log: optional tensorboard log directory.
    draft_noise: k of each deck's 30 draft picks made uniformly random (0 = off) —
        diversifies the decks the agent trains on (the opponent drafts both seats).
    shared_draft: run the shared draft variant — picks deplete the offer, first
        pick alternates by round, so the two seats get asymmetric decks.

    Imports the ML stack lazily; an ImportError means the `[ml]` extra is absent.
    """
    env = _build_env(
        opponent_spec,
        seed,
        n_envs,
        both_seat=both_seat,
        obs_mode=obs_mode,
        draft_noise=draft_noise,
        shared_draft=shared_draft,
    )
    model = _make_model(
        env,
        obs_mode=obs_mode,
        seed=seed,
        verbose=verbose,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        target_kl=target_kl,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        device=device,
        extractor_kwargs=extractor_kwargs,
        tensorboard_log=tensorboard_log,
    )

    if checkpoints:
        marks = sorted({int(m) for m in checkpoints})
        prev = 0
        saved = []
        for i, mark in enumerate(marks):
            model.learn(
                total_timesteps=mark - prev, reset_num_timesteps=(i == 0), callback=callback
            )
            path = _ckpt_path(out, mark)
            model.save(path)
            saved.append(path)
            prev = mark
        env.close()
        return saved

    model.learn(total_timesteps=steps, callback=callback)
    model.save(out)
    env.close()
    return out


# ---------------------------------------------------------------------------
# E18b learned draft: train a draft policy against a frozen battle pilot
# ---------------------------------------------------------------------------


def _battle_half(spec: str):
    """Resolve a battle policy from a registry spec ('greedy', 'ppo:path') or a
    bare model path / depot ref ('runs/x.zip', 'depot:b0k/b0k_s0.zip')."""
    from locma.policies.registry import is_policy_spec, make_policy  # noqa: PLC0415

    if is_policy_spec(spec):
        return make_policy(spec).battle
    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.policies.ppo import MaskablePPOBattlePolicy  # noqa: PLC0415

    return MaskablePPOBattlePolicy(model_path=resolve_path(spec))


def _draft_half(name: str):
    """Named opponent draft policies a learned draft can train against."""
    from locma.policies import drafts  # noqa: PLC0415

    factories = {
        "balanced": drafts.BalancedDraftPolicy,
        "weighted": drafts.WeightedDraftPolicy,
        "greedy": drafts.GreedyDraftPolicy,
        "random": lambda: drafts.RandomDraftPolicy(seed=0),
    }
    if name not in factories:
        raise ValueError(f"unknown opponent draft '{name}' (have {sorted(factories)})")
    return factories[name]()


def _make_draft_env(
    battle_spec: str,
    opponent_draft: str,
    seed: int,
    seat_random: bool = True,
    rollouts: int = 1,
):
    """Top-level draft-env factory (picklable for SubprocVecEnv spawn on Windows).

    Rebuilds the frozen battle pilot and the opponent draft from strings inside
    each subprocess. Pins BLAS/torch threads first (the pilot net runs inference
    in every env worker; N workers x default-thread-count oversubscribes the
    box — same rationale as harness init_eval_worker)."""
    import os  # noqa: PLC0415

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    from locma.envs.draft_env import DraftEnv  # noqa: PLC0415

    return DraftEnv(
        battle_pilot=_battle_half(battle_spec),
        opponent_draft=_draft_half(opponent_draft),
        seed=seed,
        seat_random=seat_random,
        rollouts=rollouts,
    )


def train_draft(
    battle_spec: str,
    steps: int = 50_000,
    out: str = "draft.zip",
    seed: int = 0,
    opponent_draft: str = "balanced",
    rollouts: int = 1,
    verbose: int = 1,
    n_envs: int = 1,
    ent_coef: float = 0.02,
    seat_random: bool = True,
    learning_rate: float = 3e-4,
    target_kl: float | None = None,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    device: str = "auto",
    callback=None,
    tensorboard_log: str | None = None,
):
    """Train a MaskablePPO DRAFT policy against a frozen battle pilot (E18b).

    The learned net picks one of the 3 offered cards for 30 rounds; the battle
    is then played out by ``battle_spec`` on BOTH seats (mirror pilot — the
    reward isolates deck quality) against a deck drafted by ``opponent_draft``.
    ``steps`` counts draft picks (30 per episode). The saved model loads back
    via ``MaskablePPODraftPolicy`` / the registry's learned-draft spec param.

    gamma defaults to 1.0 (not SB3's 0.99): the episode reward is purely
    terminal, and 30 picks of discounting would scale the first pick's signal
    by 0.99^29 ~ 0.75 for no reason (the ByteRL gamma lesson, E18a).

    Imports the ML stack lazily; an ImportError means the `[ml]` extra is absent.
    """
    from stable_baselines3.common.vec_env import (  # noqa: PLC0415
        DummyVecEnv,
        SubprocVecEnv,
        VecMonitor,
    )

    # Same per-env seed stride rationale as _build_env: episodes advance the
    # seed by 1, so spacing workers 50k apart keeps their episode streams
    # disjoint and below the 1M+ held-out eval-seed range.
    fns = [
        functools.partial(
            _make_draft_env,
            battle_spec,
            opponent_draft,
            seed + i * 50_000,
            seat_random,
            rollouts,
        )
        for i in range(n_envs)
    ]
    env = VecMonitor(DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns))
    model = _make_model(
        env,
        obs_mode="flat",  # draft obs is a flat Box -> MlpPolicy
        seed=seed,
        verbose=verbose,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        target_kl=target_kl,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        device=device,
        tensorboard_log=tensorboard_log,
    )
    model.learn(total_timesteps=steps, callback=callback)
    model.save(out)
    env.close()
    return out


# A small, code-declared "zoo" of opponents to train against back-to-back. This is
# intentionally a constant for now (no CLI list plumbing); edit it to change the
# curriculum. Order matters — training proceeds left to right. See docs/cli.md and
# the future-explorations roadmap in docs/ppo-review.md.
# "boardkeep" (E11): the disciplined winning-trades-only archetype that beat the
# 4-opponent reactive B0 outright in E10 — last so the hardest opponent gets the
# freshest gradient.
ZOO_OPPONENTS: tuple[str, ...] = ("greedy", "scripted", "max-guard", "max-attack", "boardkeep")


def train_zoo(
    opponents=ZOO_OPPONENTS,
    steps_per_opponent: int = 200_000,
    out: str = "model.zip",
    seed: int = 0,
    ent_coef: float = 0.02,
    verbose: int = 1,
    both_seat: bool = True,
    obs_mode: str = "flat",
    learning_rate: float = 3e-4,
    target_kl: float | None = None,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    device: str = "auto",
    extractor_kwargs: dict | None = None,
    n_envs: int = 1,
    callback=None,
    tensorboard_log: str | None = None,
    draft_noise: int = 0,
    shared_draft: bool = False,
):
    """Train ONE MaskablePPO model back-to-back against each opponent in turn.

    The model's weights carry across phases (a curriculum) — `set_env` swaps the
    opponent and `learn` continues without resetting the timestep counter. Total
    budget is ``steps_per_opponent * len(opponents)``. Returns the saved path.

    Parameters
    ----------
    opponents: ordered sequence of opponent spec strings (the curriculum).
    steps_per_opponent: env timesteps per opponent phase.
    out: output model path.
    obs_mode: ``"flat"`` (default) for MlpPolicy + flat Box obs; ``"token"``
        for MultiInputPolicy + TokenSetExtractor + tokenized Dict obs.
    learning_rate: PPO learning rate (default 3e-4, matching SB3's own default).
    target_kl: PPO target KL divergence for early stopping (None = off, the default).
    n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, vf_coef,
        max_grad_norm: PPO hyperparameters, forwarded to `_make_model` (defaults
        match SB3's own defaults).
    device: torch device passed to SB3 (default "auto").
    extractor_kwargs: optional kwargs for TokenSetExtractor (token obs_mode only).
    n_envs: number of parallel envs per opponent phase (CPU speedup).
    callback: optional SB3 callback (or CallbackList), passed to every `model.learn`.
    tensorboard_log: optional tensorboard log directory.
    draft_noise: k of each deck's 30 draft picks made uniformly random (0 = off) —
        diversifies the decks the agent trains on (the opponent drafts both seats).
    shared_draft: run the shared draft variant — picks deplete the offer, first
        pick alternates by round, so the two seats get asymmetric decks.

    Imports the ML stack lazily; an ImportError means the `[ml]` extra is absent.
    """
    opps = list(opponents)
    if not opps:
        raise ValueError("train_zoo needs a non-empty opponent list")

    def build(opp):
        return _build_env(
            opp,
            seed,
            n_envs,
            both_seat=both_seat,
            obs_mode=obs_mode,
            draft_noise=draft_noise,
            shared_draft=shared_draft,
        )

    model = _make_model(
        build(opps[0]),
        obs_mode=obs_mode,
        seed=seed,
        verbose=verbose,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        target_kl=target_kl,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        device=device,
        extractor_kwargs=extractor_kwargs,
        tensorboard_log=tensorboard_log,
    )
    try:
        for i, opp in enumerate(opps):
            if i > 0:
                old_env = model.env
                model.set_env(build(opp))
                old_env.close()  # else SubprocVecEnv workers from the old phase leak
            model.learn(
                total_timesteps=steps_per_opponent,
                reset_num_timesteps=(i == 0),
                callback=callback,
            )
        model.save(out)
    finally:
        # model.learn() can raise mid-phase (e.g. from an eval callback or a
        # KeyboardInterrupt) -- close whatever VecEnv is live at that point too,
        # or its SubprocVecEnv workers leak just like the phase-swap case above.
        model.env.close()
    return out
