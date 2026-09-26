"""Warm-starting a MaskablePPO model into a widened observation space (E42).

``SB3.load(path, env=...)`` refuses when the saved model's observation space
doesn't match the new env's (a hard SB3 check) — so a hist-widened ("fxh")
model can't simply resume from a plain "fx" checkpoint via the normal load
path. ``warm_start`` instead: loads the old model WITHOUT an env (no space
check), reads its hyperparameters and policy configuration, builds a FRESH
model at the new (possibly widened) obs space via
``locma.envs.training._make_model``, and copies the old policy's weights in
with ``strict=False`` — the only keys allowed to go missing are the new
``hist_mlp`` branch (see ``locma.envs.extractor.SlimTokenExtractor``, E42),
which is zero-initialized there anyway, so a warm-started net computes
EXACTLY the same distribution as the old one at step 0. The new model gets a
FRESH optimizer (no old Adam moment state carries over).
"""

from __future__ import annotations

# Every module-level prefix a policy might register a features extractor
# under. MaskableActorCriticPolicy always registers `features_extractor`; with
# `share_features_extractor=False` it additionally registers independent
# `pi_features_extractor`/`vf_features_extractor` submodules. With the
# (default) shared extractor, `pi_features_extractor`/`vf_features_extractor`
# are STILL separate nn.Module attributes aliasing the same object as
# `features_extractor` (confirmed against depot:e36/e36_gen7.zip: `policy.
# features_extractor is policy.pi_features_extractor is policy.
# vf_features_extractor` all True), so `state_dict()` lists the hist branch
# under all three prefixes either way.
_EXTRACTOR_PREFIXES = ("features_extractor.", "pi_features_extractor.", "vf_features_extractor.")


def _const(value):
    """Evaluate an SB3 hyperparameter that may be a schedule callable.

    SB3 stores ``learning_rate``/``clip_range`` as either a plain float or a
    ``Schedule`` (``progress_remaining -> value``). Evaluating at
    ``progress_remaining=1.0`` reads the value AT THE START of the old
    model's training — the natural constant to seed a fresh model with (SB3's
    own convention: ``lr_schedule(1)`` is "the initial learning rate").
    """
    return value(1.0) if callable(value) else value


def warm_start(old_path: str, env, *, obs_mode: str, device: str = "auto"):
    """Build a fresh MaskablePPO model at ``obs_mode`` warm-started from ``old_path``.

    ``old_path`` is a plain path or a ``depot:`` ref (resolved via
    ``locma.depot.resolve_path``). ``env`` must already be built at the target
    ``obs_mode`` (e.g. via ``locma.envs.training._build_env``) — this function
    does not construct one, so callers control n_envs/opponent/pool/draft.

    Reads the old model's PPO hyperparameters (learning_rate, ent_coef,
    target_kl, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range,
    vf_coef, max_grad_norm, seed) and policy configuration (pointer head vs
    dense action net; ``SlimTokenExtractor`` vs ``TokenSetExtractor``;
    ``features_extractor_kwargs``), constructs a fresh model with
    ``locma.envs.training._make_model`` using the SAME configuration, then
    copies ``old.policy.state_dict()`` into the new policy with
    ``strict=False``. Asserts there are no unexpected keys and every missing
    key is a ``hist_mlp`` parameter (the E42 branch — see
    ``locma.envs.extractor.SlimTokenExtractor``) under one of
    ``features_extractor.``/``pi_features_extractor.``/``vf_features_extractor.``.
    Warm-starting into the SAME obs_mode as the old model (the E42 control
    arm) is the degenerate case: zero missing keys, identical outputs, fresh
    optimizer only.

    Returns the new model (fresh optimizer — no old Adam moment state).
    """
    from sb3_contrib import MaskablePPO  # noqa: PLC0415 — optional [ml] dep

    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.envs.extractor import SlimTokenExtractor  # noqa: PLC0415
    from locma.envs.pointer_head import PointerMaskablePolicy  # noqa: PLC0415
    from locma.envs.training import _make_model  # noqa: PLC0415

    old = MaskablePPO.load(resolve_path(old_path), device=device)

    pointer_head = isinstance(old.policy, PointerMaskablePolicy)
    extractor_cls = (old.policy_kwargs or {}).get("features_extractor_class")
    slim = extractor_cls is SlimTokenExtractor
    extractor_kwargs = (old.policy_kwargs or {}).get("features_extractor_kwargs")

    new = _make_model(
        env,
        obs_mode=obs_mode,
        seed=old.seed,
        verbose=0,
        ent_coef=_const(old.ent_coef),
        learning_rate=_const(old.learning_rate),
        target_kl=old.target_kl,
        n_steps=old.n_steps,
        batch_size=old.batch_size,
        n_epochs=old.n_epochs,
        gamma=old.gamma,
        gae_lambda=old.gae_lambda,
        clip_range=_const(old.clip_range),
        vf_coef=_const(old.vf_coef),
        max_grad_norm=old.max_grad_norm,
        device=device,
        extractor_kwargs=dict(extractor_kwargs) if extractor_kwargs else None,
        pointer_head=pointer_head,
        slim=slim,
    )

    missing, unexpected = new.policy.load_state_dict(old.policy.state_dict(), strict=False)
    if unexpected:
        raise ValueError(f"warm_start: unexpected keys copying into the new policy: {unexpected}")
    bad = [k for k in missing if not (k.startswith(_EXTRACTOR_PREFIXES) and ".hist_mlp." in k)]
    if bad:
        raise ValueError(f"warm_start: missing keys that are not the E42 hist_mlp branch: {bad}")
    return new
