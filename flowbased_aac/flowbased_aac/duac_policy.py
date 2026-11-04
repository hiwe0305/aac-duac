"""DUAC for OpenPI pi0.5: paper equations, AAC motion lower bound, and the JAX/OpenPI sampler.

DUAC has no public code release.  Everything here follows the reference reproduction
``/mnt/ssd-storage/hactt13/UQ_action_chunking/AAC_reproduce/flowbased_aac`` (core.py,
jax_estimator.py, openpi_server.py), which implements the paper directly and reaches 95.0%
on LIBERO-10.  The motion lower bound xi reproduces the official AAC release
(``external/aac-libero/action_optimization/action_entropy_pi05.py``: ``action_magnitude`` and
``find_first_idx``); ``duac.py test`` checks both against those sources numerically.

Time convention: the paper's flow time ``tau`` increases from noise (0) to data (1); OpenPI
integrates the other way (t = 1 is noise).  The model velocity is u = noise - action, so the
paper's field is v = -u and its Jacobian trace flips sign.

    U_i(tau) = (1 - tau)^2 / tau * (D + (1 - tau) * Tr(d v_i / d a_tau_i))        (eq. 6)
    U_bar_h  = mean(U_1..U_h);  h_unc = max(argmax_h (U_bar_{h+1} - U_bar_h) + 1, 2) (eq. 8)
    h*       = max(h_unc, xi)                                                     (AAC lower bound)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
import time
import types
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class DUACConfig:
    """DUAC parameters (paper defaults) and the arm's horizon rule."""

    chunk_horizon: int = 10
    min_horizon: int = 2
    tau: float = 0.9
    num_probes: int = 40
    alpha: float = 3.0  # AAC movement threshold move_th
    valid_action_dim: int = 7  # LIBERO action dims; pi0.5 pads actions to 32
    num_denoising_steps: int = 10
    fixed_horizon: Optional[int] = None
    random_horizon_pool: Optional[List[int]] = field(default=None)
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.tau < 1.0:
            raise ValueError("tau must be strictly between 0 and 1")
        grid = self.tau * self.num_denoising_steps
        if not np.isclose(grid, round(grid), atol=1e-7):
            raise ValueError("tau * num_denoising_steps must be an integer (tau must lie on the Euler grid)")
        if self.num_probes < 1 or self.min_horizon < 1 or self.valid_action_dim < 1:
            raise ValueError("num_probes, min_horizon and valid_action_dim must be positive")


@dataclass
class DUACDecisionResult:
    """One decision: the executed horizon and both components of the adaptive rule."""

    h_selected: int  # horizon actually executed by this arm
    h_unc: int  # equation (8) horizon from the uncertainty profile
    xi: int  # AAC movement lower bound
    u_step: List[float]
    u_prefix: List[float]
    u_diff: List[float]
    m_step: List[float]
    tau: float = 0.9
    num_probes: int = 40
    alpha: float = 3.0
    mode: str = "duac"


# --------------------------------------------------------------------------- paper equations


def action_uncertainty(paper_block_divergence: Sequence[float], *, tau: float, action_dim: int) -> np.ndarray:
    """Equation (6): posterior covariance trace per action position (no clipping, as in the paper)."""

    divergence = np.asarray(paper_block_divergence, dtype=np.float64)
    scale = (1.0 - tau) ** 2 / tau
    return scale * (float(action_dim) + (1.0 - tau) * divergence)


def prefix_mean_jump_horizon(u_step: Sequence[float], min_horizon: int = 2) -> Tuple[int, List[float], List[float]]:
    """Equation (8) / AAC elbow rule: ``max(argmax(diff(prefix_means)) + 1, min_horizon)``, clipped to H.

    Exact ties use the first maximum (``np.argmax``), like the AAC release.
    """

    values = np.asarray(u_step, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("u_step must be a non-empty 1-D sequence")
    means = np.cumsum(values) / np.arange(1, values.size + 1)
    diffs = np.diff(means)
    horizon = 1 if values.size == 1 else int(np.argmax(diffs)) + 1
    horizon = min(values.size, max(horizon, int(min_horizon)))
    return horizon, means.tolist(), diffs.tolist()


# --------------------------------------------------------------------------- AAC motion lower bound


def _axisangle_to_quat(vec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vec))
    if np.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = vec / angle
    return np.r_[axis * np.sin(angle / 2.0), np.cos(angle / 2.0)]


def _quat_multiply(q1: np.ndarray, q0: np.ndarray) -> np.ndarray:
    """robosuite ``quat_multiply(q1, q0)`` (x, y, z, w), as used by the AAC release."""

    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    return np.array(
        [
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ]
    )


def _quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - quat[3] ** 2))
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * np.arccos(quat[3]) / den


def compose_rotations(delta_rotations: np.ndarray) -> np.ndarray:
    """Compose LIBERO axis-angle rotation deltas (AAC ``help_function.compose_rotations``)."""

    total = np.array([0.0, 0.0, 0.0, 1.0])
    for rotation in np.asarray(delta_rotations, dtype=np.float64):
        total = _quat_multiply(_axisangle_to_quat(rotation), total)
    return _quat_to_axisangle(total)


def action_magnitude_pi05(actions: Sequence[Sequence[float]]) -> List[float]:
    """Cumulative motion m_h for h = 1..H (AAC ``action_magnitude``, weights 1 / 1 / 0.2).

    ``actions`` are the *un-normalized* LIBERO actions of the executed chunk: XYZ delta, axis-angle
    delta, gripper command.  m_h = ||sum dpos|| + ||compose(drot)|| + 0.2 * [gripper toggles in 1..h].
    AAC only uses h >= 2; m_1 is kept for logging.
    """

    a = np.asarray(actions, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 7 or a.shape[0] == 0:
        raise ValueError(f"actions must have shape [H, >=7], got {a.shape}")
    gripper = (a[:, 6] >= 0.0).astype(np.int32)
    m_list = []
    for length in range(1, len(a) + 1):
        position = float(np.linalg.norm(np.sum(a[:length, :3], axis=0)))
        rotation = float(np.linalg.norm(compose_rotations(a[:length, 3:6])))
        toggled = float(np.any(np.diff(gripper[:length])))
        m_list.append(position + rotation + 0.2 * toggled)
    return m_list


def find_first_idx(m_step: Sequence[float], alpha: float = 3.0, min_horizon: int = 2) -> int:
    """AAC ``find_first_idx``: first horizon h >= 2 with m_h > alpha (strict), else H."""

    for index, value in enumerate(m_step):
        horizon = index + 1
        if horizon >= min_horizon and value > alpha:
            return horizon
    return max(len(m_step), 1)


def movement_lower_bound(actions: Sequence[Sequence[float]], *, threshold: float = 3.0) -> int:
    """AAC's action-dependent lower bound xi (``min_id`` in the AAC code)."""

    return find_first_idx(action_magnitude_pi05(actions), alpha=threshold, min_horizon=2)


# --------------------------------------------------------------------------- horizon rule per arm


def select_duac_horizon(
    u_step: Sequence[float],
    actions: Sequence[Sequence[float]],
    config: DUACConfig,
    rng: Optional[random.Random] = None,
) -> DUACDecisionResult:
    """Return the executed horizon for DUAC, Fixed-k or Random-h; DUAC's own proposal is always logged.

    ``actions`` must be the un-normalized chunk returned by the server.
    """

    actions = np.asarray(actions, dtype=np.float64)[:, :7]
    horizon = len(actions)
    m_step = action_magnitude_pi05(actions)
    xi = find_first_idx(m_step, alpha=config.alpha, min_horizon=2)
    h_unc, u_prefix, u_diff = prefix_mean_jump_horizon(u_step, min_horizon=config.min_horizon)

    if config.fixed_horizon is not None:
        h_sel = max(1, min(int(config.fixed_horizon), horizon))
        mode = f"fixed_{config.fixed_horizon}"
    elif config.random_horizon_pool:
        chooser = rng if rng is not None else random
        h_sel = max(1, min(int(chooser.choice(config.random_horizon_pool)), horizon))
        mode = "random_h"
    else:
        h_sel = min(horizon, max(h_unc, xi))
        mode = "duac"

    return DUACDecisionResult(
        h_selected=h_sel,
        h_unc=h_unc,
        xi=xi,
        u_step=[float(x) for x in u_step],
        u_prefix=u_prefix,
        u_diff=u_diff,
        m_step=m_step,
        tau=config.tau,
        num_probes=config.num_probes,
        alpha=config.alpha,
        mode=mode,
    )


# --------------------------------------------------------------------------- JAX / OpenPI sampler


def linearized_hutchinson_action_divergence(vector_field, x, key, *, num_probes: int, valid_action_dim: int):
    """Return ``vector_field(x)`` and a Hutchinson estimate of each action block's Jacobian trace.

    Rademacher probes span the whole [H, D] chunk; cross-position terms cancel in expectation, so
    ``mean_k sum_d eps_k[i, d] * (J eps_k)[i, d]`` estimates ``Tr(d v_i / d x_i)``.  Padded action
    dimensions (>= valid_action_dim) are masked out of the probes.
    """

    import jax
    import jax.numpy as jnp

    if x.ndim != 2:
        raise ValueError(f"x must have shape [H, D], got {x.shape}")
    action_dim = x.shape[-1]
    probes = jax.random.rademacher(key, (num_probes, *x.shape), dtype=x.dtype)
    mask = (jnp.arange(action_dim) < valid_action_dim).astype(x.dtype)
    probes = probes * mask[None, None, :]
    value, pushforward = jax.linearize(vector_field, x)
    jvps = jax.vmap(pushforward)(probes)
    divergence = jnp.mean(jnp.sum(probes * jvps, axis=-1), axis=0)
    return value, divergence


def make_duac_sampler(config: DUACConfig):
    """Pi0 ``sample_actions`` replacement returning one chunk plus its DUAC profile (batch size 1)."""

    import einops
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models import pi0 as _pi0

    target_step = int(round(config.tau * config.num_denoising_steps))
    model_tau = 1.0 - config.tau

    def sample_actions(self, rng, observation, *, num_steps=config.num_denoising_steps):
        if num_steps != config.num_denoising_steps:
            raise ValueError(f"DUAC sampler compiled for {config.num_denoising_steps} steps, got {num_steps}")
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        noise_rng, probe_rng = jax.random.split(rng)
        x = jax.random.normal(noise_rng, (1, self.action_horizon, self.action_dim))
        dt = -1.0 / config.num_denoising_steps

        # The vision/language prefix is encoded once; the flow steps and all JVP probes reuse its KV cache.
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def velocity_batched(x_t, model_time):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(model_time, (batch_size,))
            )
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
            suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            return self.action_out_proj(suffix_out[:, -self.action_horizon :])

        def euler_step(index, current):
            model_time = 1.0 + index.astype(current.dtype) * dt
            return current + dt * velocity_batched(current, model_time)

        # After target_step (= 9) Euler steps, x is a_tau at paper tau = 0.9 / OpenPI t = 0.1.
        x_tau = jax.lax.fori_loop(0, target_step, euler_step, x)

        def velocity_unbatched(chunk):
            return velocity_batched(chunk[None, ...], jnp.asarray(model_tau, chunk.dtype))[0]

        target_velocity, model_divergence = linearized_hutchinson_action_divergence(
            velocity_unbatched,
            x_tau[0],
            probe_rng,
            num_probes=config.num_probes,
            valid_action_dim=config.valid_action_dim,
        )
        paper_divergence = -model_divergence  # v = -u  =>  J_v = -J_u
        scale = (1.0 - config.tau) ** 2 / config.tau
        uncertainty = scale * (config.valid_action_dim + (1.0 - config.tau) * paper_divergence)

        # Finish the integration, reusing the velocity already evaluated at tau.
        x = x_tau + dt * target_velocity[None, ...]
        x = jax.lax.fori_loop(target_step + 1, config.num_denoising_steps, euler_step, x)
        return x[0], uncertainty, paper_divergence

    return sample_actions


class DuacPolicy:
    """OpenPI policy adapter: one chunk per call plus the DUAC uncertainty profile and horizon.

    The RNG is stateful (split once per call) unless ``seed_per_step`` is set, matching the protocol
    "policy seed 0, stateful RNG" of docs/task3010.md section 2.5.
    """

    def __init__(self, policy, config: DUACConfig, *, seed: int = 0, seed_per_step: bool = False):
        import jax
        from openpi.shared import nnx_utils

        if config.valid_action_dim > policy._model.action_dim:
            raise ValueError("valid_action_dim exceeds the model action_dim")
        self._policy = policy
        self._config = config
        self._seed = seed
        self._seed_per_step = seed_per_step
        self._rng = jax.random.key(seed)
        self._sample = nnx_utils.module_jit(types.MethodType(make_duac_sampler(config), policy._model))

    @property
    def metadata(self) -> Dict[str, Any]:
        meta = dict(self._policy.metadata)
        meta["duac"] = {
            "tau": self._config.tau,
            "num_probes": self._config.num_probes,
            "min_horizon": self._config.min_horizon,
            "valid_action_dim": self._config.valid_action_dim,
            "num_denoising_steps": self._config.num_denoising_steps,
            "seed": self._seed,
            "seed_per_step": self._seed_per_step,
        }
        return meta

    def infer(self, obs: dict) -> dict:
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        policy = self._policy
        inputs = policy._input_transform(jax.tree.map(lambda value: value, obs))
        state = np.asarray(inputs["state"])
        inputs = jax.tree.map(lambda value: jnp.asarray(value)[None], inputs)
        observation = _model.Observation.from_dict(inputs)

        if self._seed_per_step:
            self._rng = jax.random.key(self._seed)
        self._rng, sample_rng = jax.random.split(self._rng)

        started = time.monotonic()
        result = self._sample(sample_rng, observation, **policy._sample_kwargs)
        normalized, uncertainty, divergence = (np.asarray(v) for v in jax.block_until_ready(result))
        infer_ms = (time.monotonic() - started) * 1000.0

        actions = np.asarray(policy._output_transform({"state": state, "actions": normalized})["actions"])
        h_unc, _, _ = prefix_mean_jump_horizon(uncertainty, min_horizon=self._config.min_horizon)
        return {
            "actions": actions,  # (H, 7) un-normalized
            "normalized_action": normalized,  # (H, 32)
            "u_step": uncertainty,  # equation (6), length H
            "duac_uncertainty": uncertainty,
            "duac_divergence": divergence,
            "duac_horizon": np.asarray(h_unc, dtype=np.int32),
            "policy_timing": {"infer_ms": infer_ms},
        }
