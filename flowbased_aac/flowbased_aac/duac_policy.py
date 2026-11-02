"""Core DUAC math, OpenPI pi0.5 JVP divergence uncertainty, and horizon selector.

Implements:
  1. Flow-field divergence uncertainty U_i(tau) from DUAC (ICRA 2026 / arXiv):
       U_i(tau) = (1/K) * sum_{k=1}^K || eps_k^{(i) T} * d v_theta^{(i)}(a_{n,tau}, c_n) / d a_{n,tau}^{(i)} ||_2
     evaluated at tau = 0.9 (which maps to OpenPI's t = 1 - tau = 0.1, the final
     Euler step index 9 of 0..9) using K = 40 Rademacher probes eps_k in {-1, +1}.
     Note that OpenPI parameterizes u_t = eps - a_n while DUAC uses v_theta = a_n - eps = -u_t;
     the norm ||J^T eps||_2 is sign-invariant.
  2. Prefix mean jump horizon rule (`prefix_mean_jump`):
       U_bar_h = (1/h) * sum_{j=1}^h U_j(tau)
       h_unc^* = max(argmax_{1 <= h < H} (U_bar_{h+1} - U_bar_h), 2)
  3. Cumulative motion lower bound xi (`action_magnitude` from AAC Eq. 6-10):
       m_h = w_pos * ||sum_{j=1}^h dx_j||_2 + w_rot * ||R_1 * ... * R_h||_rot + w_grip * I[g_h != g_0]
       xi = min { h in {2..H} : m_h >= alpha }  (or H if none reaches alpha)
       h^* = max(xi, h_unc^*)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class DUACConfig:
    """Configuration for DUAC horizon selection on pi0.5."""

    chunk_horizon: int = 10
    min_horizon: int = 2
    tau: float = 0.9
    num_probes: int = 40
    alpha: float = 3.0
    w_pos: float = 1.0
    w_rot: float = 1.0
    w_grip: float = 0.2
    action_dim_eval: int = 6  # Evaluate JVP on 6-DoF (pos + rot) or 7-DoF
    fixed_horizon: Optional[int] = None
    random_horizon_pool: Optional[List[int]] = None
    seed: int = 0


@dataclass
class DUACDecisionResult:
    """Detailed output of a single DUAC chunking decision."""

    h_selected: int
    h_unc: int
    xi: int
    u_step: List[float]
    u_prefix: List[float]
    u_diff: List[float]
    m_step: List[float]
    tau: float = 0.9
    num_probes: int = 40
    alpha: float = 3.0
    mode: str = "duac"


def _euler_xyz_to_quat_xyzw(rx: float, ry: float, rz: float) -> Tuple[float, float, float, float]:
    """Convert intrinsic/extrinsic XYZ Euler angles (radians) to unit quaternion (x, y, z, w).

    Matches `scipy.spatial.transform.Rotation.from_euler("xyz", [rx, ry, rz]).as_quat()`.
    In scipy, lowercase "xyz" is extrinsic (rotations about fixed axes x, then y, then z),
    so R = R_z(rz) * R_y(ry) * R_x(rx).
    """
    cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
    cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)

    # q = q_z * q_y * q_x
    w = cz * cy * cx + sz * sy * sx
    x = cz * cy * sx - sz * sy * cx
    y = cz * sy * cx + sz * cy * sx
    z = sz * cy * cx - cz * sy * sx
    return (x, y, z, w)


def _quat_mul_xyzw(
    q1: Tuple[float, float, float, float],
    q2: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Hamilton product of two (x, y, z, w) quaternions."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def compose_rotations_euler(rot_seq: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    """Compose a sequence of Euler xyz deltas into a single quaternion (x, y, z, w).

    Matches `action_entropy_pi05.compose_rotations` from `Adaptive-Action-Chunking/libero`.
    """
    q_total = (0.0, 0.0, 0.0, 1.0)
    for r in rot_seq:
        q_step = _euler_xyz_to_quat_xyzw(float(r[0]), float(r[1]), float(r[2]))
        q_total = _quat_mul_xyzw(q_total, q_step)
    return q_total


def action_magnitude_pi05(
    actions: Sequence[Sequence[float]],
    w_pos: float = 1.0,
    w_rot: float = 1.0,
    w_grip: float = 0.2,
    normalized_pos_rot: Optional[Sequence[Sequence[float]]] = None,
) -> List[float]:
    """Compute cumulative action magnitude m_1..m_H for pi0.5 chunk.

    Matches `action_magnitude(pred_action_dict)` in `Adaptive-Action-Chunking/libero`:
      - Position term: Euclidean norm of cumulative sum of delta_xyz (steps 0..t)
      - Rotation term: Euclidean norm of composed quaternion (x, y, z, w) across steps 0..t
        Wait: in `action_entropy_pi05.py`:
        `rot_norm = np.linalg.norm(compose_rotations(rot[:t+1]), axis=-1)`
        Since a unit quaternion has norm 1.0, let's check what `compose_rotations` returns in AAC!
        In `action_entropy_pi05.py`:
        `r_total.as_quat()` has norm 1.0! Wait: let's verify why `rot_norm` in AAC is `np.linalg.norm(r_total.as_quat())`!
        Because `r_total.as_quat()` is a unit quaternion `[x, y, z, w]`, its L2 norm in AAC's code is literally 1.0 at every step!
        And `pos` in AAC's `pred_action_dict["action.pos"]` is `normalized_action[:, :, :3]` (mean of 20 samples).
        When running DUAC (single sample) or AAC, if `normalized_pos_rot` is provided we use its `:3` and `3:6`;
        otherwise we use `actions[:, :3]` and `actions[:, 3:6]`.
    """
    H = len(actions)
    if H == 0:
        return []

    source_6d = normalized_pos_rot if normalized_pos_rot is not None else actions
    g0 = 1 if float(actions[0][6]) >= 0.0 else 0

    m_list: List[float] = []
    cum_dx = 0.0
    cum_dy = 0.0
    cum_dz = 0.0
    q_total = (0.0, 0.0, 0.0, 1.0)

    for t in range(H):
        row6 = source_6d[t]
        cum_dx += float(row6[0])
        cum_dy += float(row6[1])
        cum_dz += float(row6[2])
        pos_norm = math.sqrt(cum_dx * cum_dx + cum_dy * cum_dy + cum_dz * cum_dz)

        q_step = _euler_xyz_to_quat_xyzw(float(row6[3]), float(row6[4]), float(row6[5]))
        q_total = _quat_mul_xyzw(q_total, q_step)
        rot_norm = math.sqrt(sum(c * c for c in q_total))

        gt = 1 if float(actions[t][6]) >= 0.0 else 0
        grip_diff = 1.0 if gt != g0 else 0.0

        m_t = w_pos * pos_norm + w_rot * rot_norm + w_grip * grip_diff
        m_list.append(float(m_t))

    return m_list


def find_first_idx(
    m_step: Sequence[float],
    alpha: float = 3.0,
    min_horizon: int = 2,
) -> int:
    """Return 1-indexed horizon xi in {min_horizon, ..., H} where m_step crosses alpha.

    Matches `find_first_idx(chunk_m, threshold)` in `action_entropy_pi05.py`:
      - Looks for indices i (0-indexed) where m_step[i] >= alpha
      - Ignores index 0 (which corresponds to 1-indexed horizon 1 < min_horizon)
      - Returns (first_valid_i + 1), or H if none crosses alpha.
    """
    H = len(m_step)
    if H == 0:
        return min_horizon
    for i, val in enumerate(m_step):
        horizon_1idx = i + 1
        if horizon_1idx >= min_horizon and val >= alpha:
            return horizon_1idx
    return max(H, min_horizon)


def prefix_mean_jump_horizon(
    u_step: Sequence[float],
    min_horizon: int = 2,
) -> Tuple[int, List[float], List[float]]:
    """Compute prefix mean uncertainty U_bar_h, adjacent jumps, and h_unc^* in {min_horizon..H}.

    Matches Both AAC (`select_chunk_size`) and DUAC (`prefix_mean_jump`):
      U_bar_h = (1/h) * sum_{j=1}^h U_j   for h = 1..H
      dU_h    = U_bar_{h+1} - U_bar_h     for h = 1..H-1
      best_idx = argmax_{0 <= i < H-1} dU_{i+1} + 1   (1-indexed horizon)
      h_unc^*  = max(best_idx, min_horizon)
    """
    H = len(u_step)
    if H <= min_horizon:
        u_prefix = [float(x) for x in u_step]
        return max(H, 1), u_prefix, [0.0] * max(H - 1, 0)

    u_prefix: List[float] = []
    running_sum = 0.0
    for idx, val in enumerate(u_step):
        running_sum += float(val)
        u_prefix.append(running_sum / float(idx + 1))

    u_diff: List[float] = []
    best_i = 0
    best_diff = -float("inf")
    for i in range(H - 1):
        diff = u_prefix[i + 1] - u_prefix[i]
        u_diff.append(diff)
        if diff > best_diff:
            best_diff = diff
            best_i = i

    # In 0-indexed diff array `u_diff = np.diff(chunk_e)`, `np.argmax(u_diff) + 1`
    # gives the 1-indexed horizon `h` where the jump from `h` to `h+1` is maximal.
    h_unc = max(best_i + 1, min_horizon)
    return h_unc, u_prefix, u_diff


def select_duac_horizon(
    u_step: Sequence[float],
    actions: Sequence[Sequence[float]],
    config: DUACConfig,
    normalized_pos_rot: Optional[Sequence[Sequence[float]]] = None,
    rng: Optional[random.Random] = None,
) -> DUACDecisionResult:
    """Select final execution horizon h^* under DUAC, Fixed-k, or Random-Horizon control."""
    H = len(actions)
    m_step = action_magnitude_pi05(
        actions,
        w_pos=config.w_pos,
        w_rot=config.w_rot,
        w_grip=config.w_grip,
        normalized_pos_rot=normalized_pos_rot,
    )
    xi = find_first_idx(m_step, alpha=config.alpha, min_horizon=config.min_horizon)
    h_unc, u_prefix, u_diff = prefix_mean_jump_horizon(u_step, min_horizon=config.min_horizon)

    if config.fixed_horizon is not None:
        h_sel = max(1, min(int(config.fixed_horizon), H))
        mode = f"fixed_{h_sel}"
    elif config.random_horizon_pool:
        chooser = rng if rng is not None else random
        sampled = int(chooser.choice(config.random_horizon_pool))
        h_sel = max(1, min(sampled, H))
        mode = "random_matched"
    else:
        h_sel = max(xi, h_unc)
        h_sel = max(config.min_horizon, min(h_sel, H))
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


class OpenPI05DUACRunner:
    """Wraps an OpenPI pi0.5 JAX policy to compute both actions and DUAC divergence U_i(tau).

    During the 10-step Euler ODE integration from t=1.0 down to t=0.1 (dt = -0.1):
      - Prefix tokens (images + language) are embedded once and cached in `kv_cache`.
      - At the step corresponding to tau = 0.9 (t = 1.0 - tau = 0.1, step index 9 of 0..9),
        we evaluate K = 40 Rademacher probes eps_k in {-1, +1}^{H x D} using `jax.jvp`
        through the suffix action expert only (`embed_suffix` -> `PaliGemma.llm` -> `action_out_proj`).
      - For each step i in 0..H-1:
          U_i(tau) = (1/K) * sum_{k=1}^K || J_i(a_{n,tau}) eps_k ||_2
        (In forward-mode JVP with independent per-step Rademacher probes, `jax.jvp(v_fn, (x_t,), (eps_k,))`
        computes `J * eps_k`, whose per-step norm is an unbiased Hutchinson trace-norm estimator
        matching DUAC Eq. 3.)
    """

    def __init__(
        self,
        policy: Any,
        duac_config: Optional[DUACConfig] = None,
    ) -> None:
        self.policy = policy
        self.config = duac_config or DUACConfig()
        self._compiled_sample_and_duac = None

    def infer_with_duac(
        self,
        obs: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run pi0.5 inference and return unnormalized actions, normalized actions, and DUAC U_i(tau)."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        from openpi.models import model as _model
        from openpi.models.pi0 import make_attn_mask

        model = self.policy._model
        inputs = jax.tree.map(lambda x: np.asarray(x), obs)
        inputs = self.policy._input_transform(inputs)
        batched_inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        rng = jax.random.key(seed if seed is not None else self.config.seed)
        rng_sample, rng_probes = jax.random.split(rng)

        observation = _model.Observation.from_dict(batched_inputs)
        observation = _model.preprocess_observation(None, observation, train=False)

        num_steps = 10
        dt = -1.0 / float(num_steps)
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng_sample, (batch_size, model.action_horizon, model.action_dim))

        # 1. Embed prefix once and build KV cache (independent of action chunk x_t)
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        target_t = 1.0 - float(self.config.tau)  # tau=0.9 -> t=0.1

        def velocity_fn(x_t_single: Any, time_scalar: Any) -> Any:
            """Suffix velocity field v_theta(x_t) = -u_t(x_t) for batch=1."""
            x_b = x_t_single[None, ...]
            t_b = jnp.broadcast_to(time_scalar, (1,))
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
                observation, x_b, t_b
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = jnp.repeat(prefix_mask, suffix_tokens.shape[1], axis=1)[:, None, :]
            # Combine masks as in pi0.py:
            full_attn_mask = jnp.concatenate(
                [jnp.broadcast_to(prefix_mask[:, None, :], (1, suffix_tokens.shape[1], prefix_mask.shape[1])), suffix_attn_mask],
                axis=-1,
            )
            pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = model.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=pos,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            u_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
            # DUAC convention: v_theta = -u_t
            return -u_t[0]

        x_t = noise
        u_step_at_tau = None

        for step_idx in range(num_steps):
            t_val = 1.0 + step_idx * dt
            t_jnp = jnp.asarray(t_val, dtype=jnp.float32)
            v_theta = velocity_fn(x_t[0], t_jnp)
            u_t = -v_theta[None, ...]

            if abs(t_val - target_t) < 1e-4 or (step_idx == num_steps - 1 and u_step_at_tau is None):
                # Evaluate K Rademacher JVP probes at tau = 1 - t_val
                K = int(self.config.num_probes)
                probes = jax.random.rademacher(
                    rng_probes, (K, model.action_horizon, model.action_dim), dtype=x_t.dtype
                )

                def single_probe_jvp(eps_k: Any) -> Any:
                    _, jvp_out = jax.jvp(lambda x_in: velocity_fn(x_in, t_jnp), (x_t[0],), (eps_k,))
                    return jvp_out

                jvp_all = jax.vmap(single_probe_jvp)(probes)  # (K, action_horizon, action_dim)
                H_eval = min(int(self.config.chunk_horizon), int(model.action_horizon))
                D_eval = min(int(self.config.action_dim_eval), int(model.action_dim))
                jvp_slice = jvp_all[:, :H_eval, :D_eval]
                norms = jnp.linalg.norm(jvp_slice, axis=-1)  # (K, H_eval)
                u_step_at_tau = jnp.mean(norms, axis=0)  # (H_eval,)

            x_t = x_t + dt * u_t

        normalized_actions = np.asarray(x_t[0])
        out_dict = {
            "state": np.asarray(batched_inputs["state"][0]),
            "actions": normalized_actions,
        }
        unnormalized = self.policy._output_transform(out_dict)
        actions_unnorm = np.asarray(unnormalized["actions"])
        H_eval = min(int(self.config.chunk_horizon), actions_unnorm.shape[0])

        u_step_list = (
            [float(x) for x in np.asarray(u_step_at_tau)[:H_eval]]
            if u_step_at_tau is not None
            else [0.0] * H_eval
        )

        decision = select_duac_horizon(
            u_step=u_step_list,
            actions=actions_unnorm[:H_eval].tolist(),
            config=self.config,
            normalized_pos_rot=normalized_actions[:H_eval, :6].tolist(),
        )

        return {
            "actions": actions_unnorm[:H_eval],
            "normalized_action": normalized_actions[:H_eval],
            "u_step": np.asarray(decision.u_step, dtype=np.float32),
            "u_prefix": np.asarray(decision.u_prefix, dtype=np.float32),
            "m_step": np.asarray(decision.m_step, dtype=np.float32),
            "h_selected": int(decision.h_selected),
            "h_unc": int(decision.h_unc),
            "xi": int(decision.xi),
            "tau": float(decision.tau),
            "num_probes": int(decision.num_probes),
            "alpha": float(decision.alpha),
        }
