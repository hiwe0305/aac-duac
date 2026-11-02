#!/usr/bin/env python3
"""Multi-sample OpenPI pi0.5 WebSocket policy server for Adaptive Action Chunking (AAC).

Serves N=20 stochastic flow-matching action chunks per observation:
  - Returns `actions` of shape (N, H, 7) (unnormalized)
  - Returns `normalized_action` of shape (N, H, 7) (normalized, used by AAC's
    `gaussian_entropy_from_samples` and `action_magnitude` on `action.pos` / `action.rot`)
  - Supports `--seed 0 --no_seed_per_step` so the JAX PRNGKey advances across
    decision steps instead of resetting every step (O-LONG Section 3.1 / 7).
"""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

if "XDG_CACHE_HOME" not in os.environ:
    default_cache = Path.home() / ".cache" / "duac"
    try:
        default_cache.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(default_cache)
    except OSError:
        os.environ["XDG_CACHE_HOME"] = f"/tmp/duac_{getpass.getuser()}"

AAC_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = AAC_ROOT.parent
for p in (
    WORKSPACE_ROOT,
    WORKSPACE_ROOT / "external" / "openpi" / "src",
    WORKSPACE_ROOT / "external" / "openpi" / "packages" / "openpi-client" / "src",
):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


class MultiSamplePi05Policy:
    """Wraps an OpenPI trained policy to sample `n_samples` chunks per observation."""

    def __init__(
        self,
        base_policy: Any,
        n_samples: int = 20,
        seed: int = 0,
        seed_per_step: bool = False,
        action_horizon_out: int = 10,
    ) -> None:
        import jax

        self.policy = base_policy
        self.n_samples = int(n_samples)
        self.seed = int(seed)
        self.seed_per_step = bool(seed_per_step)
        self.action_horizon_out = int(action_horizon_out)
        self._rng = jax.random.key(self.seed)
        self._sample_actions_vmapped = jax.jit(
            jax.vmap(self.policy._model.sample_actions, in_axes=(0, None))
        )

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        import jax
        import jax.numpy as jnp
        import numpy as np
        from openpi.models import model as _model

        inputs = jax.tree.map(lambda x: np.asarray(x), obs)
        inputs = self.policy._input_transform(inputs)
        batched_inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(batched_inputs)

        if self.seed_per_step:
            step_rng = jax.random.key(self.seed)
        else:
            self._rng, step_rng = jax.random.split(self._rng)

        sample_rngs = jax.random.split(step_rng, self.n_samples)
        # Vmap across N sample keys -> shape (N, 1, model_horizon, model_action_dim)
        raw_samples = self._sample_actions_vmapped(sample_rngs, observation)
        norm_samples = np.asarray(raw_samples[:, 0, :, :])  # (N, model_horizon, model_action_dim)

        state_np = np.asarray(batched_inputs["state"][0])
        unnorm_list = []
        for idx in range(self.n_samples):
            out_i = self.policy._output_transform(
                {
                    "state": state_np,
                    "actions": norm_samples[idx],
                }
            )
            unnorm_list.append(np.asarray(out_i["actions"]))

        unnorm_arr = np.stack(unnorm_list, axis=0)  # (N, H, 7)
        H_out = min(self.action_horizon_out, unnorm_arr.shape[1])
        return {
            "actions": unnorm_arr[:, :H_out, :7],
            "normalized_action": norm_samples[:, :H_out, :7],
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        meta = dict(getattr(self.policy, "metadata", {}) or {})
        meta.update(
            {
                "n_samples": self.n_samples,
                "seed": self.seed,
                "seed_per_step": self.seed_per_step,
            }
        )
        return meta


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve multi-sample OpenPI pi0.5 for AAC.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8015)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument(
        "--checkpoint_dir",
        default=str(
            WORKSPACE_ROOT / "checkpoints" / "openpi" / "openpi-assets" / "checkpoints" / "pi05_libero"
        ),
    )
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no_seed_per_step",
        action="store_true",
        help="Advance PRNGKey across decision steps instead of resetting every call.",
    )
    parser.add_argument("--seed_per_step", action="store_true", help="Reset PRNGKey every call.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config

    ckpt_dir = Path(args.checkpoint_dir).resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_dir}\n"
            f"Run: python3 tools/download_checkpoints.py --target pi05_libero"
        )

    seed_per_step = bool(args.seed_per_step) and not bool(args.no_seed_per_step)
    train_cfg = _config.get_config(args.config)
    base_pol = _policy_config.create_trained_policy(train_cfg, str(ckpt_dir))
    multisample_pol = MultiSamplePi05Policy(
        base_pol,
        n_samples=args.n_samples,
        seed=args.seed,
        seed_per_step=seed_per_step,
    )
    print(
        f"[serve_pi05_multisample] Serving {args.config} (N={args.n_samples}, "
        f"seed={args.seed}, seed_per_step={seed_per_step}) on {args.host}:{args.port}"
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=multisample_pol,
        host=args.host,
        port=args.port,
        metadata=multisample_pol.metadata,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())

