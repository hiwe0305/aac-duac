#!/usr/bin/env python3
"""OpenPI pi0.5 websocket server returning N sampled action chunks per call (AAC backbone).

The AAC pi0.5 client (external/aac-libero/pi05_benchmark/inference_client_action_entropy_pi05.py) expects
  actions:           (N, H, 7)  un-normalized LIBERO actions, one row per sampled chunk
  normalized_action: (N, H, 32) the same chunks in the model's normalized action space
The AAC authors did not release this server.  This follows the hactt13 reproduction
(AAC_reproduce/adaptive_action_chunking/tools/serve_pi05_multisample.py): the VLM prefix is encoded
once and its KV cache is shared by N denoising runs, each with its own flow-matching noise.
Unlike that script, the default here is a stateful RNG (``--no_seed_per_step``), the O-LONG protocol.
"""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import sys
import time
import types
from typing import Any, Dict, List, Optional

if "XDG_CACHE_HOME" not in os.environ:
    _cache = Path.home() / ".cache" / "duac"
    try:
        _cache.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(_cache)
    except OSError:
        os.environ["XDG_CACHE_HOME"] = f"/tmp/duac_{getpass.getuser()}"

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
for _p in (WORKSPACE_ROOT / "external" / "openpi" / "src",
           WORKSPACE_ROOT / "external" / "openpi" / "packages" / "openpi-client" / "src"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def make_shared_prefix_sampler(n: int):
    """Pi0.sample_actions for a batch-1 observation, returning n chunks from one prefix forward pass."""

    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models import pi0 as _pi0

    def sample_actions(self, rng, observation, *, num_steps=10):
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        noise = jax.random.normal(rng, (n, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        # KV cache is (layers, batch, ...): share the single prefix across the n samples
        kv_cache = jax.tree.map(lambda x: jnp.broadcast_to(x, (x.shape[0], n, *x.shape[2:])), kv_cache)
        prefix_mask = jnp.broadcast_to(prefix_mask, (n, prefix_mask.shape[1]))
        observation = jax.tree.map(lambda x: jnp.broadcast_to(x, (n, *x.shape[1:])), observation)

        def step(carry):
            x_t, time_ = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time_, n)
            )
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = jnp.broadcast_to(prefix_mask[:, None, :], (n, suffix_tokens.shape[1], prefix_mask.shape[1]))
            full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
            pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=pos, kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time_ + dt

        x_0, _ = jax.lax.while_loop(lambda c: c[1] >= -dt / 2, step, (noise, 1.0))
        return x_0

    return sample_actions


class MultiSamplePi05Policy:
    def __init__(self, policy, n_samples: int = 20, seed: int = 0, seed_per_step: bool = False):
        import jax
        from openpi.shared import nnx_utils

        self._policy = policy
        self._n = int(n_samples)
        self._seed = int(seed)
        self._seed_per_step = bool(seed_per_step)
        self._rng = jax.random.key(self._seed)
        self._sample = nnx_utils.module_jit(types.MethodType(make_shared_prefix_sampler(self._n), policy._model))

    @property
    def metadata(self) -> Dict[str, Any]:
        meta = dict(self._policy.metadata)
        meta.update({"n_samples": self._n, "seed": self._seed, "seed_per_step": self._seed_per_step})
        return meta

    def infer(self, obs: dict) -> dict:
        import jax
        import jax.numpy as jnp
        import numpy as np
        from openpi.models import model as _model

        p = self._policy
        inputs = p._input_transform(jax.tree.map(lambda x: x, obs))
        state = np.asarray(inputs["state"])
        if self._seed_per_step:
            self._rng = jax.random.key(self._seed)
        self._rng, sample_rng = jax.random.split(self._rng)

        start = time.monotonic()
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
        actions = np.asarray(jax.block_until_ready(
            self._sample(sample_rng, _model.Observation.from_dict(inputs), **p._sample_kwargs)
        ))  # (N, H, 32), normalized
        infer_ms = (time.monotonic() - start) * 1000.0
        per_sample = [p._output_transform({"state": state, "actions": actions[i]})["actions"] for i in range(self._n)]
        return {
            "actions": np.stack(per_sample),
            "normalized_action": actions,
            "policy_timing": {"infer_ms": infer_ms},
        }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8015)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--checkpoint_dir", default=str(WORKSPACE_ROOT / "checkpoints" / "openpi" / "openpi-assets"
                                                        / "checkpoints" / "pi05_libero"))
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--no_seed_per_step", dest="seed_per_step", action="store_false")
    group.add_argument("--seed_per_step", dest="seed_per_step", action="store_true")
    parser.set_defaults(seed_per_step=False)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config

    ckpt_dir = Path(args.checkpoint_dir).resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_dir}")
    base = policy_config.create_trained_policy(_config.get_config(args.config), str(ckpt_dir))
    policy = MultiSamplePi05Policy(base, args.n_samples, args.seed, seed_per_step=args.seed_per_step)
    print(f"[serve_pi05_multisample] {args.config} N={args.n_samples} seed={args.seed} "
          f"seed_per_step={args.seed_per_step} on {args.host}:{args.port}", flush=True)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=policy.metadata
    ).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
