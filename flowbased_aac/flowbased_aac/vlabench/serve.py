#!/usr/bin/env python3
"""OpenPI websocket server for VLABench/pi05-primitive-10task (JAX, .venv-openpi), with the O-LONG samplers.

  --mode vanilla : plain OpenPI ``Policy.infer`` (the official serving path of the checkpoint), one chunk per call.
  --mode duac    : ``flowbased_aac.duac_policy.DuacPolicy`` (same JAX sampler as the LIBERO track): one chunk plus
                   the DUAC profile U (eq. 6, tau = 0.9, K = 40 Hutchinson probes on the 7 real action dims).
                   Also serves the Fixed-k and Random-h arms so DUAC's proposal is always logged.
  --mode aac     : ``MultiSamplePi05Policy`` from adaptive_action_chunking/tools/serve_pi05_multisample.py:
                   N = 20 chunks sharing one prefix encoding (AAC backbone).

Request : the official client keys (``observation/image``, ``observation/second_image``, ``observation/wrist_image``,
          ``observation/state`` (7,), ``prompt``); images at native 480x480, resized with padding to 224 by the model
          transforms, exactly as in the official pipeline.
Response: actions (H, 7) or (N, H, 7) ABSOLUTE targets (AbsoluteActions adds the state back), normalized_action
          (model space, delta w.r.t. the state), DUAC: u_step / duac_divergence / duac_horizon;
          policy_timing: infer_ms (model only, block_until_ready), server_ms (incl. input/output transforms),
          peak_gpu_mb (JAX allocator peak since start).
The RNG is the policy's single stateful JAX key (seed 0), as in the O-LONG protocol.
"""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import sys
import time

if "XDG_CACHE_HOME" not in os.environ:
    _cache = Path.home() / ".cache" / "duac"
    try:
        _cache.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(_cache)
    except OSError:
        os.environ["XDG_CACHE_HOME"] = f"/tmp/duac_{getpass.getuser()}"

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "flowbased_aac", ROOT / "adaptive_action_chunking" / "tools", ROOT / "external" / "openpi" / "src",
           ROOT / "external" / "openpi" / "packages" / "openpi-client" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_CKPT = ROOT / "checkpoints" / "vlabench" / "pi05-primitive-10task"


def peak_gpu_mb() -> float:
    import jax

    try:
        stats = jax.devices()[0].memory_stats() or {}
        return round(stats.get("peak_bytes_in_use", 0) / 2**20, 1)
    except Exception:  # CPU backend
        return 0.0


def load_base_policy(checkpoint: str | Path, seed: int = 0):
    import jax
    from openpi.policies import policy_config as _policy_config

    from flowbased_aac.vlabench.openpi_config import get_train_config

    ckpt = Path(checkpoint).resolve()
    if not (ckpt / "params").exists():
        raise FileNotFoundError(f"{ckpt}/params not found (see docs/VLABENCH_INTEGRATION.md for the download)")
    base = _policy_config.create_trained_policy(get_train_config(), str(ckpt))
    base._rng = jax.random.key(seed)  # Policy's stateful key (OpenPI default key(0))
    return base


class TimedPolicy:
    """Adds server-side wall time and the GPU peak to any OpenPI-style policy's response."""

    def __init__(self, policy, mode: str, extra_metadata: dict):
        self._policy, self._mode, self._extra = policy, mode, extra_metadata

    @property
    def metadata(self):
        meta = dict(self._policy.metadata)
        meta.update(self._extra)
        return meta

    def infer(self, obs: dict) -> dict:
        started = time.perf_counter()
        out = dict(self._policy.infer(obs))
        timing = dict(out.get("policy_timing") or {})
        timing["server_ms"] = 1000.0 * (time.perf_counter() - started)
        timing["peak_gpu_mb"] = peak_gpu_mb()
        out["policy_timing"] = timing
        out["server_mode"] = self._mode
        return out


def build_policy(mode: str, checkpoint, *, seed: int = 0, tau: float = 0.9, num_probes: int = 40, n_samples: int = 20,
                 n_anchor: int = 0, anchor_sigma: float = 0.3):
    base = load_base_policy(checkpoint, seed)
    meta = {"benchmark": "vlabench", "mode": mode, "checkpoint": str(Path(checkpoint).resolve()), "seed": seed,
            "action_horizon": int(base._model.action_horizon), "action_dim": 7}
    if mode == "vanilla":
        return TimedPolicy(base, mode, meta)
    if mode == "duac":
        from flowbased_aac.duac_policy import DUACConfig, DuacPolicy

        cfg = DUACConfig(chunk_horizon=int(base._model.action_horizon), tau=tau, num_probes=num_probes, valid_action_dim=7)
        meta.update(tau=tau, num_probes=num_probes)
        return TimedPolicy(DuacPolicy(base, cfg, seed=seed, seed_per_step=False), mode, meta)
    if mode == "aac":
        from serve_pi05_multisample import MultiSamplePi05Policy

        meta.update(n_samples=n_samples, n_anchor=n_anchor, anchor_sigma=anchor_sigma)
        return TimedPolicy(MultiSamplePi05Policy(base, n_samples, seed, seed_per_step=False, n_anchor=n_anchor,
                                                 anchor_sigma=anchor_sigma), mode, meta)
    raise ValueError(f"unknown mode {mode!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["vanilla", "duac", "aac"], required=True)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--num-probes", type=int, default=40)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--n-anchor", type=int, default=0, help="aac mode: extra anchored samples (ccs.select_acs)")
    ap.add_argument("--anchor-sigma", type=float, default=0.3)
    args = ap.parse_args(argv)

    from openpi.serving import websocket_policy_server

    policy = build_policy(args.mode, args.checkpoint, seed=args.seed, tau=args.tau, num_probes=args.num_probes,
                          n_samples=args.n_samples, n_anchor=args.n_anchor, anchor_sigma=args.anchor_sigma)
    print(f"[vlabench serve] mode={args.mode} on {args.host}:{args.port} server listening", flush=True)
    websocket_policy_server.WebsocketPolicyServer(policy=policy, host=args.host, port=args.port,
                                                  metadata=policy.metadata).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
