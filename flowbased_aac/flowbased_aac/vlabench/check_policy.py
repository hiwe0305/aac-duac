#!/usr/bin/env python3
"""Checkpoint / sampler validation and latency benchmark for VLABench/pi05-primitive-10task (.venv-openpi, GPU).

  1. Load: config (H, action dim, quantile norm), norm stats file, parameter load through OpenPI.
  2. I/O: vanilla ``Policy.infer`` on real VLABench observations -> (10, 7) absolute actions; the chunk's first
     target lies close to the current end-effector pose (DeltaActions/AbsoluteActions round trip).
  3. Normalization: un-normalizing ``normalized_action`` by hand from norm_stats.json (quantile, as OpenPI pi0.5) and
     adding the state back reproduces the server's absolute actions (no double (un)normalization).
  4. Equivalence: with the same flow-matching noise, the DUAC sampler's chunk and AAC sample 0 equal the vanilla chunk
     (the samplers only add uncertainty computation; they do not change the policy).
  5. Latency: steady-state per-call time of vanilla / DUAC / AAC after a warm-up (JIT) call, block_until_ready,
     plus the JAX allocator peak. Results -> --output (json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "flowbased_aac"))
from flowbased_aac.vlabench import serve as vserve  # noqa: E402  (sets sys.path for openpi)
from flowbased_aac.vlabench.common import policy_request  # noqa: E402


def stats(xs):
    xs = np.asarray(xs, dtype=np.float64)
    return {"mean": round(xs.mean(), 2), "median": round(float(np.median(xs)), 2), "p95": round(float(np.percentile(xs, 95)), 2),
            "std": round(xs.std(), 2), "n": int(xs.size)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(vserve.DEFAULT_CKPT))
    ap.add_argument("--obs", default=str(ROOT / "outputs" / "vlabench" / "validation" / "sample_obs.npz"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--output", default=str(ROOT / "outputs" / "vlabench" / "validation" / "check_policy.json"))
    args = ap.parse_args(argv)

    import jax

    from flowbased_aac.duac_policy import DUACConfig, DuacPolicy
    from serve_pi05_multisample import MultiSamplePi05Policy

    report = {"device": str(jax.devices()[0]), "jax": jax.__version__}
    t0 = time.perf_counter()
    base = vserve.load_base_policy(args.checkpoint, seed=0)
    report["load_s"] = round(time.perf_counter() - t0, 1)
    model = base._model
    H = int(model.action_horizon)
    report["model"] = {"action_horizon": H, "action_dim": int(model.action_dim), "pi05": bool(getattr(model, "pi05", True)),
                       "max_token_len": int(model.max_token_len), "sample_kwargs": dict(base._sample_kwargs)}
    norm_file = Path(args.checkpoint) / "assets" / "vlabench" / "vlabench_ft_primitive" / "norm_stats.json"
    ns = json.loads(norm_file.read_text())["norm_stats"]
    q01, q99 = np.asarray(ns["actions"]["q01"][:7]), np.asarray(ns["actions"]["q99"][:7])

    data = np.load(args.obs)
    tasks = sorted(k[: -len("_rgb")] for k in data.files if k.endswith("_rgb"))
    duac = DuacPolicy(base, DUACConfig(chunk_horizon=H, tau=0.9, num_probes=40, valid_action_dim=7), seed=0)
    aac = MultiSamplePi05Policy(base, args.n_samples, 0, seed_per_step=False)
    checks = {}
    for task in tasks:
        req = policy_request(data[f"{task}_rgb"], data[f"{task}_ee_state"], str(data[f"{task}_instruction"]))
        state = req["observation/state"]
        out_v = base.infer(dict(req))
        a_v = np.asarray(out_v["actions"])
        assert a_v.shape == (H, 7), a_v.shape
        # DUAC: reproduce its noise draw, then run vanilla with that noise
        key = jax.random.key(123)
        duac._rng = key
        _, sample_rng = jax.random.split(key)
        noise_rng, _ = jax.random.split(sample_rng)
        noise = np.asarray(jax.random.normal(noise_rng, (1, H, int(model.action_dim))))
        out_d = duac.infer(dict(req))
        ref_d = np.asarray(base.infer(dict(req), noise=noise[0])["actions"])
        # AAC: sample 0 of the multi-sample draw
        aac._rng = key
        _, sample_rng = jax.random.split(key)
        noise_n = np.asarray(jax.random.normal(sample_rng, (args.n_samples, H, int(model.action_dim))))
        out_a = aac.infer(dict(req))
        ref_a = np.asarray(base.infer(dict(req), noise=noise_n[0])["actions"])
        # manual un-normalization (quantile) + AbsoluteActions on dims 0:6
        nz = np.asarray(out_d["normalized_action"])[:, :7]
        manual = (nz + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        manual[:, :6] += state[:6]
        half_range = (q99 - q01) / 2.0  # 1.0 in the model's (quantile) normalized space
        def ndiff(a, b):  # per-dim max |a - b| in normalized units
            return np.round(np.abs(np.asarray(a) - np.asarray(b)).max(0) / half_range, 5).tolist()
        other = np.asarray(base.infer(dict(req))["actions"])  # a different noise draw: scale of sampling spread
        checks[task] = {
            "duac_vs_vanilla_same_noise_norm_per_dim": ndiff(out_d["actions"], ref_d),
            "aac_sample0_vs_vanilla_same_noise_norm_per_dim": ndiff(np.asarray(out_a["actions"])[0], ref_a),
            "vanilla_vs_vanilla_other_noise_norm_per_dim": ndiff(other, ref_d),
            "instruction": str(data[f"{task}_instruction"]),
            "state": np.round(state, 4).tolist(),
            "vanilla_first_action": np.round(a_v[0], 4).tolist(),
            "first_target_minus_state_pos_m": float(np.linalg.norm(a_v[0, :3] - state[:3])),
            "gripper_values": np.round(a_v[:, 6], 3).tolist(),
            "duac_vs_vanilla_same_noise_max_abs": float(np.abs(np.asarray(out_d["actions"]) - ref_d).max()),
            "aac_sample0_vs_vanilla_same_noise_max_abs": float(np.abs(np.asarray(out_a["actions"])[0] - ref_a).max()),
            "manual_unnorm_vs_server_max_abs": float(np.abs(manual - np.asarray(out_d["actions"])).max()),
            "duac_u_step": np.round(np.asarray(out_d["u_step"]), 6).tolist(),
            "duac_divergence": np.round(np.asarray(out_d["duac_divergence"]), 3).tolist(),
            "duac_h_unc": int(out_d["duac_horizon"]),
            "aac_shapes": [list(np.asarray(out_a["actions"]).shape), list(np.asarray(out_a["normalized_action"]).shape)],
            "aac_sample_spread_pos_m": float(np.asarray(out_a["actions"])[:, :, :3].std(0).mean()),
        }
        print(json.dumps({task: checks[task]}, indent=1), flush=True)
    report["checks"] = checks

    # Latency (steady state). Warm-up calls above already compiled every sampler.
    req = policy_request(data[f"{tasks[0]}_rgb"], data[f"{tasks[0]}_ee_state"], str(data[f"{tasks[0]}_instruction"]))
    lat = {}
    for name, pol in (("vanilla", base), ("duac", duac), ("aac", aac)):
        pol.infer(dict(req))
        model_ms, call_ms = [], []
        for _ in range(args.reps):
            t = time.perf_counter()
            out = pol.infer(dict(req))
            call_ms.append(1000 * (time.perf_counter() - t))
            model_ms.append(float(out["policy_timing"]["infer_ms"]))
        lat[name] = {"model_ms": stats(model_ms), "call_ms": stats(call_ms)}
        print(name, lat[name], flush=True)
    lat["duac_overhead_vs_vanilla_ms"] = round(lat["duac"]["call_ms"]["median"] - lat["vanilla"]["call_ms"]["median"], 2)
    lat["aac_overhead_vs_vanilla_ms"] = round(lat["aac"]["call_ms"]["median"] - lat["vanilla"]["call_ms"]["median"], 2)
    report["latency"] = lat
    report["peak_gpu_mb"] = vserve.peak_gpu_mb()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    # Same-noise equivalence is judged in normalized units. The bf16 model compiled for different batch shapes
    # (1 vs N, separate jit graphs) differs by ~0.2-1.2% of the half-range on every dim (bf16 eps = 0.4%), versus
    # 3-200% between two noise draws; 2.5% (~6 bf16 eps) separates numerics from a real sampler mismatch.
    ok = all(max(c["duac_vs_vanilla_same_noise_norm_per_dim"]) < 2.5e-2
             and max(c["aac_sample0_vs_vanilla_same_noise_norm_per_dim"]) < 2.5e-2
             and c["manual_unnorm_vs_server_max_abs"] < 1e-4 for c in checks.values())
    print(f"[check_policy] {'PASS' if ok else 'FAIL'} -> {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
