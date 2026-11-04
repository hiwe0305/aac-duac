#!/usr/bin/env python3
"""DUAC CLI for pi0.5 + LIBERO / LIBERO-PRO (O-LONG, docs/task3010.md).

  doctor          workspace, LIBERO / LIBERO-PRO data and checkpoint status
  test            equations vs. the official AAC code (and the hactt13 DUAC reference when present),
                  plus a synthetic end-to-end run of the real evaluation loop
  serve-pi05      OpenPI pi0.5 websocket server returning one chunk + DUAC uncertainty (JIT, stateful RNG)
  eval-libero     evaluate on clean LIBERO (external/LIBERO)
  eval-liberopro  evaluate through LIBERO-PRO (required for libero_*_with_*; also usable for clean suites)
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import List, Optional

if "XDG_CACHE_HOME" not in os.environ:
    _cache = Path.home() / ".cache" / "duac"
    try:
        _cache.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(_cache)
    except OSError:
        os.environ["XDG_CACHE_HOME"] = f"/tmp/duac_{getpass.getuser()}"

FLOWBASED_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = FLOWBASED_ROOT.parent
for _p in (FLOWBASED_ROOT, WORKSPACE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from flowbased_aac.duac_policy import (  # noqa: E402
    DUACConfig,
    action_magnitude_pi05,
    action_uncertainty,
    find_first_idx,
    movement_lower_bound,
    prefix_mean_jump_horizon,
    select_duac_horizon,
)
from flowbased_aac.eval_libero import (  # noqa: E402
    LIBERO_ROOTS,
    evaluate_libero_subgoals,
    max_steps_for_suite,
    run_libero_evaluation,
    suite_step_budget_name,
)

HACTT13_DUAC_CORE = Path("/mnt/ssd-storage/hactt13/UQ_action_chunking/AAC_reproduce/flowbased_aac/flowbased_aac/core.py")


def cmd_doctor(args: argparse.Namespace) -> int:
    from tools.download_checkpoints import verify_checkpoints
    from tools.download_libero import configure_and_verify_benchmark

    checks = {
        "workspace_root": str(WORKSPACE_ROOT),
        "xdg_cache_home": os.environ.get("XDG_CACHE_HOME", ""),
        "openpi_repo": (WORKSPACE_ROOT / "external" / "openpi").exists(),
        "aac_libero_repo (official AAC)": (WORKSPACE_ROOT / "external" / "aac-libero").exists(),
        "libero": {k: {"source": str(s), "exists": s.exists(), "config": (c / "config.yaml").exists()}
                   for k, (s, c) in LIBERO_ROOTS.items()},
        "libero_10_benchmark_data": configure_and_verify_benchmark("libero_10"),
        "checkpoints": verify_checkpoints(WORKSPACE_ROOT / "checkpoints"),
    }
    print(json.dumps(checks, indent=2))
    return 0


def _load_module(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cmd_test(args: argparse.Namespace) -> int:
    import numpy as np

    print("[duac.py test] O-LONG / DUAC / AAC verification")
    rng = np.random.default_rng(0)

    # Step 0.1
    assert suite_step_budget_name("libero_10_with_mug") == "libero_10"
    assert suite_step_budget_name("libero_object_temp_x0.2") == "libero_object"
    assert max_steps_for_suite("libero_10") == 520 and max_steps_for_suite("libero_10_with_milk") == 520
    print("  [PASS] step 0.1: libero_10_with_* -> libero_10 budget (520)")

    # Step 0.2
    class Inner:
        parsed_problem = {"goal_state": [["turnon", "stove"], ["on", "moka_pot", "stove_region"]]}

        def _eval_predicate(self, predicate):
            return predicate[0] == "turnon"

    assert evaluate_libero_subgoals(type("W", (), {"env": Inner()})()) == (2, 1, "10")
    print("  [PASS] step 0.2: goal predicates evaluated one by one")

    # xi == official AAC action_magnitude + find_first_idx
    sys.path.insert(0, str(WORKSPACE_ROOT / "external" / "aac-libero"))
    from action_optimization import action_entropy_pi05 as aac

    for _ in range(2000):
        a = np.zeros((10, 7))
        a[:, :3] = rng.uniform(-0.7, 0.7, 3) + rng.normal(0, 0.15, (10, 3))
        a[:, 3:6] = rng.normal(0, 0.3, (10, 3))
        a[:, 6] = np.where(rng.random(10) < 0.15, 1.0, -1.0)
        official = {"action.end_effector_position": a[None, :, :3], "action.end_effector_rotation": a[None, :, 3:6],
                    "action.gripper_close": aac.convert_gripper_to_binary(a[None, :, 6])}
        chunk_m = aac.action_magnitude(official)
        assert np.allclose(action_magnitude_pi05(a)[1:], chunk_m), "m_h differs from official AAC"
        for th in (0.5, 1.5, 3.0):
            assert movement_lower_bound(a, threshold=th) == aac.find_first_idx(chunk_m, th)
    print("  [PASS] xi(alpha) identical to official AAC action_magnitude/find_first_idx (2000 chunks x 3 alphas)")

    # Equation (8) == official AAC elbow rule
    for _ in range(2000):
        u = rng.normal(size=10)
        diffs = np.diff([np.mean(u[:k]) for k in range(1, 11)])
        assert prefix_mean_jump_horizon(u, 2)[0] == max(int(np.argmax(diffs)) + 1, 2)
    print("  [PASS] prefix-mean jump horizon identical to official AAC select_chunk_size")

    # Equation (6) and the DUAC reference (hactt13), when readable on this server
    u = action_uncertainty([0.0, -10.0], tau=0.9, action_dim=7)
    assert np.allclose(u, [(0.1 ** 2 / 0.9) * 7, (0.1 ** 2 / 0.9) * (7 - 1.0)])
    if HACTT13_DUAC_CORE.exists():
        ref = _load_module("duac_reference_core", HACTT13_DUAC_CORE)
        for _ in range(500):
            a = rng.normal(0, 0.4, (10, 7))
            d = rng.normal(size=10)
            assert ref.movement_lower_bound(a, threshold=3.0) == movement_lower_bound(a, threshold=3.0)
            assert ref.select_horizon(d, min_horizon=2) == prefix_mean_jump_horizon(d, 2)[0]
            assert np.allclose(ref.action_uncertainty(d, tau=0.9, action_dim=7), action_uncertainty(d, tau=0.9, action_dim=7))
        print("  [PASS] equation (6), (8) and xi identical to the hactt13 DUAC reference")
    else:
        print(f"  [SKIP] DUAC reference not found at {HACTT13_DUAC_CORE}")

    # Hutchinson estimator on a linear field: E[div] = trace of each diagonal block
    try:
        import jax
        import jax.numpy as jnp
        from flowbased_aac.duac_policy import linearized_hutchinson_action_divergence

        m = jnp.asarray(rng.normal(size=(10 * 8, 10 * 8)), dtype=jnp.float32)
        field = lambda x: (m @ x.reshape(-1)).reshape(10, 8)  # noqa: E731
        _, div = linearized_hutchinson_action_divergence(field, jnp.zeros((10, 8)), jax.random.key(0),
                                                         num_probes=100000, valid_action_dim=7)
        blocks = np.asarray(m).reshape(10, 8, 10, 8)
        expected = [np.trace(blocks[i, :7, i, :7]) for i in range(10)]
        assert np.allclose(np.asarray(div), expected, atol=0.6), (np.asarray(div), expected)
        print("  [PASS] Hutchinson block-trace estimator (JAX) matches exact traces, padded dims masked")
    except ImportError:
        print("  [SKIP] JAX not installed in this interpreter (run with .venv-openpi to test the estimator)")

    # Arms
    actions = np.tile([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], (10, 1))
    u_jump = [0.10, 0.11, 0.12, 0.12, 0.65, 0.70, 0.72, 0.74, 0.75, 0.76]
    res = select_duac_horizon(u_jump, actions, DUACConfig())
    assert (res.h_unc, res.xi, res.h_selected) == (4, 7, 7), res  # 7 * 0.5 = 3.5 > 3 first at h = 7
    assert select_duac_horizon(u_jump, actions, DUACConfig(fixed_horizon=5)).h_selected == 5
    print("  [PASS] arms: duac = max(h_unc, xi), fixed-k = k")

    # Synthetic end-to-end run of the real loop: every arm, state bank, resume
    with tempfile.TemporaryDirectory(prefix="duac_test_") as tmp:
        tmp = Path(tmp)
        common = dict(suite="libero_10", num_trials=2, task_ids=[0, 1], synthetic_dry_run=True)
        s = run_libero_evaluation(output_dir=tmp / "duac", state_bank=True, **common)
        assert s["total_episodes"] == 4
        for npz in sorted((tmp / "duac" / "state_bank").glob("*.npz")):
            assert npz.read_bytes()[:2] == b"PK"
            bank = np.load(npz)
            assert len(bank["states"]) == len(bank["actions"]) + 1 and bank["chunks"].shape[1:] == (10, 7)
        assert run_libero_evaluation(output_dir=tmp / "f5", fixed_horizon=5, **common)["mean_h"] == 5
        rh = run_libero_evaluation(output_dir=tmp / "rh", random_horizon_from=tmp / "duac" / "decisions.csv", **common)
        assert rh["arm"] == "random_h"
        aac_s = run_libero_evaluation(output_dir=tmp / "aac", aac_selector="gaussian_bernoulli", **common)
        assert aac_s["arm"] == "aac" and aac_s["total_episodes"] == 4
        # resume: drop one finished episode and the decisions after it, then resume
        ep = (tmp / "f5" / "episodes.csv").read_text().splitlines()
        (tmp / "f5" / "episodes.csv").write_text("\n".join(ep[:-1]) + "\n")
        r = run_libero_evaluation(resume_dir=tmp / "f5", fixed_horizon=5, **common)
        assert r["total_episodes"] == 4
        import csv as _csv

        rows = list(_csv.DictReader(open(tmp / "f5" / "decisions.csv")))
        keys = [(r_["task_id"], r_["trial_id"], r_["decision_id"]) for r_ in rows]
        assert len(keys) == len(set(keys)), "duplicated decisions after resume"
        try:
            run_libero_evaluation(output_dir=tmp / "x", num_trials=60, suite="libero_10", task_ids=[0],
                                  synthetic_dry_run=True)
            raise AssertionError("expected error for too few initial states")
        except ValueError:
            pass
    print("  [PASS] synthetic run of the real loop: duac / fixed / random_h / aac, state bank, resume, init-state check")
    print("[duac.py test] ALL TESTS PASSED.")
    return 0


def cmd_serve_pi05(args: argparse.Namespace) -> int:
    for p in (WORKSPACE_ROOT / "external" / "openpi" / "src",
              WORKSPACE_ROOT / "external" / "openpi" / "packages" / "openpi-client" / "src"):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config

    from flowbased_aac.duac_policy import DuacPolicy

    ckpt_dir = Path(args.checkpoint_dir).resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_dir}")
    config = DUACConfig(tau=args.tau, num_probes=args.num_probes, min_horizon=args.min_horizon,
                        valid_action_dim=args.valid_action_dim, alpha=args.alpha)
    base = _policy_config.create_trained_policy(_config.get_config(args.config), str(ckpt_dir))
    policy = DuacPolicy(base, config, seed=args.seed, seed_per_step=args.seed_per_step)
    print(f"[duac.py serve-pi05] {args.config} from {ckpt_dir} on {args.host}:{args.port} "
          f"(tau={args.tau}, K={args.num_probes}, seed={args.seed}, seed_per_step={args.seed_per_step})", flush=True)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=policy.metadata
    ).serve_forever()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    summary = run_libero_evaluation(
        suite=args.suite,
        host=args.host,
        port=args.port,
        num_trials=args.num_trials,
        seed=args.seed,
        alpha=args.alpha,
        tau=args.tau,
        num_probes=args.num_probes,
        fixed_horizon=args.fixed_horizon,
        random_horizon_from=Path(args.random_horizon_from) if args.random_horizon_from else None,
        state_bank=args.state_bank,
        save_videos=args.save_videos,
        video_trials=args.video_trials,
        output_dir=Path(args.output_dir),
        resume_dir=Path(args.resume_dir) if args.resume_dir else None,
        task_ids=[int(x) for x in args.task_ids.split(",")] if args.task_ids else None,
        trial_ids=parse_ids(args.trial_ids),
        force_libero_pro=args.subcommand == "eval-liberopro",
        synthetic_dry_run=args.synthetic_dry_run,
    )
    print(json.dumps(summary, indent=2))
    return 0


def parse_ids(text):
    """'0,3,5' or '36-49' (inclusive) or a mix -> sorted list of ints; None if empty."""
    if not text:
        return None
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return sorted(set(out))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DUAC CLI for pi0.5 + LIBERO (O-LONG).")
    sub = parser.add_subparsers(dest="subcommand", required=True)
    sub.add_parser("doctor")
    sub.add_parser("test")

    p = sub.add_parser("serve-pi05", help="OpenPI pi0.5 server with DUAC uncertainty")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--config", default="pi05_libero")
    p.add_argument("--checkpoint-dir", default=str(WORKSPACE_ROOT / "checkpoints" / "openpi" / "openpi-assets"
                                                   / "checkpoints" / "pi05_libero"))
    p.add_argument("--tau", type=float, default=0.9)
    p.add_argument("--num-probes", type=int, default=40)
    p.add_argument("--min-horizon", type=int, default=2)
    p.add_argument("--valid-action-dim", type=int, default=7)
    p.add_argument("--alpha", type=float, default=3.0, help="logged only; xi is applied by the evaluator")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed-per-step", action="store_true",
                   help="reset the RNG at every call (NOT the O-LONG protocol, which uses a stateful RNG)")

    for name in ("eval-libero", "eval-liberopro"):
        e = sub.add_parser(name)
        e.add_argument("--suite", default="libero_10")
        e.add_argument("--host", default="127.0.0.1")
        e.add_argument("--port", type=int, default=8000)
        e.add_argument("--num-trials", "--trials", dest="num_trials", type=int, default=50)
        e.add_argument("--seed", type=int, default=7)
        e.add_argument("--alpha", "--move-th", dest="alpha", type=float, default=3.0)
        e.add_argument("--tau", type=float, default=0.9, help="logged in args.json (the server applies tau)")
        e.add_argument("--num-probes", type=int, default=40, help="logged in args.json")
        e.add_argument("--fixed-horizon", type=int, default=None)
        e.add_argument("--random-horizon-from", default=None)
        e.add_argument("--state-bank", action=argparse.BooleanOptionalAction, default=False)
        e.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=False)
        e.add_argument("--video-trials", type=int, default=2, help="record videos for trials 0..N-1 of each task")
        e.add_argument("--output-dir", "--out-dir", dest="output_dir", default="outputs/A/libero_10/duac")
        e.add_argument("--resume-dir", default=None)
        e.add_argument("--task-ids", default=None, help="comma-separated, e.g. 0,1")
        e.add_argument("--trial-ids", default=None, help="subset of trials, e.g. 36-49 or 0,5,7")
        e.add_argument("--synthetic-dry-run", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if len(raw) >= 2 and raw[0] == "eval" and raw[1] in ("libero", "liberopro"):
        raw = [f"eval-{raw[1]}", *raw[2:]]
    args = build_parser().parse_args(raw)
    if args.subcommand == "doctor":
        return cmd_doctor(args)
    if args.subcommand == "test":
        return cmd_test(args)
    if args.subcommand == "serve-pi05":
        return cmd_serve_pi05(args)
    return cmd_eval(args)


if __name__ == "__main__":
    sys.exit(main())
