#!/usr/bin/env python3
"""DUAC CLI (`duac.py`) for pi0.5 + LIBERO / RoboCasa under the O-LONG specification.

Subcommands:
  - doctor:        Verify workspace layout, OpenPI/LIBERO repos, and checkpoint status.
  - test:          Run unit & integration tests (Steps 0.1-0.4, DUAC math, CSV/NPZ schemas).
  - serve-pi05:    Launch OpenPI pi0.5 WebSocket policy server with DUAC JVP divergence U_i(tau).
  - eval-libero:   Evaluate DUAC / Fixed-k / Random-Horizon on clean LIBERO suites (default: libero_10).
  - eval-liberopro: Alias for eval-libero supporting both clean (`libero_10`) and perturbed suites.
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

# Ensure per-user XDG_CACHE_HOME to prevent /tmp/openpi permission clashes (O-LONG Section 7)
if "XDG_CACHE_HOME" not in os.environ:
    default_cache = Path.home() / ".cache" / "duac"
    try:
        default_cache.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(default_cache)
    except OSError:
        os.environ["XDG_CACHE_HOME"] = f"/tmp/duac_{getpass.getuser()}"

FLOWBASED_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = FLOWBASED_ROOT.parent
if str(FLOWBASED_ROOT) not in sys.path:
    sys.path.insert(0, str(FLOWBASED_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from flowbased_aac.duac_policy import (  # noqa: E402
    DUACConfig,
    OpenPI05DUACRunner,
    action_magnitude_pi05,
    find_first_idx,
    prefix_mean_jump_horizon,
    select_duac_horizon,
)
from flowbased_aac.eval_libero import (  # noqa: E402
    DECISIONS_CSV_COLUMNS,
    EPISODES_CSV_COLUMNS,
    SUBGOALS_CSV_COLUMNS,
    evaluate_libero_subgoals,
    max_steps_for_suite,
    run_libero_evaluation,
    suite_step_budget_name,
)


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check environment, cloned repos, LIBERO benchmark data, and checkpoint readiness."""
    from tools.download_checkpoints import verify_checkpoints
    from tools.download_libero import configure_and_verify_benchmark

    checkpoints_dir = WORKSPACE_ROOT / "checkpoints"
    ckpt_report = verify_checkpoints(checkpoints_dir)
    libero_report = configure_and_verify_benchmark("libero_10")

    checks = {
        "workspace_root": str(WORKSPACE_ROOT),
        "xdg_cache_home": os.environ.get("XDG_CACHE_HOME", ""),
        "openpi_repo": (WORKSPACE_ROOT / "external" / "openpi").exists(),
        "libero_repo": (WORKSPACE_ROOT / "external" / "LIBERO").exists(),
        "aac_libero_repo": (WORKSPACE_ROOT / "external" / "aac-libero").exists(),
        "libero_benchmark_data": libero_report,
        "checkpoints": ckpt_report,
    }
    print(json.dumps(checks, indent=2))
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    """Run unit and integration tests for DUAC math and O-LONG Steps 0.1-0.4."""
    print("[duac.py test] Running O-LONG & DUAC verification suite...")

    # 1. Step 0.1: suite_step_budget_name() and max_steps_for_suite()
    assert suite_step_budget_name("libero_10") == "libero_10"
    assert suite_step_budget_name("libero_10_with_object") == "libero_10"
    assert suite_step_budget_name("libero_10_with_swap") == "libero_10"
    assert suite_step_budget_name("libero_spatial_with_position") == "libero_spatial"
    assert max_steps_for_suite("libero_10") == 520
    assert max_steps_for_suite("libero_10_with_object") == 520
    assert max_steps_for_suite("libero_spatial") == 220
    assert max_steps_for_suite("libero_object") == 280
    assert max_steps_for_suite("libero_goal") == 300
    assert max_steps_for_suite("libero_90") == 400
    print("  [PASS] Step 0.1: suite_step_budget_name() -> 520 active steps for libero_10")

    # 2. Step 0.2: BDDL subgoal predicate evaluation helper
    class DummyInnerEnv:
        parsed_problem = {
            "goal_state": [
                ["turnon", "flat_stove_1"],
                ["on", "moka_pot_1", "flat_stove_1_cook_region"],
            ]
        }

        def _eval_predicate(self, pred):
            return pred[0] == "turnon"

    class DummyWrapperEnv:
        env = DummyInnerEnv()

    n_sub, n_sat, bits = evaluate_libero_subgoals(DummyWrapperEnv())
    assert (n_sub, n_sat, bits) == (2, 1, "10"), f"Unexpected subgoal result: {(n_sub, n_sat, bits)}"
    print("  [PASS] Step 0.2: evaluate_libero_subgoals() -> (2, 1, '10')")

    # 3. DUAC prefix_mean_jump and action_magnitude math
    u_step = [0.10, 0.11, 0.12, 0.12, 0.65, 0.70, 0.72, 0.74, 0.75, 0.76]
    h_unc, u_prefix, u_diff = prefix_mean_jump_horizon(u_step, min_horizon=2)
    assert h_unc == 4, f"Expected biggest jump after h=4 (from step 4 to 5), got h_unc={h_unc}"

    actions = [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for _ in range(10)]
    m_step = action_magnitude_pi05(actions, w_pos=1.0, w_rot=1.0, w_grip=0.2)
    # At step idx 0: pos_norm=0.5, rot_norm=1.0 -> m_1 = 1.5
    # At step idx 3 (horizon 4): pos_norm=2.0, rot_norm=1.0 -> m_4 = 3.0 >= alpha=3.0
    xi = find_first_idx(m_step, alpha=3.0, min_horizon=2)
    assert xi == 4, f"Expected xi=4, got xi={xi}"

    cfg = DUACConfig(chunk_horizon=10, min_horizon=2, tau=0.9, num_probes=40, alpha=3.0)
    res = select_duac_horizon(u_step=u_step, actions=actions, config=cfg)
    assert res.h_selected == 4 and res.h_unc == 4 and res.xi == 4
    print("  [PASS] DUAC math: prefix_mean_jump + action_magnitude + select_duac_horizon")

    # 4. Step 0.3 & 0.4: End-to-end synthetic run producing episodes.csv, decisions.csv, subgoals.csv, state_bank/*.npz,
    #    followed by --random-horizon-from matched empirical control run.
    with tempfile.TemporaryDirectory(prefix="duac_test_") as tmpdir:
        tmp_path = Path(tmpdir)
        duac_out = tmp_path / "duac_run"
        summary_duac = run_libero_evaluation(
            suite="libero_10",
            num_trials=2,
            seed=7,
            alpha=3.0,
            tau=0.9,
            num_probes=40,
            state_bank=True,
            output_dir=duac_out,
            task_ids=[0, 1],
            synthetic_dry_run=True,
        )
        assert summary_duac["total_episodes"] == 4
        assert (duac_out / "episodes.csv").exists()
        assert (duac_out / "decisions.csv").exists()
        assert (duac_out / "subgoals.csv").exists()
        npz_files = sorted((duac_out / "state_bank").glob("*.npz"))
        assert len(npz_files) == 4
        for npz_p in npz_files:
            with open(npz_p, "rb") as f:
                magic = f.read(2)
            assert magic == b"PK", f"Expected PK zip magic in {npz_p}, got {magic!r}"

        rand_out = tmp_path / "rand_run"
        summary_rand = run_libero_evaluation(
            suite="libero_10",
            num_trials=2,
            seed=7,
            random_horizon_from=duac_out / "decisions.csv",
            output_dir=rand_out,
            task_ids=[0, 1],
            synthetic_dry_run=True,
        )
        assert summary_rand["method"] == "random_h"
        assert summary_rand["total_episodes"] == 4
    print("  [PASS] Step 0.3 & 0.4: CSV schemas, PK .npz state_bank, and --random-horizon-from verified.")
    print("[duac.py test] ALL TESTS PASSED.")
    return 0


def cmd_serve_pi05(args: argparse.Namespace) -> int:
    """Start OpenPI pi0.5 WebSocket server with DUAC JVP divergence uncertainty."""
    openpi_src = WORKSPACE_ROOT / "external" / "openpi" / "src"
    openpi_client_src = WORKSPACE_ROOT / "external" / "openpi" / "packages" / "openpi-client" / "src"
    for p in (openpi_src, openpi_client_src):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config
    from openpi_client import base_policy as _base_policy

    ckpt_dir = Path(args.checkpoint_dir).resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_dir}\n"
            f"Run: python3 tools/download_checkpoints.py --target pi05_libero"
        )

    train_cfg = _config.get_config(args.config)
    base_pol = _policy_config.create_trained_policy(train_cfg, str(ckpt_dir))
    duac_cfg = DUACConfig(
        chunk_horizon=args.chunk_horizon,
        min_horizon=2,
        tau=args.tau,
        num_probes=args.num_probes,
        alpha=args.alpha,
        seed=args.seed,
    )
    duac_runner = OpenPI05DUACRunner(base_pol, duac_cfg)

    class DUACWebsocketPolicy(_base_policy.BasePolicy):
        def __init__(self, runner: OpenPI05DUACRunner, compute_duac: bool = True) -> None:
            self._runner = runner
            self._compute_duac = compute_duac
            self._step_counter = 0

        def infer(self, obs: dict) -> dict:
            self._step_counter += 1
            if not self._compute_duac:
                return self._runner.policy.infer(obs)
            return self._runner.infer_with_duac(obs, seed=self._runner.config.seed + self._step_counter)

        @property
        def metadata(self) -> dict:
            meta = dict(getattr(self._runner.policy, "metadata", {}) or {})
            meta.update(
                {
                    "duac_enabled": not args.no_duac,
                    "tau": args.tau,
                    "num_probes": args.num_probes,
                    "alpha": args.alpha,
                }
            )
            return meta

    policy_wrapper = DUACWebsocketPolicy(duac_runner, compute_duac=not args.no_duac)
    print(
        f"[duac.py serve-pi05] Serving {args.config} from {ckpt_dir} on {args.host}:{args.port} "
        f"(DUAC={not args.no_duac}, tau={args.tau}, K={args.num_probes}, alpha={args.alpha})"
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy_wrapper,
        host=args.host,
        port=args.port,
        metadata=policy_wrapper.metadata,
    )
    server.serve_forever()
    return 0


def cmd_eval_libero(args: argparse.Namespace) -> int:
    """Run LIBERO evaluation for DUAC, Fixed-k, or Random-Horizon."""
    task_ids = [int(x) for x in args.task_ids.split(",")] if args.task_ids else None
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
        output_dir=Path(args.output_dir),
        resume_dir=Path(args.resume_dir) if args.resume_dir else None,
        task_ids=task_ids,
        synthetic_dry_run=args.synthetic_dry_run,
    )
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DUAC CLI for pi0.5 + LIBERO (O-LONG pipeline).")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("doctor", help="Check workspace, repositories, and checkpoints.")
    sub.add_parser("test", help="Run DUAC & O-LONG unit and integration tests.")

    p_serve = sub.add_parser("serve-pi05", help="Serve OpenPI pi0.5 with DUAC JVP divergence.")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--config", default="pi05_libero")
    p_serve.add_argument(
        "--checkpoint-dir",
        default=str(
            WORKSPACE_ROOT / "checkpoints" / "openpi" / "openpi-assets" / "checkpoints" / "pi05_libero"
        ),
    )
    p_serve.add_argument("--tau", type=float, default=0.9)
    p_serve.add_argument("--num-probes", type=int, default=40)
    p_serve.add_argument("--alpha", type=float, default=3.0)
    p_serve.add_argument("--chunk-horizon", type=int, default=10)
    p_serve.add_argument("--seed", type=int, default=0)
    p_serve.add_argument("--no-duac", action="store_true", help="Disable JVP probes (for fast Fixed-k serving).")

    for cmd_name in ("eval-libero", "eval-liberopro"):
        p_eval = sub.add_parser(cmd_name, help=f"Evaluate on LIBERO ({cmd_name}).")
        p_eval.add_argument("--suite", default="libero_10", help="LIBERO suite name (default: libero_10).")
        p_eval.add_argument("--host", default="127.0.0.1")
        p_eval.add_argument("--port", type=int, default=8000)
        p_eval.add_argument("--num-trials", "--trials", dest="num_trials", type=int, default=50)
        p_eval.add_argument("--seed", type=int, default=7)
        p_eval.add_argument("--alpha", type=float, default=3.0)
        p_eval.add_argument("--tau", type=float, default=0.9)
        p_eval.add_argument("--num-probes", type=int, default=40)
        p_eval.add_argument("--fixed-horizon", type=int, default=None)
        p_eval.add_argument("--random-horizon-from", default=None)
        p_eval.add_argument("--state-bank", action="store_true")
        p_eval.add_argument("--no-state-bank", dest="state_bank", action="store_false")
        p_eval.set_defaults(state_bank=False)
        p_eval.add_argument("--save-videos", action="store_true")
        p_eval.add_argument("--no-save-videos", dest="save_videos", action="store_false")
        p_eval.set_defaults(save_videos=False)
        p_eval.add_argument("--output-dir", "--out-dir", "--out_dir", dest="output_dir", default="outputs/A/libero_10/duac")
        p_eval.add_argument("--resume-dir", default=None)
        p_eval.add_argument("--task-ids", default=None, help="Optional comma-separated task IDs (e.g. '0,1').")
        p_eval.add_argument(
            "--synthetic-dry-run",
            action="store_true",
            help="Run synthetic dry-run without GPU/MuJoCo to verify logging & schemas.",
        )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if len(raw_argv) >= 2 and raw_argv[0] == "eval" and raw_argv[1] in ("libero", "liberopro"):
        raw_argv = [f"eval-{raw_argv[1]}", *raw_argv[2:]]
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    if args.subcommand == "doctor":
        return cmd_doctor(args)
    if args.subcommand == "test":
        return cmd_test(args)
    if args.subcommand == "serve-pi05":
        return cmd_serve_pi05(args)
    if args.subcommand in ("eval-libero", "eval-liberopro"):
        return cmd_eval_libero(args)
    raise ValueError(f"Unknown subcommand {args.subcommand}")


if __name__ == "__main__":
    sys.exit(main())

