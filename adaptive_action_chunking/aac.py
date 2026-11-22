#!/usr/bin/env python3
"""AAC CLI for pi0.5 + LIBERO / LIBERO-PRO (O-LONG, docs/task3010.md).

The horizon rule is the official AAC release (external/aac-libero, see
flowbased_aac/flowbased_aac/aac_official.py); the episode loop, logging, subgoals and state bank
are shared with DUAC (flowbased_aac/flowbased_aac/eval_libero.py), so every arm uses the same
protocol: OffScreenRenderEnv, 10 wait steps + 520 policy steps on libero_10, env seed 7.

  serve-pi05      multi-sample OpenPI pi0.5 server (N chunks per call, stateful RNG by default)
  eval-liberopro  evaluate through LIBERO-PRO (required for libero_*_with_*, recommended for clean)
  eval-libero     evaluate on clean LIBERO (external/LIBERO)
  test            synthetic run of the AAC arm through the real loop
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

AAC_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = AAC_ROOT.parent
for _p in (AAC_ROOT, WORKSPACE_ROOT, WORKSPACE_ROOT / "flowbased_aac"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from flowbased_aac.eval_libero import run_libero_evaluation  # noqa: E402


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
    parser = argparse.ArgumentParser(description="AAC CLI for pi0.5 + LIBERO (O-LONG).")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p = sub.add_parser("serve-pi05", help="multi-sample OpenPI pi0.5 server for AAC")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8015)
    p.add_argument("--config", default="pi05_libero")
    p.add_argument("--checkpoint_dir", "--checkpoint-dir", dest="checkpoint_dir",
                   default=str(WORKSPACE_ROOT / "checkpoints" / "openpi" / "openpi-assets" / "checkpoints" / "pi05_libero"))
    p.add_argument("--n_samples", "--n-samples", dest="n_samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--no_seed_per_step", "--no-seed-per-step", dest="seed_per_step", action="store_false",
                       help="stateful RNG across calls (default, O-LONG protocol)")
    group.add_argument("--seed_per_step", "--seed-per-step", dest="seed_per_step", action="store_true",
                       help="reset the RNG every call: every decision then reuses the same noise (wrong for O-LONG)")
    p.set_defaults(seed_per_step=False)

    for name in ("eval-liberopro", "eval-libero"):
        e = sub.add_parser(name, help="evaluate AAC")
        e.add_argument("--backbone", default="pi05", choices=["pi05"])
        e.add_argument("--suite", default="libero_10")
        e.add_argument("--selector", default="gaussian_bernoulli", help="gaussian_bernoulli (AAC) or fixed_<k>")
        e.add_argument("--trials", "--num-trials", dest="trials", type=int, default=50)
        e.add_argument("--seed", type=int, default=7)
        e.add_argument("--move_th", "--move-th", dest="move_th", type=float, default=3.0)
        e.add_argument("--videos", "--save-videos", dest="save_videos", action="store_true")
        e.add_argument("--no-videos", "--no-save-videos", dest="save_videos", action="store_false")
        e.set_defaults(save_videos=False)
        e.add_argument("--video-trials", type=int, default=2)
        e.add_argument("--state-bank", action=argparse.BooleanOptionalAction, default=False)
        e.add_argument("--host", default="127.0.0.1")
        e.add_argument("--port", type=int, default=8015)
        e.add_argument("--out_dir", "--out-dir", "--output-dir", dest="out_dir", default="outputs/A/libero_10/aac")
        e.add_argument("--resume-dir", default=None)
        e.add_argument("--official_openpi_protocol", "--official-openpi-protocol", action="store_true", default=True,
                       help="always on: OffScreenRenderEnv + max_steps policy steps after the wait steps")
        e.add_argument("--task_ids", "--task-ids", dest="task_ids", default=None)
        e.add_argument("--trial-ids", "--trial_ids", dest="trial_ids", default=None)
        e.add_argument("--synthetic-dry-run", action="store_true")

    sub.add_parser("test", help="synthetic run of the AAC arm")
    return parser


def cmd_test() -> int:
    with tempfile.TemporaryDirectory(prefix="aac_test_") as tmp:
        s = run_libero_evaluation(suite="libero_10", num_trials=2, task_ids=[0, 1], aac_selector="gaussian_bernoulli",
                                  output_dir=Path(tmp) / "aac", synthetic_dry_run=True)
        assert s["arm"] == "aac" and s["total_episodes"] == 4
        s = run_libero_evaluation(suite="libero_10", num_trials=1, task_ids=[0], aac_selector="fixed_8",
                                  output_dir=Path(tmp) / "f8", synthetic_dry_run=True)
        assert s["mean_h"] == 8
    print("[aac.py test] ALL TESTS PASSED.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if len(raw) >= 2 and raw[0] == "eval" and raw[1] in ("libero", "liberopro"):
        raw = [f"eval-{raw[1]}", *raw[2:]]
    args = build_parser().parse_args(raw)
    if args.subcommand == "test":
        return cmd_test()
    if args.subcommand == "serve-pi05":
        from tools.serve_pi05_multisample import main as serve_main

        fwd = ["--host", args.host, "--port", str(args.port), "--config", args.config,
               "--checkpoint_dir", args.checkpoint_dir, "--n_samples", str(args.n_samples), "--seed", str(args.seed)]
        fwd.append("--seed_per_step" if args.seed_per_step else "--no_seed_per_step")
        return serve_main(fwd)
    summary = run_libero_evaluation(
        suite=args.suite,
        host=args.host,
        port=args.port,
        num_trials=args.trials,
        seed=args.seed,
        alpha=args.move_th,
        aac_selector=args.selector,
        state_bank=args.state_bank,
        save_videos=args.save_videos,
        video_trials=args.video_trials,
        output_dir=Path(args.out_dir),
        resume_dir=Path(args.resume_dir) if args.resume_dir else None,
        task_ids=[int(x) for x in args.task_ids.split(",")] if args.task_ids else None,
        trial_ids=parse_ids(args.trial_ids),
        force_libero_pro=args.subcommand == "eval-liberopro",
        synthetic_dry_run=args.synthetic_dry_run,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
