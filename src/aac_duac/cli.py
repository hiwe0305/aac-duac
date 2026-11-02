#!/usr/bin/env python3
"""Unified CLI (`PYTHONPATH=src python3 -m aac_duac.cli`) for the pi0.5 O-LONG pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[2]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Unified O-LONG CLI for pi0.5 (DUAC vs AAC on LIBERO & RoboCasa)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Check workspace, OpenPI/LIBERO repos, and pi0.5 checkpoints.")
    sub.add_parser("test", help="Run all unit, schema, and statistical tests.")

    dl = sub.add_parser(
        "download-checkpoints",
        help="Download and verify pi0.5 LIBERO checkpoint & PaliGemma tokenizer.",
    )
    dl.add_argument(
        "--target",
        default="pi05_libero",
        choices=["pi05_libero", "all"],
    )
    dl.add_argument("--metadata-only", action="store_true")
    dl.add_argument("--verify", action="store_true")
    dl.add_argument("--dry-run", action="store_true")

    dl_libero = sub.add_parser(
        "download-libero",
        help="Configure and verify LIBERO benchmark data (bddl_files, init_files, assets) & optional demo datasets.",
    )
    dl_libero.add_argument(
        "--suite",
        default="libero_10",
        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal", "libero_90"],
    )
    dl_libero.add_argument("--download-demos", action="store_true")

    p_duac = sub.add_parser("duac", help="Forward arguments to flowbased_aac/duac.py.")
    p_duac.add_argument("duac_args", nargs=argparse.REMAINDER)

    p_aac = sub.add_parser("aac", help="Forward arguments to adaptive_action_chunking/aac.py.")
    p_aac.add_argument("aac_args", nargs=argparse.REMAINDER)

    p_an = sub.add_parser("analyze", help="Convert/validate runs and generate O-LONG T1-T6 report.")
    p_an.add_argument("--manifest", default=str(ROOT / "manifest.json"))
    p_an.add_argument("--output-dir", default=str(ROOT / "outputs" / "analysis_report"))

    args = parser.parse_args(argv)

    if args.command == "doctor":
        return subprocess.call([sys.executable, str(ROOT / "flowbased_aac" / "duac.py"), "doctor"])

    if args.command == "test":
        for cmd in (
            [sys.executable, str(ROOT / "flowbased_aac" / "duac.py"), "test"],
            [sys.executable, str(ROOT / "adaptive_action_chunking" / "aac.py"), "test"],
            [sys.executable, str(ROOT / "analysis" / "convert.py"), "--self-test"],
            [sys.executable, str(ROOT / "analysis" / "analyze_o_long.py"), "--self-test"],
        ):
            rc = subprocess.call(cmd)
            if rc != 0:
                return rc
        return 0

    if args.command == "download-checkpoints":
        cmd = [
            sys.executable,
            str(ROOT / "tools" / "download_checkpoints.py"),
            "--target",
            args.target,
        ]
        if args.metadata_only:
            cmd.append("--metadata-only")
        if args.verify:
            cmd.append("--verify")
        if args.dry_run:
            cmd.append("--dry-run")
        return subprocess.call(cmd)

    if args.command == "download-libero":
        cmd = [
            sys.executable,
            str(ROOT / "tools" / "download_libero.py"),
            "--suite",
            args.suite,
        ]
        if args.download_demos:
            cmd.append("--download-demos")
        return subprocess.call(cmd)

    if args.command == "duac":
        fwd = [a for a in args.duac_args if a != "--"]
        return subprocess.call([sys.executable, str(ROOT / "flowbased_aac" / "duac.py"), *fwd])

    if args.command == "aac":
        fwd = [a for a in args.aac_args if a != "--"]
        return subprocess.call([sys.executable, str(ROOT / "adaptive_action_chunking" / "aac.py"), *fwd])

    if args.command == "analyze":
        rc1 = subprocess.call(
            [sys.executable, str(ROOT / "analysis" / "convert.py"), "--manifest", args.manifest]
        )
        if rc1 != 0:
            return rc1
        return subprocess.call(
            [
                sys.executable,
                str(ROOT / "analysis" / "analyze_o_long.py"),
                "--manifest",
                args.manifest,
                "--output-dir",
                args.output_dir,
            ]
        )

    return 1


if __name__ == "__main__":
    sys.exit(main())
