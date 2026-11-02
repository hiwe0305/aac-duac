#!/usr/bin/env python3
"""Setup, verify, and download LIBERO benchmark data & datasets (`tools/download_libero.py`).

What LIBERO needs for O-LONG (`pi0.5` on `libero_10`):
  1. Benchmark Simulation Data (REQUIRED for evaluation):
     - `external/LIBERO/libero/libero/bddl_files/libero_10/*.bddl` (10 long-horizon tasks)
     - `external/LIBERO/libero/libero/init_files/libero_10/*.pruned_init` (50 initial states x 10 tasks)
     - `external/LIBERO/libero/libero/assets/` (MuJoCo 3D meshes, textures, scenes, robot models)
     - `.libero/config.yaml` pointing to the active workspace's `external/LIBERO/libero/libero`
  2. Demonstration Datasets (OPTIONAL, only if inspecting offline demos in `data/libero/`):
     - Downloaded via `--download-demos` from the official LIBERO dataset release.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.request
from typing import Any, Dict, List, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
LIBERO_BENCHMARK_ROOT = WORKSPACE_ROOT / "external" / "LIBERO" / "libero" / "libero"
LIBERO_CONFIG_DIR = WORKSPACE_ROOT / ".libero"
DATA_LIBERO_DIR = WORKSPACE_ROOT / "data" / "libero"

SUITE_EXPECTED_TASKS = {
    "libero_10": 10,
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_90": 90,
}

HF_DEMO_URLS = {
    "libero_10": "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/libero_10.zip",
    "libero_spatial": "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/libero_spatial.zip",
    "libero_object": "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/libero_object.zip",
    "libero_goal": "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/libero_goal.zip",
    "libero_90": "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/libero_90.zip",
}


def configure_and_verify_benchmark(suite: str = "libero_10") -> Dict[str, Any]:
    """Write portable `.libero/config.yaml` and verify all BDDL, init_files, and 3D assets exist."""
    LIBERO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_LIBERO_DIR.mkdir(parents=True, exist_ok=True)

    bddl_root = LIBERO_BENCHMARK_ROOT / "bddl_files"
    init_root = LIBERO_BENCHMARK_ROOT / "init_files"
    assets_root = LIBERO_BENCHMARK_ROOT / "assets"

    config_yaml = LIBERO_CONFIG_DIR / "config.yaml"
    config_text = (
        f"benchmark_root: {LIBERO_BENCHMARK_ROOT}\n"
        f"bddl_files: {bddl_root}\n"
        f"init_states: {init_root}\n"
        f"datasets: {DATA_LIBERO_DIR}\n"
        f"assets: {assets_root}\n"
    )
    config_yaml.write_text(config_text, encoding="utf-8")

    suite_bddl_dir = bddl_root / suite
    suite_init_dir = init_root / suite
    bddl_files = sorted(suite_bddl_dir.glob("*.bddl")) if suite_bddl_dir.exists() else []
    pruned_inits = sorted(suite_init_dir.glob("*.pruned_init")) if suite_init_dir.exists() else []
    xml_assets = list(assets_root.rglob("*.xml")) if assets_root.exists() else []

    expected = SUITE_EXPECTED_TASKS.get(suite, 10)
    ready = (
        len(bddl_files) == expected
        and len(pruned_inits) == expected
        and len(xml_assets) > 0
    )

    return {
        "suite": suite,
        "benchmark_ready": ready,
        "config_yaml": str(config_yaml),
        "bddl_files_found": f"{len(bddl_files)}/{expected}",
        "init_files_found": f"{len(pruned_inits)}/{expected}",
        "mujoco_xml_assets": len(xml_assets),
        "datasets_dir": str(DATA_LIBERO_DIR),
    }


def download_demo_zip(suite: str, target_dir: Path) -> Path:
    """Download official LIBERO HDF5 demonstration zip into `data/libero/` with HTTP Range resume."""
    if suite not in HF_DEMO_URLS:
        raise ValueError(f"Unknown LIBERO suite: {suite}")
    url = HF_DEMO_URLS[suite]
    target_dir.mkdir(parents=True, exist_ok=True)
    dest_zip = target_dir / f"{suite}.zip"
    part_zip = target_dir / f"{suite}.zip.part"

    if dest_zip.exists() and dest_zip.stat().st_size > 0:
        print(f"[skip] {dest_zip} already exists ({dest_zip.stat().st_size} bytes)")
        return dest_zip

    existing = part_zip.stat().st_size if part_zip.exists() else 0
    headers = {"User-Agent": "aac-duac-downloader/1.0"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        print(f"[resume] {suite}.zip from byte {existing}")
    else:
        print(f"[download] {url} -> {dest_zip}")

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        mode = "ab" if (existing > 0 and getattr(resp, "status", 200) == 206) else "wb"
        with open(part_zip, mode) as f:
            while True:
                chunk = resp.read(4 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    part_zip.replace(dest_zip)
    return dest_zip


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Configure and verify LIBERO benchmark data (bddl_files, init_files, assets) & optional demo datasets."
    )
    parser.add_argument(
        "--suite",
        default="libero_10",
        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal", "libero_90"],
        help="LIBERO task suite to verify/prepare (default: libero_10).",
    )
    parser.add_argument(
        "--download-demos",
        action="store_true",
        help="Also download the optional HDF5 demonstration zip into data/libero/ (not required for evaluation).",
    )
    args = parser.parse_args(argv)

    report = configure_and_verify_benchmark(args.suite)
    if args.download_demos:
        zip_path = download_demo_zip(args.suite, DATA_LIBERO_DIR)
        report["demo_zip"] = str(zip_path)

    print(json.dumps(report, indent=2))
    return 0 if report["benchmark_ready"] else 1


if __name__ == "__main__":
    sys.exit(main())

