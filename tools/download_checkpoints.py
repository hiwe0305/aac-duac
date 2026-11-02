#!/usr/bin/env python3
"""Download and verify O-LONG pi0.5 LIBERO checkpoint & PaliGemma tokenizer (`tools/download_checkpoints.py`).

Supported targets:
  - pi05_libero (or `all`):
      1. Model checkpoint:
         Source: gs://openpi-assets/checkpoints/pi05_libero
                 (via https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero)
         Dest:   checkpoints/openpi/openpi-assets/checkpoints/pi05_libero
                 (also creates symlink at checkpoints/openpi/pi05_libero)
      2. PaliGemma SentencePiece tokenizer (required by OpenPI PaligemmaTokenizer):
         Source: gs://big_vision/paligemma_tokenizer.model
                 (via https://storage.googleapis.com/big_vision/paligemma_tokenizer.model)
         Dest:   checkpoints/openpi/big_vision/paligemma_tokenizer.model

Features:
  - Pure-Python HTTP Range resume (supports aria2c/curl if installed).
  - Exact byte-size verification against official manifest sizes.
  - `--metadata-only` flag to pull config/norm_stats/manifest/tokenizer files
    for dry-run and `duac.py doctor` verification before pulling full weights.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINTS_DIR = WORKSPACE_ROOT / "checkpoints"

GCS_OPENPI_BASE = "https://storage.googleapis.com/openpi-assets"
PALIGEMMA_TOKENIZER_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"

# Exact file manifest from gs://openpi-assets/checkpoints/pi05_libero (16 files, 11.58 GiB)
PI05_LIBERO_FILES: List[Tuple[Optional[int], str, bool]] = [
    (1914, "assets/physical-intelligence/libero/norm_stats.json", True),
    (23493, "params/_METADATA", True),
    (17952, "params/_sharding", True),
    (9062, "params/array_metadatas/process_0", True),
    (42307, "params/d/98a77a52a8eb845ae4830eb7fe983979", True),
    (120, "params/manifest.ocdbt", True),
    (322, "params/ocdbt.process_0/manifest.ocdbt", True),
    (1077, "params/ocdbt.process_0/d/2b6985f48e9da86f68627a7608c5bc25", True),
    (217, "params/ocdbt.process_0/d/bc613ff288a162563e622d01bb60622a", True),
    (2926522, "params/ocdbt.process_0/d/bda87d9791f23df771cd2d15293780cc", False),
    (2449323387, "params/ocdbt.process_0/d/0eaaecefaa9720d30a32cc56e65fd345", False),
    (2307885530, "params/ocdbt.process_0/d/155391c1cbf93a1be16266de052e5b48", False),
    (2150232827, "params/ocdbt.process_0/d/475fab3ee8821662585a8cde3eb32e22", False),
    (1608436908, "params/ocdbt.process_0/d/6c54da5a6f62c20a09c0f3f8c3329e00", False),
    (2240080987, "params/ocdbt.process_0/d/896bf93c5cf2e8ddd274a6ea0a2feec0", False),
    (1680102856, "params/ocdbt.process_0/d/efbb46173882cb35ed41ffe0c2db8a5e", False),
]


def format_bytes(num_bytes: int) -> str:
    val = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024.0 or unit == "TB":
            return f"{val:.2f} {unit}"
        val /= 1024.0
    return f"{num_bytes} B"


def download_file_with_resume(
    url: str,
    dest_path: Path,
    expected_size: Optional[int] = None,
    use_external_downloader: bool = True,
) -> None:
    """Download a single file with HTTP Range resume and exact byte-size verification."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        cur_sz = dest_path.stat().st_size
        if (expected_size is not None and cur_sz == expected_size) or (expected_size is None and cur_sz > 0):
            print(f"  [SKIP] {dest_path.name} ({format_bytes(cur_sz)}) already complete.")
            return

    part_path = dest_path.with_suffix(dest_path.suffix + ".part")

    if use_external_downloader and (expected_size or 0) > 50 * 1024 * 1024:
        aria2c = shutil.which("aria2c")
        curl = shutil.which("curl")
        if aria2c:
            cmd = [
                aria2c,
                "--continue=true",
                "--max-connection-per-server=8",
                "--split=8",
                "--min-split-size=10M",
                "--dir",
                str(part_path.parent),
                "--out",
                part_path.name,
                url,
            ]
            print(f"  [ARIA2C] {dest_path.name} ({format_bytes(expected_size or 0)})")
            res = subprocess.run(cmd, check=False)
            if res.returncode == 0 and part_path.exists():
                if expected_size is None or part_path.stat().st_size == expected_size:
                    part_path.replace(dest_path)
                    return
        elif curl:
            cmd = [
                curl,
                "-L",
                "-C",
                "-",
                "--fail",
                "--retry",
                "5",
                "--retry-delay",
                "2",
                "-o",
                str(part_path),
                url,
            ]
            print(f"  [CURL] {dest_path.name} ({format_bytes(expected_size or 0)})")
            res = subprocess.run(cmd, check=False)
            if res.returncode == 0 and part_path.exists():
                if expected_size is None or part_path.stat().st_size == expected_size:
                    part_path.replace(dest_path)
                    return

    existing_size = part_path.stat().st_size if part_path.exists() else 0
    headers = {"User-Agent": "aac-duac-pi05-downloader/1.0"}
    mode = "wb"
    if existing_size > 0 and (expected_size is None or existing_size < expected_size):
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"
    elif existing_size > 0 and expected_size is not None and existing_size >= expected_size:
        part_path.unlink(missing_ok=True)
        existing_size = 0

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = getattr(resp, "status", 200)
            if status == 200 and mode == "ab":
                mode = "wb"
                existing_size = 0
            total = expected_size
            if total is None:
                cl = resp.headers.get("Content-Length")
                if cl is not None:
                    total = existing_size + int(cl)

            downloaded = existing_size
            chunk_size = 4 * 1024 * 1024
            with open(part_path, mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total and total > 10 * 1024 * 1024:
                        pct = (downloaded / total) * 100.0
                        print(
                            f"\r  [GET] {dest_path.name}: {format_bytes(downloaded)} / {format_bytes(total)} ({pct:.1f}%)",
                            end="",
                            flush=True,
                        )
            if total and total > 10 * 1024 * 1024:
                print()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP error {exc.code} downloading {url}: {exc.reason}") from exc

    actual_size = part_path.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {dest_path}: expected {expected_size} bytes, got {actual_size} bytes"
        )
    part_path.replace(dest_path)
    print(
        f"  [OK] {dest_path.relative_to(WORKSPACE_ROOT) if dest_path.is_relative_to(WORKSPACE_ROOT) else dest_path} ({format_bytes(actual_size)})"
    )


def ensure_symlink(target: Path, link_path: Path) -> None:
    """Create or update a relative symlink so both AAC and DUAC paths resolve cleanly."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink():
            try:
                if link_path.resolve() == target.resolve():
                    return
            except OSError:
                pass
            link_path.unlink()
        else:
            return
    rel_target = os.path.relpath(target, start=link_path.parent)
    link_path.symlink_to(rel_target)


def download_pi05_libero(
    checkpoints_dir: Path,
    metadata_only: bool = False,
    dry_run: bool = False,
) -> Path:
    """Download OpenPI pi05_libero checkpoint and PaliGemma tokenizer."""
    dest_root = checkpoints_dir / "openpi" / "openpi-assets" / "checkpoints" / "pi05_libero"
    tokenizer_dest = checkpoints_dir / "openpi" / "big_vision" / "paligemma_tokenizer.model"
    files = [f for f in PI05_LIBERO_FILES if (not metadata_only or f[2])]
    total_bytes = sum(f[0] or 0 for f in files)
    print(
        f"\n=== [OpenPI JAX] pi05_libero ({len(files)} files, {format_bytes(total_bytes)}) ===\n"
        f"Destination: {dest_root}"
    )
    if dry_run:
        for size_b, rel_path, _ in files:
            print(
                f"  [DRY-RUN] {GCS_OPENPI_BASE}/checkpoints/pi05_libero/{rel_path} -> {dest_root / rel_path} ({format_bytes(size_b or 0)})"
            )
        print(f"  [DRY-RUN] {PALIGEMMA_TOKENIZER_URL} -> {tokenizer_dest}")
        return dest_root

    for size_b, rel_path, _ in files:
        url = f"{GCS_OPENPI_BASE}/checkpoints/pi05_libero/{rel_path}"
        download_file_with_resume(url, dest_root / rel_path, expected_size=size_b)

    download_file_with_resume(PALIGEMMA_TOKENIZER_URL, tokenizer_dest, expected_size=None)
    ensure_symlink(dest_root, checkpoints_dir / "openpi" / "pi05_libero")
    return dest_root


def verify_checkpoints(checkpoints_dir: Path) -> Dict[str, Dict[str, object]]:
    """Inspect local pi0.5 checkpoint directory and return completeness status."""
    root = checkpoints_dir / "openpi" / "openpi-assets" / "checkpoints" / "pi05_libero"
    tokenizer_path = checkpoints_dir / "openpi" / "big_vision" / "paligemma_tokenizer.model"

    present_meta = 0
    total_meta = sum(1 for _, _, is_m in PI05_LIBERO_FILES if is_m)
    present_weights = 0
    total_weights = sum(1 for _, _, is_m in PI05_LIBERO_FILES if not is_m)
    bytes_present = 0
    bytes_expected = sum(sz or 0 for sz, _, _ in PI05_LIBERO_FILES)

    for sz, rel_path, is_m in PI05_LIBERO_FILES:
        fp = root / rel_path
        if fp.exists() and ((sz is None and fp.stat().st_size > 0) or fp.stat().st_size == sz):
            bytes_present += fp.stat().st_size
            if is_m:
                present_meta += 1
            else:
                present_weights += 1

    return {
        "pi05_libero": {
            "path": str(root),
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_ready": tokenizer_path.exists() and tokenizer_path.stat().st_size > 0,
            "metadata_ready": present_meta == total_meta,
            "weights_ready": (present_meta == total_meta and present_weights == total_weights),
            "metadata_files": f"{present_meta}/{total_meta}",
            "weight_files": f"{present_weights}/{total_weights}",
            "bytes_present": bytes_present,
            "bytes_expected": bytes_expected,
            "progress": f"{format_bytes(bytes_present)} / {format_bytes(bytes_expected)}",
        }
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify O-LONG pi0.5 checkpoint (pi05_libero + paligemma_tokenizer.model)."
    )
    parser.add_argument(
        "--target",
        choices=["pi05_libero", "all"],
        default="pi05_libero",
        help="Checkpoint target to download (default: pi05_libero).",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=DEFAULT_CHECKPOINTS_DIR,
        help=f"Root directory for checkpoints (default: {DEFAULT_CHECKPOINTS_DIR}).",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only metadata/config/norm_stats/manifest + tokenizer files to verify layout.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only check and print local checkpoint verification status without downloading.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URLs and destination paths without downloading.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    checkpoints_dir = args.checkpoints_dir.resolve()

    if args.verify:
        report = verify_checkpoints(checkpoints_dir)
        print(json.dumps(report, indent=2))
        return 0

    download_pi05_libero(checkpoints_dir, metadata_only=args.metadata_only, dry_run=args.dry_run)

    print("\n--- pi0.5 Checkpoint Status Summary ---")
    report = verify_checkpoints(checkpoints_dir)
    for name, info in report.items():
        print(
            f"  {name:18s}: metadata_ready={info['metadata_ready']} "
            f"tokenizer_ready={info['tokenizer_ready']} "
            f"weights_ready={info['weights_ready']} ({info['progress']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
