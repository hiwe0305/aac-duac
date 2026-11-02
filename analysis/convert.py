#!/usr/bin/env python3
"""Convert raw AAC logs and validate O-LONG Section 4.2 run directories (`analysis/convert.py`).

Capabilities:
  1. Convert raw AAC `chunk_size_log.csv` + `results_log.csv` into standardized
     `episodes.csv` and `decisions.csv` if not already generated.
  2. Validate any run directory or `manifest.json` against O-LONG Section 4.2:
     - Verifies column headers of `episodes.csv`, `decisions.csv`, `subgoals.csv`
     - Verifies `state_bank/*.npz` files start with ZIP `PK` magic bytes (`50 4b`)
  3. `--self-test`: Runs an end-to-end synthetic check for O-LONG Step 0.4.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
FLOWBASED_ROOT = WORKSPACE_ROOT / "flowbased_aac"
for p in (WORKSPACE_ROOT, FLOWBASED_ROOT):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from flowbased_aac.eval_libero import (  # noqa: E402
    DECISIONS_CSV_COLUMNS,
    EPISODES_CSV_COLUMNS,
    SUBGOALS_CSV_COLUMNS,
)


def convert_raw_aac_dir(
    run_dir: Path,
    backbone: str = "pi05",
    suite: str = "libero_10",
    method: str = "aac",
    seed: int = 7,
    n_samples: int = 20,
) -> Dict[str, Any]:
    """Convert raw AAC `results_log.csv` and `chunk_size_log.csv` into `episodes.csv` and `decisions.csv`."""
    run_dir = run_dir.resolve()
    episodes_csv = run_dir / "episodes.csv"
    decisions_csv = run_dir / "decisions.csv"

    results_log = run_dir / "results_log.csv"
    chunk_log = run_dir / "chunk_size_log.csv"

    if episodes_csv.exists() and decisions_csv.exists():
        return {"status": "already_standardized", "run_dir": str(run_dir)}

    if not results_log.exists() or not chunk_log.exists():
        raise FileNotFoundError(
            f"Neither standardized CSVs nor raw AAC logs found in {run_dir}"
        )

    # Group chunk sizes by (task_description, episode_idx)
    chunks_by_ep: Dict[tuple[str, int], List[tuple[int, int]]] = {}
    with open(chunk_log, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tdesc = row["task_description"]
            ep_idx = int(row["episode_idx"])
            step_idx = int(row["step_idx"])
            csize = int(float(row["chunk_size"]))
            chunks_by_ep.setdefault((tdesc, ep_idx), []).append((step_idx, csize))

    task_desc_to_id: Dict[str, int] = {}
    with open(episodes_csv, "w", encoding="utf-8", newline="") as ef, open(
        decisions_csv, "w", encoding="utf-8", newline=""
    ) as df:
        ep_writer = csv.DictWriter(ef, fieldnames=EPISODES_CSV_COLUMNS)
        dec_writer = csv.DictWriter(df, fieldnames=DECISIONS_CSV_COLUMNS)
        ep_writer.writeheader()
        dec_writer.writeheader()

        with open(results_log, "r", encoding="utf-8", newline="") as rf:
            for row in csv.DictReader(rf):
                tid = int(row["task_id"])
                tdesc = row["task_description"]
                task_desc_to_id[tdesc] = tid
                ep_idx = int(row["episode_idx"])
                succ_str = str(row["success"]).strip().lower()
                succ = 1 if succ_str in ("true", "1", "yes") else 0

                decs = chunks_by_ep.get((tdesc, ep_idx), [])
                h_list = [c for _, c in decs]
                num_dec = len(h_list)
                env_steps = sum(h_list)
                mean_h = (env_steps / float(num_dec)) if num_dec > 0 else 0.0

                ep_writer.writerow(
                    {
                        "suite": suite,
                        "task_id": tid,
                        "trial_id": ep_idx,
                        "arm": method,
                        "success": succ,
                        "end_step": env_steps,
                        "n_decisions": num_dec,
                        "mean_h": round(mean_h, 4),
                        "wall_s": 0.0,
                        "method": method,
                        "backbone": backbone,
                        "benchmark": "libero",
                        "perturbation": "clean",
                        "task_name": tdesc,
                        "episode_id": ep_idx,
                        "init_state_id": ep_idx,
                        "seed": seed,
                        "env_steps": env_steps,
                        "num_decisions": num_dec,
                        "wall_time_s": 0.0,
                        "forward_passes": num_dec * n_samples * 10,
                        "jvp_probes": 0,
                    }
                )

                for d_i, (s_idx, c_sz) in enumerate(decs):
                    dec_row: Dict[str, Any] = {
                        "task_id": tid,
                        "trial_id": ep_idx,
                        "decision_id": d_i,
                        "sim_step": s_idx,
                        "h_executed": c_sz,
                        "h_method": c_sz,
                        "move_lower_bound": 2,
                        "infer_ms": 0.0,
                        "gripper_toggle": 0,
                        "method": method,
                        "backbone": backbone,
                        "suite": suite,
                        "episode_id": ep_idx,
                        "decision_idx": d_i,
                        "env_step": s_idx,
                        "h_selected": c_sz,
                        "h_unc": c_sz,
                        "xi": 2,
                    }
                    for j in range(10):
                        dec_row[f"u_step_{j+1}"] = 0.0
                        dec_row[f"u_prefix_{j+1}"] = 0.0
                        dec_row[f"m_step_{j+1}"] = 0.0
                    dec_writer.writerow(dec_row)

    return {"status": "converted", "run_dir": str(run_dir)}


def validate_run_dir(
    run_dir: Path,
    expected_episodes: Optional[int] = None,
    require_state_bank: bool = False,
    require_subgoals: bool = False,
) -> Dict[str, Any]:
    """Validate a single O-LONG run directory against Section 4.2 requirements."""
    run_dir = run_dir.resolve()
    episodes_csv = run_dir / "episodes.csv"
    decisions_csv = run_dir / "decisions.csv"
    subgoals_csv = run_dir / "subgoals.csv"

    if not episodes_csv.exists():
        raise AssertionError(f"Missing episodes.csv in {run_dir}")
    if not decisions_csv.exists():
        raise AssertionError(f"Missing decisions.csv in {run_dir}")

    with open(episodes_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == EPISODES_CSV_COLUMNS, (
            f"Invalid episodes.csv columns in {run_dir}: {reader.fieldnames}"
        )
        ep_rows = list(reader)

    if expected_episodes is not None and len(ep_rows) != expected_episodes:
        raise AssertionError(
            f"Episode count mismatch in {run_dir}: expected {expected_episodes}, got {len(ep_rows)}"
        )

    with open(decisions_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == DECISIONS_CSV_COLUMNS, (
            f"Invalid decisions.csv columns in {run_dir}: {reader.fieldnames}"
        )
        dec_count = sum(1 for _ in reader)

    subgoal_count = 0
    if subgoals_csv.exists():
        with open(subgoals_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == SUBGOALS_CSV_COLUMNS, (
                f"Invalid subgoals.csv columns in {run_dir}: {reader.fieldnames}"
            )
            subgoal_count = sum(1 for _ in reader)
    elif require_subgoals:
        raise AssertionError(f"Missing subgoals.csv in {run_dir}")

    npz_count = 0
    state_bank_dir = run_dir / "state_bank"
    if state_bank_dir.exists():
        for npz_file in sorted(state_bank_dir.glob("*.npz")):
            with open(npz_file, "rb") as bf:
                magic = bf.read(2)
            assert magic == b"PK", f"Corrupt .npz (missing PK magic) in {npz_file}"
            npz_count += 1
    elif require_state_bank:
        raise AssertionError(f"Missing state_bank/ directory in {run_dir}")

    if require_state_bank and expected_episodes is not None and npz_count != expected_episodes:
        raise AssertionError(
            f"state_bank .npz count ({npz_count}) != expected_episodes ({expected_episodes}) in {run_dir}"
        )

    return {
        "run_dir": str(run_dir),
        "episodes": len(ep_rows),
        "decisions": dec_count,
        "subgoals": subgoal_count,
        "state_bank_npz": npz_count,
        "valid": True,
    }


def run_self_test() -> int:
    """Run O-LONG Step 0.4 verification on both DUAC and AAC synthetic outputs."""
    from adaptive_action_chunking.aac import run_aac_libero_eval
    from flowbased_aac.eval_libero import run_libero_evaluation

    print("[analysis/convert.py --self-test] Running Step 0.4 verification...")
    with tempfile.TemporaryDirectory(prefix="convert_test_") as tmpdir:
        tmp = Path(tmpdir)
        duac_dir = tmp / "duac"
        aac_dir = tmp / "aac"

        run_libero_evaluation(
            suite="libero_10",
            num_trials=2,
            seed=7,
            state_bank=True,
            output_dir=duac_dir,
            task_ids=[0, 1],
            synthetic_dry_run=True,
        )
        run_aac_libero_eval(
            backbone="pi05",
            suite="libero_10",
            trials=2,
            seed=7,
            out_dir=aac_dir,
            task_ids=[0, 1],
            synthetic_dry_run=True,
        )

        res_duac = validate_run_dir(
            duac_dir, expected_episodes=4, require_state_bank=True, require_subgoals=True
        )
        res_aac = validate_run_dir(
            aac_dir, expected_episodes=4, require_state_bank=False, require_subgoals=True
        )
        print("  [PASS] DUAC validation:", json.dumps(res_duac))
        print("  [PASS] AAC validation: ", json.dumps(res_aac))
    print("[analysis/convert.py --self-test] ALL CHECKS PASSED.")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert and validate O-LONG run directories.")
    parser.add_argument("--manifest", type=Path, default=None, help="Path to manifest.json.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Single run directory to convert/validate.")
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument("--require-state-bank", action="store_true")
    parser.add_argument("--require-subgoals", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Run Step 0.4 self-test.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()

    if args.run_dir is not None:
        convert_raw_aac_dir(args.run_dir)
        rep = validate_run_dir(
            args.run_dir,
            expected_episodes=args.expected_episodes,
            require_state_bank=args.require_state_bank,
            require_subgoals=args.require_subgoals,
        )
        print(json.dumps(rep, indent=2))
        return 0

    if args.manifest is not None:
        manifest_data = json.loads(args.manifest.read_text(encoding="utf-8"))
        reports = []
        for entry in manifest_data.get("runs", []):
            rdir = WORKSPACE_ROOT / entry["path"]
            if not rdir.exists():
                reports.append({"id": entry.get("id"), "path": str(rdir), "valid": False, "reason": "missing_dir"})
                continue
            convert_raw_aac_dir(
                rdir,
                backbone=entry.get("backbone", "pi05"),
                suite=entry.get("suite", "libero_10"),
                method=entry.get("method", "duac"),
            )
            rep = validate_run_dir(
                rdir,
                expected_episodes=entry.get("expected_episodes"),
                require_state_bank=bool(entry.get("require_state_bank", False)),
                require_subgoals=bool(entry.get("require_subgoals", False)),
            )
            rep["id"] = entry.get("id")
            reports.append(rep)
        print(json.dumps(reports, indent=2))
        return 0

    return run_self_test()


if __name__ == "__main__":
    sys.exit(main())

