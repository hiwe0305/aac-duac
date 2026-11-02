"""DUAC LIBERO evaluator implementing `docs/task3010.md` (O-LONG specification).

Implements:
  - Step 0.1: `suite_step_budget_name()` regex `^(libero_(10|90|goal|spatial|object))_with_.+$` -> 520 active steps.
  - Step 0.2: Per-step BDDL goal predicate tracking (`inner._eval_predicate(g)` after every `env.step`),
    recording `first_true_step` for each predicate in `subgoals.csv`.
  - Step 0.3: `--fixed-horizon` (5, 8, 10) and `--random-horizon-from <decisions.csv>` (Random-h).
  - Section 4.2: Exact schemas for `episodes.csv`, `decisions.csv` (including `h_executed`, `h_method`,
    `move_lower_bound`, `infer_ms`, `gripper_toggle`), `subgoals.csv`, and `state_bank/task{TT}_trial{KK}.npz`.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import random
import re
import time
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .duac_policy import DUACConfig, DUACDecisionResult, select_duac_horizon


SUITE_MAX_STEPS: Dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# Exact columns matching `docs/task3010.md` Section 4.2 + full metadata
EPISODES_CSV_COLUMNS = [
    "suite",
    "task_id",
    "trial_id",
    "arm",
    "success",
    "end_step",
    "n_decisions",
    "mean_h",
    "wall_s",
    "method",
    "backbone",
    "benchmark",
    "perturbation",
    "task_name",
    "episode_id",
    "init_state_id",
    "seed",
    "env_steps",
    "num_decisions",
    "wall_time_s",
    "forward_passes",
    "jvp_probes",
]

DECISIONS_CSV_COLUMNS = [
    "task_id",
    "trial_id",
    "decision_id",
    "sim_step",
    "h_executed",
    "h_method",
    "move_lower_bound",
    "infer_ms",
    "gripper_toggle",
    "method",
    "backbone",
    "suite",
    "episode_id",
    "decision_idx",
    "env_step",
    "h_selected",
    "h_unc",
    "xi",
    "u_step_1",
    "u_step_2",
    "u_step_3",
    "u_step_4",
    "u_step_5",
    "u_step_6",
    "u_step_7",
    "u_step_8",
    "u_step_9",
    "u_step_10",
    "u_prefix_1",
    "u_prefix_2",
    "u_prefix_3",
    "u_prefix_4",
    "u_prefix_5",
    "u_prefix_6",
    "u_prefix_7",
    "u_prefix_8",
    "u_prefix_9",
    "u_prefix_10",
    "m_step_1",
    "m_step_2",
    "m_step_3",
    "m_step_4",
    "m_step_5",
    "m_step_6",
    "m_step_7",
    "m_step_8",
    "m_step_9",
    "m_step_10",
]

# Section 4.2: 1 goal predicate x 1 episode
SUBGOALS_CSV_COLUMNS = [
    "task_id",
    "trial_id",
    "arm",
    "predicate_idx",
    "predicate",
    "first_true_step",
]


def suite_step_budget_name(suite_name: str) -> str:
    """Normalize a suite name to its base LIBERO suite for step-budget lookup (Step 0.1)."""
    s = suite_name.strip().lower()
    m = re.match(r"^(libero_(10|90|goal|spatial|object))_with_.+$", s)
    if m:
        return m.group(1)
    m_temp = re.match(r"^(libero_(10|90|goal|spatial|object))_temp_.+$", s)
    if m_temp:
        return m_temp.group(1)
    if s in ("libero_long", "libero-10", "libero10"):
        return "libero_10"
    return s


def max_steps_for_suite(suite_name: str) -> int:
    """Return official active step budget for a LIBERO suite (520 for libero_10)."""
    base = suite_step_budget_name(suite_name)
    if base not in SUITE_MAX_STEPS:
        raise ValueError(f"Unsupported suite {suite_name!r} (normalized: {base!r}).")
    return SUITE_MAX_STEPS[base]


def parse_perturbation_from_suite(suite_name: str) -> str:
    """Extract perturbation label from suite name (`clean` for standard LIBERO suites)."""
    s = suite_name.strip().lower()
    m = re.match(r"^libero_(?:10|90|goal|spatial|object)_with_(.+)$", s)
    if m:
        return m.group(1)
    return "clean"


def get_libero_goal_predicates(env: Any) -> List[Any]:
    """Return list of BDDL goal_state predicates from a LIBERO environment."""
    inner = getattr(env, "env", env)
    parsed = getattr(inner, "parsed_problem", None)
    if not isinstance(parsed, dict):
        return []
    return list(parsed.get("goal_state", []))


def eval_libero_goal_predicates(env: Any) -> Tuple[List[str], List[bool]]:
    """Evaluate each BDDL goal predicate after `env.step` (Step 0.2 of `docs/task3010.md`).

    Returns:
        (predicate_strings, flags)
    """
    inner = getattr(env, "env", env)
    goal_state = get_libero_goal_predicates(env)
    if not goal_state or not hasattr(inner, "_eval_predicate"):
        return [], []
    pred_strs: List[str] = []
    flags: List[bool] = []
    for g in goal_state:
        pred_strs.append(" ".join(str(x) for x in g) if isinstance(g, (list, tuple)) else str(g))
        try:
            flags.append(bool(inner._eval_predicate(g)))
        except Exception:
            flags.append(False)
    return pred_strs, flags


def evaluate_libero_subgoals(env: Any) -> Tuple[int, int, str]:
    """Helper returning (num_subgoals, num_satisfied, subgoal_bits) for a LIBERO environment."""
    _, flags = eval_libero_goal_predicates(env)
    n_sub = len(flags)
    n_sat = sum(1 for f in flags if f)
    bits = "".join("1" if f else "0" for f in flags)
    return n_sub, n_sat, bits


def load_random_horizon_pools(decisions_csv_path: Path) -> Tuple[Dict[int, List[int]], List[int]]:
    """Load empirical h_executed / h_selected pools per task_id from a DUAC decisions.csv (Step 0.3)."""
    per_task: Dict[int, List[int]] = {}
    global_pool: List[int] = []
    with open(decisions_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(row["task_id"])
            h_str = row.get("h_executed") or row.get("h_selected")
            if h_str is None:
                continue
            h_val = int(float(h_str))
            per_task.setdefault(tid, []).append(h_val)
            global_pool.append(h_val)
    if not global_pool:
        raise ValueError(f"No decision rows found in {decisions_csv_path}")
    return per_task, global_pool


def _pad_or_trim_10(vals: Sequence[float]) -> List[float]:
    out = [float(x) for x in vals[:10]]
    while len(out) < 10:
        out.append(out[-1] if out else 0.0)
    return out


def _has_gripper_toggle(actions: Sequence[Sequence[float]], h_exec: int) -> int:
    """Return 1 if gripper command changes sign within the executed prefix (or full 10-step chunk)."""
    if not actions:
        return 0
    g0 = 1 if float(actions[0][6]) >= 0.0 else -1
    limit = min(len(actions), max(h_exec, 2))
    for t in range(1, limit):
        gt = 1 if float(actions[t][6]) >= 0.0 else -1
        if gt != g0:
            return 1
    return 0


def write_npz_without_numpy(path: Path, arrays: Dict[str, Tuple[Tuple[int, ...], List[float]]]) -> None:
    """Fallback writer producing a valid `.npz` (ZIP with `.npy` entries, PK magic bytes) when numpy is absent."""
    import struct

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, (shape, flat_vals) in arrays.items():
            header_dict = f"{{'descr': '<f4', 'fortran_order': False, 'shape': {shape}, }}"
            magic = b"\x93NUMPY\x01\x00"
            pad_len = (16 - ((len(magic) + 2 + len(header_dict) + 1) % 16)) % 16
            header_bytes = (header_dict + (" " * pad_len) + "\n").encode("latin1")
            header_len = struct.pack("<H", len(header_bytes))
            payload = struct.pack(f"<{len(flat_vals)}f", *[float(v) for v in flat_vals])
            zf.writestr(f"{name}.npy", magic + header_len + header_bytes + payload)


def save_state_bank_entry(
    path: Path,
    u_step_rows: List[List[float]],
    u_prefix_rows: List[List[float]],
    m_step_rows: List[List[float]],
    h_executed_rows: List[int],
    h_method_rows: List[int],
    xi_rows: List[int],
    sim_step_rows: List[int],
) -> None:
    """Save per-episode DUAC decision arrays to compressed `.npz` (`state_bank/task{TT}_trial{KK}.npz`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np

        np.savez_compressed(
            path,
            u_step=np.asarray(u_step_rows, dtype=np.float32),
            u_prefix=np.asarray(u_prefix_rows, dtype=np.float32),
            m_step=np.asarray(m_step_rows, dtype=np.float32),
            h_executed=np.asarray(h_executed_rows, dtype=np.int32),
            h_selected=np.asarray(h_executed_rows, dtype=np.int32),
            h_method=np.asarray(h_method_rows, dtype=np.int32),
            xi=np.asarray(xi_rows, dtype=np.int32),
            sim_step=np.asarray(sim_step_rows, dtype=np.int32),
        )
    except ImportError:
        n_dec = len(h_executed_rows)
        h_dim = len(u_step_rows[0]) if u_step_rows else 10
        write_npz_without_numpy(
            path,
            {
                "u_step": ((n_dec, h_dim), [x for r in u_step_rows for x in r]),
                "u_prefix": ((n_dec, h_dim), [x for r in u_prefix_rows for x in r]),
                "m_step": ((n_dec, h_dim), [x for r in m_step_rows for x in r]),
                "h_executed": ((n_dec,), [float(x) for x in h_executed_rows]),
                "h_selected": ((n_dec,), [float(x) for x in h_executed_rows]),
                "h_method": ((n_dec,), [float(x) for x in h_method_rows]),
                "xi": ((n_dec,), [float(x) for x in xi_rows]),
                "sim_step": ((n_dec,), [float(x) for x in sim_step_rows]),
            },
        )


def _quat2axisangle(quat: Sequence[float]) -> List[float]:
    """Convert (x, y, z, w) quaternion from robosuite obs to 3D axis-angle vector."""
    x, y, z, w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    if w > 1.0:
        w = 1.0
    elif w < -1.0:
        w = -1.0
    den = math.sqrt(max(1.0 - w * w, 0.0))
    if math.isclose(den, 0.0):
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.acos(w)
    return [(x * angle) / den, (y * angle) / den, (z * angle) / den]


def run_libero_evaluation(
    suite: str = "libero_10",
    host: str = "127.0.0.1",
    port: int = 8000,
    num_trials: int = 50,
    seed: int = 7,
    alpha: float = 3.0,
    tau: float = 0.9,
    num_probes: int = 40,
    fixed_horizon: Optional[int] = None,
    random_horizon_from: Optional[Path] = None,
    state_bank: bool = False,
    save_videos: bool = False,
    output_dir: Path = Path("outputs/A/libero_10/duac"),
    resume_dir: Optional[Path] = None,
    task_ids: Optional[List[int]] = None,
    synthetic_dry_run: bool = False,
) -> Dict[str, Any]:
    """Run DUAC (or Fixed-h / Random-h) evaluation on LIBERO and write `docs/task3010.md` Section 4 artifacts."""
    out_dir = (resume_dir if resume_dir is not None else output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    state_bank_dir = out_dir / "state_bank"
    if state_bank:
        state_bank_dir.mkdir(parents=True, exist_ok=True)

    base_suite = suite_step_budget_name(suite)
    max_steps = max_steps_for_suite(suite)
    perturbation = parse_perturbation_from_suite(suite)
    benchmark_name = "libero" if perturbation == "clean" else "liberopro"

    random_pools_by_task: Dict[int, List[int]] = {}
    random_pool_global: List[int] = []
    if random_horizon_from is not None:
        random_pools_by_task, random_pool_global = load_random_horizon_pools(Path(random_horizon_from))

    if fixed_horizon is not None:
        method_name = f"fixed_{fixed_horizon}"
    elif random_horizon_from is not None:
        method_name = "random_h"
    else:
        method_name = "duac"

    args_payload = {
        "arm": method_name,
        "method": method_name,
        "backbone": "pi05",
        "benchmark": benchmark_name,
        "suite": suite,
        "base_suite": base_suite,
        "perturbation": perturbation,
        "max_steps": max_steps,
        "host": host,
        "port": port,
        "trials": num_trials,
        "seed": seed,
        "alpha": alpha,
        "tau": tau,
        "num_probes": num_probes,
        "fixed_horizon": fixed_horizon,
        "random_horizon_from": str(random_horizon_from) if random_horizon_from else None,
        "state_bank": state_bank,
        "save_videos": save_videos,
        "out_dir": str(out_dir),
        "synthetic_dry_run": synthetic_dry_run,
    }
    (out_dir / "args.json").write_text(json.dumps(args_payload, indent=2), encoding="utf-8")

    episodes_csv_path = out_dir / "episodes.csv"
    decisions_csv_path = out_dir / "decisions.csv"
    subgoals_csv_path = out_dir / "subgoals.csv"

    completed_episodes: set[Tuple[int, int]] = set()
    existing_episode_rows: List[Dict[str, Any]] = []
    if resume_dir is not None and episodes_csv_path.exists():
        with open(episodes_csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                tid = int(row["task_id"])
                epid = int(row.get("trial_id", row.get("episode_id", 0)))
                completed_episodes.add((tid, epid))
                existing_episode_rows.append(row)

    ep_mode = "a" if completed_episodes and episodes_csv_path.exists() else "w"
    dec_mode = "a" if completed_episodes and decisions_csv_path.exists() else "w"
    sub_mode = "a" if completed_episodes and subgoals_csv_path.exists() else "w"

    ep_file = open(episodes_csv_path, ep_mode, encoding="utf-8", newline="")
    dec_file = open(decisions_csv_path, dec_mode, encoding="utf-8", newline="")
    sub_file = open(subgoals_csv_path, sub_mode, encoding="utf-8", newline="")

    ep_writer = csv.DictWriter(ep_file, fieldnames=EPISODES_CSV_COLUMNS)
    dec_writer = csv.DictWriter(dec_file, fieldnames=DECISIONS_CSV_COLUMNS)
    sub_writer = csv.DictWriter(sub_file, fieldnames=SUBGOALS_CSV_COLUMNS)

    if ep_mode == "w":
        ep_writer.writeheader()
    if dec_mode == "w":
        dec_writer.writeheader()
    if sub_mode == "w":
        sub_writer.writeheader()

    rng = random.Random(seed)

    if synthetic_dry_run:
        t_list = task_ids if task_ids is not None else list(range(10 if base_suite == "libero_10" else 2))
        total_eps = 0
        total_succ = 0
        all_h: List[int] = []

        for tid in t_list:
            t_name = f"{base_suite}_task_{tid:02d}"
            task_pool = random_pools_by_task.get(tid, random_pool_global) if random_horizon_from else None
            cfg = DUACConfig(
                chunk_horizon=10,
                min_horizon=2,
                tau=tau,
                num_probes=num_probes,
                alpha=alpha,
                fixed_horizon=fixed_horizon,
                random_horizon_pool=task_pool,
                seed=seed + tid,
            )
            for ep_idx in range(num_trials):
                if (tid, ep_idx) in completed_episodes:
                    continue
                t0 = time.time()
                sim_step = 0
                dec_idx = 0
                ep_h_list: List[int] = []
                u_step_rows: List[List[float]] = []
                u_prefix_rows: List[List[float]] = []
                m_step_rows: List[List[float]] = []
                h_exec_rows: List[int] = []
                h_method_rows: List[int] = []
                xi_rows: List[int] = []
                sim_step_rows: List[int] = []

                target_steps = min(max_steps, 48)
                pred_names = [f"turnon stove_{tid}", f"on moka_pot_{tid} stove_{tid}_region"]
                first_true: List[Optional[int]] = [None, None]

                while sim_step < target_steps:
                    t_infer0 = time.time()
                    jump_at = 6 + ((tid + ep_idx + dec_idx) % 4)
                    u_step = [
                        0.12 + 0.015 * j + (0.40 if j >= jump_at else 0.0) + 0.005 * rng.random()
                        for j in range(10)
                    ]
                    grip_flip = (dec_idx == 2) or (dec_idx == 5)
                    actions = [
                        [0.35, 0.18, -0.12, 0.04, -0.02, 0.02, (-1.0 if (grip_flip and j >= 4) else 1.0)]
                        for j in range(10)
                    ]
                    res: DUACDecisionResult = select_duac_horizon(
                        u_step=u_step,
                        actions=actions,
                        config=cfg,
                        rng=rng,
                    )
                    infer_ms = max((time.time() - t_infer0) * 1000.0, 0.05)
                    h_method_proposed = max(res.xi, res.h_unc)
                    h_exec = res.h_selected
                    g_toggle = _has_gripper_toggle(actions, h_exec)

                    u10 = _pad_or_trim_10(res.u_step)
                    up10 = _pad_or_trim_10(res.u_prefix)
                    m10 = _pad_or_trim_10(res.m_step)

                    dec_row: Dict[str, Any] = {
                        "task_id": tid,
                        "trial_id": ep_idx,
                        "decision_id": dec_idx,
                        "sim_step": sim_step,
                        "h_executed": h_exec,
                        "h_method": h_method_proposed,
                        "move_lower_bound": res.xi,
                        "infer_ms": round(infer_ms, 3),
                        "gripper_toggle": g_toggle,
                        "method": method_name,
                        "backbone": "pi05",
                        "suite": suite,
                        "episode_id": ep_idx,
                        "decision_idx": dec_idx,
                        "env_step": sim_step,
                        "h_selected": h_exec,
                        "h_unc": res.h_unc,
                        "xi": res.xi,
                    }
                    for j in range(10):
                        dec_row[f"u_step_{j+1}"] = round(u10[j], 6)
                        dec_row[f"u_prefix_{j+1}"] = round(up10[j], 6)
                        dec_row[f"m_step_{j+1}"] = round(m10[j], 6)
                    dec_writer.writerow(dec_row)

                    u_step_rows.append(u10)
                    u_prefix_rows.append(up10)
                    m_step_rows.append(m10)
                    h_exec_rows.append(h_exec)
                    h_method_rows.append(h_method_proposed)
                    xi_rows.append(res.xi)
                    sim_step_rows.append(sim_step)

                    for _ in range(h_exec):
                        if sim_step >= target_steps:
                            break
                        sim_step += 1
                        if first_true[0] is None and sim_step >= target_steps // 3:
                            first_true[0] = sim_step
                        if first_true[1] is None and sim_step >= (target_steps * 4) // 5 and ((tid + ep_idx) % 3 != 0):
                            first_true[1] = sim_step

                    ep_h_list.append(h_exec)
                    all_h.append(h_exec)
                    dec_idx += 1

                for p_i, p_str in enumerate(pred_names):
                    sub_writer.writerow(
                        {
                            "task_id": tid,
                            "trial_id": ep_idx,
                            "arm": method_name,
                            "predicate_idx": p_i,
                            "predicate": p_str,
                            "first_true_step": "" if first_true[p_i] is None else first_true[p_i],
                        }
                    )

                if state_bank:
                    npz_path = state_bank_dir / f"task{tid:02d}_trial{ep_idx:02d}.npz"
                    save_state_bank_entry(
                        npz_path,
                        u_step_rows,
                        u_prefix_rows,
                        m_step_rows,
                        h_exec_rows,
                        h_method_rows,
                        xi_rows,
                        sim_step_rows,
                    )

                succ = 1 if (first_true[0] is not None and first_true[1] is not None) else 0
                total_eps += 1
                total_succ += succ
                mean_h = sum(ep_h_list) / max(len(ep_h_list), 1)
                wall_s = max(time.time() - t0, 1e-4)
                jvp_per_dec = 0 if (fixed_horizon is not None or random_horizon_from is not None) else num_probes

                ep_writer.writerow(
                    {
                        "suite": suite,
                        "task_id": tid,
                        "trial_id": ep_idx,
                        "arm": method_name,
                        "success": succ,
                        "end_step": sim_step,
                        "n_decisions": dec_idx,
                        "mean_h": round(mean_h, 4),
                        "wall_s": round(wall_s, 4),
                        "method": method_name,
                        "backbone": "pi05",
                        "benchmark": benchmark_name,
                        "perturbation": perturbation,
                        "task_name": t_name,
                        "episode_id": ep_idx,
                        "init_state_id": ep_idx,
                        "seed": seed,
                        "env_steps": sim_step,
                        "num_decisions": dec_idx,
                        "wall_time_s": round(wall_s, 4),
                        "forward_passes": dec_idx * 10,
                        "jvp_probes": dec_idx * jvp_per_dec,
                    }
                )
        ep_file.close()
        dec_file.close()
        sub_file.close()

        sr = (total_succ / total_eps) if total_eps > 0 else 0.0
        summary = {
            "arm": method_name,
            "method": method_name,
            "backbone": "pi05",
            "suite": suite,
            "base_suite": base_suite,
            "max_steps": max_steps,
            "total_episodes": total_eps,
            "total_successes": total_succ,
            "success_rate": round(sr, 6),
            "mean_h": round(sum(all_h) / max(len(all_h), 1), 4),
            "synthetic_dry_run": True,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out_dir / "success_rate.txt").write_text(f"{sr:.6f}\n", encoding="utf-8")
        return summary

    # Real LIBERO evaluation with OpenPI WebSocket client + Official OpenPI 224x224 protocol
    from tools.download_libero import configure_and_verify_benchmark
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(Path(__file__).resolve().parents[2] / ".libero"))
    configure_and_verify_benchmark(base_suite)

    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools, websocket_client_policy

    benchmark_dict = benchmark.get_benchmark_dict()
    if suite in benchmark_dict:
        task_suite = benchmark_dict[suite]()
    elif base_suite in benchmark_dict:
        task_suite = benchmark_dict[base_suite]()
    else:
        raise ValueError(f"Unsupported suite {suite!r} not found in LIBERO benchmark_dict.")

    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    num_tasks = task_suite.n_tasks
    t_list = task_ids if task_ids is not None else list(range(num_tasks))

    total_eps = len(existing_episode_rows)
    total_succ = sum(int(r["success"]) for r in existing_episode_rows)
    all_h = [float(r["mean_h"]) for r in existing_episode_rows]
    dummy_action = [0.0] * 6 + [-1.0]
    num_steps_wait = 10
    resize_size = 224

    for tid in t_list:
        task = task_suite.get_task(tid)
        initial_states = task_suite.get_task_init_states(tid)
        task_description = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=256,
            camera_widths=256,
        )
        env.seed(seed)

        task_pool = random_pools_by_task.get(tid, random_pool_global) if random_horizon_from else None
        cfg = DUACConfig(
            chunk_horizon=10,
            min_horizon=2,
            tau=tau,
            num_probes=num_probes,
            alpha=alpha,
            fixed_horizon=fixed_horizon,
            random_horizon_pool=task_pool,
            seed=seed + tid,
        )

        n_ep = min(num_trials, len(initial_states))
        for ep_idx in range(n_ep):
            if (tid, ep_idx) in completed_episodes:
                continue

            env.reset()
            obs = env.set_init_state(initial_states[ep_idx])
            t = 0
            done = False
            dec_idx = 0
            ep_h_list = []
            u_step_rows = []
            u_prefix_rows = []
            m_step_rows = []
            h_exec_rows = []
            h_method_rows = []
            xi_rows = []
            sim_step_rows = []
            t0 = time.time()

            while t < num_steps_wait:
                obs, _, done, _ = env.step(dummy_action)
                t += 1

            pred_strs_init, flags_init = eval_libero_goal_predicates(env)
            first_true_steps: List[Optional[int]] = [
                0 if f else None for f in flags_init
            ]

            active_steps = 0
            while active_steps < max_steps and not done:
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, resize_size, resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
                )

                state_vec = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state_vec,
                    "prompt": str(task_description),
                }

                t_infer0 = time.time()
                resp = client.infer(element)
                infer_ms = (time.time() - t_infer0) * 1000.0

                actions_arr = np.asarray(resp["actions"])
                norm_arr = np.asarray(resp.get("normalized_action", actions_arr))
                u_step_srv = resp.get("u_step")
                if u_step_srv is not None:
                    u_step_list = [float(x) for x in np.asarray(u_step_srv).tolist()]
                else:
                    u_step_list = [0.0] * len(actions_arr)

                actions_list = actions_arr.tolist()
                res = select_duac_horizon(
                    u_step=u_step_list,
                    actions=actions_list,
                    config=cfg,
                    normalized_pos_rot=norm_arr[:, :6].tolist(),
                    rng=rng,
                )
                h_method_proposed = max(res.xi, res.h_unc)
                h_exec = res.h_selected
                g_toggle = _has_gripper_toggle(actions_list, h_exec)

                u10 = _pad_or_trim_10(res.u_step)
                up10 = _pad_or_trim_10(res.u_prefix)
                m10 = _pad_or_trim_10(res.m_step)
                dec_row = {
                    "task_id": tid,
                    "trial_id": ep_idx,
                    "decision_id": dec_idx,
                    "sim_step": active_steps,
                    "h_executed": h_exec,
                    "h_method": h_method_proposed,
                    "move_lower_bound": res.xi,
                    "infer_ms": round(infer_ms, 3),
                    "gripper_toggle": g_toggle,
                    "method": method_name,
                    "backbone": "pi05",
                    "suite": suite,
                    "episode_id": ep_idx,
                    "decision_idx": dec_idx,
                    "env_step": active_steps,
                    "h_selected": h_exec,
                    "h_unc": res.h_unc,
                    "xi": res.xi,
                }
                for j in range(10):
                    dec_row[f"u_step_{j+1}"] = round(u10[j], 6)
                    dec_row[f"u_prefix_{j+1}"] = round(up10[j], 6)
                    dec_row[f"m_step_{j+1}"] = round(m10[j], 6)
                dec_writer.writerow(dec_row)

                u_step_rows.append(u10)
                u_prefix_rows.append(up10)
                m_step_rows.append(m10)
                h_exec_rows.append(h_exec)
                h_method_rows.append(h_method_proposed)
                xi_rows.append(res.xi)
                sim_step_rows.append(active_steps)

                ep_h_list.append(h_exec)
                dec_idx += 1

                for step_i in range(h_exec):
                    if active_steps >= max_steps or done:
                        break
                    act = actions_list[step_i]
                    obs, _, done, _ = env.step(act)
                    active_steps += 1
                    # Step 0.2 of docs/task3010.md: check each goal predicate after every env.step
                    _, step_flags = eval_libero_goal_predicates(env)
                    for p_i, flag_val in enumerate(step_flags):
                        if flag_val and p_i < len(first_true_steps) and first_true_steps[p_i] is None:
                            first_true_steps[p_i] = active_steps
                    if done:
                        break

            for p_i, p_str in enumerate(pred_strs_init):
                sub_writer.writerow(
                    {
                        "task_id": tid,
                        "trial_id": ep_idx,
                        "arm": method_name,
                        "predicate_idx": p_i,
                        "predicate": p_str,
                        "first_true_step": "" if first_true_steps[p_i] is None else first_true_steps[p_i],
                    }
                )

            if state_bank:
                npz_path = state_bank_dir / f"task{tid:02d}_trial{ep_idx:02d}.npz"
                save_state_bank_entry(
                    npz_path,
                    u_step_rows,
                    u_prefix_rows,
                    m_step_rows,
                    h_exec_rows,
                    h_method_rows,
                    xi_rows,
                    sim_step_rows,
                )

            succ = 1 if done else 0
            total_eps += 1
            total_succ += succ
            mean_h = sum(ep_h_list) / max(len(ep_h_list), 1)
            all_h.append(mean_h)
            wall_s = max(time.time() - t0, 1e-4)
            jvp_per_dec = 0 if (fixed_horizon is not None or random_horizon_from is not None) else num_probes

            ep_writer.writerow(
                {
                    "suite": suite,
                    "task_id": tid,
                    "trial_id": ep_idx,
                    "arm": method_name,
                    "success": succ,
                    "end_step": active_steps,
                    "n_decisions": dec_idx,
                    "mean_h": round(mean_h, 4),
                    "wall_s": round(wall_s, 4),
                    "method": method_name,
                    "backbone": "pi05",
                    "benchmark": benchmark_name,
                    "perturbation": perturbation,
                    "task_name": task.name,
                    "episode_id": ep_idx,
                    "init_state_id": ep_idx,
                    "seed": seed,
                    "env_steps": active_steps,
                    "num_decisions": dec_idx,
                    "wall_time_s": round(wall_s, 4),
                    "forward_passes": dec_idx * 10,
                    "jvp_probes": dec_idx * jvp_per_dec,
                }
            )
            ep_file.flush()
            dec_file.flush()
            sub_file.flush()

        env.close()

    ep_file.close()
    dec_file.close()
    sub_file.close()

    sr = (total_succ / total_eps) if total_eps > 0 else 0.0
    summary = {
        "arm": method_name,
        "method": method_name,
        "backbone": "pi05",
        "suite": suite,
        "base_suite": base_suite,
        "max_steps": max_steps,
        "total_episodes": total_eps,
        "total_successes": total_succ,
        "success_rate": round(sr, 6),
        "mean_h": round(sum(all_h) / max(len(all_h), 1), 4),
        "synthetic_dry_run": False,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "success_rate.txt").write_text(f"{sr:.6f}\n", encoding="utf-8")
    return summary
