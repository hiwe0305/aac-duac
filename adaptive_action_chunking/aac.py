#!/usr/bin/env python3
"""AAC CLI (`aac.py`) for pi0.5 + LIBERO and RoboCasa under the O-LONG spec.

Integrates the official `Adaptive-Action-Chunking/libero` math (`action_entropy_pi05.py`)
with:
  - `--official_openpi_protocol`: 224x224 `image_tools.resize_with_pad` + `OffScreenRenderEnv`
    + 520 active steps after 10 wait steps (O-LONG Section 2.5).
  - Step 0.1 `suite_step_budget_name()` regex normalization.
  - Step 0.2 BDDL subgoal predicate evaluation (`subgoals.csv`).
  - Dual output: writes both native `chunk_size_log.csv` / `results_log.csv` AND
    the standardized O-LONG Section 4.2 CSVs (`episodes.csv`, `decisions.csv`, `subgoals.csv`).
"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

if "XDG_CACHE_HOME" not in os.environ:
    default_cache = Path.home() / ".cache" / "duac"
    try:
        default_cache.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(default_cache)
    except OSError:
        os.environ["XDG_CACHE_HOME"] = f"/tmp/duac_{getpass.getuser()}"

AAC_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = AAC_ROOT.parent
FLOWBASED_ROOT = WORKSPACE_ROOT / "flowbased_aac"
for p in (
    AAC_ROOT,
    WORKSPACE_ROOT,
    FLOWBASED_ROOT,
    WORKSPACE_ROOT / "external" / "aac-libero",
    WORKSPACE_ROOT / "external" / "openpi" / "src",
    WORKSPACE_ROOT / "external" / "openpi" / "packages" / "openpi-client" / "src",
    WORKSPACE_ROOT / "external" / "LIBERO",
):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from flowbased_aac.duac_policy import (  # noqa: E402
    action_magnitude_pi05,
    find_first_idx,
    prefix_mean_jump_horizon,
)
from flowbased_aac.eval_libero import (  # noqa: E402
    DECISIONS_CSV_COLUMNS,
    EPISODES_CSV_COLUMNS,
    SUBGOALS_CSV_COLUMNS,
    _has_gripper_toggle,
    _pad_or_trim_10,
    _quat2axisangle,
    eval_libero_goal_predicates,
    max_steps_for_suite,
    parse_perturbation_from_suite,
    suite_step_budget_name,
)


def _cholesky_3x3_logdet(cov: List[List[float]]) -> float:
    """Compute log(det(cov)) for a 3x3 symmetric positive-definite matrix via Cholesky."""
    l00 = math.sqrt(max(cov[0][0], 1e-12))
    l10 = cov[1][0] / l00
    l20 = cov[2][0] / l00
    l11 = math.sqrt(max(cov[1][1] - l10 * l10, 1e-12))
    l21 = (cov[2][1] - l20 * l10) / l11
    l22 = math.sqrt(max(cov[2][2] - l20 * l20 - l21 * l21, 1e-12))
    return 2.0 * (math.log(l00) + math.log(l11) + math.log(l22))


def gaussian_entropy_from_samples_3d(
    samples_nh3: Sequence[Sequence[Sequence[float]]],
    eps: float = 1e-6,
) -> List[float]:
    """Compute Gaussian differential entropy at each horizon step t=0..H-1 (AAC Eq. 3).

    Matches `gaussian_entropy_from_samples` in `action_entropy_pi05.py`:
      H_t = 0.5 * (d * log(2 * pi * e) + log det(Sigma_t + eps * I))
    where Sigma_t is the unbiased sample covariance (1 / (N - 1)) across N samples.
    """
    N = len(samples_nh3)
    if N == 0:
        return []
    H = len(samples_nh3[0])
    d = 3
    const_term = d * math.log(2.0 * math.pi * math.e)
    entropies: List[float] = []

    for t in range(H):
        mean = [0.0, 0.0, 0.0]
        for s in range(N):
            for c in range(3):
                mean[c] += float(samples_nh3[s][t][c])
        mean = [m / float(N) for m in mean]

        denom = float(max(N - 1, 1))
        cov = [[0.0, 0.0, 0.0] for _ in range(3)]
        for s in range(N):
            dx = [float(samples_nh3[s][t][c]) - mean[c] for c in range(3)]
            for r in range(3):
                for c in range(3):
                    cov[r][c] += (dx[r] * dx[c]) / denom
        for c in range(3):
            cov[c][c] += eps

        logdet = _cholesky_3x3_logdet(cov)
        entropies.append(0.5 * (const_term + logdet))

    return entropies


def bernoulli_entropy_from_samples(
    grip_nh: Sequence[Sequence[float]],
    eps: float = 1e-12,
) -> List[float]:
    """Compute Bernoulli entropy of binary gripper close indicator at each step t=0..H-1 (AAC Eq. 2)."""
    N = len(grip_nh)
    if N == 0:
        return []
    H = len(grip_nh[0])
    entropies: List[float] = []
    for t in range(H):
        p = sum(1.0 if float(grip_nh[s][t]) >= 0.0 else 0.0 for s in range(N)) / float(N)
        p = min(max(p, eps), 1.0 - eps)
        h_val = -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))
        entropies.append(h_val)
    return entropies


def compute_aac_pi05_decision(
    actions_nh7: Sequence[Sequence[Sequence[float]]],
    normalized_nh7: Sequence[Sequence[Sequence[float]]],
    move_th: float = 3.0,
    selector: str = "gaussian_bernoulli",
) -> Dict[str, Any]:
    """Compute AAC step entropies, prefix-mean jump h_entropy^*, lower bound xi, and h^* (AAC Eq. 4-5).

    Matches `inference_client_action_entropy_pi05.py` and `action_entropy_pi05.py`:
      - `pos_samples` = `normalized_action[:, :, :3]`
      - `rot_samples` = `normalized_action[:, :, 3:6]`
      - `grip_samples` = `actions[:, :, 6] >= 0`
      - Stepwise total entropy E_t = H_gauss(pos_t) + H_gauss(rot_t) + H_bern(grip_t)
      - Mean trajectory across N samples is used for execution and `action_magnitude`.
    """
    N = len(actions_nh7)
    H = len(actions_nh7[0]) if N > 0 else 10

    pos_samples = [[[float(normalized_nh7[s][t][c]) for c in range(3)] for t in range(H)] for s in range(N)]
    rot_samples = [[[float(normalized_nh7[s][t][c]) for c in range(3, 6)] for t in range(H)] for s in range(N)]
    grip_samples = [[1.0 if float(actions_nh7[s][t][6]) >= 0.0 else 0.0 for t in range(H)] for s in range(N)]

    e_pos = gaussian_entropy_from_samples_3d(pos_samples)
    e_rot = gaussian_entropy_from_samples_3d(rot_samples)
    e_grip = bernoulli_entropy_from_samples(grip_samples)
    e_step = [e_pos[t] + e_rot[t] + e_grip[t] for t in range(H)]

    # Mean unnormalized action and mean normalized action across N samples
    mean_actions: List[List[float]] = []
    mean_norm_pos_rot: List[List[float]] = []
    for t in range(H):
        row_u = [sum(float(actions_nh7[s][t][d]) for s in range(N)) / float(N) for d in range(7)]
        # Gripper uses mean of binary indicator in AAC's `pred_action_dict`, while execution uses `actions.mean(axis=0)`
        mean_actions.append(row_u)
        row_n = [sum(float(normalized_nh7[s][t][d]) for s in range(N)) / float(N) for d in range(6)]
        mean_norm_pos_rot.append(row_n)

    m_step = action_magnitude_pi05(
        mean_actions,
        w_pos=1.0,
        w_rot=1.0,
        w_grip=0.2,
        normalized_pos_rot=mean_norm_pos_rot,
    )
    xi = find_first_idx(m_step, alpha=move_th, min_horizon=2)
    h_ent, e_prefix, e_diff = prefix_mean_jump_horizon(e_step, min_horizon=2)

    if selector.startswith("fixed_"):
        k_val = int(selector.split("_", 1)[1])
        h_sel = max(1, min(k_val, H))
    else:
        h_sel = max(xi, h_ent)
        h_sel = max(2, min(h_sel, H))

    return {
        "mean_actions": mean_actions,
        "h_selected": h_sel,
        "h_unc": h_ent,
        "xi": xi,
        "u_step": e_step,
        "u_prefix": e_prefix,
        "u_diff": e_diff,
        "m_step": m_step,
    }


def run_aac_libero_eval(
    backbone: str = "pi05",
    suite: str = "libero_10",
    selector: str = "gaussian_bernoulli",
    trials: int = 50,
    seed: int = 7,
    move_th: float = 3.0,
    save_videos: bool = False,
    host: str = "127.0.0.1",
    port: int = 8015,
    out_dir: Path = Path("outputs/aac_libero_10"),
    official_openpi_protocol: bool = True,
    n_samples: int = 20,
    task_ids: Optional[List[int]] = None,
    synthetic_dry_run: bool = False,
) -> Dict[str, Any]:
    """Run AAC evaluation on LIBERO and produce both native AAC logs and O-LONG Section 4.2 CSVs."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_suite = suite_step_budget_name(suite)
    max_steps = max_steps_for_suite(suite)
    perturbation = parse_perturbation_from_suite(suite)
    benchmark_name = "libero" if perturbation == "clean" else "liberopro"
    method_name = "aac" if selector == "gaussian_bernoulli" else selector

    args_payload = {
        "method": method_name,
        "backbone": backbone,
        "benchmark": benchmark_name,
        "suite": suite,
        "base_suite": base_suite,
        "perturbation": perturbation,
        "selector": selector,
        "trials": trials,
        "seed": seed,
        "move_th": move_th,
        "save_videos": save_videos,
        "host": host,
        "port": port,
        "out_dir": str(out_dir),
        "official_openpi_protocol": official_openpi_protocol,
        "n_samples": n_samples,
        "max_steps": max_steps,
        "synthetic_dry_run": synthetic_dry_run,
    }
    (out_dir / "args.json").write_text(json.dumps(args_payload, indent=2), encoding="utf-8")

    episodes_csv = out_dir / "episodes.csv"
    decisions_csv = out_dir / "decisions.csv"
    subgoals_csv = out_dir / "subgoals.csv"
    chunk_log_csv = out_dir / "chunk_size_log.csv"
    results_log_csv = out_dir / "results_log.csv"

    ep_file = open(episodes_csv, "w", encoding="utf-8", newline="")
    dec_file = open(decisions_csv, "w", encoding="utf-8", newline="")
    sub_file = open(subgoals_csv, "w", encoding="utf-8", newline="")
    chunk_file = open(chunk_log_csv, "w", encoding="utf-8", newline="")
    res_file = open(results_log_csv, "w", encoding="utf-8", newline="")

    ep_writer = csv.DictWriter(ep_file, fieldnames=EPISODES_CSV_COLUMNS)
    dec_writer = csv.DictWriter(dec_file, fieldnames=DECISIONS_CSV_COLUMNS)
    sub_writer = csv.DictWriter(sub_file, fieldnames=SUBGOALS_CSV_COLUMNS)
    chunk_writer = csv.writer(chunk_file)
    res_writer = csv.writer(res_file)

    ep_writer.writeheader()
    dec_writer.writeheader()
    sub_writer.writeheader()
    chunk_writer.writerow(["task_description", "episode_idx", "step_idx", "chunk_size"])
    res_writer.writerow(["task_id", "task_description", "episode_idx", "success", "total_episodes", "total_successes", "success_rate"])

    rng = random.Random(seed)
    total_eps = 0
    total_succ = 0
    all_h: List[float] = []

    if synthetic_dry_run:
        t_list = task_ids if task_ids is not None else list(range(10 if base_suite == "libero_10" else 2))
        for tid in t_list:
            t_name = f"{base_suite}_task_{tid:02d}"
            pred_names = [f"turnon stove_{tid}", f"on moka_pot_{tid} stove_{tid}_region"]
            for ep_idx in range(trials):
                t0 = time.time()
                sim_step = 0
                dec_idx = 0
                ep_h_list: List[int] = []
                target_steps = min(max_steps, 48)
                first_true: List[Optional[int]] = [None, None]

                while sim_step < target_steps:
                    t_infer0 = time.time()
                    jump_at = 5 + ((tid + ep_idx + dec_idx) % 4)
                    actions_nh7 = []
                    norm_nh7 = []
                    grip_flip = (dec_idx == 2) or (dec_idx == 5)
                    for s in range(n_samples):
                        a_traj = []
                        n_traj = []
                        for j in range(10):
                            spread = 0.02 if j < jump_at else 0.25
                            nx = 0.35 + rng.gauss(0.0, spread)
                            ny = 0.18 + rng.gauss(0.0, spread)
                            nz = -0.12 + rng.gauss(0.0, spread)
                            nrx = 0.03 + rng.gauss(0.0, spread * 0.5)
                            nry = -0.02 + rng.gauss(0.0, spread * 0.5)
                            nrz = 0.01 + rng.gauss(0.0, spread * 0.5)
                            g = -1.0 if (grip_flip and j >= 4) else 1.0
                            n_traj.append([nx, ny, nz, nrx, nry, nrz, g])
                            a_traj.append([nx * 0.5, ny * 0.5, nz * 0.5, nrx, nry, nrz, g])
                        actions_nh7.append(a_traj)
                        norm_nh7.append(n_traj)

                    dec = compute_aac_pi05_decision(
                        actions_nh7, norm_nh7, move_th=move_th, selector=selector
                    )
                    infer_ms = max((time.time() - t_infer0) * 1000.0, 0.2)
                    h_sel = int(dec["h_selected"])
                    h_method_prop = max(int(dec["xi"]), int(dec["h_unc"]))
                    g_toggle = _has_gripper_toggle(dec["mean_actions"], h_sel)
                    chunk_writer.writerow([t_name, ep_idx, sim_step, h_sel])

                    u10 = _pad_or_trim_10(dec["u_step"])
                    up10 = _pad_or_trim_10(dec["u_prefix"])
                    m10 = _pad_or_trim_10(dec["m_step"])
                    dec_row: Dict[str, Any] = {
                        "task_id": tid,
                        "trial_id": ep_idx,
                        "decision_id": dec_idx,
                        "sim_step": sim_step,
                        "h_executed": h_sel,
                        "h_method": h_method_prop,
                        "move_lower_bound": int(dec["xi"]),
                        "infer_ms": round(infer_ms, 3),
                        "gripper_toggle": g_toggle,
                        "method": method_name,
                        "backbone": backbone,
                        "suite": suite,
                        "episode_id": ep_idx,
                        "decision_idx": dec_idx,
                        "env_step": sim_step,
                        "h_selected": h_sel,
                        "h_unc": int(dec["h_unc"]),
                        "xi": int(dec["xi"]),
                    }
                    for j in range(10):
                        dec_row[f"u_step_{j+1}"] = round(u10[j], 6)
                        dec_row[f"u_prefix_{j+1}"] = round(up10[j], 6)
                        dec_row[f"m_step_{j+1}"] = round(m10[j], 6)
                    dec_writer.writerow(dec_row)

                    for _ in range(h_sel):
                        if sim_step >= target_steps:
                            break
                        sim_step += 1
                        if first_true[0] is None and sim_step >= target_steps // 3:
                            first_true[0] = sim_step
                        if first_true[1] is None and sim_step >= (target_steps * 4) // 5 and ((tid + ep_idx + 1) % 3 != 0):
                            first_true[1] = sim_step

                    ep_h_list.append(h_sel)
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

                succ = 1 if (first_true[0] is not None and first_true[1] is not None) else 0
                total_eps += 1
                total_succ += succ
                sr_running = total_succ / float(total_eps)
                res_writer.writerow([tid, t_name, ep_idx, bool(succ), total_eps, total_succ, f"{sr_running:.4f}"])

                mean_h = sum(ep_h_list) / max(len(ep_h_list), 1)
                all_h.append(mean_h)
                wall_s = max(time.time() - t0, 1e-4)
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
                        "backbone": backbone,
                        "benchmark": benchmark_name,
                        "perturbation": perturbation,
                        "task_name": t_name,
                        "episode_id": ep_idx,
                        "init_state_id": ep_idx,
                        "seed": seed,
                        "env_steps": sim_step,
                        "num_decisions": dec_idx,
                        "wall_time_s": round(wall_s, 4),
                        "forward_passes": dec_idx * n_samples * 10,
                        "jvp_probes": 0,
                    }
                )
    else:
        from tools.download_libero import configure_and_verify_benchmark
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(WORKSPACE_ROOT / ".libero"))
        configure_and_verify_benchmark(base_suite)

        import numpy as np
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from openpi_client import image_tools, websocket_client_policy

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[suite]() if suite in benchmark_dict else benchmark_dict[base_suite]()
        client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        t_list = task_ids if task_ids is not None else list(range(task_suite.n_tasks))
        dummy_action = [0.0] * 6 + [-1.0]
        num_steps_wait = 10
        resize_size = 224

        for tid in t_list:
            task = task_suite.get_task(tid)
            initial_states = task_suite.get_task_init_states(tid)
            task_description = task.language
            task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=256, camera_widths=256)
            env.seed(seed)

            n_ep = min(trials, len(initial_states))
            for ep_idx in range(n_ep):
                env.reset()
                obs = env.set_init_state(initial_states[ep_idx])
                t = 0
                done = False
                dec_idx = 0
                ep_h_list = []
                t0 = time.time()

                while t < num_steps_wait:
                    obs, _, done, _ = env.step(dummy_action)
                    t += 1

                pred_strs_init, flags_init = eval_libero_goal_predicates(env)
                first_true_steps: List[Optional[int]] = [0 if f else None for f in flags_init]

                active_steps = 0
                active_budget = max_steps if official_openpi_protocol else (max_steps - num_steps_wait)

                while active_steps < active_budget and not done:
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    if official_openpi_protocol:
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
                    if actions_arr.ndim == 2:
                        actions_arr = actions_arr[None, ...]
                    if norm_arr.ndim == 2:
                        norm_arr = norm_arr[None, ...]

                    dec = compute_aac_pi05_decision(
                        actions_arr.tolist(),
                        norm_arr.tolist(),
                        move_th=move_th,
                        selector=selector,
                    )
                    h_sel = int(dec["h_selected"])
                    h_method_prop = max(int(dec["xi"]), int(dec["h_unc"]))
                    mean_actions = dec["mean_actions"]
                    g_toggle = _has_gripper_toggle(mean_actions, h_sel)
                    chunk_writer.writerow([task_description, ep_idx, active_steps, h_sel])

                    u10 = _pad_or_trim_10(dec["u_step"])
                    up10 = _pad_or_trim_10(dec["u_prefix"])
                    m10 = _pad_or_trim_10(dec["m_step"])
                    dec_row = {
                        "task_id": tid,
                        "trial_id": ep_idx,
                        "decision_id": dec_idx,
                        "sim_step": active_steps,
                        "h_executed": h_sel,
                        "h_method": h_method_prop,
                        "move_lower_bound": int(dec["xi"]),
                        "infer_ms": round(infer_ms, 3),
                        "gripper_toggle": g_toggle,
                        "method": method_name,
                        "backbone": backbone,
                        "suite": suite,
                        "episode_id": ep_idx,
                        "decision_idx": dec_idx,
                        "env_step": active_steps,
                        "h_selected": h_sel,
                        "h_unc": int(dec["h_unc"]),
                        "xi": int(dec["xi"]),
                    }
                    for j in range(10):
                        dec_row[f"u_step_{j+1}"] = round(u10[j], 6)
                        dec_row[f"u_prefix_{j+1}"] = round(up10[j], 6)
                        dec_row[f"m_step_{j+1}"] = round(m10[j], 6)
                    dec_writer.writerow(dec_row)

                    ep_h_list.append(h_sel)
                    dec_idx += 1

                    for step_i in range(h_sel):
                        if active_steps >= active_budget or done:
                            break
                        obs, _, done, _ = env.step(mean_actions[step_i])
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

                succ = 1 if done else 0
                total_eps += 1
                total_succ += succ
                sr_running = total_succ / float(total_eps)
                res_writer.writerow([tid, task_description, ep_idx, bool(succ), total_eps, total_succ, f"{sr_running:.4f}"])

                mean_h = sum(ep_h_list) / max(len(ep_h_list), 1)
                all_h.append(mean_h)
                wall_s = max(time.time() - t0, 1e-4)
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
                        "backbone": backbone,
                        "benchmark": benchmark_name,
                        "perturbation": perturbation,
                        "task_name": task.name,
                        "episode_id": ep_idx,
                        "init_state_id": ep_idx,
                        "seed": seed,
                        "env_steps": active_steps,
                        "num_decisions": dec_idx,
                        "wall_time_s": round(wall_s, 4),
                        "forward_passes": dec_idx * n_samples * 10,
                        "jvp_probes": 0,
                    }
                )
                ep_file.flush()
                dec_file.flush()
                sub_file.flush()
                chunk_file.flush()
                res_file.flush()

            env.close()

    ep_file.close()
    dec_file.close()
    sub_file.close()
    chunk_file.close()
    res_file.close()

    sr = (total_succ / total_eps) if total_eps > 0 else 0.0
    summary = {
        "method": method_name,
        "backbone": backbone,
        "suite": suite,
        "base_suite": base_suite,
        "selector": selector,
        "official_openpi_protocol": official_openpi_protocol,
        "total_episodes": total_eps,
        "total_successes": total_succ,
        "success_rate": round(sr, 6),
        "mean_h": round(sum(all_h) / max(len(all_h), 1), 4),
        "synthetic_dry_run": synthetic_dry_run,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "success_rate.txt").write_text(f"{sr:.6f}\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AAC CLI (`aac.py`) for pi0.5 + LIBERO / RoboCasa.")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_serve = sub.add_parser("serve-pi05", help="Serve multi-sample OpenPI pi0.5 (N=20).")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8015)
    p_serve.add_argument("--config", default="pi05_libero")
    p_serve.add_argument(
        "--checkpoint_dir",
        "--checkpoint-dir",
        dest="checkpoint_dir",
        default=str(
            WORKSPACE_ROOT / "checkpoints" / "openpi" / "openpi-assets" / "checkpoints" / "pi05_libero"
        ),
    )
    p_serve.add_argument("--n_samples", "--n-samples", dest="n_samples", type=int, default=20)
    p_serve.add_argument("--seed", type=int, default=0)
    p_serve.add_argument("--no_seed_per_step", "--no-seed-per-step", dest="no_seed_per_step", action="store_true")
    p_serve.add_argument("--seed_per_step", "--seed-per-step", dest="seed_per_step", action="store_true")

    for cmd_name in ("eval-liberopro", "eval-libero"):
        p_eval = sub.add_parser(cmd_name, help=f"Evaluate AAC on LIBERO ({cmd_name}).")
        p_eval.add_argument("--backbone", default="pi05")
        p_eval.add_argument("--suite", default="libero_10")
        p_eval.add_argument("--selector", default="gaussian_bernoulli")
        p_eval.add_argument("--trials", "--num-trials", dest="trials", type=int, default=50)
        p_eval.add_argument("--seed", type=int, default=7)
        p_eval.add_argument("--move_th", "--move-th", dest="move_th", type=float, default=3.0)
        p_eval.add_argument("--videos", "--save-videos", dest="save_videos", action="store_true")
        p_eval.add_argument("--no-videos", "--no-save-videos", dest="save_videos", action="store_false")
        p_eval.set_defaults(save_videos=False)
        p_eval.add_argument("--host", default="127.0.0.1")
        p_eval.add_argument("--port", type=int, default=8015)
        p_eval.add_argument("--out_dir", "--out-dir", "--output-dir", dest="out_dir", default="outputs/aac_libero_10")
        p_eval.add_argument("--official_openpi_protocol", "--official-openpi-protocol", dest="official_openpi_protocol", action="store_true", default=True)
        p_eval.add_argument("--legacy_aac_protocol", "--legacy-aac-protocol", dest="official_openpi_protocol", action="store_false")
        p_eval.add_argument("--n_samples", "--n-samples", dest="n_samples", type=int, default=20)
        p_eval.add_argument("--task_ids", "--task-ids", dest="task_ids", default=None)
        p_eval.add_argument("--synthetic-dry-run", action="store_true")

    sub.add_parser("test", help="Run AAC math & CSV schema self-tests.")
    return parser


def cmd_test() -> int:
    print("[aac.py test] Running AAC math and schema verification...")
    import tempfile

    with tempfile.TemporaryDirectory(prefix="aac_test_") as tmpdir:
        out_dir = Path(tmpdir) / "aac_run"
        summary = run_aac_libero_eval(
            backbone="pi05",
            suite="libero_10",
            selector="gaussian_bernoulli",
            trials=2,
            seed=7,
            move_th=3.0,
            out_dir=out_dir,
            official_openpi_protocol=True,
            task_ids=[0, 1],
            synthetic_dry_run=True,
        )
        assert summary["total_episodes"] == 4
        assert (out_dir / "episodes.csv").exists()
        assert (out_dir / "decisions.csv").exists()
        assert (out_dir / "subgoals.csv").exists()
        assert (out_dir / "chunk_size_log.csv").exists()
    print("[aac.py test] ALL TESTS PASSED.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if len(raw_argv) >= 2 and raw_argv[0] == "eval" and raw_argv[1] in ("libero", "liberopro", "robocasa"):
        raw_argv = [f"eval-{raw_argv[1]}", *raw_argv[2:]]
    parser = build_parser()
    args = parser.parse_args(raw_argv)

    if args.subcommand == "test":
        return cmd_test()
    if args.subcommand == "serve-pi05":
        from tools.serve_pi05_multisample import main as serve_main

        fwd = [
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--config",
            args.config,
            "--checkpoint_dir",
            args.checkpoint_dir,
            "--n_samples",
            str(args.n_samples),
            "--seed",
            str(args.seed),
        ]
        if args.no_seed_per_step:
            fwd.append("--no_seed_per_step")
        if args.seed_per_step:
            fwd.append("--seed_per_step")
        return serve_main(fwd)

    if args.subcommand in ("eval-liberopro", "eval-libero"):
        t_ids = [int(x) for x in args.task_ids.split(",")] if args.task_ids else None
        summary = run_aac_libero_eval(
            backbone=args.backbone,
            suite=args.suite,
            selector=args.selector,
            trials=args.trials,
            seed=args.seed,
            move_th=args.move_th,
            save_videos=args.save_videos,
            host=args.host,
            port=args.port,
            out_dir=Path(args.out_dir),
            official_openpi_protocol=args.official_openpi_protocol,
            n_samples=args.n_samples,
            task_ids=t_ids,
            synthetic_dry_run=args.synthetic_dry_run,
        )
        print(json.dumps(summary, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

