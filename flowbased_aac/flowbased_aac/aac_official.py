"""AAC decision rule taken verbatim from the official release (no re-implementation).

Source: https://github.com/Adaptive-Action-Chunking/libero, cloned at ``external/aac-libero``.
The pi0.5 client ``pi05_benchmark/inference_client_action_entropy_pi05.py`` builds
``pred_action_dict`` from the multi-sample server response, calls
``action_optimization.action_entropy_pi05.select_chunk_size`` and executes the first
``chunk_size`` actions of sample ``chunk_id = 0`` (continuous gripper value).  This module does
exactly the same and additionally exposes the components (entropy profile, xi) for logging.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np

AAC_LIBERO_ROOT = Path(__file__).resolve().parents[2] / "external" / "aac-libero"


def _official():
    if not (AAC_LIBERO_ROOT / "action_optimization" / "action_entropy_pi05.py").exists():
        raise FileNotFoundError(
            f"Official AAC code not found in {AAC_LIBERO_ROOT}. Clone it with:\n"
            "  git clone --depth 1 https://github.com/Adaptive-Action-Chunking/libero.git external/aac-libero"
        )
    if str(AAC_LIBERO_ROOT) not in sys.path:
        sys.path.insert(0, str(AAC_LIBERO_ROOT))
    from action_optimization import action_entropy_pi05

    return action_entropy_pi05


def aac_decision(actions: np.ndarray, normalized_action: np.ndarray, *, selector: str, move_th: float) -> Dict[str, Any]:
    """Run the official AAC selector on one multi-sample response.

    Args:
        actions: (N, H, 7) un-normalized chunks from the multi-sample server.
        normalized_action: (N, H, >=7) the same chunks in normalized model space.
        selector: ``gaussian_bernoulli`` (AAC) or ``fixed_<k>``.
        move_th: AAC movement threshold (alpha = 3).

    Returns the executed horizon, the chunk of sample 0 to execute, and the logged components.
    """

    aac = _official()
    actions = np.asarray(actions)
    normalized_action = np.asarray(normalized_action)
    if actions.ndim != 3 or actions.shape[-1] < 7:
        raise RuntimeError(f"AAC expects (N, H, 7) actions from the multi-sample server, got {actions.shape}")
    pred_action_dict = {
        "normalized_action": normalized_action,
        "action.end_effector_position": actions[:, :, :3],
        "action.end_effector_rotation": actions[:, :, 3:6],
        "action.gripper_close": actions[:, :, 6],
    }

    # Components, computed with the official functions on a copy (select_chunk_size binarizes the gripper).
    binarized = dict(pred_action_dict)
    binarized["action.gripper_close"] = aac.convert_gripper_to_binary(actions[:, :, 6])
    breakdown = aac.action_entropy_gausssian_bernoulli(binarized)
    chunk_m = aac.action_magnitude(binarized)  # m_h for h = 2..H
    xi = int(aac.find_first_idx(chunk_m, move_th))
    diffs = np.diff(breakdown["chunk_mean"])
    h_entropy = max(int(np.argmax(diffs)) + 1, 2)

    if selector.startswith("fixed_"):
        chunk_size = int(selector.split("_", 1)[1])
    elif selector == "gaussian_bernoulli":
        chunk_size, _ = aac.select_chunk_size(pred_action_dict, method=selector, move_th=move_th)
        chunk_size = int(chunk_size)
        assert chunk_size == max(xi, h_entropy), "official select_chunk_size disagrees with its components"
    else:
        raise ValueError(f"unsupported AAC selector {selector!r}")

    horizon = actions.shape[1]
    chunk_size = max(1, min(chunk_size, horizon))
    m_step = [float(np.linalg.norm(actions[0, 0, :3]) + np.linalg.norm(actions[0, 0, 3:6]))] + [float(m) for m in chunk_m]
    return {
        "h_selected": chunk_size,
        "h_unc": h_entropy,
        "xi": xi,
        "u_step": [float(x) for x in breakdown["total"]],  # per-step entropy, the AAC uncertainty profile
        "u_prefix": [float(x) for x in breakdown["chunk_mean"]],
        "m_step": m_step,
        "chunk": actions[0, :, :7],  # chunk_id = 0, as in the official client
        "normalized_chunk": normalized_action[0],
        # Full official outputs, for algo_log.jsonl
        "detail": {
            "H_pos": [float(x) for x in breakdown["H_pos"]],
            "H_rot": [float(x) for x in breakdown["H_rot"]],
            "H_grip": [float(x) for x in breakdown["H_grip"]],
            "H_total": [float(x) for x in breakdown["total"]],
            "H_prefix_mean": [float(x) for x in breakdown["chunk_mean"]],
            "H_prefix_diff": [float(x) for x in diffs],
            "h_entropy": h_entropy,
            "motion_m_h2_to_H": [float(m) for m in chunk_m],
            "xi": xi,
            "h_aac": max(xi, h_entropy),
            "chunk_size_official": chunk_size,
            "n_samples": int(actions.shape[0]),
            "gripper_close_fraction": [float(x) for x in binarized["action.gripper_close"].mean(axis=0)],
            "sample_spread_pos": [float(x) for x in actions[:, :, :3].std(axis=0).mean(axis=-1)],
        },
    }
