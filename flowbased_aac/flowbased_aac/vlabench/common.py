"""VLABench constants and the action/state conventions of ``VLABench/pi05-primitive-10task`` (numpy + scipy only).

Everything here is benchmark-side glue shared by the server (.venv-openpi) and the evaluator (.venv-vlabench).
The conventions are taken from the official client ``external/openpi-vlabench/examples/vlabench/eval.py``
(Shiduo-zh/openpi, branch pi05) and the official evaluator ``VLABench/evaluation/evaluator/base.py``:

  cameras  env.get_observation()["rgb"] = [right, left, forward, wrist] (4 x 480 x 480 x 3 uint8);
           policy "image" = forward, "second_image" = right, "wrist_image" = wrist. The server resizes (pad) to 224.
  state    [ee_pos - ROBOT_FRAME (3), euler_xyz(ee_quat_wxyz) (3), gripper_open (1)]       (7, padded to 32 by the model)
  action   [target_pos (3), target_euler_xyz (3), gripper (1)], ABSOLUTE: the model predicts deltas w.r.t. the state
           (DeltaActions on dims 0:6 during training) and the server adds the state back (AbsoluteActions).
           Execution: pos + ROBOT_FRAME, quat = euler_to_quaternion(euler), IK -> 7 joint targets,
           fingers = 0.04 (open) if gripper >= 0.1 else 0.0 (closed). One action = one env.step (10 Hz).

AAC and DUAC were written for LIBERO actions (per-step OSC deltas, robosuite units: 1.0 = 0.05 m / 0.5 rad, gripper
+1 = close). Their *uncertainty* parts work in the model's normalized action space and need no conversion. Only the
AAC motion lower bound xi (``action_magnitude`` / ``find_first_idx``, alpha = 3) reads un-normalized actions, so
``to_aac_frame`` re-expresses an absolute VLABench chunk as LIBERO-style per-step deltas in LIBERO units. alpha
therefore keeps its physical meaning (3 units = 15 cm of translation-equivalent motion). See docs/VLABENCH_INTEGRATION.md.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

# --------------------------------------------------------------------------- registry (verified against the tracks)
PRIMITIVE_TASKS: List[str] = [
    "add_condiment", "insert_flower", "select_book", "select_chemistry_tube", "select_drink",
    "select_fruit", "select_mahjong", "select_painting", "select_poker", "select_toy",
]
TRACKS: Dict[str, str] = {
    "T1": "track_1_in_distribution",
    "T2": "track_2_cross_category",
    "T3": "track_3_common_sense",
    "T4": "track_4_semantic_instruction",
    "T6": "track_6_unseen_texture",  # VLABench repo T6 (unseen texture), not the paper's long-horizon T6
}
# Track 3 has no select_poker / select_mahjong; its two common-sense variants take their place
# (the checkpoint's model card reports them in the select_poker / select_mahjong columns).
TRACK_TASK_ALIASES: Dict[str, Dict[str, str]] = {
    "track_3_common_sense": {"select_poker": "select_nth_largest_poker", "select_mahjong": "select_unique_type_mahjong"},
}
# Official reference success rates of VLABench/pi05-primitive-10task (model card, 50 episodes per task).
REFERENCE_SR: Dict[str, Dict[str, float]] = {
    "track_1_in_distribution": dict(add_condiment=0.56, insert_flower=0.22, select_book=0.545, select_chemistry_tube=0.367,
                                    select_drink=0.44, select_fruit=0.42, select_mahjong=0.38, select_painting=0.3,
                                    select_poker=0.52, select_toy=0.3),
    "track_2_cross_category": dict(add_condiment=0.06, insert_flower=0.04, select_book=0.02, select_chemistry_tube=0.16,
                                   select_drink=0.10, select_fruit=0.36, select_mahjong=0.27, select_painting=0.32,
                                   select_poker=0.5, select_toy=0.24),
    "track_3_common_sense": dict(add_condiment=0.2, insert_flower=0.1, select_book=0.383, select_chemistry_tube=0.32,
                                 select_drink=0.1, select_fruit=0.2, select_mahjong=0.0, select_painting=0.24,
                                 select_poker=0.2, select_toy=0.06),
    "track_4_semantic_instruction": dict(add_condiment=0.02, insert_flower=0.1, select_book=0.333,
                                         select_chemistry_tube=0.245, select_drink=0.30, select_fruit=0.14,
                                         select_mahjong=0.205, select_painting=0.12, select_poker=0.04, select_toy=0.074),
    "track_6_unseen_texture": dict(add_condiment=0.48, insert_flower=0.1, select_book=0.383, select_chemistry_tube=0.354,
                                   select_drink=0.18, select_fruit=0.2, select_mahjong=0.217, select_painting=0.3,
                                   select_poker=0.32, select_toy=0.08),
}

# --------------------------------------------------------------------------- policy I/O conventions
ROBOT_FRAME = np.array([0.0, -0.4, 0.78])  # hard-coded by the official client; equals env.get_robot_frame_position()
CAM_RIGHT, CAM_LEFT, CAM_FORWARD, CAM_WRIST = 0, 1, 2, 3
GRIPPER_OPEN_THRESHOLD = 0.1  # official client: fingers open iff gripper >= 0.1
FINGER_OPEN = 0.04
ACTION_HORIZON = 10  # Pi0Config(pi05=True, action_horizon=10) of pi05_ft_vlabench_primitive
OFFICIAL_REPLAN_STEPS = 5  # official client default (Args.replan_steps) = the vanilla execution horizon
# robosuite OSC_POSE defaults used by LIBERO: input 1.0 -> 0.05 m translation / 0.5 rad rotation
LIBERO_POS_UNIT = 0.05
LIBERO_ROT_UNIT = 0.5


def resolve_track(track: str) -> str:
    return TRACKS.get(track, track)


def track_task(track: str, task: str) -> str:
    """Name of ``task`` inside ``track`` (identity except for the two Track 3 substitutions)."""

    return TRACK_TASK_ALIASES.get(resolve_track(track), {}).get(task, task)


def resolve_tasks(spec: str, track: str) -> List[str]:
    """``primitive_10`` / ``all`` / comma list -> task names valid inside ``track``."""

    names = PRIMITIVE_TASKS if spec in ("primitive_10", "all") else [t.strip() for t in spec.split(",") if t.strip()]
    return [track_task(track, t) for t in names]


def quat_wxyz_to_euler_xyz(quat: Sequence[float]) -> np.ndarray:
    """VLABench ``quaternion_to_euler`` (scipy extrinsic 'xyz')."""

    from scipy.spatial.transform import Rotation

    q = np.asarray(quat, dtype=np.float64)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")


def euler_xyz_to_quat_wxyz(euler: Sequence[float]) -> np.ndarray:
    """VLABench ``euler_to_quaternion(roll, pitch, yaw)`` (same rotation as scipy extrinsic 'xyz')."""

    roll, pitch, yaw = (float(x) for x in euler)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def policy_state(ee_state: Sequence[float]) -> np.ndarray:
    """Official client state: [pos - ROBOT_FRAME, euler(quat), gripper] from ``obs["ee_state"]`` (pos, quat wxyz, open)."""

    ee = np.asarray(ee_state, dtype=np.float64)
    return np.concatenate([ee[:3] - ROBOT_FRAME, quat_wxyz_to_euler_xyz(ee[3:7]), ee[-1:]]).astype(np.float64)


def policy_request(rgb: np.ndarray, ee_state: Sequence[float], instruction: str) -> Dict[str, object]:
    """Request in the keys of the official client / VLABenchInputs (images sent at native 480x480)."""

    return {
        "observation/image": np.ascontiguousarray(rgb[CAM_FORWARD]),
        "observation/second_image": np.ascontiguousarray(rgb[CAM_RIGHT]),
        "observation/wrist_image": np.ascontiguousarray(rgb[CAM_WRIST]),
        "observation/state": policy_state(ee_state),
        "prompt": instruction,
    }


def gripper_is_open(g: float) -> bool:
    return bool(g >= GRIPPER_OPEN_THRESHOLD)


def to_env_target(action: Sequence[float]):
    """Model action -> (world target pos, target quat wxyz, finger targets (2,)), as the official client + evaluator."""

    a = np.asarray(action, dtype=np.float64)
    fingers = np.full(2, FINGER_OPEN) if gripper_is_open(a[6]) else np.zeros(2)
    return a[:3] + ROBOT_FRAME, euler_xyz_to_quat_wxyz(a[3:6]), fingers


def to_aac_frame(chunk: np.ndarray, state: Sequence[float], *, units: str = "libero") -> np.ndarray:
    """Absolute VLABench chunk(s) (..., H, 7) -> LIBERO/AAC layout: per-step deltas + gripper in {-1 (open), +1 (close)}.

    delta_pos_k = p_k - p_{k-1}, delta_rot_k = axis-angle of R_k R_{k-1}^T (world frame, so the official
    ``compose_rotations`` of steps 1..h gives the rotation from the current pose to target h), with p_{-1}, R_{-1}
    the current end-effector pose from ``state``. ``units='libero'`` divides by 0.05 m / 0.5 rad (robosuite OSC
    scaling used by LIBERO); ``units='raw'`` keeps metres / radians (then alpha = 3 practically never binds).
    """

    from scipy.spatial.transform import Rotation

    chunk = np.asarray(chunk, dtype=np.float64)
    squeeze = chunk.ndim == 2
    batch = chunk[None] if squeeze else chunk
    state = np.asarray(state, dtype=np.float64)
    out = np.empty(batch.shape[:-1] + (7,), dtype=np.float64)
    for n, c in enumerate(batch):
        pos = np.vstack([state[None, :3], c[:, :3]])
        rots = Rotation.from_euler("xyz", np.vstack([state[None, 3:6], c[:, 3:6]]))
        out[n, :, 0:3] = np.diff(pos, axis=0)
        out[n, :, 3:6] = (rots[1:] * rots[:-1].inv()).as_rotvec()
        out[n, :, 6] = np.where(c[:, 6] >= GRIPPER_OPEN_THRESHOLD, -1.0, 1.0)
    if units == "libero":
        out[..., 0:3] /= LIBERO_POS_UNIT
        out[..., 3:6] /= LIBERO_ROT_UNIT
    elif units != "raw":
        raise ValueError(f"units must be 'libero' or 'raw', got {units!r}")
    return out[0] if squeeze else out
