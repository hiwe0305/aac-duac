#!/usr/bin/env python3
"""RoboCasa365 evaluator for π0.5 with the O-LONG arms (duac, fixed_k, random_h, aac).

Environment: LeRobot's official wrapper ``lerobot.envs.robocasa.RoboCasaEnv`` (RoboCasa365 RoboCasaGymEnv,
256x256 cameras robot0_agentview_left / eye_in_hand / agentview_right, 16-d state, 12-d action
[base_motion 0:4, control_mode 4, eef_pos 5:8, eef_rot 8:11, gripper 11]; the policy uses the native
robosuite order, see ARM_IDX below). Episode length = the horizon
registered by RoboCasa for the task (unless --max-steps). Seed of trial k = --seed + k, identical for every arm.

Horizon rules operate on the arm part of the action, reordered to the LIBERO/AAC layout [pos, rot, gripper]:
  duac      h = max(h_unc, xi) from the server's U profile (equation 8) and AAC's motion bound xi(alpha)
  fixed_k   h = k                    random_h  h ~ DUAC's executed-h distribution of the same task
  aac       official AAC selector (external/aac-libero) on N samples; executes sample 0
Base / mode dims are executed unchanged and do not enter the entropy or xi (documented choice).

Outputs per run: args.json, episodes.csv, decisions.csv, algo_log.jsonl, summary.json,
state_bank/<task>_trial<KK>.npz (MuJoCo state at every decision, chunks, executed actions), videos/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "flowbased_aac", ROOT / "external" / "openpi" / "packages" / "openpi-client" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from flowbased_aac.duac_policy import DUACConfig, select_duac_horizon  # noqa: E402

# The checkpoint (lerobot/pi05_robocasa) was trained in the native robosuite PandaOmron layout, verified from its
# normalization stats (gripper dim 6 std 0.99, torso dim 10 std 0, base_pos at state 7:10 with mean (2.67, -2.24, 0.70)):
#   action [eef_pos 0:3, eef_rot 3:6, gripper 6, base 7:10, torso 10, control_mode 11]
#   state  [base_to_eef_pos 0:3, base_to_eef_quat 3:7, base_pos 7:10, base_quat 10:14, gripper_qpos 14:16]
# The LeRobot RoboCasaEnv wrapper uses another order, so state and action are converted here.
ARM_IDX = np.r_[0:3, 3:6, 6]  # model action -> [eef_pos, eef_rot, gripper] (AAC / xi layout)
STATE_FROM_WRAPPER = np.r_[7:10, 10:14, 0:3, 3:7, 14:16]  # wrapper agent_pos -> model state


def to_wrapper_action(a: np.ndarray) -> np.ndarray:
    """Model action (native order) -> LeRobot RoboCasaEnv flat action [base 0:4, mode 4, pos 5:8, rot 8:11, grip 11].

    Gripper and control mode were trained on {-1, 1} (threshold 0); the RoboCasa365 gym wrapper thresholds at 0.5,
    so they are binarized at 0 here.
    """
    a = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)
    grip = np.float32(1.0 if a[6] > 0 else -1.0)
    mode = np.float32(1.0 if a[11] > 0 else -1.0)
    return np.concatenate([a[7:11], [mode], a[0:3], a[3:6], [grip]]).astype(np.float32)
CAMERAS = ["robot0_agentview_left", "robot0_eye_in_hand", "robot0_agentview_right"]
EPISODE_COLS = ["task", "trial", "seed", "arm", "success", "end_step", "max_steps", "n_decisions", "mean_h", "wall_s",
                "instruction"]
DECISION_COLS = ["task", "trial", "decision_id", "sim_step", "h_executed", "h_method", "h_unc", "move_lower_bound",
                 "infer_ms", "roundtrip_ms", "gripper_toggle", "arm"]


def arm_part(actions: np.ndarray) -> np.ndarray:
    return np.asarray(actions)[..., ARM_IDX]


def append_csv(path: Path, cols: List[str], row: Dict[str, Any]) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return [r for r in csv.DictReader(handle) if None not in r.values()]


def make_env(task: str, max_steps: Optional[int], obj_registries):
    from lerobot.envs.robocasa import RoboCasaEnv

    return RoboCasaEnv(task=task, camera_name=",".join(CAMERAS), obs_type="pixels_agent_pos",
                       observation_width=256, observation_height=256, episode_length=max_steps,
                       obj_registries=obj_registries)


class SyntheticEnv:
    """CPU stand-in with the RoboCasaEnv interface (for --synthetic-dry-run)."""

    def __init__(self, task, max_steps, *_):
        self.task, self._max_episode_steps, self.task_description, self._t = task, max_steps or 200, f"synthetic {task}", 0

    def _obs(self):
        return {"pixels": {c: np.zeros((8, 8, 3), np.uint8) for c in CAMERAS}, "agent_pos": np.zeros(16, np.float32)}

    def reset(self, seed=None):
        self._t = 0
        return self._obs(), {"is_success": False}

    def step(self, action):
        self._t += 1
        ok = self._t >= 150
        return self._obs(), float(ok), ok, False, {"is_success": ok}

    def close(self):
        pass


class SyntheticClient:
    def __init__(self, multi, n=20, h=50):
        self.multi, self.n, self.h, self.rng = multi, n, h, np.random.default_rng(0)

    def infer(self, _):
        base = np.zeros((self.h, 12)); base[:, 0:3] = 0.3; base[:, 6] = -1; base[:, 11] = -1
        if self.multi:
            a = base[None] + self.rng.normal(size=(self.n, self.h, 12)) * np.linspace(0.02, 0.3, self.h)[None, :, None]
            return {"actions": a, "normalized_action": a, "policy_timing": {"infer_ms": 1.0}}
        u = 0.01 + 0.001 * np.arange(self.h) + np.where(np.arange(self.h) > self.rng.integers(5, 40), 0.02, 0)
        return {"actions": base + self.rng.normal(size=base.shape) * 0.05, "normalized_action": base, "u_step": u,
                "duac_divergence": np.zeros(self.h), "duac_horizon": 10, "policy_timing": {"infer_ms": 1.0}}


def run(args) -> Dict[str, Any]:
    if args.aac_selector:
        arm = "aac" if args.aac_selector == "gaussian_bernoulli" else f"aac_{args.aac_selector}"
    elif args.fixed_horizon:
        arm = f"fixed_{args.fixed_horizon}"
    elif args.random_horizon_from:
        arm = "random_h"
    else:
        arm = "duac"
    out = Path(args.resume_dir or args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in ("resume_dir", "output_dir")}
    payload["arm"] = arm
    if args.resume_dir:
        prev = json.loads((out / "args.json").read_text())
        diff = [k for k in payload if k not in ("host", "port") and prev.get(k) != payload[k]]
        if diff:
            raise ValueError(f"cannot resume {out}: different {diff}")
    elif (out / "episodes.csv").exists():
        raise FileExistsError(f"{out} already has results; use --resume-dir")
    (out / "args.json").write_text(json.dumps(payload, indent=2) + "\n")

    done_eps = {(r["task"], int(r["trial"])) for r in read_rows(out / "episodes.csv")}
    if args.resume_dir:  # drop decisions / algo log lines of the episode that was interrupted
        rows = [r for r in read_rows(out / "decisions.csv") if (r["task"], int(r["trial"])) in done_eps]
        with open(out / "decisions.csv", "w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=DECISION_COLS); w.writeheader(); w.writerows(rows)
        if (out / "algo_log.jsonl").exists():
            keep = [l for l in (out / "algo_log.jsonl").read_text().splitlines()
                    if l.strip() and (json.loads(l)["task"], json.loads(l)["trial"]) in done_eps]
            (out / "algo_log.jsonl").write_text("".join(k + "\n" for k in keep))

    if args.synthetic_dry_run:
        client = SyntheticClient(bool(args.aac_selector))
        env_factory = SyntheticEnv
    else:
        from openpi_client import websocket_client_policy

        client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
        env_factory = make_env
    if args.aac_selector:
        from flowbased_aac.aac_official import aac_decision

    pool: Dict[str, List[int]] = {}
    if args.random_horizon_from:
        for r in read_rows(Path(args.random_horizon_from)):
            pool.setdefault(r["task"], []).append(int(r["h_executed"]))
    rng = random.Random(args.seed)
    obj_registries = tuple(args.obj_registries.split(","))

    for task in args.tasks.split(","):
        env = env_factory(task, args.max_steps, obj_registries)
        cfg = DUACConfig(tau=args.tau, num_probes=args.num_probes, alpha=args.alpha, fixed_horizon=args.fixed_horizon,
                         random_horizon_pool=pool.get(task) if args.random_horizon_from else None)
        for trial in range(args.trials):
            if (task, trial) in done_eps:
                continue
            seed = args.seed + trial
            started = time.time()
            obs, _ = env.reset(seed=seed)
            max_steps = int(env._max_episode_steps)
            instruction = env.task_description
            bank = {k: [] for k in ("mujoco_state", "decision_steps", "chunks", "normalized_chunks", "chunk_sizes",
                                    "executed_actions", "aac_samples", "duac_uncertainty", "duac_divergence")}
            frames, step, hs, success = [], 0, [], False
            while step < max_steps and not success:
                request = {"images": [obs["pixels"][c] for c in CAMERAS], "state": np.asarray(obs["agent_pos"])[STATE_FROM_WRAPPER], "task": instruction}
                sent = time.time()
                reply = client.infer(request)
                roundtrip_ms = 1000 * (time.time() - sent)
                infer_ms = (reply.get("policy_timing") or {}).get("infer_ms", roundtrip_ms)
                if args.aac_selector:
                    samples = np.asarray(reply["actions"])
                    d = aac_decision(arm_part(samples), arm_part(np.asarray(reply["normalized_action"])),
                                     selector=args.aac_selector, move_th=args.alpha)
                    chunk = samples[0]
                    h, h_unc, xi = d["h_selected"], d["h_unc"], d["xi"]
                    detail = {"algorithm": "aac", **d["detail"]}
                else:
                    chunk = np.asarray(reply["actions"])
                    res = select_duac_horizon(np.asarray(reply["u_step"]).tolist(), arm_part(chunk), cfg, rng=rng)
                    h, h_unc, xi = res.h_selected, res.h_unc, res.xi
                    detail = {"algorithm": "duac", "tau": args.tau, "num_probes": args.num_probes, "U": res.u_step,
                              "divergence": np.asarray(reply["duac_divergence"]).tolist(), "U_prefix_mean": res.u_prefix,
                              "U_prefix_diff": res.u_diff, "h_unc": h_unc, "motion_m": res.m_step, "xi": xi,
                              "h_duac": min(len(chunk), max(h_unc, xi))}
                h_method = min(len(chunk), max(h_unc, xi))
                grip = arm_part(chunk)[:h, 6] >= 0
                append_csv(out / "decisions.csv", DECISION_COLS, {
                    "task": task, "trial": trial, "decision_id": len(hs), "sim_step": step, "h_executed": h,
                    "h_method": h_method, "h_unc": h_unc, "move_lower_bound": xi, "infer_ms": round(float(infer_ms), 2),
                    "roundtrip_ms": round(roundtrip_ms, 2), "gripper_toggle": int(grip.any() != grip.all()), "arm": arm})
                detail.update({"task": task, "trial": trial, "decision_id": len(hs), "sim_step": step, "arm": arm,
                               "h_executed": h, "infer_ms": round(float(infer_ms), 2),
                               "executed_chunk": np.round(chunk[:h], 5).tolist()})
                with open(out / "algo_log.jsonl", "a") as handle:
                    handle.write(json.dumps(detail) + "\n")
                if args.state_bank and not args.synthetic_dry_run:
                    bank["mujoco_state"].append(env._env.env.sim.get_state().flatten())
                bank["decision_steps"].append(step)
                bank["chunks"].append(chunk)
                bank["normalized_chunks"].append(np.asarray(reply["normalized_action"])[0] if args.aac_selector
                                                 else np.asarray(reply["normalized_action"]))
                bank["chunk_sizes"].append(h)
                if args.aac_selector:
                    bank["aac_samples"].append(np.asarray(reply["actions"], dtype=np.float32))
                else:
                    bank["duac_uncertainty"].append(np.asarray(reply["u_step"]))
                    bank["duac_divergence"].append(np.asarray(reply["duac_divergence"]))
                hs.append(h)
                for action in chunk[:h]:
                    obs, _, terminated, truncated, info = env.step(to_wrapper_action(action))
                    bank["executed_actions"].append(action)
                    step += 1
                    if args.save_videos and trial < args.video_trials:
                        frames.append(np.asarray(obs["pixels"][CAMERAS[0]]))
                    if info.get("is_success"):
                        success = True
                        break
                    if terminated or truncated or step >= max_steps:
                        break
            if args.state_bank:
                arrays = {k: np.asarray(v, dtype=np.float32) for k, v in bank.items() if v}
                (out / "state_bank").mkdir(exist_ok=True)
                np.savez_compressed(out / "state_bank" / f"{task}_trial{trial:02d}.npz", success=success, seed=seed, **arrays)
            if frames:
                import imageio.v2 as imageio
                (out / "videos").mkdir(exist_ok=True)
                imageio.mimwrite(out / "videos" / f"{task}_trial{trial:02d}_{'success' if success else 'failure'}.mp4", frames, fps=20)
            append_csv(out / "episodes.csv", EPISODE_COLS, {
                "task": task, "trial": trial, "seed": seed, "arm": arm, "success": int(success), "end_step": step,
                "max_steps": max_steps, "n_decisions": len(hs), "mean_h": round(float(np.mean(hs)), 3) if hs else 0,
                "wall_s": round(time.time() - started, 1), "instruction": instruction})
            print(f"[{arm}] {task} trial={trial} success={int(success)} steps={step}/{max_steps} decisions={len(hs)} "
                  f"mean_h={np.mean(hs) if hs else 0:.1f}", flush=True)
        env.close()

    eps = read_rows(out / "episodes.csv")
    per_task: Dict[str, List[int]] = {}
    for r in eps:
        per_task.setdefault(r["task"], []).append(int(r["success"]))
    summary = {"arm": arm, "episodes": len(eps), "success_rate": round(np.mean([int(r["success"]) for r in eps]), 4) if eps else 0,
               "mean_h": round(float(np.mean([float(r["mean_h"]) for r in eps])), 3) if eps else 0,
               "tasks": {t: round(float(np.mean(v)), 3) for t, v in per_task.items()}, "synthetic_dry_run": args.synthetic_dry_run}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True, help="comma-separated RoboCasa365 task names")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None, help="default: RoboCasa registry horizon of the task")
    ap.add_argument("--obj-registries", default="lightwheel")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--alpha", "--move-th", dest="alpha", type=float, default=3.0)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--num-probes", type=int, default=40)
    ap.add_argument("--fixed-horizon", type=int, default=None)
    ap.add_argument("--random-horizon-from", default=None)
    ap.add_argument("--aac-selector", default=None, help="gaussian_bernoulli (AAC, needs --mode aac server) or fixed_<k>")
    ap.add_argument("--state-bank", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--video-trials", type=int, default=2)
    ap.add_argument("--output-dir", default="outputs/robocasa/run")
    ap.add_argument("--resume-dir", default=None)
    ap.add_argument("--synthetic-dry-run", action="store_true")
    return ap


def main(argv=None) -> int:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
