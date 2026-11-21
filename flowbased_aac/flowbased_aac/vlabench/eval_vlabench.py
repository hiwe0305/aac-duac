#!/usr/bin/env python3
"""VLABench evaluator for VLABench/pi05-primitive-10task with the O-LONG arms (.venv-vlabench).

Episodes: the official evaluation tracks (VLABench/configs/evaluation/tracks/<track>.json); episode i of a task is
loaded deterministically from its stored config, exactly as ``VLABench.evaluation.evaluator.Evaluator``. Python
``random`` and numpy are additionally seeded with ``--seed + episode`` before loading, so any residual randomness
(e.g. instruction sampling) is identical for every arm. Max episode length = the task's ``evaluation.max_episode_length``
(add_condiment 300, select_poker / select_painting 100, others 200). Success = ``timestep.last()`` (task conditions met).
Progress and intention scores are the official ``env.get_task_progress()`` / ``env.get_intention_score(0.1)``.

Control: one action = one ``env.step`` (10 Hz). Target pose -> IK (``robot.get_qpos_from_ee_pos``) -> joint targets,
fingers 0.04 if gripper >= 0.1 else 0 (official client + evaluator). The observation is rendered (RGB only, 480x480)
at decision points; the official loop renders every step, which does not change the simulation state.

Arms (server mode in brackets):
  vanilla      [vanilla]  h = 5 (official replan_steps), plain OpenPI inference: reproduces the official evaluation
  fixed_k      [duac]     h = k <= 10 (DUAC's proposal is still logged); [vanilla] -> arm vanilla_fixed_k
  duac         [duac]     h = max(h_unc, xi): eq. 8 on the server's U profile, xi = AAC motion bound (alpha = 3)
  random_h     [duac]     h ~ DUAC's executed-h distribution of the same task (--random-horizon-from)
  aac          [aac]      official AAC selector (external/aac-libero) on N = 20 samples; executes sample 0
The AAC/DUAC code is called unchanged. xi reads the chunk re-expressed as LIBERO-style per-step deltas in LIBERO
units (common.to_aac_frame); everything else (entropy, U) works in the model's normalized action space.

Outputs per run (same layout as the LIBERO / RoboCasa evaluators): args.json, episodes.csv, decisions.csv,
algo_log.jsonl, summary.json, state_bank/<task>_ep<KK>.npz (optional), videos/ (optional).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "flowbased_aac", ROOT / "external" / "openpi" / "packages" / "openpi-client" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault("VLABENCH_ROOT", str(ROOT / "external" / "VLABench" / "VLABench"))

from flowbased_aac.duac_policy import DUACConfig, action_magnitude_pi05, find_first_idx, select_duac_horizon  # noqa: E402
from flowbased_aac.vlabench import common  # noqa: E402

EPISODE_COLS = ["track", "task", "episode", "seed", "arm", "success", "progress_score", "intention_score", "end_step",
                "max_steps", "n_decisions", "mean_h", "std_h", "min_h", "max_h", "infer_ms_total", "server_ms_total",
                "roundtrip_ms_total", "render_ms_total", "select_ms_total", "env_ms_total", "wall_s", "peak_gpu_mb",
                "error", "instruction"]
DECISION_COLS = ["track", "task", "episode", "decision_id", "sim_step", "prediction_horizon", "h_executed", "h_method",
                 "h_unc", "move_lower_bound", "u_mean", "u_first", "infer_ms", "server_ms", "roundtrip_ms", "render_ms",
                 "select_ms", "env_ms", "sample_count", "gripper_toggle", "arm"]


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


def load_track(track: str) -> Dict[str, list]:
    path = Path(os.environ["VLABENCH_ROOT"]) / "configs" / "evaluation" / "tracks" / f"{track}.json"
    return json.loads(path.read_text())


def max_episode_length(task: str) -> int:
    """Official rule (Evaluator.evaluate): task_config.json[<series>].evaluation.max_episode_length, default 200."""

    from VLABench.configs import name2config
    from VLABench.utils.utils import find_key_by_value

    cfg = json.loads((Path(os.environ["VLABENCH_ROOT"]) / "configs" / "task_config.json").read_text())
    return int(cfg.get(find_key_by_value(name2config, task), {}).get("evaluation", {}).get("max_episode_length", 200))


def software_versions() -> Dict[str, str]:
    out = {"python": platform.python_version(), "host": platform.node()}
    for mod in ("numpy", "mujoco", "dm_control", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    for name, path in (("vlabench_commit", ROOT / "external" / "VLABench"), ("repo_commit", ROOT),
                       ("aac_libero_commit", ROOT / "external" / "aac-libero")):
        try:
            out[name] = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------- environment


class VLABenchEnv:
    """Thin wrapper reproducing the official evaluator's per-episode protocol."""

    def __init__(self, task: str, episode_config: dict):
        from VLABench.envs import load_env
        from VLABench.robots import franka  # noqa: F401  (registers the robot)
        import VLABench.tasks  # noqa: F401  (registers the tasks)

        self.env = load_env(task, episode_config=episode_config, random_init=False, eval=False, run_mode="eval")
        self.env.reset()
        frame = np.asarray(self.env.get_robot_frame_position(), dtype=np.float64)
        if not np.allclose(frame, common.ROBOT_FRAME):
            raise RuntimeError(f"robot frame {frame} differs from the official client's constant {common.ROBOT_FRAME}")

    def observe(self, cameras=(0, 1, 2, 3)):
        e = self.env
        rgb = np.zeros((4, e.render_options["height"], e.render_options["width"], 3), np.uint8)
        for c in cameras:
            rgb[c] = e.render(camera_id=c, **e.render_options)
        return rgb, np.asarray(e.robot.get_ee_state(e.physics), dtype=np.float64), e.task.get_instruction()

    def step(self, action) -> bool:
        pos, quat, fingers = common.to_env_target(action)
        _, qpos = self.env.robot.get_qpos_from_ee_pos(physics=self.env.physics, pos=pos, quat=quat)
        return bool(self.env.step(np.concatenate([qpos, fingers])).last())

    def scores(self):
        return float(self.env.get_task_progress()), float(self.env.get_intention_score(threshold=0.1))

    def mujoco_state(self):
        return self.env.physics.get_state().copy()

    def close(self):
        self.env.close()


class SyntheticEnv:
    """CPU stand-in with the VLABenchEnv interface (no simulator, for --synthetic-dry-run)."""

    def __init__(self, task, episode_config):
        self.t, self.state = 0, np.array([0.0, 0.2, 0.4, 3.1, 0.0, -1.6, 1.0])

    def observe(self, cameras=(0, 1, 2, 3)):
        quat = common.euler_xyz_to_quat_wxyz(self.state[3:6])
        ee = np.concatenate([self.state[:3] + common.ROBOT_FRAME, quat, [self.state[6]]])
        return np.zeros((4, 8, 8, 3), np.uint8), ee, "synthetic instruction"

    def step(self, action):
        self.t += 1
        self.state = np.asarray(action, dtype=np.float64).copy()
        return self.t >= 45

    def scores(self):
        return min(1.0, self.t / 45), 1.0

    def mujoco_state(self):
        return np.zeros(4)

    def close(self):
        pass


class SyntheticClient:
    """Fake server replies with the real reply schema (absolute chunks, normalized deltas, U profile)."""

    def __init__(self, mode, n=20, h=10):
        self.mode, self.n, self.h, self.rng = mode, n, h, np.random.default_rng(0)

    def get_server_metadata(self):
        return {"mode": self.mode, "action_horizon": self.h, "n_samples": self.n}

    def infer(self, req):
        s = np.asarray(req["observation/state"])
        steps = np.arange(1, self.h + 1)[:, None]
        base = np.tile(s, (self.h, 1)); base[:, :3] += 0.01 * steps * np.array([1, 0, -0.5]); base[:, 6] = 1.0
        timing = {"infer_ms": 1.0, "server_ms": 1.2, "peak_gpu_mb": 0.0}
        if self.mode == "aac":
            a = base[None] + self.rng.normal(size=(self.n, self.h, 7)) * np.linspace(0.001, 0.02, self.h)[None, :, None]
            a[..., 6] = 1.0
            return {"actions": a, "normalized_action": np.pad(a - s, ((0, 0), (0, 0), (0, 25))), "policy_timing": timing}
        out = {"actions": base, "normalized_action": np.pad(base - s, ((0, 0), (0, 25))), "policy_timing": timing}
        if self.mode == "duac":
            u = 0.01 + 0.001 * np.arange(self.h) + np.where(np.arange(self.h) > self.rng.integers(2, 8), 0.02, 0)
            out.update(u_step=u, duac_divergence=np.zeros(self.h), duac_horizon=np.int32(4))
        return out


# --------------------------------------------------------------------------- evaluation


class ServerError(RuntimeError):
    """The policy server failed (OOM, crash, connection timeout). Never recorded as a policy failure."""


def open_env(env_factory, task, episode_config, seed: int, attempts: int = 5):
    """Build the episode's env and render once, retrying start-up failures before any policy interaction.

    Concurrent clients occasionally fail to create an EGL offscreen framebuffer ("Offscreen framebuffer is not
    complete"). Such a failure happens before the first decision, so the episode is rebuilt from scratch with the same
    seed; a render does not change the physics. Failures during the episode are not retried (recorded as errors).
    """

    for attempt in range(attempts):
        random.seed(seed); np.random.seed(seed)
        env = None
        try:
            env = env_factory(task, episode_config)
            env.observe()
            return env
        except Exception as exc:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            if attempt == attempts - 1:
                raise
            print(f"[open_env] {task}: start-up failure ({type(exc).__name__}: {str(exc)[:80]}), retry {attempt + 1}", flush=True)
            time.sleep(2 + 3 * attempt + random.random())


def arm_name(args, server_mode: str) -> str:
    if args.select:  # chunk selection among the N samples at a fixed horizon (sample0 = AAC's executed chunk)
        return f"{args.select}_h{args.fixed_horizon}"
    if args.no_xi:  # ablation: h = h_unc only (validated in run())
        return "aac_noxi" if args.aac_selector else "duac_noxi"
    if args.aac_selector:
        return "aac" if args.aac_selector == "gaussian_bernoulli" else f"aac_{args.aac_selector}"
    if args.random_horizon_from:
        return "random_h"
    if server_mode == "vanilla":
        k = args.fixed_horizon or common.OFFICIAL_REPLAN_STEPS
        return "vanilla" if k == common.OFFICIAL_REPLAN_STEPS else f"vanilla_fixed_{k}"
    if args.fixed_horizon:
        return f"fixed_{args.fixed_horizon}"
    return "duac"


def required_mode(args) -> str:
    if args.aac_selector or args.select:
        return "aac"
    if args.vanilla:
        return "vanilla"
    if args.fixed_horizon:
        return "vanilla|duac"
    return "duac"


def run(args) -> Dict[str, Any]:
    track = common.resolve_track(args.track)
    tasks = common.resolve_tasks(args.tasks, track)
    out = Path(args.resume_dir or args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.synthetic_dry_run:
        mode = "aac" if (args.aac_selector or args.select) else ("vanilla" if args.vanilla else "duac")
        client = SyntheticClient(mode)
        env_factory = SyntheticEnv
        track_cfg = {t: [{}] * 50 for t in tasks}
    else:
        from openpi_client import websocket_client_policy

        for attempt in range(30):  # a busy server can miss the websocket handshake deadline; retry before giving up
            try:
                client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
                break
            except TimeoutError as exc:
                if attempt == 29:
                    raise ServerError(f"handshake failed 30 times: {exc}") from exc
                print(f"[connect] handshake timeout, retry {attempt + 1}", flush=True)
                time.sleep(5 + random.random() * 5)
        env_factory = VLABenchEnv
        track_cfg = load_track(track)
    server = dict(client.get_server_metadata() or {})
    mode = str(server.get("mode", "?"))
    if mode not in required_mode(args).split("|"):
        raise ValueError(f"arm needs a server in mode {required_mode(args)}, but port {args.port} serves mode={mode!r}")
    if args.no_xi and (args.fixed_horizon or args.random_horizon_from or args.vanilla
                       or (args.aac_selector and args.aac_selector != "gaussian_bernoulli")):
        raise ValueError("--no-xi applies to the duac and aac (gaussian_bernoulli) arms only")
    if args.select and (not args.fixed_horizon or args.aac_selector or args.no_xi or args.random_horizon_from or args.vanilla):
        raise ValueError("--select needs --fixed-horizon and excludes --aac-selector / --no-xi / --random-horizon-from / --vanilla")
    arm = arm_name(args, mode)
    horizon_cap = int(server.get("action_horizon", common.ACTION_HORIZON))
    if args.fixed_horizon and not 1 <= args.fixed_horizon <= horizon_cap:
        raise ValueError(f"--fixed-horizon must be in [1, {horizon_cap}] (prediction horizon)")

    payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in ("resume_dir", "output_dir")}
    payload.update(arm=arm, track_resolved=track, tasks_resolved=tasks)
    if args.resume_dir:
        prev = json.loads((out / "args.json").read_text())
        diff = [k for k in payload if k not in ("host", "port") and prev.get(k) != payload[k]]
        if diff:
            raise ValueError(f"cannot resume {out}: different {diff}")
    elif (out / "episodes.csv").exists():
        raise FileExistsError(f"{out} already has results; use --resume-dir")
    (out / "args.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "run_info.json").write_text(json.dumps({"server": server, "software": software_versions(),
                                                   "started": time.strftime("%F %T")}, indent=2, default=str) + "\n")

    done = {(r["task"], int(r["episode"])) for r in read_rows(out / "episodes.csv")}
    if args.resume_dir:  # drop decisions / algo-log lines of an interrupted episode
        rows = [r for r in read_rows(out / "decisions.csv") if (r["task"], int(r["episode"])) in done]
        with open(out / "decisions.csv", "w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=DECISION_COLS); w.writeheader(); w.writerows(rows)
        if (out / "algo_log.jsonl").exists():
            keep = [l for l in (out / "algo_log.jsonl").read_text().splitlines()
                    if l.strip() and (json.loads(l)["task"], json.loads(l)["episode"]) in done]
            (out / "algo_log.jsonl").write_text("".join(k + "\n" for k in keep))

    if args.aac_selector:
        from flowbased_aac.aac_official import aac_decision
    if args.select:
        from flowbased_aac.ccs import VARIANTS, select_chunk
    pool: Dict[str, List[int]] = {}
    if args.random_horizon_from:
        for r in read_rows(Path(args.random_horizon_from)):
            pool.setdefault(r["task"], []).append(int(r["h_executed"]))
    rng = random.Random(args.seed)
    episodes = parse_ids(args.episode_ids) if args.episode_ids else list(range(args.episodes))

    for task in tasks:
        if task not in track_cfg:
            raise KeyError(f"{task} is not in {track} (has {sorted(track_cfg)})")
        max_steps = args.max_steps or (200 if args.synthetic_dry_run else max_episode_length(task))
        cfg = DUACConfig(chunk_horizon=horizon_cap, tau=args.tau, num_probes=args.num_probes, alpha=args.alpha,
                         fixed_horizon=args.fixed_horizon if mode == "duac" else None,
                         random_horizon_pool=pool.get(task) if args.random_horizon_from else None)
        if args.random_horizon_from and not cfg.random_horizon_pool:
            raise ValueError(f"no DUAC decisions for {task} in {args.random_horizon_from}")
        for ep in episodes:
            if (task, ep) in done:
                continue
            if ep >= len(track_cfg[task]):
                raise IndexError(f"{track}/{task} has {len(track_cfg[task])} episodes, asked for {ep}")
            seed = args.seed + ep
            random.seed(seed); np.random.seed(seed)
            started = time.time()
            bank = {k: [] for k in ("mujoco_state", "decision_steps", "states", "chunks", "normalized_chunks",
                                    "chunk_sizes", "executed_actions", "aac_samples", "duac_uncertainty",
                                    "duac_divergence")}
            frames, step, hs, success, error, instruction = [], 0, [], False, "", ""
            tot = dict(infer=0.0, server=0.0, roundtrip=0.0, render=0.0, select=0.0, env=0.0)
            peak_gpu, progress, intention, env = 0.0, 0.0, 0.0, None
            prev_chunk, prev_h, pick = None, 0, 0  # --select: previously executed chunk and its executed length
            try:
                env = open_env(env_factory, task, track_cfg[task][ep], seed)
                while step < max_steps and not success:
                    t = time.perf_counter()
                    rgb, ee_state, instruction = env.observe()
                    render_ms = 1000 * (time.perf_counter() - t)
                    request = common.policy_request(rgb, ee_state, instruction)
                    state = request["observation/state"]
                    t = time.perf_counter()
                    try:
                        reply = client.infer(request)
                    except Exception as exc:  # server / connection failure: not a policy outcome
                        raise ServerError(f"{type(exc).__name__}: {exc}") from exc
                    roundtrip_ms = 1000 * (time.perf_counter() - t)
                    timing = reply.get("policy_timing") or {}
                    infer_ms, server_ms = float(timing.get("infer_ms", roundtrip_ms)), float(timing.get("server_ms", roundtrip_ms))
                    peak_gpu = max(peak_gpu, float(timing.get("peak_gpu_mb", 0.0)))

                    t = time.perf_counter()
                    if args.select:
                        samples = np.asarray(reply["actions"], dtype=np.float64)
                        h = args.fixed_horizon
                        if args.select == "sample0":
                            pick, sel = 0, {}
                        else:
                            res = select_chunk(samples, h, prev_chunk, prev_h, VARIANTS[args.select])
                            pick, sel = res.index, res.as_log()
                        chunk = samples[pick]
                        conv = common.to_aac_frame(chunk, state, units=args.xi_units)
                        m = action_magnitude_pi05(conv)
                        xi, h_unc, u = find_first_idx(m, alpha=args.alpha, min_horizon=2), -1, []
                        detail = {"algorithm": args.select, "pick": pick, **sel, "xi": xi}
                        n_samples = samples.shape[0]
                    elif args.aac_selector:
                        samples = np.asarray(reply["actions"], dtype=np.float64)
                        conv = common.to_aac_frame(samples, state, units=args.xi_units)
                        d = aac_decision(conv, np.asarray(reply["normalized_action"]), selector=args.aac_selector,
                                         move_th=args.alpha)
                        chunk = samples[0]
                        h, h_unc, xi, u = d["h_selected"], d["h_unc"], d["xi"], d["u_step"]
                        if args.no_xi:  # ablation: entropy elbow alone (official components; xi still logged)
                            h = h_unc
                        detail = {"algorithm": "aac", **d["detail"]}
                        n_samples = samples.shape[0]
                    else:
                        chunk = np.asarray(reply["actions"], dtype=np.float64)
                        conv = common.to_aac_frame(chunk, state, units=args.xi_units)
                        n_samples = 1
                        if "u_step" in reply:
                            res = select_duac_horizon(np.asarray(reply["u_step"]).tolist(), conv, cfg, rng=rng)
                            h_unc, xi, u = res.h_unc, res.xi, res.u_step
                            h = res.h_unc if args.no_xi else res.h_selected  # ablation: eq. 8 alone
                            detail = {"algorithm": "duac", "tau": args.tau, "num_probes": args.num_probes, "U": res.u_step,
                                      "divergence": np.asarray(reply["duac_divergence"]).tolist(),
                                      "U_prefix_mean": res.u_prefix, "U_prefix_diff": res.u_diff, "h_unc": h_unc,
                                      "motion_m": res.m_step, "xi": xi, "h_duac": min(len(chunk), max(h_unc, xi))}
                        else:  # vanilla server: no uncertainty; xi is still logged
                            m = action_magnitude_pi05(conv)
                            xi, h_unc, u = find_first_idx(m, alpha=args.alpha, min_horizon=2), -1, []
                            h = min(len(chunk), args.fixed_horizon or common.OFFICIAL_REPLAN_STEPS)
                            detail = {"algorithm": "vanilla", "motion_m": m, "xi": xi}
                    select_ms = 1000 * (time.perf_counter() - t)
                    h = int(max(1, min(int(h), len(chunk))))
                    h_method = min(len(chunk), max(h_unc, xi)) if h_unc >= 0 else h
                    prev_chunk, prev_h = chunk, h

                    decision_step = step
                    t = time.perf_counter()
                    for action in chunk[:h]:
                        success = env.step(action)
                        bank["executed_actions"].append(action)
                        step += 1
                        if args.save_videos and ep < args.video_episodes:
                            frame = env.observe(cameras=(common.CAM_FORWARD, common.CAM_WRIST))[0]
                            frames.append(np.hstack([frame[common.CAM_FORWARD], frame[common.CAM_WRIST]]))
                        if success or step >= max_steps:
                            break
                    env_ms = 1000 * (time.perf_counter() - t)

                    grip = (conv[0] if conv.ndim == 3 else conv)[:h, 6] > 0  # executed chunk (AAC: sample 0)
                    append_csv(out / "decisions.csv", DECISION_COLS, {
                        "track": track, "task": task, "episode": ep, "decision_id": len(hs), "sim_step": decision_step,
                        "prediction_horizon": len(chunk), "h_executed": h, "h_method": h_method, "h_unc": h_unc,
                        "move_lower_bound": xi, "u_mean": round(float(np.mean(u)), 6) if len(u) else "",
                        "u_first": round(float(u[0]), 6) if len(u) else "", "infer_ms": round(infer_ms, 2),
                        "server_ms": round(server_ms, 2), "roundtrip_ms": round(roundtrip_ms, 2),
                        "render_ms": round(render_ms, 2), "select_ms": round(select_ms, 3), "env_ms": round(env_ms, 2),
                        "sample_count": n_samples, "gripper_toggle": int(grip.any() != grip.all()), "arm": arm})
                    detail.update({"track": track, "task": task, "episode": ep, "decision_id": len(hs),
                                   "sim_step": decision_step, "arm": arm, "h_executed": h, "infer_ms": round(infer_ms, 2),
                                   "state": np.round(state, 5).tolist(),
                                   "executed_chunk": np.round(chunk[:h], 5).tolist(),
                                   "aac_frame_chunk": np.round(conv[:h] if conv.ndim == 2 else conv[0, :h], 5).tolist()})
                    with open(out / "algo_log.jsonl", "a") as handle:
                        handle.write(json.dumps(detail) + "\n")
                    if args.state_bank:
                        bank["mujoco_state"].append(env.mujoco_state())
                        bank["decision_steps"].append(decision_step)
                        bank["states"].append(state)
                        bank["chunks"].append(chunk)
                        bank["normalized_chunks"].append(np.asarray(reply["normalized_action"])[pick] if args.select
                                                         else np.asarray(reply["normalized_action"])[0] if args.aac_selector
                                                         else np.asarray(reply.get("normalized_action", np.zeros(1))))
                        bank["chunk_sizes"].append(h)
                        if args.select:
                            bank.setdefault("picks", []).append(pick)
                        if args.aac_selector or args.select:
                            bank["aac_samples"].append(np.asarray(reply["actions"], dtype=np.float32))
                        elif "u_step" in reply:
                            bank["duac_uncertainty"].append(np.asarray(reply["u_step"]))
                            bank["duac_divergence"].append(np.asarray(reply["duac_divergence"]))
                    hs.append(h)
                    for k, v in (("infer", infer_ms), ("server", server_ms), ("roundtrip", roundtrip_ms),
                                 ("render", render_ms), ("select", select_ms), ("env", env_ms)):
                        tot[k] += v
                progress, intention = env.scores()
            except ServerError:
                # Infrastructure failure: do not record the episode; the job exits non-zero and a rerun
                # (--resume-dir) redoes this episode from scratch (its partial decisions are dropped on resume).
                raise
            except Exception as exc:  # the official evaluator also catches unstable episodes (and drops them)
                error = f"{type(exc).__name__}: {exc}"[:300]
                traceback.print_exc()
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
            if args.state_bank:
                arrays = {k: np.asarray(v, dtype=np.float32) for k, v in bank.items() if v}
                (out / "state_bank").mkdir(exist_ok=True)
                np.savez_compressed(out / "state_bank" / f"{task}_ep{ep:02d}.npz", success=success, seed=seed, **arrays)
            if frames:
                import imageio.v2 as imageio
                (out / "videos").mkdir(exist_ok=True)
                imageio.mimwrite(out / "videos" / f"{task}_ep{ep:02d}_{'success' if success else 'failure'}.mp4", frames, fps=10)
            hs_arr = np.asarray(hs if hs else [0])
            append_csv(out / "episodes.csv", EPISODE_COLS, {
                "track": track, "task": task, "episode": ep, "seed": seed, "arm": arm, "success": int(success),
                "progress_score": round(progress, 4), "intention_score": round(intention, 4), "end_step": step,
                "max_steps": max_steps, "n_decisions": len(hs), "mean_h": round(float(hs_arr.mean()), 3),
                "std_h": round(float(hs_arr.std()), 3), "min_h": int(hs_arr.min()), "max_h": int(hs_arr.max()),
                **{f"{k}_ms_total": round(v, 1) for k, v in tot.items()}, "wall_s": round(time.time() - started, 1),
                "peak_gpu_mb": peak_gpu, "error": error, "instruction": instruction})
            print(f"[{arm}] {track} {task} ep={ep} success={int(success)} progress={progress:.2f} steps={step}/{max_steps} "
                  f"decisions={len(hs)} mean_h={hs_arr.mean():.2f}{' ERROR ' + error if error else ''}", flush=True)

    eps = read_rows(out / "episodes.csv")
    ok = [r for r in eps if not r["error"]]
    per_task: Dict[str, List[int]] = {}
    for r in eps:
        per_task.setdefault(r["task"], []).append(int(r["success"]))
    summary = {"arm": arm, "track": track, "episodes": len(eps), "errors": len(eps) - len(ok),
               "success_rate": round(float(np.mean([int(r["success"]) for r in eps])), 4) if eps else 0.0,
               "success_rate_excl_errors": round(float(np.mean([int(r["success"]) for r in ok])), 4) if ok else 0.0,
               "progress_score": round(float(np.mean([float(r["progress_score"]) for r in eps])), 4) if eps else 0.0,
               "mean_h": round(float(np.mean([float(r["mean_h"]) for r in eps])), 3) if eps else 0.0,
               "tasks": {t: round(float(np.mean(v)), 3) for t, v in per_task.items()},
               "synthetic_dry_run": args.synthetic_dry_run}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_ids(text: str) -> List[int]:
    ids: List[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            ids.extend(range(int(a), int(b) + 1))
        elif part.strip():
            ids.append(int(part))
    return ids


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default="T1", help="T1|T2|T3|T4|T6 or the official track name")
    ap.add_argument("--tasks", default="primitive_10", help="primitive_10 or comma-separated task names")
    ap.add_argument("--episodes", type=int, default=50, help="evaluate episodes 0..N-1 of the track config")
    ap.add_argument("--episode-ids", default=None, help="explicit subset, e.g. 0-9 or 3,7")
    ap.add_argument("--seed", type=int, default=0, help="python/numpy seed of episode i is seed + i")
    ap.add_argument("--max-steps", type=int, default=None, help="default: official per-task max_episode_length")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--vanilla", action="store_true", help="official execution horizon 5 (needs --mode vanilla server)")
    ap.add_argument("--fixed-horizon", type=int, default=None)
    ap.add_argument("--random-horizon-from", default=None)
    ap.add_argument("--aac-selector", default=None, help="gaussian_bernoulli (AAC, needs --mode aac server)")
    ap.add_argument("--alpha", "--move-th", dest="alpha", type=float, default=3.0)
    ap.add_argument("--xi-units", choices=["libero", "raw"], default="libero")
    ap.add_argument("--no-xi", action="store_true",
                    help="ablation: execute h = h_unc (DUAC eq. 8 / AAC entropy elbow) without the AAC motion bound xi; "
                         "xi is still computed and logged. Arms duac_noxi / aac_noxi")
    ap.add_argument("--select", choices=["sample0", "ccs", "medoid", "coherent", "gripvote"], default=None,
                    help="chunk selection among the N samples of an aac server at --fixed-horizon h (flowbased_aac/ccs.py); "
                         "sample0 = the chunk AAC executes (control). Arm <select>_h<h>")
    ap.add_argument("--tau", type=float, default=0.9, help="logged; the server applies tau")
    ap.add_argument("--num-probes", type=int, default=40, help="logged; the server applies K")
    ap.add_argument("--state-bank", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--video-episodes", type=int, default=2)
    ap.add_argument("--output-dir", default="outputs/vlabench/run")
    ap.add_argument("--resume-dir", default=None)
    ap.add_argument("--synthetic-dry-run", action="store_true")
    return ap


def main(argv=None) -> int:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
