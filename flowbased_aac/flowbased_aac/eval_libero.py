"""One LIBERO / LIBERO-PRO evaluator for every O-LONG Track A arm (docs/task3010.md).

Arms
  duac      DUAC server (``duac.py serve-pi05``), h = max(h_unc, xi)
  fixed_k   same server, h = k (DUAC's proposal is still logged as h_method)
  random_h  same server, h drawn from the h distribution of a DUAC run (``--random-horizon-from``)
  aac       multi-sample server (``aac.py serve-pi05``), official AAC selector, executes sample 0

Protocol (reference evaluator of hactt13 / official OpenPI LIBERO example)
  OffScreenRenderEnv 256x256, env.seed(seed), init state = trial id, 10 dummy wait steps, then
  ``max_steps`` (520 for libero_10) policy steps; raw images are sent and the server resizes to 224.

Step 0.1  ``libero_<base>_with_*`` suites use the base suite's step budget and are loaded from
          LIBERO-PRO (``external/LIBERO-PRO``); unknown suites or too few init states raise.
Step 0.2  every goal predicate of the BDDL problem is evaluated after every ``env.step``; the
          first step at which it holds is written to ``subgoals.csv``.
State bank  per-step MuJoCo state, gripper command, warm start, executed actions and chunks
          (same format as the reference ``--state-bank``), so episodes can be restored/branched.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .duac_policy import DUACConfig, select_duac_horizon

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
LIBERO_ROOTS = {
    "libero": (WORKSPACE_ROOT / "external" / "LIBERO", WORKSPACE_ROOT / ".libero"),
    "liberopro": (WORKSPACE_ROOT / "external" / "LIBERO-PRO", WORKSPACE_ROOT / ".libero-pro"),
}

SUITE_MAX_STEPS: Dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
CLEAN_SUITES = set(SUITE_MAX_STEPS)
NUM_STEPS_WAIT = 10
ENV_RESOLUTION = 256
DUMMY_ACTION = [0.0] * 6 + [-1.0]
CHUNK_LOG_LEN = 10  # pi0.5 LIBERO chunk H = 10

EPISODES_CSV_COLUMNS = [
    "suite", "task_id", "trial_id", "arm", "success", "end_step", "n_decisions", "mean_h", "wall_s",
    "task_name", "instruction", "benchmark", "perturbation", "seed",
]
DECISIONS_CSV_COLUMNS = (
    ["task_id", "trial_id", "decision_id", "sim_step", "h_executed", "h_method", "h_unc",
     "move_lower_bound", "infer_ms", "roundtrip_ms", "gripper_toggle", "arm", "suite"]
    + [f"u_step_{j}" for j in range(1, CHUNK_LOG_LEN + 1)]
    + [f"u_prefix_{j}" for j in range(1, CHUNK_LOG_LEN + 1)]
    + [f"m_step_{j}" for j in range(1, CHUNK_LOG_LEN + 1)]
)
SUBGOALS_CSV_COLUMNS = ["task_id", "trial_id", "arm", "predicate_idx", "predicate", "first_true_step"]

# Arguments that must match for a resumed run to remain one experiment.
RESUME_CHECKED_ARGS = (
    "suite", "arm", "seed", "trials", "max_steps", "alpha", "tau", "num_probes", "fixed_horizon",
    "random_horizon_from", "selector", "libero_variant", "trial_ids",
)

PRO_LANGUAGE_VARIANT = re.compile(r"^libero_(spatial|object|goal|10)_(lan|object|swap|task|env)$")


# --------------------------------------------------------------------------- suite helpers


def suite_step_budget_name(suite_name: str) -> str:
    """Step 0.1: map ``libero_10_with_mug`` (or ``*_temp_x0.2``) to the base suite of its step budget."""

    s = suite_name.strip().lower()
    for pattern in (
        r"^(libero_(?:10|90|goal|spatial|object))_with_.+$",
        r"^(libero_(?:10|90|goal|spatial|object))_temp_[xy][0-9.]+$",
        r"^(libero_(?:10|goal|spatial|object))_(?:lan|object|swap|task|env)$",
    ):
        match = re.match(pattern, s)
        if match:
            return match.group(1)
    return s


def max_steps_for_suite(suite_name: str) -> int:
    base = suite_step_budget_name(suite_name)
    if base not in SUITE_MAX_STEPS:
        raise ValueError(f"unsupported suite {suite_name!r} (base {base!r})")
    return SUITE_MAX_STEPS[base]


def parse_perturbation_from_suite(suite_name: str) -> str:
    s = suite_name.strip().lower()
    if s in CLEAN_SUITES:
        return "clean"
    base = suite_step_budget_name(s)
    return s[len(base) + 1 :] if s.startswith(base + "_") else s


def libero_variant_for(suite_name: str, force_pro: bool = False) -> str:
    """LIBERO-PRO is needed for every non-clean suite; ``force_pro`` also uses it for clean suites."""

    return "liberopro" if force_pro or suite_name not in CLEAN_SUITES else "libero"


def activate_libero(variant: str) -> None:
    """Point ``import libero`` and its config at LIBERO or LIBERO-PRO (must run before the import)."""

    source, config_dir = LIBERO_ROOTS[variant]
    if not (source / "libero" / "libero").exists():
        raise FileNotFoundError(f"{variant} source not found at {source}")
    if not (config_dir / "config.yaml").exists():
        raise FileNotFoundError(f"missing {config_dir / 'config.yaml'} (run tools/download_libero.py)")
    loaded = sys.modules.get("libero")
    if loaded is not None:
        loaded_from = Path(getattr(loaded, "__file__", "") or list(loaded.__path__)[0]).resolve()
        if source.resolve() not in loaded_from.parents:
            raise RuntimeError(f"libero already imported from {loaded_from}; cannot switch to {variant} in-process")
        return
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    sys.path.insert(0, str(source))
    import libero.libero

    loaded_from = Path(libero.libero.__file__).resolve()
    if source.resolve() not in loaded_from.parents:
        raise RuntimeError(
            f"wanted {variant} from {source} but imported {loaded_from}; "
            f"make sure {source / 'libero' / '__init__.py'} exists"
        )


# --------------------------------------------------------------------------- step 0.2: subgoals


def get_libero_goal_predicates(env: Any) -> List[Any]:
    inner = getattr(env, "env", env)
    parsed = getattr(inner, "parsed_problem", None)
    return list(parsed.get("goal_state", [])) if isinstance(parsed, dict) else []


def eval_libero_goal_predicates(env: Any) -> Tuple[List[str], List[bool]]:
    """Evaluate each BDDL goal predicate (``inner._eval_predicate`` as in ``_check_success``)."""

    inner = getattr(env, "env", env)
    goal_state = get_libero_goal_predicates(env)
    if not goal_state:
        raise RuntimeError("environment exposes no BDDL goal_state; cannot log subgoals")
    names = [" ".join(str(x) for x in g) if isinstance(g, (list, tuple)) else str(g) for g in goal_state]
    flags = [bool(inner._eval_predicate(g)) for g in goal_state]
    return names, flags


def evaluate_libero_subgoals(env: Any) -> Tuple[int, int, str]:
    _, flags = eval_libero_goal_predicates(env)
    return len(flags), sum(flags), "".join("1" if f else "0" for f in flags)


# --------------------------------------------------------------------------- helpers


def load_random_horizon_pools(decisions_csv_path: Path) -> Tuple[Dict[int, List[int]], List[int]]:
    """Empirical DUAC executed-horizon distribution, per task and pooled (Random-h arm)."""

    per_task: Dict[int, List[int]] = {}
    pooled: List[int] = []
    with open(decisions_csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            h = int(float(row["h_executed"]))
            per_task.setdefault(int(row["task_id"]), []).append(h)
            pooled.append(h)
    if not pooled:
        raise ValueError(f"no decisions in {decisions_csv_path}")
    return per_task, pooled


def _pad_or_trim_10(values: Sequence[float]) -> List[float]:
    out = [float(x) for x in list(values)[:CHUNK_LOG_LEN]]
    return out + [float("nan")] * (CHUNK_LOG_LEN - len(out))


def _has_gripper_toggle(actions: Sequence[Sequence[float]], h_exec: int) -> int:
    """1 if the binarized gripper command changes within the executed prefix."""

    signs = [float(a[6]) >= 0.0 for a in list(actions)[: max(h_exec, 1)]]
    return int(any(s != signs[0] for s in signs))


def _quat2axisangle(quat: Sequence[float]) -> List[float]:
    x, y, z, w = (float(q) for q in quat)
    w = min(1.0, max(-1.0, w))
    den = math.sqrt(max(1.0 - w * w, 0.0))
    if math.isclose(den, 0.0):
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.acos(w)
    return [x * angle / den, y * angle / den, z * angle / den]


def save_state_bank(path: Path, record: Dict[str, Any]) -> None:
    """Reference state-bank format: ``states[t]`` is the MuJoCo state the observation at step t was
    rendered from and ``actions[t]`` the action applied at step t (wait steps included)."""

    import numpy as np

    arrays = {
        "states": np.asarray(record["states"], dtype=np.float64),
        "gripper_actions": np.asarray(record["gripper_actions"], dtype=np.float64),
        "qacc_warmstart": np.asarray(record["qacc_warmstart"], dtype=np.float64),
        "initial_joint": np.asarray(record["initial_joint"], dtype=np.float64),
        "actions": np.asarray(record["actions"], dtype=np.float64).reshape(-1, 7),
        "decision_steps": np.asarray(record["decision_steps"], dtype=np.int32),
        "chunks": np.asarray(record["chunks"], dtype=np.float32),
        "normalized_chunks": np.asarray(record["normalized_chunks"], dtype=np.float32),
        "chunk_sizes": np.asarray(record["chunk_sizes"], dtype=np.int32),
        "num_steps_wait": np.asarray(NUM_STEPS_WAIT, dtype=np.int32),
        "success": np.asarray(record["success"]),
    }
    for key in ("aac_samples", "aac_normalized_samples", "duac_uncertainty", "duac_divergence"):
        if record.get(key):
            arrays[key] = np.asarray(record[key], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _read_complete_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return [r for r in csv.DictReader(handle) if None not in r and None not in r.values()]


def _rewrite(path: Path, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_resume(out_dir: Path) -> Dict[Tuple[int, int], Dict[str, str]]:
    """Keep finished episodes only; drop decisions/subgoals of the episode that was interrupted."""

    episodes = _read_complete_rows(out_dir / "episodes.csv")
    done = {(int(r["task_id"]), int(r["trial_id"])): r for r in episodes}
    for name, columns in (("decisions.csv", DECISIONS_CSV_COLUMNS), ("subgoals.csv", SUBGOALS_CSV_COLUMNS)):
        rows = [r for r in _read_complete_rows(out_dir / name) if (int(r["task_id"]), int(r["trial_id"])) in done]
        _rewrite(out_dir / name, columns, rows)
    _rewrite(out_dir / "episodes.csv", EPISODES_CSV_COLUMNS, list(done.values()))
    log = out_dir / "algo_log.jsonl"
    if log.exists():
        kept = []
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # line cut by the interruption
                continue
            if (int(rec["task_id"]), int(rec["trial_id"])) in done:
                kept.append(line)
        log.write_text("".join(k + "\n" for k in kept), encoding="utf-8")
    return done


# --------------------------------------------------------------------------- simulator backend


class LiberoBackend:
    """Real LIBERO / LIBERO-PRO tasks, environments and the websocket policy client."""

    def __init__(self, suite: str, variant: str, host: str, port: int):
        activate_libero(variant)
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from openpi_client import websocket_client_policy

        registry = benchmark.get_benchmark_dict()
        if suite not in registry:
            raise ValueError(f"suite {suite!r} is not registered in {variant} ({LIBERO_ROOTS[variant][0]})")
        self.suite = suite
        self.task_suite = registry[suite]()
        self._bddl_root = Path(get_libero_path("bddl_files"))
        self._env_cls = OffScreenRenderEnv
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

    def init_states(self, task_id: int):
        if "_temp_" in self.suite:
            return self.task_suite.get_task_init_states(task_id, self.suite)
        return self.task_suite.get_task_init_states(task_id)

    def make_env(self, task, seed: int):
        folder = self.suite if "_temp_" in self.suite else task.problem_folder
        env = self._env_cls(
            bddl_file_name=str(self._bddl_root / folder / task.bddl_file),
            camera_heights=ENV_RESOLUTION,
            camera_widths=ENV_RESOLUTION,
        )
        env.seed(seed)
        instruction = env.language_instruction if PRO_LANGUAGE_VARIANT.match(self.suite) else task.language
        return env, instruction


# --------------------------------------------------------------------------- evaluation loop


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
    aac_selector: Optional[str] = None,
    state_bank: bool = False,
    save_videos: bool = False,
    video_trials: int = 2,
    output_dir: Path = Path("outputs/A/libero_10/duac"),
    resume_dir: Optional[Path] = None,
    task_ids: Optional[List[int]] = None,
    trial_ids: Optional[List[int]] = None,
    force_libero_pro: bool = False,
    max_steps: Optional[int] = None,
    synthetic_dry_run: bool = False,
    backend: Any = None,
) -> Dict[str, Any]:
    """Run one arm on one suite and write args.json, episodes/decisions/subgoals CSVs, summary, state bank."""

    import numpy as np

    if aac_selector is not None:
        arm = "aac" if aac_selector == "gaussian_bernoulli" else f"aac_{aac_selector}"
    elif fixed_horizon is not None:
        if fixed_horizon < 1:
            raise ValueError("fixed_horizon must be positive")
        arm = f"fixed_{fixed_horizon}"
    elif random_horizon_from is not None:
        arm = "random_h"
    else:
        arm = "duac"

    variant = libero_variant_for(suite, force_libero_pro)
    max_steps = max_steps or max_steps_for_suite(suite)
    perturbation = parse_perturbation_from_suite(suite)
    out_dir = Path(resume_dir if resume_dir is not None else output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    args_payload = {
        "suite": suite, "arm": arm, "libero_variant": variant, "seed": seed, "trials": num_trials,
        "max_steps": max_steps, "num_steps_wait": NUM_STEPS_WAIT, "alpha": alpha, "tau": tau,
        "num_probes": num_probes, "fixed_horizon": fixed_horizon,
        "random_horizon_from": str(random_horizon_from) if random_horizon_from else None,
        "selector": aac_selector, "task_ids": task_ids, "trial_ids": trial_ids, "state_bank": state_bank,
        "save_videos": save_videos, "video_trials": video_trials, "host": host, "port": port,
        "synthetic_dry_run": synthetic_dry_run,
    }
    completed: Dict[Tuple[int, int], Dict[str, str]] = {}
    if resume_dir is not None and (out_dir / "args.json").exists():
        previous = json.loads((out_dir / "args.json").read_text())
        mismatched = [k for k in RESUME_CHECKED_ARGS if previous.get(k) != args_payload.get(k)]
        if mismatched:
            raise ValueError(f"cannot resume {out_dir}: different {', '.join(mismatched)}")
        completed = _prepare_resume(out_dir)
    else:
        if (out_dir / "episodes.csv").exists():
            raise FileExistsError(f"{out_dir} already has results; use --resume-dir or a new --output-dir")
        (out_dir / "args.json").write_text(json.dumps(args_payload, indent=2) + "\n")
        for name, columns in (("episodes.csv", EPISODES_CSV_COLUMNS), ("decisions.csv", DECISIONS_CSV_COLUMNS),
                              ("subgoals.csv", SUBGOALS_CSV_COLUMNS)):
            _rewrite(out_dir / name, columns, [])

    if backend is None:
        if synthetic_dry_run:
            from .synthetic import SyntheticBackend

            backend = SyntheticBackend(suite, multi_sample=aac_selector is not None)
        else:
            backend = LiberoBackend(suite, variant, host, port)

    random_pools: Dict[int, List[int]] = {}
    random_pool_all: List[int] = []
    if random_horizon_from is not None:
        random_pools, random_pool_all = load_random_horizon_pools(Path(random_horizon_from))
    if aac_selector is not None:
        from .aac_official import aac_decision

    task_suite = backend.task_suite
    t_list = task_ids if task_ids is not None else list(range(task_suite.n_tasks))
    rng = random.Random(seed)
    np.random.seed(seed)

    def append(name: str, columns: List[str], row: Dict[str, Any]) -> None:
        with open(out_dir / name, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=columns).writerow(row)

    for tid in t_list:
        task = task_suite.get_task(tid)
        initial_states = backend.init_states(tid)
        trials = list(trial_ids) if trial_ids is not None else list(range(num_trials))
        if max(trials) >= len(initial_states) or num_trials > len(initial_states):
            raise ValueError(
                f"{suite} task {tid} has only {len(initial_states)} initial states, {num_trials} trials requested"
            )
        env, instruction = backend.make_env(task, seed)
        cfg = DUACConfig(
            tau=tau, num_probes=num_probes, alpha=alpha, fixed_horizon=fixed_horizon,
            random_horizon_pool=(random_pools.get(tid) or random_pool_all) if random_horizon_from else None,
        )

        for trial in trials:
            if (tid, trial) in completed:
                continue
            started = time.time()
            env.reset()
            obs = env.set_init_state(initial_states[trial])
            done = False
            bank = None
            if state_bank:
                bank = {k: [] for k in ("states", "gripper_actions", "qacc_warmstart", "actions",
                                        "decision_steps", "chunks", "normalized_chunks", "chunk_sizes",
                                        "aac_samples", "aac_normalized_samples", "duac_uncertainty",
                                        "duac_divergence")}
                bank["initial_joint"] = np.array(env.robots[0].controller.initial_joint)

            def record_state() -> None:
                bank["states"].append(env.get_sim_state())
                bank["gripper_actions"].append(
                    np.broadcast_to(env.robots[0].gripper.current_action, (2,)).astype(np.float64)
                )
                bank["qacc_warmstart"].append(np.array(env.sim.data.qacc_warmstart))

            if bank is not None:
                record_state()
            for _ in range(NUM_STEPS_WAIT):
                obs, _, done, _ = env.step(DUMMY_ACTION)
                if bank is not None:
                    bank["actions"].append(DUMMY_ACTION)
                    record_state()

            predicates, flags = eval_libero_goal_predicates(env)
            first_true: List[Optional[int]] = [0 if f else None for f in flags]
            frames = []
            want_video = save_videos and trial < video_trials
            active = 0
            decisions: List[int] = []

            while active < max_steps and not done:
                request = {
                    "observation/image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
                    "observation/wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
                    "observation/state": np.concatenate(
                        (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                    "prompt": str(instruction),
                }
                sent = time.time()
                response = backend.client.infer(request)
                roundtrip_ms = (time.time() - sent) * 1000.0
                infer_ms = (response.get("policy_timing") or {}).get("infer_ms", roundtrip_ms)

                if aac_selector is not None:
                    d = aac_decision(response["actions"], response["normalized_action"],
                                     selector=aac_selector, move_th=alpha)
                    chunk, normalized_chunk = d["chunk"], d["normalized_chunk"]
                    h_exec, h_unc, xi = d["h_selected"], d["h_unc"], d["xi"]
                    u_step, u_prefix, m_step = d["u_step"], d["u_prefix"], d["m_step"]
                    detail = {"algorithm": "aac", "selector": aac_selector, **d["detail"]}
                else:
                    chunk = np.asarray(response["actions"])[:, :7]
                    normalized_chunk = np.asarray(response["normalized_action"])
                    if chunk.ndim != 2:
                        raise RuntimeError(f"expected (H, 7) actions from the DUAC server, got {chunk.shape}")
                    res = select_duac_horizon(np.asarray(response["u_step"]).tolist(), chunk, cfg, rng=rng)
                    h_exec, h_unc, xi = res.h_selected, res.h_unc, res.xi
                    u_step, u_prefix, m_step = res.u_step, res.u_prefix, res.m_step
                    detail = {
                        "algorithm": "duac", "tau": tau, "num_probes": num_probes,
                        "U": res.u_step,  # equation (6), per action position
                        "divergence": np.asarray(response.get("duac_divergence", [])).tolist(),  # Tr(dv_i/da_i)
                        "U_prefix_mean": res.u_prefix, "U_prefix_diff": res.u_diff, "h_unc": h_unc,
                        "motion_m": res.m_step, "xi": xi, "h_duac": min(len(chunk), max(h_unc, xi)),
                        "server_h_unc": int(np.asarray(response.get("duac_horizon", -1))),
                    }
                detail.update({
                    "task_id": tid, "trial_id": trial, "decision_id": len(decisions), "sim_step": active,
                    "arm": arm, "suite": suite, "h_executed": h_exec, "infer_ms": round(float(infer_ms), 3),
                    "executed_chunk": np.round(chunk[:h_exec], 6).tolist(),
                })
                with open(out_dir / "algo_log.jsonl", "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(detail) + "\n")

                row = {
                    "task_id": tid, "trial_id": trial, "decision_id": len(decisions), "sim_step": active,
                    "h_executed": h_exec, "h_method": min(len(chunk), max(h_unc, xi)), "h_unc": h_unc,
                    "move_lower_bound": xi, "infer_ms": round(float(infer_ms), 3),
                    "roundtrip_ms": round(roundtrip_ms, 3), "gripper_toggle": _has_gripper_toggle(chunk, h_exec),
                    "arm": arm, "suite": suite,
                }
                for j, (u, up, m) in enumerate(zip(_pad_or_trim_10(u_step), _pad_or_trim_10(u_prefix),
                                                   _pad_or_trim_10(m_step)), start=1):
                    row[f"u_step_{j}"], row[f"u_prefix_{j}"], row[f"m_step_{j}"] = u, up, m
                append("decisions.csv", DECISIONS_CSV_COLUMNS, row)
                decisions.append(h_exec)
                if bank is not None:
                    bank["decision_steps"].append(NUM_STEPS_WAIT + active)
                    bank["chunks"].append(chunk)
                    bank["normalized_chunks"].append(normalized_chunk)
                    bank["chunk_sizes"].append(h_exec)
                    if aac_selector is not None:  # all N samples, for offline re-analysis of AAC
                        bank["aac_samples"].append(np.asarray(response["actions"])[:, :, :7])
                        bank["aac_normalized_samples"].append(np.asarray(response["normalized_action"])[:, :, :7])
                    else:
                        bank["duac_uncertainty"].append(np.asarray(u_step))
                        bank["duac_divergence"].append(np.asarray(response.get("duac_divergence", np.zeros(len(chunk)))))

                for action in chunk[:h_exec]:
                    obs, _, done, _ = env.step(action.tolist())
                    active += 1
                    if bank is not None:
                        bank["actions"].append(action)
                        record_state()
                    _, flags = eval_libero_goal_predicates(env)
                    for i, flag in enumerate(flags):
                        if flag and first_true[i] is None:
                            first_true[i] = active
                    if want_video:
                        frames.append(np.asarray(obs["agentview_image"])[::-1, ::-1])
                    if done or active >= max_steps:
                        break

            success = int(bool(done))
            for i, predicate in enumerate(predicates):
                append("subgoals.csv", SUBGOALS_CSV_COLUMNS, {
                    "task_id": tid, "trial_id": trial, "arm": arm, "predicate_idx": i, "predicate": predicate,
                    "first_true_step": "" if first_true[i] is None else first_true[i],
                })
            if bank is not None:
                bank["success"] = bool(success)
                save_state_bank(out_dir / "state_bank" / f"task{tid:02d}_trial{trial:02d}.npz", bank)
            if want_video and frames:
                import imageio.v2 as imageio

                video_dir = out_dir / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                status = "success" if success else "failure"
                imageio.mimwrite(video_dir / f"task{tid:02d}_trial{trial:02d}_{status}.mp4", frames, fps=10)
            append("episodes.csv", EPISODES_CSV_COLUMNS, {
                "suite": suite, "task_id": tid, "trial_id": trial, "arm": arm, "success": success,
                "end_step": active, "n_decisions": len(decisions),
                "mean_h": round(sum(decisions) / max(len(decisions), 1), 4),
                "wall_s": round(time.time() - started, 3), "task_name": task.name, "instruction": instruction,
                "benchmark": variant, "perturbation": perturbation, "seed": seed,
            })
            print(f"[{arm}] {suite} task={tid} trial={trial} success={success} steps={active} "
                  f"decisions={len(decisions)}", flush=True)
        env.close()

    episodes = _read_complete_rows(out_dir / "episodes.csv")
    per_task: Dict[int, List[int]] = {}
    for r in episodes:
        per_task.setdefault(int(r["task_id"]), []).append(int(r["success"]))
    n_dec = sum(int(r["n_decisions"]) for r in episodes)
    h_total = sum(float(r["mean_h"]) * int(r["n_decisions"]) for r in episodes)
    total = len(episodes)
    successes = sum(int(r["success"]) for r in episodes)
    summary = {
        "arm": arm, "suite": suite, "libero_variant": variant, "max_steps": max_steps,
        "total_episodes": total, "total_successes": successes,
        "success_rate": round(successes / total, 6) if total else 0.0,
        "mean_h": round(h_total / n_dec, 4) if n_dec else 0.0,
        "decisions_per_episode": round(n_dec / total, 3) if total else 0.0,
        "tasks": {str(t): {"episodes": len(v), "success_rate": round(sum(v) / len(v), 4)} for t, v in sorted(per_task.items())},
        "synthetic_dry_run": synthetic_dry_run,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "success_rate.txt").write_text(
        f"Total success count:{successes}, total num_trials:{total}, total success_rate:{summary['success_rate']}\n"
    )
    return summary


__all__ = [
    "DECISIONS_CSV_COLUMNS", "EPISODES_CSV_COLUMNS", "SUBGOALS_CSV_COLUMNS", "LiberoBackend",
    "eval_libero_goal_predicates", "evaluate_libero_subgoals", "libero_variant_for", "max_steps_for_suite",
    "parse_perturbation_from_suite", "run_libero_evaluation", "suite_step_budget_name",
]
