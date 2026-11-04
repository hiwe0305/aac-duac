#!/usr/bin/env python3
"""Track A scheduler: finish every (suite, arm) run with a shared work queue instead of one serial job per run.

Work is split into units = (suite, arm, task, missing trials). Each unit writes its own shard directory
<suite>/<arm>/shards/tTT_pK/; after a stage, the shards of every arm are merged into <suite>/<arm>/
(episodes.csv, decisions.csv, subgoals.csv, algo_log.jsonl, state_bank/, videos/, summary.json), which is the
layout the analysis expects. Episodes are fixed by (task, trial, env seed 7), so splitting does not change them.

  stage 1: 2 DUAC servers  -> duac, fixed_5, fixed_8, fixed_10
  stage 2: 1 DUAC server   -> random_h (needs the merged DUAC decisions of the same suite)
           1 AAC server    -> aac (N = 20)
Re-running the script resumes: completed episodes of all shards are kept, only missing trials are queued.

    python tools/schedule_track_a.py [--clients 14] [--stage 1|2|all]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "A"
LOGS = OUT / "logs_sched"
SUITES = ["libero_10", "libero_10_object", "libero_10_env"]
TASKS = list(range(10))
TRIALS = 50
STAGE1 = ["duac", "fixed_5", "fixed_8", "fixed_10"]
COST = {"fixed_5": 2.0, "fixed_8": 1.25, "fixed_10": 1.0, "duac": 1.2, "random_h": 1.2, "aac": 1.6}
SUITE_COST = {"libero_10": 1.0, "libero_10_object": 1.4, "libero_10_env": 1.3}
PY_OPENPI = ROOT / ".venv-openpi" / "bin" / "python"
PY_LIBERO = ROOT / ".venv-libero" / "bin" / "python"
RUN_FILES = ["episodes.csv", "decisions.csv", "subgoals.csv", "algo_log.jsonl", "args.json", "summary.json",
             "success_rate.txt"]


def log(msg: str) -> None:
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "scheduler.txt", "a") as h:
        h.write(line + "\n")


def complete_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as h:
        return [r for r in csv.DictReader(h) if None not in r and None not in r.values()]


def adopt_legacy(arm_dir: Path) -> None:
    """Move a run written directly into <arm>/ (the old one-job-per-run layout) into shards/legacy/."""
    shards = arm_dir / "shards"
    if (arm_dir / "episodes.csv").exists() and not (shards / "legacy").exists() and not (arm_dir / ".merged").exists():
        dest = shards / "legacy"
        dest.mkdir(parents=True)
        for name in RUN_FILES + ["state_bank", "videos"]:
            if (arm_dir / name).exists():
                shutil.move(str(arm_dir / name), str(dest / name))
        log(f"adopted {arm_dir.relative_to(OUT)} -> shards/legacy")


def done_trials(arm_dir: Path) -> Dict[int, Set[int]]:
    done: Dict[int, Set[int]] = {}
    for ep in sorted((arm_dir / "shards").glob("*/episodes.csv")):
        for r in complete_rows(ep):
            done.setdefault(int(r["task_id"]), set()).add(int(r["trial_id"]))
    return done


def compact(ids: List[int]) -> str:
    out, i = [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        out.append(f"{ids[i]}-{ids[j]}" if j > i else str(ids[i]))
        i = j + 1
    return ",".join(out)


def units_for(arms: List[str]) -> List[dict]:
    units = []
    for suite in SUITES:
        for arm in arms:
            arm_dir = OUT / suite / arm
            adopt_legacy(arm_dir)
            done = done_trials(arm_dir)
            for task in TASKS:
                missing = [t for t in range(TRIALS) if t not in done.get(task, set())]
                if not missing:
                    continue
                k = 0
                while (arm_dir / "shards" / f"t{task:02d}_p{k}").exists():
                    k += 1
                units.append({"suite": suite, "arm": arm, "task": task, "trials": missing,
                              "dir": arm_dir / "shards" / f"t{task:02d}_p{k}",
                              "cost": len(missing) * COST[arm] * SUITE_COST[suite]})
    return sorted(units, key=lambda u: -u["cost"])


def command(u: dict, port: int) -> List[str]:
    common = ["--suite", u["suite"], "--trials", str(TRIALS), "--seed", "7", "--port", str(port), "--state-bank"]
    ids = ["--task-ids", str(u["task"]), "--trial-ids", compact(u["trials"])]
    if u["arm"] == "aac":
        return [str(PY_LIBERO), "adaptive_action_chunking/aac.py", "eval-liberopro", "--backbone", "pi05",
                "--selector", "gaussian_bernoulli", "--move_th", "3", "--videos", *common, *ids, "--out_dir", str(u["dir"])]
    arm_args = {"duac": [], "fixed_5": ["--fixed-horizon", "5"], "fixed_8": ["--fixed-horizon", "8"],
                "fixed_10": ["--fixed-horizon", "10"],
                "random_h": ["--random-horizon-from", str(OUT / u["suite"] / "duac" / "decisions.csv")]}[u["arm"]]
    return [str(PY_LIBERO), "flowbased_aac/duac.py", "eval-liberopro", *common, *ids, *arm_args, "--save-videos",
            "--output-dir", str(u["dir"])]


def start_server(kind: str, port: int) -> subprocess.Popen:
    args = ([str(PY_OPENPI), "flowbased_aac/duac.py", "serve-pi05", "--seed", "0", "--port", str(port)] if kind == "duac"
            else [str(PY_OPENPI), "adaptive_action_chunking/aac.py", "serve-pi05", "--n_samples", "20", "--seed", "0",
                  "--no_seed_per_step", "--port", str(port)])
    logf = open(LOGS / f"server_{kind}_{port}.log", "w")
    proc = subprocess.Popen(args, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, env=os.environ.copy())
    while "server listening" not in (LOGS / f"server_{kind}_{port}.log").read_text():
        if proc.poll() is not None:
            raise RuntimeError(f"{kind} server on {port} died, see {LOGS}/server_{kind}_{port}.log")
        time.sleep(3)
    log(f"{kind} server ready on {port}")
    return proc


def run_queue(units: List[dict], ports: Dict[str, List[int]], clients: Dict[str, int]) -> None:
    """ports/clients per server kind ('duac' or 'aac'); every unit runs on a server of its kind."""
    pending = {k: [u for u in units if ("aac" if u["arm"] == "aac" else "duac") == k] for k in ports}
    running: List[Tuple[subprocess.Popen, dict, str, int]] = []
    env = dict(os.environ, OMP_NUM_THREADS="2")
    load = {p: 0 for ps in ports.values() for p in ps}
    total = sum(len(v) for v in pending.values())
    finished = failed = 0
    while any(pending.values()) or running:
        for kind in ports:
            while pending[kind] and sum(1 for r in running if r[2] == kind) < clients[kind]:
                u = pending[kind].pop(0)
                port = min(ports[kind], key=lambda p: load[p])
                load[port] += 1
                name = f"{u['suite']}__{u['arm']}__{u['dir'].name}"
                logf = open(LOGS / f"{name}.log", "w")
                proc = subprocess.Popen(command(u, port), cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, env=env)
                running.append((proc, u, kind, port))
                log(f"START {name} trials={compact(u['trials'])} port={port}")
        time.sleep(5)
        for item in list(running):
            proc, u, kind, port = item
            if proc.poll() is None:
                continue
            running.remove(item)
            load[port] -= 1
            finished += 1
            failed += proc.returncode != 0
            log(f"END rc={proc.returncode} {u['suite']}/{u['arm']}/{u['dir'].name} ({finished}/{total} units, {failed} failed)")


def merge(arm_dir: Path) -> int:
    """Merge shards into <arm>/; returns the number of episodes."""
    shards = sorted(p for p in (arm_dir / "shards").glob("*") if (p / "episodes.csv").exists())
    if not shards:
        return 0
    seen: Set[Tuple[int, int]] = set()
    eps, decs, subs, algo = [], [], [], []
    for sh in shards:
        own = []
        for r in complete_rows(sh / "episodes.csv"):
            key = (int(r["task_id"]), int(r["trial_id"]))
            if key not in seen:
                seen.add(key); own.append(key); eps.append(r)
        own = set(own)
        decs += [r for r in complete_rows(sh / "decisions.csv") if (int(r["task_id"]), int(r["trial_id"])) in own]
        subs += [r for r in complete_rows(sh / "subgoals.csv") if (int(r["task_id"]), int(r["trial_id"])) in own]
        if (sh / "algo_log.jsonl").exists():
            for line in (sh / "algo_log.jsonl").read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (int(rec["task_id"]), int(rec["trial_id"])) in own:
                    algo.append(line)
        for sub in ("state_bank", "videos"):
            for f in sorted((sh / sub).glob("*")) if (sh / sub).exists() else []:
                tid, tr = int(f.name[4:6]), int(f.name[12:14])  # task{TT}_trial{KK}...
                if (tid, tr) in own:
                    (arm_dir / sub).mkdir(exist_ok=True)
                    dst = arm_dir / sub / f.name
                    if not dst.exists():
                        try:
                            os.link(f, dst)
                        except OSError:
                            shutil.copy2(f, dst)
    order = lambda r: (int(r["task_id"]), int(r["trial_id"]), int(r.get("decision_id", r.get("predicate_idx", 0)) or 0))
    for name, rows in (("episodes.csv", eps), ("decisions.csv", decs), ("subgoals.csv", subs)):
        rows = sorted(rows, key=order)
        cols = list(rows[0].keys()) if rows else []
        with open(arm_dir / name, "w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=cols); w.writeheader(); w.writerows(rows)
    (arm_dir / "algo_log.jsonl").write_text("".join(l + "\n" for l in algo))
    args = json.loads((shards[0] / "args.json").read_text())
    args.update({"task_ids": None, "trial_ids": None, "merged_from": [s.name for s in shards]})
    (arm_dir / "args.json").write_text(json.dumps(args, indent=2) + "\n")
    n, k = len(eps), sum(int(r["success"]) for r in eps)
    per_task: Dict[int, List[int]] = {}
    for r in eps:
        per_task.setdefault(int(r["task_id"]), []).append(int(r["success"]))
    nd = sum(int(r["n_decisions"]) for r in eps)
    summary = {"arm": args.get("arm"), "suite": args.get("suite"), "total_episodes": n, "total_successes": k,
               "success_rate": round(k / n, 6) if n else 0.0,
               "mean_h": round(sum(float(r["mean_h"]) * int(r["n_decisions"]) for r in eps) / nd, 4) if nd else 0.0,
               "tasks": {str(t): {"episodes": len(v), "success_rate": round(sum(v) / len(v), 4)} for t, v in sorted(per_task.items())}}
    (arm_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (arm_dir / "success_rate.txt").write_text(f"Total success count:{k}, total num_trials:{n}, total success_rate:{summary['success_rate']}\n")
    (arm_dir / ".merged").write_text(time.strftime("%F %T") + "\n")
    return n


def merge_all(arms: List[str]) -> None:
    for suite in SUITES:
        for arm in arms:
            n = merge(OUT / suite / arm)
            log(f"merged {suite}/{arm}: {n}/{TRIALS * len(TASKS)} episodes")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clients", type=int, default=14)
    ap.add_argument("--stage", choices=["1", "2", "all"], default="all")
    args = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    if args.stage in ("1", "all"):
        units = units_for(STAGE1)
        log(f"stage 1: {len(units)} units, {sum(len(u['trials']) for u in units)} episodes")
        if units:
            servers = [start_server("duac", 8000), start_server("duac", 8001)]
            try:
                run_queue(units, {"duac": [8000, 8001]}, {"duac": args.clients})
            finally:
                for s in servers:
                    s.terminate()
                for s in servers:
                    s.wait()
        merge_all(STAGE1)
        if units_for(STAGE1):
            log("stage 1 incomplete (failed units); fix and re-run before stage 2"); return 1
    if args.stage in ("2", "all"):
        units = units_for(["random_h", "aac"])
        log(f"stage 2: {len(units)} units, {sum(len(u['trials']) for u in units)} episodes")
        if units:
            servers = [start_server("duac", 8000), start_server("aac", 8015)]
            n_aac = max(1, round(args.clients * 0.55))
            try:
                run_queue(units, {"duac": [8000], "aac": [8015]}, {"duac": args.clients - n_aac, "aac": n_aac})
            finally:
                for s in servers:
                    s.terminate()
                for s in servers:
                    s.wait()
        merge_all(["random_h", "aac"])
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
