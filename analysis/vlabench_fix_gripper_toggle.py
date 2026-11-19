"""Recompute decisions.csv `gripper_toggle` of AAC runs from algo_log.jsonl (one-off fix, idempotent).

eval_vlabench.py before 2026-11-19 computed the AAC toggle on the (N, H, 7) sample array, i.e. along samples instead of
time. algo_log.jsonl stores the executed chunk in the AAC frame (`aac_frame_chunk`, sample 0, gripper +1 close / -1
open), so the correct value is recomputed from it. Only this diagnostic column changes; AAC's decisions did not use it.

    python analysis/vlabench_fix_gripper_toggle.py outputs/vlabench/full
"""
import csv, glob, json, sys

import numpy as np

run = sys.argv[1]
n_files = n_changed = 0
for dec_path in sorted(glob.glob(f"{run}/**/aac/*/decisions.csv", recursive=True)):
    log_path = dec_path.replace("decisions.csv", "algo_log.jsonl")
    toggle = {}
    for line in open(log_path):
        d = json.loads(line)
        g = np.asarray(d["aac_frame_chunk"])[:, 6] > 0
        toggle[(d["task"], str(d["episode"]), str(d["decision_id"]))] = int(g.any() != g.all())
    rows = list(csv.DictReader(open(dec_path)))
    for r in rows:
        new = toggle[(r["task"], r["episode"], r["decision_id"])]
        n_changed += int(str(new) != r["gripper_toggle"])
        r["gripper_toggle"] = new
    with open(dec_path, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n_files += 1
print(f"fixed {n_files} AAC decisions.csv files, {n_changed} rows changed")
