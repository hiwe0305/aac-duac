#!/usr/bin/env python3
"""Exploratory look at DUAC/AAC decision logs of finished runs (no paired arm comparison).

For every run directory (one containing episodes.csv, decisions.csv, subgoals.csv) it reports:
  - success, mean executed h, the h the adaptive rule proposes (h_method), its h_unc / xi components
  - proposed h by episode phase: before subgoal 1, between subgoals, after subgoal 2 (success only)
  - proposed h when the chunk toggles the gripper vs not
  - whether the uncertainty profile separates failed from successful episodes (AUROC of mean U)

    python analysis/explore_runs.py outputs/calibration [outputs/A ...]
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import statistics as st
import sys


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(score of a random positive > random negative); ties count 1/2."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def mean(xs):
    return st.fmean(xs) if xs else float("nan")


def analyse(run: Path) -> None:
    eps = {(r["task_id"], r["trial_id"]): r for r in csv.DictReader(open(run / "episodes.csv"))}
    if not eps:
        return
    sub = defaultdict(list)
    for r in csv.DictReader(open(run / "subgoals.csv")):
        if r["first_true_step"] != "":
            sub[(r["task_id"], r["trial_id"])].append(int(r["first_true_step"]))
    phase_h = defaultdict(list)
    grip_h = defaultdict(list)
    u_ep = defaultdict(list)
    hm, hu, xi, he = [], [], [], []
    for d in csv.DictReader(open(run / "decisions.csv")):
        key = (d["task_id"], d["trial_id"])
        h = int(d["h_method"])
        hm.append(h); hu.append(int(d["h_unc"])); xi.append(int(d["move_lower_bound"])); he.append(int(d["h_executed"]))
        u = [float(d[f"u_step_{j}"]) for j in range(1, 11)]
        u_ep[key].append(mean(u))
        firsts = sorted(sub.get(key, []))
        step = int(d["sim_step"])
        if not firsts or step < firsts[0]:
            phase = "1_truoc_buoc_con_1"
        elif len(firsts) < 2 or step < firsts[1]:
            phase = "2_giua_2_buoc_con"
        else:
            phase = "3_sau_buoc_con_2"
        phase_h[phase].append(h)
        grip_h["co_dong_mo_gripper" if d["gripper_toggle"] == "1" else "khong"].append(h)

    succ = [k for k, r in eps.items() if r["success"] == "1"]
    fail = [k for k, r in eps.items() if r["success"] == "0"]
    arm = next(iter(eps.values()))["arm"]
    print(f"\n### {run.relative_to(run.parents[1])}  (arm={arm}, {len(eps)} ep, success {100 * len(succ) / len(eps):.1f}%)")
    print(f"  h thuc thi trung binh {mean(he):.2f} | h ma thuat toan de xuat (h_method) {mean(hm):.2f} "
          f"= max(h_unc {mean(hu):.2f}, xi {mean(xi):.2f}); de xuat < 10: {100 * sum(h < 10 for h in hm) / len(hm):.0f}% decision")
    print("  h de xuat theo giai doan: " + ", ".join(f"{p[2:]}={mean(v):.2f} (n={len(v)})" for p, v in sorted(phase_h.items())))
    print("  h de xuat theo gripper:   " + ", ".join(f"{p}={mean(v):.2f} (n={len(v)})" for p, v in sorted(grip_h.items())))
    if fail and succ:
        # higher uncertainty in failed episodes? (positive class = failure)
        a = auroc([mean(u_ep[k]) for k in fail], [mean(u_ep[k]) for k in succ])
        late = auroc([mean(u_ep[k][-5:]) for k in fail], [mean(u_ep[k][-5:]) for k in succ])
        print(f"  U trung binh: fail {mean([mean(u_ep[k]) for k in fail]):.5f} vs success {mean([mean(u_ep[k]) for k in succ]):.5f} "
              f"| AUROC(U -> fail) ca episode {a:.2f}, 5 decision cuoi {late:.2f}")


def main() -> int:
    roots = [Path(p) for p in (sys.argv[1:] or ["outputs/calibration"])]
    for root in roots:
        for ep in sorted(root.glob("**/episodes.csv")):
            analyse(ep.parent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
