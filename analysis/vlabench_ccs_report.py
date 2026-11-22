#!/usr/bin/env python3
"""VLABench: Consensus-Coherent Chunk Selection (CCS) vs its controls -> <run>/REPORT_CCS.md + report_ccs.json.

Arms (all on matched (track, task, episode)):
  ccs_h8      CCS (flowbased_aac/ccs.py) over the AAC server's 20 samples, h = 8
  sample0_h8  same server and h, executes sample 0 (the chunk AAC executes): the "no selection" control
  medoid_h8   ablation: consensus only (no gripper vote, no coherence)
  fixed_8, fixed_10, vanilla, aac, duac   earlier arms of the full run (reference)

Primary test (pre-stated): ccs_h8 vs sample0_h8 and ccs_h8 vs the best fixed arm, pooled over the 5 tracks
(paired, McNemar exact, bootstrap stratified by track x task, Holm over the primary family).

    .venv-vlabench/bin/python analysis/vlabench_ccs_report.py outputs/vlabench/full
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robocasa_report import holm, mcnemar_exact, strat_bootstrap, wilson  # noqa: E402

TRACKS = ["T1", "T2", "T3", "T4", "T6"]
NEW = ["ccs_h8", "sample0_h8", "medoid_h8", "egc_h8", "xdc_h8", "medoid_h5", "medoid_h10", "mcons_h10",
       "keystone_h8", "sample0_h8_rl", "medoid_h8_rl"]
REF = ["vanilla", "fixed_8", "fixed_10", "aac", "duac"]
PRIMARY = [("ccs_h8", "sample0_h8"), ("ccs_h8", "fixed_8"), ("ccs_h8", "fixed_10")]
SECONDARY = [("medoid_h8", "sample0_h8"), ("ccs_h8", "medoid_h8"), ("sample0_h8", "fixed_8"),
             ("egc_h8", "medoid_h8"), ("xdc_h8", "medoid_h8"), ("egc_h8", "fixed_8"), ("xdc_h8", "fixed_8"),
             ("mcons_h10", "medoid_h10"), ("mcons_h10", "medoid_h8"), ("medoid_h5", "medoid_h8"), ("medoid_h10", "medoid_h8"),
             ("mcons_h10", "aac"),
             ("medoid_h8", "keystone_h8"), ("keystone_h8", "sample0_h8"),                       # P0: prior work
             ("sample0_h8_rl", "sample0_h8"), ("medoid_h8", "sample0_h8_rl"), ("medoid_h8_rl", "medoid_h8"),
             ("medoid_h8_rl", "sample0_h8_rl"),                                                 # P0: 2x2 smoothing
             ("ccs_h8", "aac"), ("ccs_h8", "duac"), ("ccs_h8", "vanilla")]


def read_csv(pattern):
    return [r for f in sorted(glob.glob(pattern)) for r in csv.DictReader(open(f)) if None not in r.values()]


def load(run):
    ep = {}
    for a in NEW + REF:
        for tr in TRACKS:
            for r in read_csv(f"{run}/{tr}/{a}/*/episodes.csv"):
                ep.setdefault(a, {})[(tr, r["task"], int(r["episode"]))] = r
    return {a: v for a, v in ep.items() if v}


def compare(S, strata, a, b):
    x, y = S[a], S[b]
    bb = int(((x == 1) & (y == 0)).sum()); cc = int(((x == 0) & (y == 1)).sum())
    lo, hi = strat_bootstrap(x, y, strata)
    return {"a": a, "b": b, "diff_pp": 100 * float(x.mean() - y.mean()), "ci95_pp": [100 * lo, 100 * hi],
            "a_only": bb, "b_only": cc, "p_mcnemar": mcnemar_exact(bb, cc)}


def table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt_cmp(c):
    v = "WORK" if c["ci95_pp"][0] > 0 else "WORSE" if c["ci95_pp"][1] < 0 else "inconclusive"
    return [f"{c['a']} vs {c['b']}", f"{c['diff_pp']:+.1f}", f"[{c['ci95_pp'][0]:+.1f}, {c['ci95_pp'][1]:+.1f}]",
            c["a_only"], c["b_only"], f"{c['p_mcnemar']:.3f}", f"{c.get('p_holm', float('nan')):.3f}", v]


def motion(run, arm):
    """Per episode, from the executed absolute targets: target jump at chunk seams vs inside chunks (m), mean |second
    difference| of the target position (m/step^2), gripper open/close toggles."""
    per = {}
    for tr in TRACKS:
        for f in glob.glob(f"{run}/{tr}/{arm}/*/algo_log.jsonl"):
            for line in open(f):
                if line.strip():
                    d = json.loads(line)
                    per.setdefault((tr, d["task"], int(d["episode"])), []).append(
                        (int(d["decision_id"]), np.asarray(d["executed_chunk"], dtype=np.float64)))
    out = {}
    for k, dec in per.items():
        chunks = [c for _, c in sorted(dec, key=lambda x: x[0]) if len(c)]
        if len(chunks) < 2:
            continue
        a = np.concatenate(chunks)
        start = np.zeros(len(a), bool); start[np.cumsum([0] + [len(c) for c in chunks[:-1]])] = True
        dp = np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1)
        g = a[:, 6] >= 0.1
        out[k] = {"seam": dp[start[1:]].mean(), "inner": dp[~start[1:]].mean(),
                  "acc": np.linalg.norm(np.diff(a[:, :3], n=2, axis=0), axis=1).mean(), "toggles": int((g[1:] != g[:-1]).sum())}
    return out


def ccs_diagnostics(run, arm):
    logs = []
    for f in glob.glob(f"{run}/T*/{arm}/*/algo_log.jsonl"):
        for line in open(f):
            if line.strip():
                d = json.loads(line)
                logs.append((d.get("pick", 0), d.get("ccs_kept", 20), d.get("ccs_overlap", 0)))
    if not logs:
        return {}
    p, k, o = map(np.asarray, zip(*logs))
    return {"decisions": len(p), "pick_not_0": float((p != 0).mean()), "gripper_vote_dropped_any": float((k < 20).mean()),
            "mean_kept": float(k.mean()), "overlap_2": float((o == 2).mean())}


def main(run):
    run = Path(run)
    ep = load(run)
    arms = [a for a in NEW + REF if a in ep]
    keys = sorted(set.intersection(*(set(ep[a]) for a in arms)))
    S = {a: np.array([int(ep[a][k]["success"]) for k in keys]) for a in arms}
    strata = np.array([f"{k[0]}/{k[1]}" for k in keys])
    trk = np.array([k[0] for k in keys])
    rep = {"arms": arms, "n_matched": len(keys)}

    md = ["# VLABench: Consensus-Coherent Chunk Selection (CCS)", "",
          f"*Generated by `analysis/vlabench_ccs_report.py` from `{run}`. {len(keys)} matched episodes per arm "
          f"(5 tracks x 10 tasks x 50, T2 insert_flower 10).*", ""]

    # 1. success per track
    rows = []
    for tr in TRACKS:
        m = trk == tr
        if m.any():
            rows.append([tr] + [f"{100 * S[a][m].mean():.1f}%" for a in arms])
    pooled = []
    for a in arms:
        lo, hi = wilson(int(S[a].sum()), len(keys))
        pooled.append(f"**{100 * S[a].mean():.1f}%** [{100 * lo:.0f}-{100 * hi:.0f}]")
    rows.append(["**pooled**"] + pooled)
    rep["success"] = {a: float(S[a].mean()) for a in arms}
    md += ["## 1. Success rate", "", table(rows, ["track"] + arms), ""]
    fixed = [a for a in ("vanilla", "fixed_8", "fixed_10") if a in arms]
    if fixed:
        best = max(fixed, key=lambda a: S[a].sum()); rep["best_fixed"] = best
        md += [f"Best fixed arm (pooled): **{best}**.", ""]

    # 2. paired comparisons
    prim = [compare(S, strata, a, b) for a, b in PRIMARY if a in S and b in S]
    for c, p in zip(prim, holm([c["p_mcnemar"] for c in prim])):
        c["p_holm"] = p
    sec = [compare(S, strata, a, b) for a, b in SECONDARY if a in S and b in S]
    for c, p in zip(sec, holm([c["p_mcnemar"] for c in sec])):
        c["p_holm"] = p
    rep["primary"], rep["secondary"] = prim, sec
    cols = ["comparison", "diff (pp)", "95% CI", "only A succeeds", "only B succeeds", "p McNemar", "p Holm", "verdict"]
    md += ["## 2. Paired comparisons (pooled; bootstrap stratified by track x task)", "",
           "**Primary (Holm family 1)**", "", table([fmt_cmp(c) for c in prim], cols), "",
           "**Secondary (Holm family 2)**", "", table([fmt_cmp(c) for c in sec], cols), ""]

    # per-track primary
    rows = []
    for tr in TRACKS:
        m = trk == tr
        for a, b in PRIMARY[:2]:
            if a in S and b in S and m.any():
                c = compare({a: S[a][m], b: S[b][m]}, strata[m], a, b)
                rows.append([tr, f"{a} vs {b}", f"{c['diff_pp']:+.1f}", f"[{c['ci95_pp'][0]:+.1f}, {c['ci95_pp'][1]:+.1f}]",
                             c["a_only"], c["b_only"], f"{c['p_mcnemar']:.3f}"])
    md += ["**Per track**", "", table(rows, ["track", "comparison", "diff (pp)", "95% CI", "only A", "only B", "p McNemar"]), ""]

    # 3. per task (pooled over tracks)
    tasks = sorted({k[1] for k in keys}); tk = np.array([k[1] for k in keys])
    rows = [[t] + [f"{int(S[a][tk == t].sum())}/{int((tk == t).sum())}" for a in arms] for t in tasks]
    md += ["## 3. Per task (summed over tracks)", "", table(rows, ["task"] + arms), ""]

    # 4. efficiency + selection diagnostics
    rows = []
    for a in arms:
        rs = [ep[a][k] for k in keys]
        succ_steps = [int(r["end_step"]) for r in rs if r["success"] == "1"]
        rows.append([a, f"{np.mean([float(r['n_decisions']) for r in rs]):.1f}", f"{np.mean([float(r['mean_h']) for r in rs]):.2f}",
                     f"{np.mean(succ_steps):.0f}" if succ_steps else "-",
                     f"{np.mean([float(r['progress_score']) for r in rs]):.3f}", sum(1 for r in rs if r["error"])])
    md += ["## 4. Efficiency", "", table(rows, ["arm", "decisions/episode", "mean h", "steps to success", "progress score", "sim errors"]), ""]
    mot = {a: motion(run, a) for a in arms}
    rows = []
    for a in arms:
        v = [mot[a][k] for k in keys if k in mot[a]]
        if v:
            g = lambda f: float(np.mean([x[f] for x in v]))
            rows.append([a, len(v), f"{100 * g('seam'):.2f}", f"{100 * g('inner'):.2f}", f"{g('seam') / g('inner'):.2f}",
                         f"{1000 * g('acc'):.2f}", f"{g('toggles'):.2f}",
                         f"{np.mean([float(ep[a][k]['intention_score']) for k in keys]):.3f}"])
    rep["motion"] = rows
    md += ["## 4b. Motion quality and task understanding", "",
           "seam = jump of the commanded end-effector target when a new chunk starts (cm); inside = step-to-step target "
           "change within a chunk (cm); seam / inside > 1 = discontinuity at chunk boundaries. accel = mean |second "
           "difference| of the target (mm/step^2). toggles = gripper switches per episode. intention = official VLABench "
           "intention score (did the robot go for the right object).", "",
           table(rows, ["arm", "episodes", "seam (cm)", "inside (cm)", "seam / inside", "accel", "gripper toggles", "intention"]), ""]
    diag = {a: ccs_diagnostics(run, a) for a in ("ccs_h8", "medoid_h8", "egc_h8", "xdc_h8") if a in arms}
    rep["selection"] = diag
    rows = [[a, d["decisions"], f"{100 * d['pick_not_0']:.1f}%", f"{100 * d['gripper_vote_dropped_any']:.1f}%",
             f"{d['mean_kept']:.1f}", f"{100 * d['overlap_2']:.1f}%"] for a, d in diag.items() if d]
    md += ["## 5. What the selector did", "",
           table(rows, ["arm", "decisions", "picked != sample 0", "gripper vote dropped >= 1 sample", "mean kept", "boundary overlap = 2 steps"]), ""]

    (run / "REPORT_CCS.md").write_text("\n".join(md) + "\n")
    (run / "report_ccs.json").write_text(json.dumps(rep, indent=2) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/vlabench/full")
