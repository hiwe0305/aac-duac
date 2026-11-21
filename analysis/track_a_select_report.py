#!/usr/bin/env python3
"""Track A (LIBERO-10 clean + 2 noisy suites): chunk selection (medoid) vs every earlier arm, compared on many axes.

Arms: fixed_5, fixed_8, fixed_10, duac, aac, random_h (main run) and medoid_h8, sample0_h8, medoid_h10
(tools/schedule_track_a.py --stage select). All episodes are matched by (suite, task, trial) (env seed 7).

Axes
  1. success            per suite, noisy pooled (object + env), all pooled; Wilson CI
  2. paired tests       McNemar exact + bootstrap clustered by task, Holm per family; the Track A section-5
                        criterion (manifest.json) applied to the selection arms, incl. leave-one-task-out
  3. per task           where the gains come from
  4. sub-goals          predicates reached, failure modes (none / partial / all-but-not-together / success)
  5. speed and compute  steps to success (paired, both succeed), decisions per episode, server infer ms
  6. motion quality     action change at chunk seams vs inside chunks, jerk, gripper toggles, path length
  7. selection          how often the medoid differs from sample 0

    .venv-libero/bin/python analysis/track_a_select_report.py  [-> outputs/A/analysis/SELECT_REPORT.md + .json]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robocasa_report import holm, mcnemar_exact, strat_bootstrap, wilson  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "A"
SUITES = ["libero_10", "libero_10_object", "libero_10_env"]
NOISY = SUITES[1:]
FIXED = ["fixed_5", "fixed_8", "fixed_10"]
OLD = FIXED + ["duac", "aac", "random_h"]
SEL = ["medoid_h8", "sample0_h8", "medoid_h10"]
POS_UNIT = 0.05


def rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as h:
        return [r for r in csv.DictReader(h) if None not in r.values()]


def load_arm(suite, arm):
    d = OUT / suite / arm
    ep = {(suite, int(r["task_id"]), int(r["trial_id"])): r for r in rows(d / "episodes.csv")}
    sub = {}
    for r in rows(d / "subgoals.csv"):
        sub.setdefault((suite, int(r["task_id"]), int(r["trial_id"])), []).append(r["first_true_step"])
    dec = {}
    for r in rows(d / "decisions.csv"):
        dec.setdefault((suite, int(r["task_id"]), int(r["trial_id"])), []).append(float(r["infer_ms"]))
    return ep, sub, dec


def motion(suite, arm):
    """Per episode: executed actions in order, chunk-start flags, selection pick (if logged)."""
    per = {}
    path = OUT / suite / arm / "algo_log.jsonl"
    if not path.exists():
        return per
    with open(path) as h:
        for line in h:
            if not line.strip():
                continue
            d = json.loads(line)
            k = (suite, int(d["task_id"]), int(d["trial_id"]))
            e = per.setdefault(k, {"dec": [], "pick": []})
            e["dec"].append((int(d["decision_id"]), np.asarray(d["executed_chunk"], dtype=np.float64)))
            if "pick" in d:
                e["pick"].append(int(d["pick"]))
    out = {}
    for k, e in per.items():
        chunks = [c for _, c in sorted(e["dec"], key=lambda x: x[0]) if len(c)]
        if not chunks:
            continue
        a = np.concatenate(chunks)[:, :7]
        start = np.zeros(len(a), bool)
        start[np.cumsum([0] + [len(c) for c in chunks[:-1]])] = True
        da = np.linalg.norm(np.diff(a[:, :6], axis=0), axis=1)          # change between consecutive actions
        seam = da[start[1:]]                                            # first action of a new chunk vs last of previous
        inner = da[~start[1:]]
        jerk = np.linalg.norm(np.diff(a[:, :6], n=2, axis=0), axis=1)
        g = a[:, 6] >= 0
        out[k] = {"seam": float(seam.mean()) if len(seam) else np.nan, "inner": float(inner.mean()) if len(inner) else np.nan,
                  "jerk": float(jerk.mean()) if len(jerk) else np.nan, "toggles": int((g[1:] != g[:-1]).sum()),
                  "path_m": float(np.linalg.norm(a[:, :3], axis=1).sum() * POS_UNIT),
                  "pick_not0": float(np.mean(np.asarray(e["pick"]) != 0)) if e["pick"] else np.nan}
    return out


def table(rws, cols):
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rws])


def compare(S, strata, a, b, m=None):
    x, y = (S[a], S[b]) if m is None else (S[a][m], S[b][m])
    st = strata if m is None else strata[m]
    bb = int(((x == 1) & (y == 0)).sum()); cc = int(((x == 0) & (y == 1)).sum())
    lo, hi = strat_bootstrap(x, y, st)
    return {"a": a, "b": b, "diff_pp": 100 * float(x.mean() - y.mean()), "ci": [100 * lo, 100 * hi],
            "a_only": bb, "b_only": cc, "p": mcnemar_exact(bb, cc)}


def fmt(c):
    v = "better" if c["ci"][0] > 0 else "worse" if c["ci"][1] < 0 else "inconclusive"
    return [f"{c['a']} vs {c['b']}", f"{c['diff_pp']:+.1f}", f"[{c['ci'][0]:+.1f}, {c['ci'][1]:+.1f}]", c["a_only"], c["b_only"],
            f"{c['p']:.3g}", f"{c.get('p_holm', float('nan')):.3g}", v]


def main():
    arms = [a for a in OLD + SEL if all((OUT / s / a / "episodes.csv").exists() for s in SUITES)]
    EP, SUB, DEC, MOT = {}, {}, {}, {}
    for a in arms:
        EP[a], SUB[a], DEC[a], MOT[a] = {}, {}, {}, {}
        for s in SUITES:
            e, sb, dc = load_arm(s, a)
            EP[a].update(e); SUB[a].update(sb); DEC[a].update(dc); MOT[a].update(motion(s, a))
    keys = sorted(set.intersection(*(set(EP[a]) for a in arms)))
    S = {a: np.array([int(EP[a][k]["success"]) for k in keys]) for a in arms}
    suite = np.array([k[0] for k in keys]); task = np.array([f"{k[0]}/{k[1]}" for k in keys])
    noisy = np.isin(suite, NOISY); clean = suite == "libero_10"
    rep = {"arms": arms, "n_matched": len(keys)}
    md = ["# Track A: chunk selection (medoid) compared on many axes", "",
          f"*`analysis/track_a_select_report.py`; {len(keys)} matched episodes per arm "
          f"(3 suites x 10 tasks x 50 trials, env seed 7). π0.5 `pi05_libero`, chunk H = 10.*", "",
          "Arms: `medoid_h8` / `medoid_h10` = execute the medoid of the AAC server's 20 samples at h = 8 / 10; "
          "`sample0_h8` = same server and h, execute sample 0 (the chunk AAC executes): the no-selection control. "
          "Earlier arms: Fixed-5/8/10, DUAC, AAC, Random-h.", ""]

    # 1. success
    rw = []
    for a in arms:
        cells = []
        for m in [suite == s for s in SUITES] + [noisy, np.ones(len(keys), bool)]:
            k_, n_ = int(S[a][m].sum()), int(m.sum()); lo, hi = wilson(k_, n_)
            cells.append(f"{100 * k_ / n_:.1f} [{100 * lo:.0f}-{100 * hi:.0f}]")
        rw.append([f"**{a}**" if a in SEL else a] + cells)
    rep["success"] = {a: {"clean": float(S[a][clean].mean()), "noisy": float(S[a][noisy].mean()), "all": float(S[a].mean())} for a in arms}
    md += ["## 1. Success rate (%, Wilson 95% CI)", "", table(rw, ["arm", "libero_10 (clean)", "object", "env", "**noisy pooled**", "all"]), ""]

    # 2. paired tests
    md += ["## 2. Paired comparisons", "", "Bootstrap clustered by (suite, task); Holm within each block.", ""]
    rep["paired"] = {}
    for label, m in (("noisy pooled (object + env, n = 1000)", noisy), ("clean libero_10 (n = 500)", clean),
                     ("all 3 suites (n = 1500)", np.ones(len(keys), bool))):
        comps = []
        for s_arm in [a for a in ("medoid_h8", "medoid_h10") if a in arms]:
            for b in [x for x in arms if x != s_arm and not (s_arm == "medoid_h10" and x == "medoid_h8")]:
                comps.append(compare(S, task, s_arm, b, m))
        for c, p in zip(comps, holm([c["p"] for c in comps])):
            c["p_holm"] = p
        rep["paired"][label] = comps
        md += [f"**{label}**", "", table([fmt(c) for c in comps],
               ["comparison", "diff (pp)", "95% CI", "only A succeeds", "only B succeeds", "p McNemar", "p Holm", "verdict"]), ""]

    # Track A section-5 criterion (manifest.json), applied to the selection arms
    md += ["**Track A criterion (manifest, section 5)**: an arm WORKS if on the noisy pool it beats every fixed arm by "
           ">= 3 pp with Holm p < 0.05, still beats them by >= 1 pp after dropping its most favourable task, and on clean "
           "is not more than 2 pp below the best fixed arm.", ""]
    rw = []
    for s_arm in [a for a in SEL if a.startswith("medoid") and a in arms]:
        cs = [c for c in rep["paired"]["noisy pooled (object + env, n = 1000)"] if c["a"] == s_arm and c["b"] in FIXED]
        ok1 = all(c["diff_pp"] >= 3 and c["p_holm"] < 0.05 for c in cs)
        gains = {t: S[s_arm][noisy & (task == t)].mean() - max(S[f][noisy & (task == t)].mean() for f in FIXED)
                 for t in np.unique(task[noisy])}
        drop = max(gains, key=gains.get); mk = noisy & (task != drop)
        loo = min(100 * (S[s_arm][mk].mean() - S[f][mk].mean()) for f in FIXED)
        best_clean = max(FIXED, key=lambda f: S[f][clean].mean())
        dclean = 100 * (S[s_arm][clean].mean() - S[best_clean][clean].mean())
        verdict = "WORKS" if ok1 and loo >= 1 and dclean >= -2 else "does not meet"
        rw.append([s_arm, f"{min(c['diff_pp'] for c in cs):+.1f} (max Holm p {max(c['p_holm'] for c in cs):.2g})",
                   f"{drop}: {loo:+.1f}", f"{dclean:+.1f} vs {best_clean}", verdict])
        rep.setdefault("criterion", {})[s_arm] = verdict
    md += [table(rw, ["arm", "min gain vs Fixed-5/8/10 (noisy)", "leave-best-task-out: min gain", "clean vs best fixed", "verdict"]), ""]

    # 3. per task
    rw = []
    for t in np.unique(task):
        m = task == t
        rw.append([t.replace("libero_10_", "").replace("libero_10/", "clean/")] + [int(S[a][m].sum()) for a in arms])
    md += ["## 3. Successes per task (out of 50)", "", table(rw, ["suite/task"] + arms), ""]

    # 4. sub-goals and failure modes
    rw = []
    for a in arms:
        reached, modes = [], {"success": 0, "all predicates, not together": 0, "partial": 0, "none": 0}
        for k in keys:
            ft = SUB[a].get(k, [])
            r_ = [x != "" for x in ft]
            frac = float(np.mean(r_)) if r_ else 0.0
            reached.append(frac)
            if int(EP[a][k]["success"]):
                modes["success"] += 1
            elif r_ and all(r_):
                modes["all predicates, not together"] += 1
            elif any(r_):
                modes["partial"] += 1
            else:
                modes["none"] += 1
        rw.append([a, f"{100 * np.mean(reached):.1f}%"] + [modes[x] for x in modes])
    md += ["## 4. Sub-goals and failure modes (all 1500 episodes)", "",
           "A predicate counts as reached if it held at any step. 'all predicates, not together' = every goal predicate "
           "held at some point but never simultaneously (e.g. an object placed and then knocked off).", "",
           table(rw, ["arm", "predicates reached", "success", "all predicates, not together", "partial", "none"]), ""]

    # 5. speed and compute
    rw = []
    for a in arms:
        both = [k for k in keys if all(int(EP[x][k]["success"]) for x in (a, "fixed_10"))]
        st = np.mean([int(EP[a][k]["end_step"]) for k in both]) if both else np.nan
        st10 = np.mean([int(EP["fixed_10"][k]["end_step"]) for k in both]) if both else np.nan
        ndec = np.mean([int(EP[a][k]["n_decisions"]) for k in keys])
        inf = np.median([x for k in keys for x in DEC[a].get(k, [])])
        rw.append([a, f"{st:.0f} vs {st10:.0f} ({len(both)} ep)", f"{ndec:.1f}", f"{inf:.0f}", f"{ndec * inf / 1000:.2f}"])
    md += ["## 5. Speed and compute", "",
           "Steps to success: paired with Fixed-10 on episodes both solve. infer ms = median server-side model time per "
           "call as logged during the run (DUAC server for Fixed/DUAC/Random-h; AAC N = 20 server for AAC and the selection "
           "arms). Each run had its own number of concurrent clients, so infer ms and policy s / episode are **not comparable
across runs** (the selection arms ran later with 12 clients on one server); use a clean single-client measurement
for cost. Decisions / episode is comparable.", "",
           table(rw, ["arm", "steps to success (arm vs Fixed-10)", "decisions / episode", "infer ms / call", "policy s / episode"]), ""]

    # 6. motion quality
    rw = []
    for a in arms:
        v = [MOT[a][k] for k in keys if k in MOT[a]]
        if not v:
            continue
        g = lambda f: np.nanmean([x[f] for x in v])
        rw.append([a, len(v), f"{g('seam'):.3f}", f"{g('inner'):.3f}", f"{g('seam') / g('inner'):.2f}", f"{g('jerk'):.3f}",
                   f"{g('toggles'):.2f}", f"{np.mean([MOT[a][k]['path_m'] for k in keys if k in MOT[a] and int(EP[a][k]['success'])]):.2f}"])
    md += ["## 6. Motion quality (executed actions, robosuite units)", "",
           "seam = |a_t+1 - a_t| when a_t+1 starts a new chunk; inside = same within a chunk; seam / inside > 1 means "
           "jumps at chunk boundaries (mode switching). jerk = mean |second difference|. toggles = gripper open/close "
           "switches per episode. commanded path = sum of commanded translation (|dpos| x 0.05 m) over successful episodes.", "",
           table(rw, ["arm", "episodes", "seam jump", "inside-chunk change", "seam / inside", "jerk", "gripper toggles", "commanded path (m, successes)"]), ""]

    # 7. selection
    rw = []
    for a in SEL:
        if a in arms:
            v = [MOT[a][k]["pick_not0"] for k in keys if k in MOT[a]]
            rw.append([a, f"{100 * np.nanmean(v):.1f}%"])
    md += ["## 7. What the selector did", "", table(rw, ["arm", "decisions where the executed chunk != sample 0"]), ""]

    (OUT / "analysis").mkdir(exist_ok=True)
    (OUT / "analysis" / "SELECT_REPORT.md").write_text("\n".join(md) + "\n")
    (OUT / "analysis" / "select_report.json").write_text(json.dumps(rep, indent=2, default=float) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
