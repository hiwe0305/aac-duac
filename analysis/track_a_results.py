#!/usr/bin/env python3
"""Track A results against docs/task3010.md section 5 (T1, T2, T3 and the decision rule).

Pairing: episode (suite, task, trial) is identical across arms (env seed 7, init state = trial id).
  T1  success per suite x arm with Wilson 95% CI
  T2  paired difference adaptive - other arm: exact McNemar per suite and pooled over the two noisy suites,
      pooled 95% CI by cluster bootstrap over (task, trial) (5000 resamples), Holm over the 6 comparisons
      {AAC, DUAC} x {Fixed-5, Fixed-8, Fixed-10}
  T3  per-task difference and the difference without the task with the largest gain
Decision (section 5, on the two noisy suites pooled):
  work      : adaptive > all three Fixed by >= 3 pp, Holm p < 0.05, still >= 1 pp without the best task,
              and on clean not worse than the best Fixed by more than 2 pp
  not work  : upper CI bound of (adaptive - best Fixed) < 3 pp
  otherwise : inconclusive

    python analysis/track_a_results.py [--out outputs/A/analysis]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "outputs" / "A"
CLEAN = "libero_10"
NOISY = ["libero_10_object", "libero_10_env"]
ARMS = ["duac", "aac", "fixed_5", "fixed_8", "fixed_10", "random_h"]
ADAPTIVE = ["aac", "duac"]
FIXED = ["fixed_5", "fixed_8", "fixed_10"]


def load(suite: str, arm: str) -> dict:
    rows = list(csv.DictReader(open(A / suite / arm / "episodes.csv")))
    return {(int(r["task_id"]), int(r["trial_id"])): int(r["success"]) for r in rows}


def wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def mcnemar(a: dict, b: dict, keys) -> tuple:
    only_a = sum(a[k] and not b[k] for k in keys)
    only_b = sum(b[k] and not a[k] for k in keys)
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return only_a, only_b, p


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(A / "analysis"))
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    data = {(s, a): load(s, a) for s in [CLEAN] + NOISY for a in ARMS}
    for (s, a), d in data.items():
        assert len(d) == 500, f"{s}/{a}: {len(d)} episodes"
    keys = sorted(data[(CLEAN, "duac")])  # (task, trial), identical in every run
    lines, res = [], {"T1": {}, "T2": {}, "T3": {}}

    lines.append("## T1. Success (Wilson 95% CI), 500 episodes per cell\n")
    lines.append("| arm | " + " | ".join([CLEAN] + NOISY + ["2 noisy pooled"]) + " |")
    lines.append("|---|" + "---|" * (len(NOISY) + 2))
    for a in ARMS:
        cells = []
        for s in [CLEAN] + NOISY + ["pooled"]:
            v = [data[(x, a)][k] for x in NOISY for k in keys] if s == "pooled" else [data[(s, a)][k] for k in keys]
            k_, n = sum(v), len(v)
            lo, hi = wilson(k_, n)
            res["T1"][f"{s}/{a}"] = {"success": k_ / n, "ci": [lo, hi], "n": n}
            cells.append(f"{100 * k_ / n:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")

    # T2: pooled noisy suites, paired, cluster bootstrap over (task, trial)
    lines.append("\n## T2. Paired difference on the 2 noisy suites pooled (pp; McNemar exact; bootstrap CI over task x trial)\n")
    lines.append("| comparison | diff | 95% CI | only A / only B | p | p Holm |")
    lines.append("|---|---|---|---|---|---|")
    pooled_keys = [(s, k) for s in NOISY for k in keys]
    mat = {a: np.array([[data[(s, a)][k] for s in NOISY] for k in keys], dtype=float) for a in ARMS}  # (500, 2)
    idx = rng.integers(0, len(keys), size=(args.boot, len(keys)))
    comps = [(x, y) for x in ADAPTIVE for y in FIXED + ["random_h"]] + [("aac", "duac")]
    raw = {}
    for x, y in comps:
        diff = mat[x] - mat[y]
        d = diff.mean() * 100
        boots = diff[idx].mean(axis=(1, 2)) * 100
        lo, hi = np.percentile(boots, [2.5, 97.5])
        oa, ob, p = mcnemar({(s, k): data[(s, x)][k] for s, k in pooled_keys}, {(s, k): data[(s, y)][k] for s, k in pooled_keys}, pooled_keys)
        res["T2"][f"{x}-{y}"] = {"diff_pp": d, "ci_pp": [lo, hi], "only_a": oa, "only_b": ob, "p": p}
        if y in FIXED:
            raw[f"{x}-{y}"] = p
    adj = holm(raw)
    for x, y in comps:
        r = res["T2"][f"{x}-{y}"]
        r["p_holm"] = adj.get(f"{x}-{y}")
        ph = f"{r['p_holm']:.3f}" if r["p_holm"] is not None else "—"
        lines.append(f"| {x} − {y} | {r['diff_pp']:+.1f} | [{r['ci_pp'][0]:+.1f}, {r['ci_pp'][1]:+.1f}] | {r['only_a']} / {r['only_b']} | {r['p']:.3f} | {ph} |")

    # per suite (incl. clean)
    lines.append("\n### T2 per suite (pp, McNemar p)\n")
    lines.append("| comparison | " + " | ".join([CLEAN] + NOISY) + " |")
    lines.append("|---|" + "---|" * (1 + len(NOISY)))
    for x, y in comps:
        cells = []
        for s in [CLEAN] + NOISY:
            oa, ob, p = mcnemar(data[(s, x)], data[(s, y)], keys)
            cells.append(f"{100 * (oa - ob) / len(keys):+.1f} (p={p:.2f})")
            res["T2"][f"{s}:{x}-{y}"] = {"diff_pp": 100 * (oa - ob) / len(keys), "only_a": oa, "only_b": ob, "p": p}
        lines.append(f"| {x} − {y} | " + " | ".join(cells) + " |")

    # T3: per task, and without the task with the largest gain, vs each Fixed
    lines.append("\n## T3. Per task (2 noisy suites pooled; pp) and difference without the task with the largest gain\n")
    tasks = sorted({t for t, _ in keys})
    lines.append("| comparison | " + " | ".join(f"t{t}" for t in tasks) + " | without best task |")
    lines.append("|---|" + "---|" * (len(tasks) + 1))
    for x, y in comps:
        per = {}
        for t in tasks:
            kk = [k for k in keys if k[0] == t]
            per[t] = 100 * np.mean([data[(s, x)][k] - data[(s, y)][k] for s in NOISY for k in kk])
        best = max(per, key=per.get)
        rest = [k for k in keys if k[0] != best]
        wo = 100 * np.mean([data[(s, x)][k] - data[(s, y)][k] for s in NOISY for k in rest])
        res["T3"][f"{x}-{y}"] = {"per_task_pp": per, "best_task": best, "without_best_pp": wo}
        lines.append(f"| {x} − {y} | " + " | ".join(f"{per[t]:+.0f}" for t in tasks) + f" | {wo:+.1f} (bỏ t{best}) |")

    # decision rule
    lines.append("\n## Section 5 decision (2 noisy suites pooled)\n")
    clean_best = max(FIXED, key=lambda a: res["T1"][f"{CLEAN}/{a}"]["success"])
    noisy_best = max(FIXED, key=lambda a: res["T1"][f"pooled/{a}"]["success"])
    verdicts = {}
    for x in ADAPTIVE:
        beats = all(res["T2"][f"{x}-{y}"]["diff_pp"] >= 3 and res["T2"][f"{x}-{y}"]["p_holm"] < 0.05 for y in FIXED)
        robust = all(res["T3"][f"{x}-{y}"]["without_best_pp"] >= 1 for y in FIXED)
        clean_gap = 100 * (res["T1"][f"{CLEAN}/{x}"]["success"] - res["T1"][f"{CLEAN}/{clean_best}"]["success"])
        upper = res["T2"][f"{x}-{noisy_best}"]["ci_pp"][1]
        if beats and robust and clean_gap >= -2:
            v = "ADAPTIVE WORK"
        elif upper < 3:
            v = "KHÔNG WORK"
        else:
            v = "CHƯA KẾT LUẬN ĐƯỢC"
        verdicts[x] = {"verdict": v, "beats_all_fixed_3pp_holm": beats, "robust_without_best_task": robust,
                       "clean_gap_vs_best_fixed_pp": clean_gap, "best_fixed_noisy": noisy_best,
                       "upper_ci_vs_best_fixed_pp": upper,
                       "vs_fixed8_pp": res["T2"][f"{x}-fixed_8"]["diff_pp"], "vs_random_h_pp": res["T2"][f"{x}-random_h"]["diff_pp"]}
        lines.append(f"- **{x.upper()}: {v}**. Vs best Fixed on noisy ({noisy_best}): {res['T2'][f'{x}-{noisy_best}']['diff_pp']:+.1f} pp, "
                     f"CI upper {upper:+.1f} pp; beats all 3 Fixed by ≥3 pp after Holm: {beats}; ≥1 pp without best task: {robust}; "
                     f"clean vs best Fixed ({clean_best}): {clean_gap:+.1f} pp; vs Fixed-8 {verdicts[x]['vs_fixed8_pp']:+.1f} pp, "
                     f"vs Random-h {verdicts[x]['vs_random_h_pp']:+.1f} pp.")
    res["decision"] = verdicts
    (out / "track_a_results.md").write_text("\n".join(lines) + "\n")
    (out / "track_a_results.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
