"""Summary, statistics and figures for VLABench runs (layout <run>/<arm>/<task>/{episodes,decisions}.csv,
any depth below <arm>; several tracks may be mixed, rows carry their track).

    .venv-vlabench/bin/python analysis/vlabench_summary.py outputs/vlabench/validate [--out DIR] [--baseline vanilla]

Writes <out>/SUMMARY.md, <out>/summary.json, <out>/per_task.csv and <out>/figures/V*.{png,pdf}:
  V1 success per arm (Wilson CI)        V2 success per task x arm (+ official reference for vanilla)
  V3 executed-h distribution            V4 policy latency per call and per executed action (compute vs success)
  V5 uncertainty vs executed h          V6 U and h over episode time       V7 mean U of successful vs failed episodes
Statistics reuse analysis/robocasa_report.py (Wilson, exact McNemar, Holm, paired stratified bootstrap, AUROC).
Failure-prediction target: episode failure; score: the episode's mean U (DUAC) / mean entropy (AAC) — a whole-episode
statistic, i.e. diagnostic, not a forecast.
"""
import argparse, csv, glob, json, os, statistics as st, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flowbased_aac"))
from robocasa_report import auroc, holm, mcnemar_exact, strat_bootstrap, wilson  # noqa: E402
from flowbased_aac.vlabench.common import REFERENCE_SR, TRACK_TASK_ALIASES  # noqa: E402

ORDER = ["vanilla", "fixed_1", "fixed_2", "fixed_4", "fixed_5", "fixed_8", "fixed_10", "random_h", "duac", "aac"]
COL = {"vanilla": "#555555", "duac": "#d9534f", "aac": "#e8a33d", "random_h": "#8e7cc3"}


def read(pattern):
    return [r for f in sorted(glob.glob(pattern, recursive=True)) for r in csv.DictReader(open(f)) if None not in r.values()]


def ref_sr(track, task):
    inv = {v: k for k, v in TRACK_TASK_ALIASES.get(track, {}).items()}
    return REFERENCE_SR.get(track, {}).get(inv.get(task, task))


def save(fig, out, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / "figures" / f"{name}.{ext}", dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline", default="vanilla")
    args = ap.parse_args(argv)
    run, out = Path(args.run), Path(args.out or Path(args.run) / "analysis")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    arms = [a for a in ORDER if (run / a).is_dir()] + sorted(
        p.name for p in run.iterdir() if p.is_dir() and p.name not in ORDER and p.name not in ("logs", "analysis"))
    ep = {a: read(f"{run}/{a}/**/episodes.csv") for a in arms}
    dec = {a: read(f"{run}/{a}/**/decisions.csv") for a in arms}
    arms = [a for a in arms if ep[a]]
    tracks = sorted({r["track"] for a in arms for r in ep[a]})
    tasks = sorted({r["task"] for a in arms for r in ep[a]})
    L, report = [f"# VLABench summary: `{run}`\n"], {"arms": {}}

    # ---- success per arm / task / track
    L.append("## Success rate (episodes with simulator errors count as failures; official evaluator drops them)\n")
    L.append("| arm | track | success | rate | 95% Wilson CI | progress | errors |\n|---|---|---|---|---|---|---|")
    for a in arms:
        for tr in tracks:
            R = [r for r in ep[a] if r["track"] == tr]
            if not R:
                continue
            k, n = sum(int(r["success"]) for r in R), len(R)
            lo, hi = wilson(k, n)
            prog = st.mean(float(r["progress_score"]) for r in R)
            err = sum(1 for r in R if r["error"])
            L.append(f"| {a} | {tr} | {k}/{n} | {k/n:.3f} | {lo:.3f}–{hi:.3f} | {prog:.3f} | {err} |")
            report["arms"].setdefault(a, {})[tr] = {"k": k, "n": n, "rate": k / n, "ci": [lo, hi], "progress": prog, "errors": err}
    per_task_rows = []
    for tr in tracks:
        L.append(f"\n### Per task, {tr} (k/n; `ref` = official model-card success rate, 50 episodes)\n")
        L.append("| task | ref | " + " | ".join(arms) + " |\n|---|---|" + "---|" * len(arms))
        for t in tasks:
            cells, ref = [], ref_sr(tr, t)
            for a in arms:
                R = [r for r in ep[a] if r["track"] == tr and r["task"] == t]
                k, n = sum(int(r["success"]) for r in R), len(R)
                cells.append(f"{k}/{n}" if n else "–")
                if n:
                    per_task_rows.append({"track": tr, "task": t, "arm": a, "k": k, "n": n, "rate": round(k / n, 4),
                                          "reference": ref, "progress": round(st.mean(float(r["progress_score"]) for r in R), 4),
                                          "intention": round(st.mean(float(r["intention_score"]) for r in R), 4)})
            if any(c != "–" for c in cells):
                L.append(f"| {t} | {'' if ref is None else ref} | " + " | ".join(cells) + " |")
        refs = [ref_sr(tr, t) for t in tasks if ref_sr(tr, t) is not None]
        if refs:
            L.append(f"| **mean** | **{np.mean(refs):.3f}** | " + " | ".join(
                f"**{np.mean([x['rate'] for x in per_task_rows if x['arm'] == a and x['track'] == tr]):.3f}**"
                if any(x["arm"] == a and x["track"] == tr for x in per_task_rows) else "–" for a in arms) + " |")
    with open(out / "per_task.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(per_task_rows[0])); w.writeheader(); w.writerows(per_task_rows)

    # ---- paired comparisons on (track, task, episode)
    base = args.baseline if args.baseline in arms else None
    if base and len(arms) > 1:
        L.append(f"\n## Paired vs `{base}` on matched (track, task, episode)\n")
        L.append("| arm | n | wins | losses | Δ success | 95% stratified bootstrap | McNemar p | Holm p |\n|---|---|---|---|---|---|---|---|")
        sb = {(r["track"], r["task"], r["episode"]): int(r["success"]) for r in ep[base]}
        rows_, ps = [], []
        for a in [x for x in arms if x != base]:
            sa = {(r["track"], r["task"], r["episode"]): int(r["success"]) for r in ep[a]}
            keys = sorted(sa.keys() & sb.keys())
            if not keys:
                continue
            x, y = [sa[k] for k in keys], [sb[k] for k in keys]
            wins, losses = sum(i > j for i, j in zip(x, y)), sum(i < j for i, j in zip(x, y))
            ci = strat_bootstrap(x, y, [k[0] + k[1] for k in keys])
            ps.append(mcnemar_exact(wins, losses))
            rows_.append([a, len(keys), wins, losses, np.mean(x) - np.mean(y), ci])
        for row, p, ph in zip(rows_, ps, holm(ps) if ps else []):
            a, n, w_, l_, d, ci = row
            L.append(f"| {a} | {n} | {w_} | {l_} | {d:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {p:.3f} | {ph:.3f} |")
            report["arms"][a]["paired_vs_" + base] = {"n": n, "wins": w_, "losses": l_, "delta": d, "ci": ci, "p": p, "p_holm": ph}

    # ---- horizon, uncertainty, cost
    L.append("\n## Horizon, uncertainty and cost (per decision; ms are medians)\n")
    L.append("| arm | decisions | mean h ± sd | min–max | mean h_unc | mean ξ | ξ = H | ξ > h_unc | samples | model ms | server ms | "
             "round-trip ms | policy ms / action | env ms / action |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        R = dec[a]
        if not R:
            continue
        h = np.array([int(r["h_executed"]) for r in R]); H = np.array([int(r["prediction_horizon"]) for r in R])
        hu = np.array([int(r["h_unc"]) for r in R]); xi = np.array([int(r["move_lower_bound"]) for r in R])
        has_u = (hu >= 0).all() and a not in ("vanilla",) and not a.startswith("vanilla")
        rt = np.array([float(r["roundtrip_ms"]) for r in R])
        per_action = rt.sum() / h.sum()
        env_per_action = sum(float(r["env_ms"]) for r in R) / h.sum()
        L.append(f"| {a} | {len(R)} | {h.mean():.2f} ± {h.std():.2f} | {h.min()}–{h.max()} | "
                 f"{hu.mean():.2f} | {xi.mean():.2f} | {np.mean(xi == H):.0%} | " if has_u else
                 f"| {a} | {len(R)} | {h.mean():.2f} ± {h.std():.2f} | {h.min()}–{h.max()} | – | {xi.mean():.2f} | {np.mean(xi == H):.0%} | ")
        L[-1] += (f"{np.mean(xi > hu):.0%} | " if has_u else "– | ") + (
            f"{int(R[0]['sample_count'])} | {np.median([float(r['infer_ms']) for r in R]):.1f} | "
            f"{np.median([float(r['server_ms']) for r in R]):.1f} | {np.median(rt):.1f} | {per_action:.1f} | {env_per_action:.1f} |")
        report["arms"][a]["horizon"] = {"mean": float(h.mean()), "sd": float(h.std()), "hist": np.bincount(h).tolist(),
                                       "policy_ms_per_action": per_action, "roundtrip_ms_median": float(np.median(rt))}
    peak = {a: max([float(r["peak_gpu_mb"]) for r in ep[a]] or [0]) for a in arms}
    L.append("\nPeak GPU memory of the serving process (MB): " + ", ".join(f"{a} {v:.0f}" for a, v in peak.items()))
    L.append("\n`policy ms / action` = total client round-trip time / executed actions (includes the uncertainty "
             "computation and the extra samples of AAC); fewer calls only help if this number drops.")

    # ---- uncertainty vs failure (diagnostic)
    L.append("\n## Episode-mean uncertainty vs failure (AUROC, label = failure; diagnostic, not a forecast)\n")
    for a in arms:
        R = [r for r in dec[a] if r["u_mean"] not in ("", None)]
        if not R:
            continue
        per_ep = defaultdict(list)
        for r in R:
            per_ep[(r["track"], r["task"], r["episode"])].append(float(r["u_mean"]))
        succ = {(r["track"], r["task"], r["episode"]): int(r["success"]) for r in ep[a]}
        keys = [k for k in per_ep if k in succ]
        if len({succ[k] for k in keys}) == 2:
            au = auroc([np.mean(per_ep[k]) for k in keys], [1 - succ[k] for k in keys])
            L.append(f"- {a}: AUROC {au:.3f} over {len(keys)} episodes")
            report["arms"][a]["auroc_mean_u_failure"] = au

    # ---- figures
    if arms:
        fig, ax = plt.subplots(figsize=(1.0 + 0.8 * len(arms), 3.2))
        for i, a in enumerate(arms):
            k = sum(int(r["success"]) for r in ep[a]); n = len(ep[a]); lo, hi = wilson(k, n)
            ax.bar(i, k / n, color=COL.get(a, "#5b84c4")); ax.errorbar(i, k / n, [[k / n - lo], [hi - k / n]], color="k", capsize=3)
        ax.set_xticks(range(len(arms)), arms, rotation=30, ha="right"); ax.set_ylabel("success rate"); ax.set_title("V1 success (95% Wilson)")
        save(fig, out, "V1_success")
        fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(tasks)), 3.4)); wbar = 0.8 / (len(arms) + 1)
        for i, a in enumerate(arms):
            vals = [next((x["rate"] for x in per_task_rows if x["arm"] == a and x["task"] == t), np.nan) for t in tasks]
            ax.bar(np.arange(len(tasks)) + i * wbar, vals, wbar, label=a, color=COL.get(a))
        refs = [ref_sr(tracks[0], t) or np.nan for t in tasks]
        ax.bar(np.arange(len(tasks)) + len(arms) * wbar, refs, wbar, label="official ref", color="none", edgecolor="k", hatch="//")
        ax.set_xticks(np.arange(len(tasks)) + 0.4, tasks, rotation=35, ha="right", fontsize=7); ax.legend(fontsize=6)
        ax.set_ylabel("success rate"); ax.set_title(f"V2 per task ({', '.join(tracks)})")
        save(fig, out, "V2_success_per_task")
        fig, ax = plt.subplots(figsize=(5, 3))
        for a in arms:
            if dec[a]:
                h = [int(r["h_executed"]) for r in dec[a]]
                ax.hist(h, bins=np.arange(0.5, 11.5), histtype="step", density=True, label=a, color=COL.get(a))
        ax.set_xlabel("executed h"); ax.set_ylabel("fraction"); ax.legend(fontsize=6); ax.set_title("V3 executed horizon")
        save(fig, out, "V3_horizon")
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        for a in arms:
            if dec[a] and a in report["arms"] and "horizon" in report["arms"][a]:
                x = report["arms"][a]["horizon"]["policy_ms_per_action"]; y = np.mean([int(r["success"]) for r in ep[a]])
                ax.scatter(x, y, color=COL.get(a, "#5b84c4")); ax.annotate(a, (x, y), fontsize=7)
        ax.set_xlabel("policy ms per executed action"); ax.set_ylabel("success rate"); ax.set_title("V4 success vs compute")
        save(fig, out, "V4_success_vs_cost")
        for a in ("duac", "aac"):
            R = [r for r in dec.get(a, []) if r["u_mean"] not in ("", None)]
            if not R:
                continue
            fig, axs = plt.subplots(1, 3, figsize=(11, 3))
            axs[0].scatter([float(r["u_mean"]) for r in R], [int(r["h_executed"]) + np.random.uniform(-.2, .2) for r in R], s=4, alpha=.4)
            axs[0].set_xlabel("chunk mean U" if a == "duac" else "chunk mean entropy"); axs[0].set_ylabel("executed h"); axs[0].set_title(f"V5 {a}")
            first = sorted({(r["track"], r["task"], r["episode"]) for r in R})[:6]
            for k in first:
                rr = [r for r in R if (r["track"], r["task"], r["episode"]) == k]
                axs[1].plot([int(r["sim_step"]) for r in rr], [float(r["u_mean"]) for r in rr], lw=.8)
                axs[2].step([int(r["sim_step"]) for r in rr], [int(r["h_executed"]) for r in rr], lw=.8, where="post")
            axs[1].set_title("V6 uncertainty over time"); axs[2].set_title("V6 executed h over time")
            axs[1].set_xlabel("env step"); axs[2].set_xlabel("env step")
            save(fig, out, f"V5_V6_{a}")
            per_ep = defaultdict(list)
            for r in R:
                per_ep[(r["track"], r["task"], r["episode"])].append(float(r["u_mean"]))
            succ = {(r["track"], r["task"], r["episode"]): int(r["success"]) for r in ep[a]}
            fig, ax = plt.subplots(figsize=(3.5, 3))
            ax.boxplot([[np.mean(v) for k, v in per_ep.items() if succ.get(k) == s] for s in (1, 0)], tick_labels=["success", "failure"])
            ax.set_ylabel("episode mean uncertainty"); ax.set_title(f"V7 {a}")
            save(fig, out, f"V7_{a}")

    (out / "SUMMARY.md").write_text("\n".join(L) + "\n")
    (out / "summary.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print("\n".join(L))
    print(f"\n[vlabench_summary] wrote {out}")


if __name__ == "__main__":
    main()
