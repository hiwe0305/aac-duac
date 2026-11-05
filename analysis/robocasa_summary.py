"""Summarize a RoboCasa run dir (<run>/<arm>/<task>/{episodes,decisions}.csv) as markdown.

    python analysis/robocasa_summary.py outputs/robocasa/main
"""
import csv, glob, math, os, statistics as st, sys

ARMS = ["fixed_10", "fixed_25", "fixed_50", "duac", "aac"]


def rows(path):
    return [r for f in sorted(glob.glob(path)) for r in csv.DictReader(open(f))]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - w) / d, (c + w) / d


def main(run):
    arms = [a for a in ARMS if os.path.isdir(f"{run}/{a}")]
    tasks = sorted({os.path.basename(p) for a in arms for p in glob.glob(f"{run}/{a}/*") if os.path.isdir(p)})
    ep = {a: {t: rows(f"{run}/{a}/{t}/episodes.csv") for t in tasks} for a in arms}
    print(f"# RoboCasa summary: `{run}`\n\n## Success (k/n)\n")
    print("| task | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms))
    for t in tasks:
        print(f"| {t} | " + " | ".join(f"{sum(int(r['success']) for r in ep[a][t])}/{len(ep[a][t])}" for a in arms) + " |")
    print("| **total** | " + " | ".join(
        "**{}/{}** ({:.0%}, CI {:.0%}–{:.0%})".format(k, n, k / max(n, 1), *wilson(k, n))
        for a in arms for k, n in [(sum(int(r["success"]) for t in tasks for r in ep[a][t]), sum(len(ep[a][t]) for t in tasks))]) + " |")
    # paired comparison vs every fixed arm on (task, trial)
    succ = {a: {(r["task"], r["trial"]): int(r["success"]) for t in tasks for r in ep[a][t]} for a in arms}
    print("\n## Paired (task, trial): adaptive wins / fixed wins\n")
    for a in [x for x in ("duac", "aac") if x in arms]:
        for f in [x for x in arms if x.startswith("fixed")]:
            keys = succ[a].keys() & succ[f].keys()
            print(f"- {a} vs {f}: {sum(succ[a][k] > succ[f][k] for k in keys)} / {sum(succ[f][k] > succ[a][k] for k in keys)} (n={len(keys)})")
    print("\n## Horizon\n\n| arm | decisions | mean h | median h | mean h_unc | mean ξ | ξ > h_unc | infer ms |\n|---|---|---|---|---|---|---|---|")
    for a in arms:
        R = rows(f"{run}/{a}/*/decisions.csv")
        if not R:
            continue
        h = [int(r["h_executed"]) for r in R]
        hu = [int(r["h_unc"]) for r in R if r.get("h_unc")]
        xi = [float(r["move_lower_bound"]) for r in R if r.get("move_lower_bound")]
        bind = (f"{sum(x > u for u, x in zip(hu, xi)) / len(hu):.0%}" if hu and xi and a in ("duac", "aac") else "–")
        print(f"| {a} | {len(R)} | {st.mean(h):.1f} | {st.median(h):.0f} | "
              f"{st.mean(hu):.1f} | {st.mean(xi):.1f} | {bind} | {st.mean(float(r['infer_ms']) for r in R):.0f} |"
              if hu and xi and a in ("duac", "aac") else
              f"| {a} | {len(R)} | {st.mean(h):.1f} | {st.median(h):.0f} | – | – | – | {st.mean(float(r['infer_ms']) for r in R):.0f} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/robocasa/main")
