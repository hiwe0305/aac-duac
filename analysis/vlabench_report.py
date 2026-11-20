"""Statistics, diagnostics, figures and REPORT.md for the VLABench runs (layout <run>/<track>/<arm>/<task>/...).

    .venv-vlabench/bin/python analysis/vlabench_report.py outputs/vlabench/full [--tracks T1 T2 T3 T4 T6]

Same analysis as analysis/robocasa_report.py (RoboCasa) and the Track A report (LIBERO):
  success + Wilson CI per task / track; post-hoc best fixed per task (oracle upper bound); paired comparisons of every
  adaptive arm vs every fixed arm on matched (task, episode): exact McNemar, Holm, task-stratified bootstrap CI;
  verdict vs the best fixed arm (WORK if CI > 0, NOT WORK if CI < 0, else INCONCLUSIVE); h / h_unc / xi statistics;
  uncertainty profiles; lag-1 autocorrelation of h_unc; h at gripper toggles; AUROC of the mean uncertainty of the first
  10 decisions for episode failure (pooled and within task); cost (decisions, clean latency, policy ms per action).
Fixed arms: vanilla (h = 5, official), fixed_8, fixed_10. Adaptive arms: duac, aac. Works on partial results (only
episodes present in every arm of a track are compared).
Writes <run>/REPORT.md, <run>/report.json and <run>/<track>/analysis/{report.json, figures/R1..R8.{png,pdf}},
<run>/figures/X1..X2 (cross-track).
"""
import argparse, csv, glob, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flowbased_aac"))
from robocasa_report import auroc, holm, mcnemar_exact, strat_bootstrap, wilson  # noqa: E402
from flowbased_aac.vlabench.common import REFERENCE_SR, TRACK_TASK_ALIASES, resolve_track  # noqa: E402

FIXED_ORDER = ["vanilla", "fixed_8", "fixed_10"]
ADAPT_ORDER = ["duac", "aac", "duac_noxi", "aac_noxi", "random_h"]  # *_noxi: ablation h = h_unc (no xi); random_h: h ~ duac_noxi
MAIN_ARMS = ("duac", "aac")  # Holm family of the main comparisons; ablation / control arms form a second family


def duac_family(a):  # arms whose algo log carries DUAC fields (U, h_unc); random_h runs on the DUAC server
    return a.startswith("duac") or a == "random_h"
ABLATION = [("duac_noxi", "duac"), ("aac_noxi", "aac"), ("duac_noxi", "random_h")]
H_OF = {"vanilla": 5, "fixed_8": 8, "fixed_10": 10}
COL = {"vanilla": "#9db4d6", "fixed_8": "#5b84c4", "fixed_10": "#1f4e9c", "duac": "#d9534f", "aac": "#e8a33d", "duac_noxi": "#f2a29f", "aac_noxi": "#f5cf8e", "random_h": "#8e7cc3"}
ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def read_csv(pattern):
    return [r for f in sorted(glob.glob(pattern)) for r in csv.DictReader(open(f)) if None not in r.values()]


def read_logs(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            if line.strip():
                d = json.loads(line)
                for k in ("executed_chunk", "aac_frame_chunk", "state"):
                    d.pop(k, None)
                rows.append(d)
    return rows


def ref_of(track, task):
    inv = {v: k for k, v in TRACK_TASK_ALIASES.get(track, {}).items()}
    return REFERENCE_SR.get(track, {}).get(inv.get(task, task))


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(f"{path}.png"); fig.savefig(f"{path}.pdf"); plt.close(fig)


def pct(x):
    return f"{100 * x:.1f}%"


def analyze_track(run, tr):
    track = resolve_track(tr)
    d = Path(run) / tr
    arms = [a for a in FIXED_ORDER + ADAPT_ORDER if read_csv(f"{d}/{a}/*/episodes.csv")]
    if not arms:
        return None
    ep = {a: {(r["task"], int(r["episode"])): r for r in read_csv(f"{d}/{a}/*/episodes.csv")} for a in arms}
    keys = sorted(set.intersection(*(set(ep[a]) for a in arms)))
    if not keys:
        return None
    FIXED = [a for a in FIXED_ORDER if a in arms]
    ADAPT = [a for a in ADAPT_ORDER if a in arms]
    tasks = sorted({k[0] for k in keys})
    S = {a: np.array([int(ep[a][k]["success"]) for k in keys]) for a in arms}
    T = np.array([k[0] for k in keys])
    rep = {"track": track, "arms": arms, "tasks": tasks, "n_matched": len(keys),
           "n_per_task": {t: int((T == t).sum()) for t in tasks},
           "errors": {a: sum(1 for k in keys if ep[a][k]["error"]) for a in arms}}
    rep["success"] = {a: {"k": int(S[a].sum()), "n": len(keys), "rate": float(S[a].mean()),
                          "wilson": wilson(int(S[a].sum()), len(keys)),
                          "progress": float(np.mean([float(ep[a][k]["progress_score"]) for k in keys])),
                          "intention": float(np.mean([float(ep[a][k]["intention_score"]) for k in keys]))} for a in arms}
    rep["per_task"] = {t: {a: int(S[a][T == t].sum()) for a in arms} for t in tasks}
    rep["reference"] = {t: ref_of(track, t) for t in tasks}
    if FIXED:
        best = max(FIXED, key=lambda a: S[a].sum())
        rep["best_fixed"] = best
        oracle = np.zeros(len(keys)); rep["oracle_fixed_per_task"] = {}
        for t in tasks:
            m = T == t; bt = max(FIXED, key=lambda a: S[a][m].sum()); oracle[m] = S[bt][m]; rep["oracle_fixed_per_task"][t] = bt
        rep["oracle_fixed_rate"] = float(oracle.mean())
    comps, ps = [], []
    for a in ADAPT:
        for f in FIXED:
            b = int(((S[a] == 1) & (S[f] == 0)).sum()); c = int(((S[a] == 0) & (S[f] == 1)).sum())
            lo, hi = strat_bootstrap(S[a], S[f], T)
            comps.append({"adaptive": a, "fixed": f, "diff_pp": 100 * float(S[a].mean() - S[f].mean()),
                          "ci95_pp": [100 * lo, 100 * hi], "adaptive_only": b, "fixed_only": c, "p_mcnemar": mcnemar_exact(b, c)})
            ps.append(comps[-1]["p_mcnemar"])
    for fam in (False, True):
        idx = [i for i, c_ in enumerate(comps) if (c_["adaptive"] not in MAIN_ARMS) == fam]
        for i, p in zip(idx, holm([ps[i] for i in idx])):
            comps[i]["p_holm"] = p
    rep["paired"] = comps
    # ablation: h = h_unc (no xi) vs the full method, matched episodes
    abl, aps = [], []
    for a, full in ABLATION:
        if a in arms and full in arms:
            b = int(((S[a] == 1) & (S[full] == 0)).sum()); c = int(((S[a] == 0) & (S[full] == 1)).sum())
            lo, hi = strat_bootstrap(S[a], S[full], T)
            abl.append({"ablation": a, "full": full, "diff_pp": 100 * float(S[a].mean() - S[full].mean()), "ci95_pp": [100 * lo, 100 * hi],
                        "ablation_only": b, "full_only": c, "p_mcnemar": mcnemar_exact(b, c)})
            aps.append(abl[-1]["p_mcnemar"])
    for c_, p in zip(abl, holm(aps)):
        c_["p_holm"] = p
    rep["ablation"] = abl
    rep["verdict"] = {}
    for a in ADAPT:
        if FIXED:
            c_ = next(x for x in comps if x["adaptive"] == a and x["fixed"] == rep["best_fixed"])
            rep["verdict"][a] = "WORK" if c_["ci95_pp"][0] > 0 else "NOT WORK (worse)" if c_["ci95_pp"][1] < 0 else "INCONCLUSIVE"

    # decisions, cost
    dec = {a: [r for r in read_csv(f"{d}/{a}/*/decisions.csv") if (r["task"], int(r["episode"])) in set(keys)] for a in arms}
    rep["efficiency"] = {}
    for a in arms:
        R = dec[a]; h = np.array([int(r["h_executed"]) for r in R]) if R else np.zeros(1)
        succ = [ep[a][k] for k in keys if ep[a][k]["success"] == "1"]
        rep["efficiency"][a] = {
            "decisions_per_episode": float(np.mean([int(ep[a][k]["n_decisions"]) for k in keys])),
            "steps_to_success_mean": float(np.mean([int(r["end_step"]) for r in succ])) if succ else None,
            "h_mean": float(h.mean()), "h_sd": float(h.std()),
            "infer_ms_median": float(np.median([float(r["infer_ms"]) for r in R])) if R else None,
            "roundtrip_ms_median": float(np.median([float(r["roundtrip_ms"]) for r in R])) if R else None,
            "wall_s_mean": float(np.mean([float(ep[a][k]["wall_s"]) for k in keys])),
        }
    # algo logs (adaptive arms)
    logs = {a: [x for x in read_logs(f"{d}/{a}/*/algo_log.jsonl") if (x["task"], int(x["episode"])) in set(keys)] for a in ADAPT}
    HH = {}
    for a in ADAPT:
        L = logs[a]
        if not L:
            continue
        h = np.array([x["h_executed"] for x in L]); xi = np.array([x["xi"] for x in L])
        hu = np.array([x["h_unc"] if duac_family(a) else x["h_entropy"] for x in L]); Hc = len(L[0]["U" if duac_family(a) else "H_total"])
        HH[a] = (h, hu, xi, Hc)
        rep[f"horizon_{a}"] = {"h_mean": float(h.mean()), "h_median": float(np.median(h)), "h_unc_mean": float(hu.mean()),
                               "h_unc_median": float(np.median(hu)), "xi_mean": float(xi.mean()),
                               "frac_xi_eq_H": float((xi >= Hc).mean()), "frac_xi_binds": float((xi > hu).mean()),
                               "frac_h_unc_binds": float((hu >= xi).mean()), "frac_h_unc_edge": float((hu >= Hc - 1).mean()),
                               "frac_h_unc_le_2": float((hu <= 2).mean()), "frac_h_eq_H": float((h >= Hc).mean()),
                               "h_unc_top": Counter(hu.tolist()).most_common(5),
                               "h_mean_per_task": {t: float(np.mean([x["h_executed"] for x in L if x["task"] == t])) for t in tasks}}
    if "duac" in logs and logs["duac"]:
        U = np.array([x["U"] for x in logs["duac"]])
        rep["duac_U_profile"] = {"mean": U.mean(0).tolist(), "frac_U_last_gt_prev": float((U[:, -1] > U[:, -2]).mean()),
                                 "frac_negative_U": float((U < 0).mean())}
    if "aac" in logs and logs["aac"]:
        Hent = np.array([x["H_total"] for x in logs["aac"]])
        rep["aac_entropy_profile"] = {"mean": Hent.mean(0).tolist()}

    def lag1(a, field):
        by = defaultdict(list)
        for x in logs[a]:
            by[(x["task"], x["episode"])].append((x["decision_id"], x[field]))
        xs, ys = [], []
        for v in by.values():
            v = [z for _, z in sorted(v)]; xs += v[:-1]; ys += v[1:]
        return float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 and np.std(xs) > 0 and np.std(ys) > 0 else None
    rep["lag1_autocorr"] = {a: lag1(a, "h_unc" if duac_family(a) else "h_entropy") for a in ADAPT if logs.get(a)}
    rep["gripper"] = {}
    for a in ADAPT:
        R = dec[a]
        if not R:
            continue
        g = np.array([int(r["gripper_toggle"]) for r in R]); h = np.array([int(r["h_executed"]) for r in R])
        hu = np.array([int(r["h_unc"]) for r in R]); xi = np.array([int(float(r["move_lower_bound"])) for r in R])
        rep["gripper"][a] = {"frac_decisions_toggle": float(g.mean()),
                             "h_mean_toggle": float(h[g == 1].mean()) if g.any() else None, "h_mean_no_toggle": float(h[g == 0].mean()),
                             "xi_binds_toggle": float((xi[g == 1] > hu[g == 1]).mean()) if g.any() else None,
                             "xi_binds_no_toggle": float((xi[g == 0] > hu[g == 0]).mean())}
    rep["auroc_fail"] = {}
    for a, field in [(x, "U" if duac_family(x) else "H_total") for x in ADAPT]:
        if not logs.get(a):
            continue
        by = defaultdict(list)
        for x in logs[a]:
            by[(x["task"], int(x["episode"]))].append((x["decision_id"], float(np.mean(x[field]))))
        ks = [k for k in by if k in ep[a]]
        sc = [np.mean([z for _, z in sorted(by[k])][:10]) for k in ks]; lab = [1 - int(ep[a][k]["success"]) for k in ks]
        per = {t: auroc([s for s, k in zip(sc, ks) if k[0] == t], [l for l, k in zip(lab, ks) if k[0] == t]) for t in tasks}
        valid = [v for v in per.values() if not np.isnan(v)]
        rep["auroc_fail"][a] = {"pooled": auroc(sc, lab), "within_task_mean": float(np.mean(valid)) if valid else None, "per_task": per}
    rep["h_vs_success"] = {a: {"mean_h_success": float(np.mean([float(ep[a][k]["mean_h"]) for k in keys if S[a][keys.index(k)]]) or 0),
                               "mean_h_failure": float(np.mean([float(ep[a][k]["mean_h"]) for k in keys if not S[a][keys.index(k)]]) or 0)}
                           for a in ADAPT if S[a].any() and (~S[a].astype(bool)).any()}

    # ---------------- figures ----------------
    figs = d / "analysis" / "figures"; figs.mkdir(parents=True, exist_ok=True)
    n_t = rep["n_per_task"]
    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(tasks) + 2), 3.6)); w = 0.8 / (len(arms) + 1); x = np.arange(len(tasks) + 1)
    for i, a in enumerate(arms):
        ax.bar(x + i * w, [S[a][T == t].mean() * 100 for t in tasks] + [S[a].mean() * 100], w, color=COL[a], label=a)
        lo, hi = rep["success"][a]["wilson"]
        ax.errorbar(len(tasks) + i * w, S[a].mean() * 100, yerr=[[S[a].mean() * 100 - lo * 100], [hi * 100 - S[a].mean() * 100]], color="k", lw=.8, capsize=2)
    refs = [rep["reference"][t] for t in tasks]
    if any(r is not None for r in refs):
        rr = [100 * (r if r is not None else np.nan) for r in refs] + [100 * np.mean([r for r in refs if r is not None])]
        ax.bar(x + len(arms) * w, rr, w, color="none", edgecolor="k", hatch="//", label="official (vanilla)")
    ax.set_xticks(x + 0.4, tasks + ["TỔNG"], rotation=30, ha="right", fontsize=7); ax.set_ylabel("success (%)")
    ax.legend(ncol=6, frameon=False, fontsize=7, loc="upper left"); ax.set_title(f"R1. {tr}: success theo task và tổng (Wilson 95%)")
    savefig(fig, figs / "R1_success_per_task")
    if FIXED:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        fx = [H_OF[a] for a in FIXED]; fy = [S[a].mean() * 100 for a in FIXED]
        fe = [[(S[a].mean() - rep["success"][a]["wilson"][0]) * 100 for a in FIXED], [(rep["success"][a]["wilson"][1] - S[a].mean()) * 100 for a in FIXED]]
        ax.errorbar(fx, fy, yerr=fe, color=COL["fixed_10"], marker="o", capsize=3, label="Fixed-h (vanilla = 5)")
        for a in ADAPT:
            mh = rep["efficiency"][a]["h_mean"]; y = S[a].mean() * 100; lo, hi = rep["success"][a]["wilson"]
            ax.errorbar([mh], [y], yerr=[[y - lo * 100], [hi * 100 - y]], color=COL[a], marker="s", capsize=3, label=f"{a.upper()} (h̄={mh:.1f})")
        ax.set_xlabel("h trung bình được thực thi"); ax.set_ylabel("success (%)"); ax.legend(frameon=False, fontsize=7)
        ax.set_title(f"R2. {tr}: success theo h trung bình"); savefig(fig, figs / "R2_success_vs_mean_h")
    if HH:
        fig, axs = plt.subplots(1, len(HH), figsize=(5 * len(HH), 3.4), sharey=True); axs = np.atleast_1d(axs)
        for ax, (a, (h, hu, xi, Hc)) in zip(axs, HH.items()):
            bins = np.arange(0.5, Hc + 1.5)
            ax.hist(hu, bins, alpha=.55, color="#888", density=True, label="h_unc" if duac_family(a) else "h_entropy")
            ax.hist(xi, bins, alpha=.55, color="#2a9d8f", density=True, label="ξ (chuyển động)")
            ax.hist(h, bins, histtype="step", lw=1.5, color=COL[a], density=True, label="h thực thi")
            ax.set_title(f"R3. {a.upper()}: ξ > h_unc ở {rep[f'horizon_{a}']['frac_xi_binds']:.0%}", fontsize=8); ax.set_xlabel("h"); ax.legend(frameon=False, fontsize=7)
        savefig(fig, figs / "R3_h_distributions")
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
        for ax, a, key, title in ((axs[0], "duac", "U", "R4a. DUAC: U_i"), (axs[1], "aac", "H_total", "R4b. AAC: entropy H_total(i)")):
            if logs.get(a):
                P = np.array([x[key] for x in logs[a]]); s = np.arange(1, P.shape[1] + 1)
                ax.plot(s, P.mean(0), color=COL[a]); ax.fill_between(s, np.percentile(P, 25, 0), np.percentile(P, 75, 0), color=COL[a], alpha=.2)
            ax.set_title(title + " (trung bình, IQR)", fontsize=8); ax.set_xlabel("bước i trong chunk")
        savefig(fig, figs / "R4_uncertainty_profiles")
    if comps:
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(comps) + 1.2)); lab = []
        lim = max(max(c_["adaptive_only"], c_["fixed_only"]) for c_ in comps) + 12
        for yy, c_ in enumerate(comps):
            ax.barh(yy, c_["adaptive_only"], color=COL[c_["adaptive"]]); ax.barh(yy, -c_["fixed_only"], color=COL[c_["fixed"]])
            ax.text(c_["adaptive_only"] + .5, yy, f"+{c_['adaptive_only']}", va="center", fontsize=7)
            ax.text(-c_["fixed_only"] - .5, yy, f"−{c_['fixed_only']}  p={c_['p_mcnemar']:.2f}", va="center", ha="right", fontsize=7)
            lab.append(f"{c_['adaptive']} vs {c_['fixed']}")
        ax.axvline(0, color="k", lw=.6); ax.set_yticks(range(len(lab)), lab); ax.set_xlim(-lim, lim)
        ax.set_xlabel("số cặp (task, episode): ← chỉ fixed thành công | chỉ adaptive thành công →", fontsize=7)
        ax.set_title(f"R5. {tr}: so sánh ghép cặp (McNemar exact)"); savefig(fig, figs / "R5_paired")
    fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(tasks)), 0.5 * len(arms) + 1.5))
    M = np.array([[S[a][T == t].sum() for t in tasks] for a in arms]); nmax = max(n_t.values())
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=nmax)
    for i in range(len(arms)):
        for j in range(len(tasks)):
            ax.text(j, i, int(M[i, j]), ha="center", va="center", color="white" if M[i, j] > nmax / 2 else "black", fontsize=8)
    ax.set_xticks(range(len(tasks)), tasks, fontsize=7, rotation=30, ha="right"); ax.set_yticks(range(len(arms)), arms)
    ax.set_title(f"R6. {tr}: số episode thành công (trên {nmax})"); fig.colorbar(im, ax=ax, fraction=.03); savefig(fig, figs / "R6_task_heatmap")
    if HH:
        fig, axs = plt.subplots(1, len(HH), figsize=(5 * len(HH), 3.2)); axs = np.atleast_1d(axs)
        for ax, a in zip(axs, HH):
            field = "h_unc" if duac_family(a) else "h_entropy"
            k0 = sorted({(x["task"], x["episode"]) for x in logs[a]})[0]
            L = sorted([x for x in logs[a] if (x["task"], x["episode"]) == k0], key=lambda x: x["decision_id"])
            ax.plot([x["sim_step"] for x in L], [x[field] for x in L], ".-", color="#888", lw=.8, label=field)
            ax.plot([x["sim_step"] for x in L], [x["xi"] for x in L], ".-", color="#2a9d8f", lw=.8, label="ξ")
            ax.plot([x["sim_step"] for x in L], [x["h_executed"] for x in L], "-", color=COL[a], lw=1.4, label="h")
            ac = rep["lag1_autocorr"].get(a)
            ax.set_title(f"R7. {a.upper()} {k0[0]} ep {k0[1]} (lag-1 autocorr {field}: {ac if ac is None else round(ac, 2)})", fontsize=7)
            ax.set_xlabel("sim step"); ax.legend(frameon=False, fontsize=7)
        savefig(fig, figs / "R7_h_sequence")
    if rep["gripper"]:
        fig, ax = plt.subplots(figsize=(5.5, 3.2)); xx = np.arange(2)
        for i, a in enumerate(rep["gripper"]):
            g = rep["gripper"][a]; ax.bar(xx + (i - .5) * .35, [g["h_mean_no_toggle"], g["h_mean_toggle"] or 0], .35, color=COL[a], label=a)
        ax.set_xticks(xx, ["chunk không đổi gripper", "chunk có đổi gripper"]); ax.set_ylabel("h trung bình"); ax.legend(frameon=False)
        ax.set_title(f"R8. {tr}: h quanh sự kiện gripper"); savefig(fig, figs / "R8_gripper")
    (d / "analysis" / "report.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    rep["_paired_data"] = {"keys": [f"{tr}|{k[0]}" for k in keys], "S": {a: S[a].tolist() for a in arms}}
    return rep


def md_track(tr, rep, latency):
    A = rep["arms"]; FIX = [a for a in A if a in FIXED_ORDER]; ADP = [a for a in A if a in ADAPT_ORDER]
    L = [f"## {tr} — `{rep['track']}`\n",
         f"{rep['n_matched']} episode ghép cặp/arm ({', '.join(f'{t} {n}' for t, n in rep['n_per_task'].items())}). "
         f"Episode lỗi mô phỏng (tính là thất bại): " + ", ".join(f"{a} {rep['errors'][a]}" for a in A) + ".\n",
         f"### {tr}.1 Success\n", "| task | official (vanilla) | " + " | ".join(A) + " |", "|---|---|" + "---|" * len(A)]
    for t in rep["tasks"]:
        best = max(rep["per_task"][t][a] for a in A)
        ref = rep["reference"][t]
        L.append(f"| {t} | {'' if ref is None else ref} | " + " | ".join(
            (f"**{rep['per_task'][t][a]}**" if rep["per_task"][t][a] == best else str(rep["per_task"][t][a])) + f"/{rep['n_per_task'][t]}" for a in A) + " |")
    refs = [r for r in rep["reference"].values() if r is not None]
    L.append(f"| **Tổng** | {pct(np.mean(refs)) if refs else ''} | " + " | ".join(
        f"**{pct(rep['success'][a]['rate'])}** [{100 * rep['success'][a]['wilson'][0]:.0f}–{100 * rep['success'][a]['wilson'][1]:.0f}]" for a in A) + " |")
    L.append("| progress score | | " + " | ".join(f"{rep['success'][a]['progress']:.3f}" for a in A) + " |")
    if "best_fixed" in rep:
        L.append(f"\nFixed tốt nhất: **{rep['best_fixed']}**. Chọn fixed tốt nhất riêng từng task (sau khi biết kết quả, cận trên mọi lựa chọn cố định): {pct(rep['oracle_fixed_rate'])}.\n")
    L.append(f"![R1]({tr}/analysis/figures/R1_success_per_task.png)\n")
    if rep["paired"]:
        L += [f"### {tr}.2 So sánh ghép cặp (cùng task, episode)\n",
              "| so sánh | chênh lệch | CI 95% (bootstrap theo task) | chỉ adaptive thành công | chỉ fixed thành công | p McNemar | p Holm |",
              "|---|---|---|---|---|---|---|"]
        for c in rep["paired"]:
            L.append(f"| {c['adaptive']} vs {c['fixed']} | {c['diff_pp']:+.1f} pp | [{c['ci95_pp'][0]:+.1f}, {c['ci95_pp'][1]:+.1f}] | "
                     f"{c['adaptive_only']} | {c['fixed_only']} | {c['p_mcnemar']:.3f} | {c['p_holm']:.3f} |")
        L.append("\nKết luận so với fixed tốt nhất (" + rep.get("best_fixed", "–") + "): " +
                 ", ".join(f"**{a.upper()}: {v}**" for a, v in rep["verdict"].items()) + ".\n")
        L.append(f"![R5]({tr}/analysis/figures/R5_paired.png) ![R2]({tr}/analysis/figures/R2_success_vs_mean_h.png)\n")
    if rep.get("ablation"):
        L += [f"**Ablation bỏ ξ (h = h_unc) so với bản đầy đủ, và so với Random-h cùng phân phối h:**\n",
              "| so sánh | chênh lệch | CI 95% | chỉ arm 1 thành công | chỉ arm 2 thành công | p McNemar | p Holm |",
              "|---|---|---|---|---|---|---|"]
        for c in rep["ablation"]:
            L.append(f"| {c['ablation']} vs {c['full']} | {c['diff_pp']:+.1f} pp | [{c['ci95_pp'][0]:+.1f}, {c['ci95_pp'][1]:+.1f}] | "
                     f"{c['ablation_only']} | {c['full_only']} | {c['p_mcnemar']:.3f} | {c['p_holm']:.3f} |")
        L.append("")
    L += [f"### {tr}.3 Horizon và chi phí\n",
          "| arm | h̄ ± sd | decision/episode | bước tới khi thành công | gọi model (ms, đo sạch) | policy ms / action (ước tính) | thời gian/episode |",
          "|---|---|---|---|---|---|---|"]
    for a in A:
        e = rep["efficiency"][a]
        server = "duac" if a.startswith("fixed") else a  # fixed arms run on the DUAC server
        ms = latency[server]["call_ms"]["median"] if server in latency else None
        steps = "" if e["steps_to_success_mean"] is None else f"{e['steps_to_success_mean']:.0f}"
        call = "" if ms is None else f"{ms:.1f}"
        per_action = "" if ms is None else f"{ms / e['h_mean']:.1f}"
        L.append(f"| {a} | {e['h_mean']:.2f} ± {e['h_sd']:.2f} | {e['decisions_per_episode']:.1f} | {steps} | {call} | "
                 f"{per_action} | {e['wall_s_mean']:.0f} s |")
    L.append("\n`gọi model` lấy từ `check_policy.py` (một process, sau warm-up, có cả bước tính bất định). Fixed-8/10 chạy trên server DUAC "
             "nên mang chi phí của DUAC; nếu chạy bằng server vanilla thì bằng chi phí vanilla. `policy ms / action` = gọi model / h̄.\n")
    for a in ADP:
        if f"horizon_{a}" in rep:
            h = rep[f"horizon_{a}"]
            L.append(f"- **{a.upper()}**: h_unc trung bình {h['h_unc_mean']:.2f} (trung vị {h['h_unc_median']:.0f}; hay gặp nhất {h['h_unc_top'][:3]}), "
                     f"ξ trung bình {h['xi_mean']:.2f}, **ξ = H ở {pct(h['frac_xi_eq_H'])}**, ξ > h_unc ở {pct(h['frac_xi_binds'])}, "
                     f"h = H ở {pct(h['frac_h_eq_H'])}, h_unc ≤ 2 ở {pct(h['frac_h_unc_le_2'])}, h_unc ≥ H−1 ở {pct(h['frac_h_unc_edge'])}.")
    L.append(f"\n![R3]({tr}/analysis/figures/R3_h_distributions.png) ![R4]({tr}/analysis/figures/R4_uncertainty_profiles.png)\n")
    L += [f"### {tr}.4 Tín hiệu bất định\n"]
    for a in ADP:
        au = rep["auroc_fail"].get(a); g = rep["gripper"].get(a); ac = rep["lag1_autocorr"].get(a)
        if au:
            L.append(f"- **{a.upper()}**: AUROC (độ bất định trung bình của 10 decision đầu → thất bại) gộp {au['pooled']:.2f}, "
                     f"trung bình trong task {au['within_task_mean'] if au['within_task_mean'] is None else round(au['within_task_mean'], 2)}; "
                     f"lag-1 autocorr h_unc {ac if ac is None else round(ac, 2)}; "
                     f"h ở chunk đổi gripper {g['h_mean_toggle'] if g['h_mean_toggle'] is None else round(g['h_mean_toggle'], 2)} so với {g['h_mean_no_toggle']:.2f}.")
    L.append(f"\n![R6]({tr}/analysis/figures/R6_task_heatmap.png) ![R7]({tr}/analysis/figures/R7_h_sequence.png) ![R8]({tr}/analysis/figures/R8_gripper.png)\n")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("--tracks", nargs="+", default=["T1", "T2", "T3", "T4", "T6"])
    ap.add_argument("--latency", default=str(ROOT / "outputs" / "vlabench" / "validation" / "check_policy.json"))
    args = ap.parse_args(argv)
    run = Path(args.run)
    latency = json.loads(Path(args.latency).read_text())["latency"] if Path(args.latency).exists() else {}
    reps = {}
    for tr in args.tracks:
        r = analyze_track(run, tr)
        if r:
            reps[tr] = r
            print(f"[{tr}] n={r['n_matched']} " + " ".join(f"{a}={r['success'][a]['rate']:.3f}" for a in r["arms"]), flush=True)
    if not reps:
        print("no results yet"); return 1
    arms = [a for a in FIXED_ORDER + ADAPT_ORDER if any(a in r["arms"] for r in reps.values())]

    # cross-track figures
    (run / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(1.6 * len(reps) + 2, 3.6)); w = 0.8 / (len(arms) + 1); x = np.arange(len(reps))
    for i, a in enumerate(arms):
        ys = [100 * r["success"][a]["rate"] if a in r["arms"] else np.nan for r in reps.values()]
        er = [[100 * (r["success"][a]["rate"] - r["success"][a]["wilson"][0]) if a in r["arms"] else 0 for r in reps.values()],
              [100 * (r["success"][a]["wilson"][1] - r["success"][a]["rate"]) if a in r["arms"] else 0 for r in reps.values()]]
        ax.bar(x + i * w, ys, w, color=COL[a], label=a); ax.errorbar(x + i * w, ys, yerr=er, fmt="none", color="k", lw=.7, capsize=2)
    refs = [100 * np.mean([v for v in REFERENCE_SR[r["track"]].values()]) for r in reps.values()]
    ax.bar(x + len(arms) * w, refs, w, color="none", edgecolor="k", hatch="//", label="official (vanilla)")
    ax.set_xticks(x + 0.4, list(reps)); ax.set_ylabel("success (%)"); ax.legend(ncol=6, frameon=False, fontsize=7)
    ax.set_title("X1. Success theo track (Wilson 95%)"); savefig(fig, run / "figures" / "X1_success_per_track")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for a in ADAPT_ORDER:
        for j, tr in enumerate(reps):
            for c in reps[tr]["paired"]:
                if c["adaptive"] == a and c["fixed"] == "vanilla":
                    off = {"duac": -0.2, "aac": -0.1, "duac_noxi": 0.0, "aac_noxi": 0.1, "random_h": 0.2}[a]
                    ax.errorbar(j + off, c["diff_pp"], yerr=[[c["diff_pp"] - c["ci95_pp"][0]], [c["ci95_pp"][1] - c["diff_pp"]]],
                                color=COL[a], marker="s", capsize=3, label=a.upper() if j == 0 else None)
    ax.axhline(0, color="k", lw=.6); ax.set_xticks(range(len(reps)), list(reps)); ax.set_ylabel("Δ success vs vanilla (pp)")
    ax.legend(frameon=False); ax.set_title("X2. Chênh lệch ghép cặp so với vanilla (CI 95%)"); savefig(fig, run / "figures" / "X2_paired_vs_vanilla")

    # REPORT.md
    tot = {a: (sum(r["success"][a]["k"] for r in reps.values() if a in r["arms"]), sum(r["success"][a]["n"] for r in reps.values() if a in r["arms"])) for a in arms}
    L = ["# Báo cáo VLABench: AAC và DUAC so với fixed horizon (π0.5 `VLABench/pi05-primitive-10task`, chunk 10)\n",
         "*Tự sinh bởi `analysis/vlabench_report.py`; số liệu đọc từ `" + str(run) + "`.*\n",
         "## 1. Thiết lập\n", "| | |", "|---|---|",
         "| Policy | `VLABench/pi05-primitive-10task` (OpenPI JAX, bf16). Chunk H = 10, action 7 chiều (pose tuyệt đối + gripper), 3 camera 480×480 → 224 |",
         "| Task | 10 primitive task (T3 thay `select_poker`/`select_mahjong` bằng `select_nth_largest_poker`/`select_unique_type_mahjong`) |",
         "| Track | " + ", ".join(tr + " `" + r["track"] + "`" for tr, r in reps.items()) + " |",
         "| Episode | Episode 0..N−1 của config track chính thức. Mọi arm cùng episode nên ghép cặp được theo (task, episode) |",
         "| Arm | vanilla (h = 5, cấu hình chính thức), Fixed-8, Fixed-10, DUAC (τ = 0.9, K = 40), AAC (N = 20, `gaussian_bernoulli`). DUAC và AAC: h = max(h_unc, ξ), α = 3 (ξ theo đơn vị LIBERO) |",
         "| Độ dài episode | Chính thức: add_condiment 300, select_poker/select_painting 100, còn lại 200 bước (10 Hz) |",
         "| Phần cứng | 1 GPU RTX 5090. Chạy tuần tự từng track, mỗi lúc một server (vanilla; DUAC, cũng phục vụ Fixed-8/10 như LIBERO/RoboCasa; AAC); 8 client (+6 ở pipeline thứ hai) |\n",
         "## 2. Tổng hợp theo track\n", "| track | official (vanilla) | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
    for tr, r in reps.items():
        L.append(f"| {tr} | {pct(np.mean(list(REFERENCE_SR[r['track']].values())))} | " + " | ".join(
            f"{pct(r['success'][a]['rate'])} ({r['success'][a]['k']}/{r['success'][a]['n']})" if a in r["arms"] else "–" for a in arms) + " |")
    L.append("| **gộp** | | " + " | ".join(f"**{pct(k / n)}** [{100 * wilson(k, n)[0]:.0f}–{100 * wilson(k, n)[1]:.0f}]" if n else "–" for a, (k, n) in tot.items()) + " |")
    ADP = [a for a in ADAPT_ORDER if a in arms]
    L += ["\nKết luận so với fixed tốt nhất của từng track:\n", "| track | fixed tốt nhất | " + " | ".join(a.upper() for a in ADP) + " |",
          "|---|---|" + "---|" * len(ADP)]
    for tr, r in reps.items():
        L.append(f"| {tr} | {r.get('best_fixed', '–')} | " + " | ".join(r["verdict"].get(a, "–") for a in ADP) + " |")
    # pooled paired comparison over all tracks (strata = track x task)
    if all("_paired_data" in r for r in reps.values()):
        common = [a for a in arms if all(a in r["arms"] for r in reps.values())]
        SS = {a: np.concatenate([np.asarray(r["_paired_data"]["S"][a]) for r in reps.values()]) for a in common}
        TT = np.concatenate([np.asarray(r["_paired_data"]["keys"]) for r in reps.values()])
        pooled, ps = [], []
        for a in [x for x in ADAPT_ORDER if x in common]:
            for f in [x for x in FIXED_ORDER if x in common]:
                b_ = int(((SS[a] == 1) & (SS[f] == 0)).sum()); c_ = int(((SS[a] == 0) & (SS[f] == 1)).sum())
                lo, hi = strat_bootstrap(SS[a], SS[f], TT)
                pooled.append({"adaptive": a, "fixed": f, "diff_pp": 100 * float(SS[a].mean() - SS[f].mean()), "ci95_pp": [100 * lo, 100 * hi],
                               "adaptive_only": b_, "fixed_only": c_, "p_mcnemar": mcnemar_exact(b_, c_), "n": len(TT)})
                ps.append(pooled[-1]["p_mcnemar"])
        for fam in (False, True):
            idx = [i for i, c_ in enumerate(pooled) if (c_["adaptive"] not in MAIN_ARMS) == fam]
            for i, p in zip(idx, holm([ps[i] for i in idx])):
                pooled[i]["p_holm"] = p
        pooled_abl, aps = [], []
        for a, full in ABLATION:
            if a in common and full in common:
                b_ = int(((SS[a] == 1) & (SS[full] == 0)).sum()); c_ = int(((SS[a] == 0) & (SS[full] == 1)).sum())
                lo, hi = strat_bootstrap(SS[a], SS[full], TT)
                pooled_abl.append({"ablation": a, "full": full, "diff_pp": 100 * float(SS[a].mean() - SS[full].mean()),
                                   "ci95_pp": [100 * lo, 100 * hi], "ablation_only": b_, "full_only": c_, "p_mcnemar": mcnemar_exact(b_, c_)})
                aps.append(pooled_abl[-1]["p_mcnemar"])
        for c_, p in zip(pooled_abl, holm(aps)):
            c_["p_holm"] = p
        L += [f"\n### Gộp cả {len(reps)} track: so sánh ghép cặp ({len(TT)} cặp/arm; bootstrap phân tầng theo track × task)\n",
              "| so sánh | chênh lệch | CI 95% | chỉ adaptive thành công | chỉ fixed thành công | p McNemar | p Holm |", "|---|---|---|---|---|---|---|"]
        for c in pooled:
            L.append(f"| {c['adaptive']} vs {c['fixed']} | {c['diff_pp']:+.1f} pp | [{c['ci95_pp'][0]:+.1f}, {c['ci95_pp'][1]:+.1f}] | "
                     f"{c['adaptive_only']} | {c['fixed_only']} | {c['p_mcnemar']:.3f} | {c['p_holm']:.3f} |")
        if pooled_abl:
            L += [f"\n### Ablation bỏ ξ và đối chứng Random-h, gộp cả {len(reps)} track\n",
                  "| so sánh | chênh lệch | CI 95% | chỉ arm 1 thành công | chỉ arm 2 thành công | p McNemar | p Holm |",
                  "|---|---|---|---|---|---|---|"]
            for c in pooled_abl:
                L.append(f"| {c['ablation']} vs {c['full']} | {c['diff_pp']:+.1f} pp | [{c['ci95_pp'][0]:+.1f}, {c['ci95_pp'][1]:+.1f}] | "
                         f"{c['ablation_only']} | {c['full_only']} | {c['p_mcnemar']:.3f} | {c['p_holm']:.3f} |")
            hs = {a: [(r["efficiency"][a]["h_mean"], r["efficiency"][a]["decisions_per_episode"]) for r in reps.values() if a in r["arms"]]
                  for a in [x for x in ADAPT_ORDER if x in common]}
            L.append("\nh̄ / decision mỗi episode (trung bình các track): " + ", ".join(
                f"{a} {np.mean([h for h, _ in v]):.2f} / {np.mean([d for _, d in v]):.1f}" for a, v in hs.items()) + "\n")
        for r in reps.values():
            r.pop("_paired_data", None)
        reps_pooled = {"vs_fixed": pooled, "ablation": pooled_abl}
    else:
        reps_pooled = {}
    L.append("\n![X1](figures/X1_success_per_track.png) ![X2](figures/X2_paired_vs_vanilla.png)\n")
    if latency:
        L += ["**Chi phí một lần gọi model** (`check_policy.py`, trung vị 20 lần sau warm-up): " +
              ", ".join(f"{a} {latency[a]['call_ms']['median']:.1f} ms" for a in ("vanilla", "duac", "aac") if a in latency) +
              f" (DUAC +{latency.get('duac_overhead_vs_vanilla_ms', 0):.1f} ms, AAC +{latency.get('aac_overhead_vs_vanilla_ms', 0):.1f} ms so với vanilla).\n"]
    for tr, r in reps.items():
        L += md_track(tr, r, latency)
    L += ["## Hạn chế\n",
          "- Tham chiếu chính thức là vanilla trên 50 episode; các arm khác chưa có số chính thức.",
          "- α = 3 lấy theo LIBERO (ξ đổi sang đơn vị LIBERO); chưa có ablation bỏ ξ, nên chưa tách được đóng góp riêng của phần bất định.",
          "- Thời gian trong `decisions.csv` có lẫn thời gian chờ khi nhiều client chung một server; chi phí so sánh dùng số đo sạch.",
          "- Episode lỗi mô phỏng (PhysicsError) được tính là thất bại; evaluator chính thức bỏ chúng khỏi trung bình."]
    (run / "REPORT.md").write_text("\n".join(L) + "\n")
    (run / "report.json").write_text(json.dumps({"tracks": reps, "pooled_paired": reps_pooled}, indent=1, default=float) + "\n")
    print(f"[vlabench_report] wrote {run / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
