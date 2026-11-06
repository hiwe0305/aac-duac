"""Statistics, diagnostics and figures for a RoboCasa run (default: outputs/robocasa/main).

    .venv-robocasa/bin/python analysis/robocasa_report.py outputs/robocasa/main outputs/robocasa/main/analysis

Writes <out>/report.json and <out>/figures/R1..R8.png.
"""
import csv, glob, json, math, os, sys
from collections import Counter, defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIXED = ["fixed_10", "fixed_25", "fixed_50"]
ADAPT = ["duac", "aac"]
ARMS = FIXED + ADAPT
COL = {"fixed_10": "#9db4d6", "fixed_25": "#5b84c4", "fixed_50": "#1f4e9c", "duac": "#d9534f", "aac": "#e8a33d"}
RNG = np.random.default_rng(0)


def read_csv(pattern):
    return [r for f in sorted(glob.glob(pattern)) for r in csv.DictReader(open(f))]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - w) / d, (c + w) / d


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def strat_bootstrap(a, b, tasks, n_boot=5000):
    """Bootstrap of mean(a-b) resampling trials within each task (paired)."""
    a, b, tasks = map(np.asarray, (a, b, tasks)); d = a - b; out = np.empty(n_boot)
    groups = [np.where(tasks == t)[0] for t in np.unique(tasks)]
    for i in range(n_boot):
        out[i] = np.mean(np.concatenate([d[RNG.choice(g, len(g))] for g in groups]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def auroc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, int)
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def main(run, out):
    figs = os.path.join(out, "figures"); os.makedirs(figs, exist_ok=True)
    tasks = sorted(os.listdir(f"{run}/duac"))
    ep = {a: {(r["task"], int(r["trial"])): r for r in read_csv(f"{run}/{a}/*/episodes.csv")} for a in ARMS}
    keys = sorted(ep["duac"])
    S = {a: np.array([int(ep[a][k]["success"]) for k in keys]) for a in ARMS}
    T = np.array([k[0] for k in keys])
    rep = {"run": run, "tasks": tasks, "n_per_arm": len(keys)}

    # ---------- T1: success + Wilson CI, per task ----------
    rep["success"] = {a: {"k": int(S[a].sum()), "n": len(keys), "rate": float(S[a].mean()), "wilson": wilson(int(S[a].sum()), len(keys))} for a in ARMS}
    rep["per_task"] = {t: {a: int(S[a][T == t].sum()) for a in ARMS} for t in tasks}
    best_fixed = max(FIXED, key=lambda a: S[a].sum())
    rep["best_fixed"] = best_fixed
    # oracle: best fixed horizon per task (chosen post hoc -> upper bound for any fixed choice)
    oracle = np.zeros(len(keys))
    rep["oracle_fixed_per_task"] = {}
    for t in tasks:
        m = T == t; bt = max(FIXED, key=lambda a: S[a][m].sum()); oracle[m] = S[bt][m]; rep["oracle_fixed_per_task"][t] = bt
    rep["oracle_fixed_rate"] = float(oracle.mean())

    # ---------- T2: paired tests vs every fixed arm ----------
    comps, ps = [], []
    for a in ADAPT:
        for f in FIXED:
            b = int(((S[a] == 1) & (S[f] == 0)).sum()); c = int(((S[a] == 0) & (S[f] == 1)).sum())
            lo, hi = strat_bootstrap(S[a], S[f], T)
            comps.append({"adaptive": a, "fixed": f, "diff_pp": 100 * float(S[a].mean() - S[f].mean()),
                          "ci95_pp": [100 * lo, 100 * hi], "adaptive_only": b, "fixed_only": c, "p_mcnemar": mcnemar_exact(b, c)})
            ps.append(comps[-1]["p_mcnemar"])
    for c_, p in zip(comps, holm(ps)):
        c_["p_holm"] = p
    rep["paired"] = comps
    # decision rule (same as Track A section 5): works if lower CI vs best fixed > 0
    rep["verdict"] = {}
    for a in ADAPT:
        c_ = next(x for x in comps if x["adaptive"] == a and x["fixed"] == best_fixed)
        rep["verdict"][a] = ("WORK" if c_["ci95_pp"][0] > 0 else "NOT WORK (worse)" if c_["ci95_pp"][1] < 0 else "INCONCLUSIVE")

    # ---------- efficiency ----------
    rep["efficiency"] = {}
    for a in ARMS:
        succ = [ep[a][k] for k in keys if ep[a][k]["success"] == "1"]
        rep["efficiency"][a] = {
            "decisions_per_episode": float(np.mean([int(ep[a][k]["n_decisions"]) for k in keys])),
            "end_step_success_mean": float(np.mean([int(r["end_step"]) for r in succ])) if succ else None,
            "wall_min_mean": float(np.mean([float(ep[a][k]["wall_s"]) for k in keys]) / 60),
        }

    # ---------- decisions + algo logs ----------
    dec = {a: read_csv(f"{run}/{a}/*/decisions.csv") for a in ARMS}
    for a in ARMS:
        rep["efficiency"][a]["infer_ms"] = float(np.mean([float(r["infer_ms"]) for r in dec[a]]))
        rep["efficiency"][a]["decisions_total"] = len(dec[a])
    logs = {}
    for a in ADAPT:
        rows = []
        for f in sorted(glob.glob(f"{run}/{a}/*/algo_log.jsonl")):
            for line in open(f):
                d = json.loads(line); d.pop("executed_chunk", None); rows.append(d)
        logs[a] = rows

    H = {}
    for a in ADAPT:
        L = logs[a]
        h = np.array([d["h_executed"] for d in L]); xi = np.array([d["xi"] for d in L])
        hu = np.array([d["h_unc"] if a == "duac" else d["h_entropy"] for d in L])
        H[a] = (h, hu, xi)
        rep[f"horizon_{a}"] = {
            "h_mean": float(h.mean()), "h_median": float(np.median(h)),
            "h_unc_mean": float(hu.mean()), "h_unc_median": float(np.median(hu)), "xi_mean": float(xi.mean()),
            "frac_xi_binds": float((xi > hu).mean()), "frac_h_unc_binds": float((hu >= xi).mean()),
            "frac_h_unc_edge_ge_48": float((hu >= 48).mean()), "frac_h_unc_le_3": float((hu <= 3).mean()),
            "frac_h_eq_50": float((h >= 49).mean()),
            "h_unc_top": Counter(hu.tolist()).most_common(6),
        }

    # profile shapes
    U = np.array([d["U"] for d in logs["duac"]]); Hent = np.array([d["H_total"] for d in logs["aac"]])
    rep["duac_U_profile"] = {"mean": U.mean(0).tolist(), "frac_U_last_gt_U_49": float((U[:, -1] > U[:, -2]).mean()),
                             "frac_negative_U": float((U < 0).mean())}
    # lag-1 autocorrelation of decision-to-decision h_unc within episode (stability of the signal)
    def lag1(a, field):
        by = defaultdict(list)
        for d in logs[a]:
            by[(d["task"], d["trial"])].append((d["decision_id"], d[field]))
        x, y = [], []
        for v in by.values():
            v = [z for _, z in sorted(v)]; x += v[:-1]; y += v[1:]
        return float(np.corrcoef(x, y)[0, 1])
    rep["lag1_autocorr"] = {"duac_h_unc": lag1("duac", "h_unc"), "aac_h_entropy": lag1("aac", "h_entropy")}

    # h near gripper toggles: decisions whose executed chunk toggles the gripper
    rep["gripper"] = {}
    for a in ADAPT:
        g = np.array([int(r["gripper_toggle"]) for r in dec[a]]); h = np.array([int(r["h_executed"]) for r in dec[a]])
        hu = np.array([int(r["h_unc"]) for r in dec[a]]); xi = np.array([int(float(r["move_lower_bound"])) for r in dec[a]])
        rep["gripper"][a] = {"frac_decisions_toggle": float(g.mean()),
                             "h_mean_toggle": float(h[g == 1].mean()) if g.any() else None, "h_mean_no_toggle": float(h[g == 0].mean()),
                             "xi_binds_toggle": float((xi[g == 1] > hu[g == 1]).mean()) if g.any() else None,
                             "xi_binds_no_toggle": float((xi[g == 0] > hu[g == 0]).mean())}

    # does uncertainty predict failure? episode-level AUROC of mean uncertainty in first 10 decisions
    rep["auroc_fail"] = {}
    for a, field in (("duac", "U"), ("aac", "H_total")):
        by = defaultdict(list)
        for d in logs[a]:
            by[(d["task"], d["trial"])].append((d["decision_id"], float(np.mean(d[field][:10]))))
        sc, lab = [], []
        for k, v in by.items():
            v = [z for _, z in sorted(v)][:10]; sc.append(np.mean(v)); lab.append(1 - int(ep[a][k]["success"]))
        rep["auroc_fail"][a] = {"all_tasks": auroc(sc, lab)}
        # within-task (removes task difficulty confound)
        ks = list(by.keys())
        rep["auroc_fail"][a]["per_task"] = {t: auroc([s for s, k in zip(sc, ks) if k[0] == t], [l for l, k in zip(lab, ks) if k[0] == t]) for t in tasks}

    # episode mean h vs success (within adaptive arms)
    rep["h_vs_success"] = {}
    for a in ADAPT:
        mh = np.array([float(ep[a][k]["mean_h"]) for k in keys])
        rep["h_vs_success"][a] = {"mean_h_success": float(mh[S[a] == 1].mean()), "mean_h_failure": float(mh[S[a] == 0].mean())}

    # ---------------- figures ----------------
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    # R1 success per task
    fig, ax = plt.subplots(figsize=(9, 3.6)); w = 0.16; x = np.arange(len(tasks) + 1)
    for i, a in enumerate(ARMS):
        vals = [S[a][T == t].mean() * 100 for t in tasks] + [S[a].mean() * 100]
        ax.bar(x + (i - 2) * w, vals, w, color=COL[a], label=a)
    lo_hi = [rep["success"][a]["wilson"] for a in ARMS]
    for i, a in enumerate(ARMS):
        lo, hi = lo_hi[i]; ax.errorbar(len(tasks) + (i - 2) * w, S[a].mean() * 100, yerr=[[S[a].mean() * 100 - lo * 100], [hi * 100 - S[a].mean() * 100]], color="k", lw=0.8, capsize=2)
    ax.set_xticks(x, tasks + ["TỔNG"]); ax.set_ylabel("success (%)"); ax.legend(ncol=5, frameon=False, loc="upper left")
    ax.set_title("R1. Success theo task (20 trial/task) và tổng (100 trial, Wilson 95%)")
    fig.tight_layout(); fig.savefig(f"{figs}/R1_success_per_task.png"); plt.close(fig)

    # R2 success vs average h: fixed curve + adaptive points
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    fx = [10, 25, 50]; fy = [S[a].mean() * 100 for a in FIXED]
    fe = [[(S[a].mean() - rep["success"][a]["wilson"][0]) * 100 for a in FIXED], [(rep["success"][a]["wilson"][1] - S[a].mean()) * 100 for a in FIXED]]
    ax.errorbar(fx, fy, yerr=fe, color=COL["fixed_50"], marker="o", capsize=3, label="Fixed-h")
    for a in ADAPT:
        mh = rep[f"horizon_{a}"]["h_mean"]; y = S[a].mean() * 100; lo, hi = rep["success"][a]["wilson"]
        ax.errorbar([mh], [y], yerr=[[y - lo * 100], [hi * 100 - y]], color=COL[a], marker="s", capsize=3, label=f"{a.upper()} (h̄={mh:.1f})")
    ax.set_xlabel("h trung bình được thực thi"); ax.set_ylabel("success (%)"); ax.legend(frameon=False)
    ax.set_title("R2. Success theo h trung bình")
    fig.tight_layout(); fig.savefig(f"{figs}/R2_success_vs_mean_h.png"); plt.close(fig)

    # R3 distributions of h, h_unc, xi
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True); bins = np.arange(0.5, 51.5, 1)
    for ax, a in zip(axs, ADAPT):
        h, hu, xi = H[a]
        ax.hist(hu, bins, alpha=0.55, color="#888", label="h_unc" if a == "duac" else "h_entropy", density=True)
        ax.hist(xi, bins, alpha=0.55, color="#2a9d8f", label="ξ (movement)", density=True)
        ax.hist(h, bins, histtype="step", lw=1.5, color=COL[a], label="h thực thi = max", density=True)
        ax.set_title(f"R3. {a.upper()}: ξ > h_unc ở {rep[f'horizon_{a}']['frac_xi_binds']:.0%} decision"); ax.set_xlabel("h"); ax.legend(frameon=False)
    axs[0].set_ylabel("mật độ"); fig.tight_layout(); fig.savefig(f"{figs}/R3_h_distributions.png"); plt.close(fig)

    # R4 profiles
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4)); s = np.arange(1, 51)
    m, q1, q3 = U.mean(0), np.percentile(U, 25, 0), np.percentile(U, 75, 0)
    axs[0].plot(s, m, color=COL["duac"]); axs[0].fill_between(s, q1, q3, color=COL["duac"], alpha=0.2)
    axs[0].set_title("R4a. DUAC: profile U_i (trung bình, IQR)"); axs[0].set_xlabel("bước i trong chunk")
    m, q1, q3 = Hent.mean(0), np.percentile(Hent, 25, 0), np.percentile(Hent, 75, 0)
    axs[1].plot(s, m, color=COL["aac"]); axs[1].fill_between(s, q1, q3, color=COL["aac"], alpha=0.2)
    axs[1].set_title("R4b. AAC: entropy H_total(i) (trung bình, IQR)"); axs[1].set_xlabel("bước i trong chunk")
    fig.tight_layout(); fig.savefig(f"{figs}/R4_uncertainty_profiles.png"); plt.close(fig)

    # R5 paired wins/losses vs each fixed arm
    fig, ax = plt.subplots(figsize=(7, 3.2)); labels = []; yy = 0
    for c_ in comps:
        ax.barh(yy, c_["adaptive_only"], color=COL[c_["adaptive"]]); ax.barh(yy, -c_["fixed_only"], color=COL[c_["fixed"]])
        ax.text(c_["adaptive_only"] + 0.3, yy, f"+{c_['adaptive_only']}", va="center", fontsize=8)
        ax.text(-c_["fixed_only"] - 0.3, yy, f"−{c_['fixed_only']}  p={c_['p_mcnemar']:.2f}", va="center", ha="right", fontsize=8)
        labels.append(f"{c_['adaptive']} vs {c_['fixed']}"); yy += 1
    ax.axvline(0, color="k", lw=0.6); ax.set_yticks(range(len(labels)), labels); ax.set_xlim(-30, 25)
    ax.set_xlabel("số cặp (task, trial): ← chỉ fixed thành công | chỉ adaptive thành công →")
    ax.set_title("R5. So sánh ghép cặp (McNemar exact)"); fig.tight_layout(); fig.savefig(f"{figs}/R5_paired.png"); plt.close(fig)

    # R6 per-task heatmap of success difference vs fixed_50 / oracle
    fig, ax = plt.subplots(figsize=(7, 2.8))
    M = np.array([[S[a][T == t].sum() for t in tasks] for a in ARMS])
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=20)
    for i in range(len(ARMS)):
        for j in range(len(tasks)):
            ax.text(j, i, int(M[i, j]), ha="center", va="center", color="white" if M[i, j] > 10 else "black", fontsize=9)
    ax.set_xticks(range(len(tasks)), tasks, fontsize=8); ax.set_yticks(range(len(ARMS)), ARMS)
    ax.set_title("R6. Số trial thành công (trên 20)"); fig.colorbar(im, ax=ax, fraction=0.03); fig.tight_layout()
    fig.savefig(f"{figs}/R6_task_heatmap.png"); plt.close(fig)

    # R7 decision-to-decision h_unc sequence example + autocorr
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, a, field in ((axs[0], "duac", "h_unc"), (axs[1], "aac", "h_entropy")):
        L = [d for d in logs[a] if d["task"] == "PreSoakPan" and d["trial"] == 0]
        L.sort(key=lambda d: d["decision_id"])
        ax.plot([d["sim_step"] for d in L], [d[field] for d in L], ".-", color="#888", lw=0.8, label=field)
        ax.plot([d["sim_step"] for d in L], [d["xi"] for d in L], ".-", color="#2a9d8f", lw=0.8, label="ξ")
        ax.plot([d["sim_step"] for d in L], [d["h_executed"] for d in L], "-", color=COL[a], lw=1.4, label="h")
        ax.set_title(f"R7. {a.upper()} PreSoakPan trial 0 (lag-1 autocorr {field}: {rep['lag1_autocorr'][a + ('_h_unc' if a == 'duac' else '_h_entropy')]:.2f})", fontsize=8)
        ax.set_xlabel("sim step"); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(f"{figs}/R7_h_sequence.png"); plt.close(fig)

    # R8 h at gripper toggles
    fig, ax = plt.subplots(figsize=(5.5, 3.2)); x = np.arange(2)
    for i, a in enumerate(ADAPT):
        g = rep["gripper"][a]; ax.bar(x + (i - 0.5) * 0.35, [g["h_mean_no_toggle"], g["h_mean_toggle"]], 0.35, color=COL[a], label=a)
    ax.set_xticks(x, ["chunk không đổi gripper", "chunk có đổi gripper"]); ax.set_ylabel("h trung bình"); ax.legend(frameon=False)
    ax.set_title("R8. h quanh sự kiện gripper"); fig.tight_layout(); fig.savefig(f"{figs}/R8_gripper.png"); plt.close(fig)

    json.dump(rep, open(f"{out}/report.json", "w"), indent=1, default=float)
    print(json.dumps({k: rep[k] for k in ("success", "best_fixed", "oracle_fixed_rate", "paired", "verdict", "efficiency", "horizon_duac", "horizon_aac",
                                          "lag1_autocorr", "gripper", "auroc_fail", "h_vs_success", "oracle_fixed_per_task")}, indent=1, default=float))
    print("U last>prev", rep["duac_U_profile"]["frac_U_last_gt_U_49"], "neg U", rep["duac_U_profile"]["frac_negative_U"])


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "outputs/robocasa/main"
    main(run, sys.argv[2] if len(sys.argv) > 2 else os.path.join(run, "analysis"))
