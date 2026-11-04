#!/usr/bin/env python3
"""Why AAC / DUAC do not beat fixed horizons on Track A: figures + quantitative evidence.

Reads outputs/A/<suite>/<arm>/{episodes,decisions,subgoals}.csv and algo_log.jsonl and writes
outputs/A/analysis/figures/*.png and outputs/A/analysis/diagnosis.json.

    python analysis/diagnose_track_a.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "outputs" / "A"
FIG = A / "analysis" / "figures"
SUITES = ["libero_10", "libero_10_object", "libero_10_env"]
NOISY = ["libero_10_object", "libero_10_env"]
ARMS = ["duac", "aac", "fixed_5", "fixed_8", "fixed_10", "random_h"]
COL = {"duac": "#1f77b4", "aac": "#d62728", "fixed_5": "#bdbdbd", "fixed_8": "#8c8c8c", "fixed_10": "#2b2b2b",
       "random_h": "#9467bd"}
LABEL = {"duac": "DUAC", "aac": "AAC", "fixed_5": "Fixed-5", "fixed_8": "Fixed-8", "fixed_10": "Fixed-10",
         "random_h": "Random-h"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
EV = {}


def rows(p):
    with open(p, newline="") as h:
        return list(csv.DictReader(h))


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - w, c + w


def load_algo(suite, arm, keys):
    out = []
    with open(A / suite / arm / "algo_log.jsonl") as h:
        for line in h:
            r = json.loads(line)
            out.append({k: r.get(k) for k in keys})
    return out


def episode_info(suite, arm):
    eps = {(int(r["task_id"]), int(r["trial_id"])): r for r in rows(A / suite / arm / "episodes.csv")}
    sub = defaultdict(list)
    for r in rows(A / suite / arm / "subgoals.csv"):
        sub[(int(r["task_id"]), int(r["trial_id"]))].append(int(r["first_true_step"]) if r["first_true_step"] != "" else None)
    return eps, sub


def phase_of(step, firsts):
    f = sorted(x for x in firsts if x is not None)
    if not f or step < f[0]:
        return 0
    if len(f) < 2 or len(f) < len(firsts) or step < f[-1]:
        return 1
    return 2


def gripper_events(algo):
    """Steps (active) at which the executed gripper command changes sign, per episode."""
    ev = defaultdict(list)
    last = {}
    for r in algo:
        key = (r["task_id"], r["trial_id"])
        for j, a in enumerate(r["executed_chunk"]):
            s = a[6] >= 0
            if key in last and s != last[key]:
                ev[key].append(r["sim_step"] + j)
            last[key] = s
    return ev


# ------------------------------------------------------------------ F1 success
def fig_success():
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.0), sharey=False)
    for i, s in enumerate(SUITES + ["pooled"]):
        for j, a in enumerate(ARMS):
            if s == "pooled":
                v = [int(r["success"]) for x in NOISY for r in rows(A / x / a / "episodes.csv")]
            else:
                v = [int(r["success"]) for r in rows(A / s / a / "episodes.csv")]
            k, n = sum(v), len(v); lo, hi = wilson(k, n)
            ax[i].bar(j, 100 * k / n, color=COL[a]); ax[i].errorbar(j, 100 * k / n, yerr=[[100 * (k / n - lo)], [100 * (hi - k / n)]], color="k", capsize=2, lw=0.8)
        ax[i].set_xticks(range(len(ARMS)), [LABEL[a] for a in ARMS], rotation=45, ha="right")
        ax[i].set_title("2 suite nhiễu gộp" if s == "pooled" else s)
        lo_y = 88 if s == "libero_10" else 58
        ax[i].set_ylim(lo_y, lo_y + 12 if s == "libero_10" else lo_y + 20); ax[i].set_ylabel("success (%)" if i == 0 else "")
    fig.suptitle("F1. Success theo arm (Wilson 95% CI), 500 episode/ô (gộp: 1.000)")
    fig.tight_layout(); fig.savefig(FIG / "F1_success.png"); plt.close(fig)


# ------------------------------------------------------------------ F2/F3: what decides h
def fig_mechanism():
    stats = {}
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    for col, (arm, hk) in enumerate([("duac", "h_unc"), ("aac", "h_unc"), ("fixed_10", "h_unc")]):
        h_unc, xi, hm, grip = [], [], [], []
        for s in NOISY:
            for r in rows(A / s / arm / "decisions.csv"):
                h_unc.append(int(r["h_unc"])); xi.append(int(r["move_lower_bound"])); hm.append(int(r["h_method"])); grip.append(r["gripper_toggle"] == "1")
        h_unc, xi, hm, grip = map(np.array, (h_unc, xi, hm, grip))
        bins = np.arange(1.5, 11.5)
        ax[0, col].hist([h_unc, xi], bins=bins, label=["h_unc (bất định)", "ξ (chuyển động)"], color=["#1f77b4", "#ff7f0e"])
        ax[0, col].set_title(f"{LABEL[arm]}: phân bố h_unc và ξ (suite nhiễu)"); ax[0, col].legend(fontsize=7); ax[0, col].set_xlabel("h")
        det = {"ξ quyết định (ξ > h_unc)": xi > h_unc, "h_unc quyết định (h_unc > ξ)": h_unc > xi, "bằng nhau": h_unc == xi}
        for g, name in ((~grip, "không đóng/mở"), (grip, "có đóng/mở gripper")):
            stats.setdefault(arm, {})[name] = {k: float(np.mean(v[g])) for k, v in det.items()} | {
                "mean_h": float(hm[g].mean()), "mean_h_unc": float(h_unc[g].mean()), "mean_xi": float(xi[g].mean()), "n": int(g.sum())}
        names = list(det)
        x = np.arange(2)
        bottom = np.zeros(2)
        for k, c in zip(names, ["#ff7f0e", "#1f77b4", "#7f7f7f"]):
            vals = np.array([stats[arm]["không đóng/mở"][k], stats[arm]["có đóng/mở gripper"][k]]) * 100
            ax[1, col].bar(x, vals, bottom=bottom, color=c, label=k); bottom += vals
        ax[1, col].set_xticks(x, ["chunk không\nđóng/mở gripper", "chunk có\nđóng/mở gripper"])
        ax[1, col].set_ylabel("% decision"); ax[1, col].set_title(f"{LABEL[arm]}: thành phần nào quyết định h")
        if col == 0:
            ax[1, col].legend(fontsize=7, loc="lower left")
        stats[arm]["frac_h_unc_eq_9_or_10"] = float(np.mean(h_unc >= 9))
    fig.suptitle("F3. h = max(h_unc, ξ): ai quyết định?  (Fixed-10: đề xuất của DUAC tại các state mà Fixed-10 đi qua)")
    fig.tight_layout(); fig.savefig(FIG / "F3_mechanism.png"); plt.close(fig)
    EV["mechanism"] = stats


# ------------------------------------------------------------------ F4: shape of U / entropy profile
def fig_profiles():
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    U = []
    for s in NOISY:
        for r in rows(A / s / "duac" / "decisions.csv"):
            U.append([float(r[f"u_step_{j}"]) for j in range(1, 11)])
    U = np.array(U)
    rel = U / U.mean(1, keepdims=True)
    q = np.percentile(rel, [10, 50, 90], axis=0)
    ax[0].fill_between(range(1, 11), q[0], q[2], color="#1f77b4", alpha=0.2, label="p10–p90")
    ax[0].plot(range(1, 11), q[1], "o-", color="#1f77b4", label="trung vị")
    ax[0].set_title("DUAC: U_i / mean(U) theo vị trí trong chunk"); ax[0].set_xlabel("vị trí i trong chunk"); ax[0].legend(fontsize=7)
    prefix = np.cumsum(U, 1) / np.arange(1, 11)
    jump = np.argmax(np.diff(prefix, axis=1), axis=1) + 1
    ax[1].hist(jump, bins=np.arange(0.5, 10.5), color="#1f77b4")
    ax[1].set_title("DUAC: vị trí bước nhảy lớn nhất (argmax ΔŪ_h)"); ax[1].set_xlabel("h")
    # share of variance: between-position (profile shape) vs between-decision
    tot = rel.var(); within = rel.var(axis=1).mean()
    EV["duac_profile"] = {"median_rel_profile": q[1].tolist(), "jump_hist": np.bincount(jump, minlength=11)[1:].tolist(),
                          "frac_jump_at_9": float(np.mean(jump == 9)), "corr_profile_to_mean_profile":
                          float(np.median([np.corrcoef(u, q[1])[0, 1] for u in rel[:20000]]))}
    H = defaultdict(list)
    for s in NOISY:
        for r in load_algo(s, "aac", ["H_pos", "H_rot", "H_grip", "H_total"]):
            for k in H.keys() | {"H_pos", "H_rot", "H_grip", "H_total"}:
                H[k].append(r[k])
    for k, c in (("H_total", "k"), ("H_pos", "#2ca02c"), ("H_rot", "#ff7f0e"), ("H_grip", "#d62728")):
        m = np.array(H[k]).mean(0)
        ax[2].plot(range(1, 11), m, "o-", color=c, label=k, ms=3)
    ax[2].set_title("AAC: entropy trung bình theo vị trí"); ax[2].set_xlabel("vị trí i"); ax[2].legend(fontsize=7)
    Ht = np.array(H["H_total"]); jumpA = np.argmax(np.diff(np.cumsum(Ht, 1) / np.arange(1, 11), axis=1), axis=1) + 1
    Hg = np.array(H["H_grip"])
    EV["aac_profile"] = {"mean_H": {k: np.array(v).mean(0).tolist() for k, v in H.items()},
                         "jump_hist": np.bincount(jumpA, minlength=11)[1:].tolist(),
                         "frac_decisions_grip_entropy_gt_0.1": float(np.mean(Hg.max(1) > 0.1))}
    fig.suptitle("F4. Hình dạng profile bất định trong chunk (suite nhiễu)")
    fig.tight_layout(); fig.savefig(FIG / "F4_profiles.png"); plt.close(fig)


# ------------------------------------------------------------------ F5: event-aligned h
def fig_event_aligned():
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    bins = np.arange(-60, 61, 10)
    res = {}
    for arm in ["duac", "aac", "fixed_10"]:
        acc = {k: defaultdict(list) for k in ("h_method", "h_unc", "xi")}
        acc_sg = defaultdict(list)
        for s in NOISY:
            algo = load_algo(s, arm, ["task_id", "trial_id", "sim_step", "executed_chunk"])
            gev = gripper_events(algo)
            _, sub = episode_info(s, arm)
            for r in rows(A / s / arm / "decisions.csv"):
                key = (int(r["task_id"]), int(r["trial_id"])); t = int(r["sim_step"])
                ev = gev.get(key, [])
                if ev:
                    off = min((t - e for e in ev), key=abs)
                    if -60 <= off < 60:
                        b = int(np.digitize(off, bins)) - 1
                        acc["h_method"][b].append(int(r["h_method"])); acc["h_unc"][b].append(int(r["h_unc"])); acc["xi"][b].append(int(r["move_lower_bound"]))
                sg = [x for x in sub.get(key, []) if x is not None]
                if sg:
                    off = min((t - e for e in sg), key=abs)
                    if -60 <= off < 60:
                        acc_sg[int(np.digitize(off, bins)) - 1].append(int(r["h_method"]))
        centers = bins[:-1] + 5
        res[arm] = {k: [float(np.mean(v[b])) if v[b] else None for b in range(len(centers))] for k, v in acc.items()}
        res[arm]["h_method_vs_subgoal"] = [float(np.mean(acc_sg[b])) if acc_sg[b] else None for b in range(len(centers))]
        if arm in ("duac", "aac"):
            i = 0 if arm == "duac" else 1
            for k, c, ls in (("h_method", COL[arm], "-"), ("h_unc", "#1f77b4", "--"), ("xi", "#ff7f0e", "--")):
                ax[i].plot(centers, res[arm][k], ls, marker="o", ms=3, color=c, label={"h_method": "h chọn", "h_unc": "h_unc", "xi": "ξ"}[k])
            ax[i].axvline(0, color="grey", lw=0.8); ax[i].set_title(f"{LABEL[arm]}: h quanh lúc đóng/mở gripper")
            ax[i].set_xlabel("bước so với lúc gripper đổi trạng thái"); ax[i].legend(fontsize=7)
    for arm in ["duac", "aac"]:
        ax[2].plot(bins[:-1] + 5, res[arm]["h_method_vs_subgoal"], "o-", ms=3, color=COL[arm], label=LABEL[arm])
    ax[2].axvline(0, color="grey", lw=0.8); ax[2].set_title("h chọn quanh lúc một bước con hoàn thành"); ax[2].set_xlabel("bước so với lúc predicate thành đúng"); ax[2].legend(fontsize=7)
    fig.suptitle("F5. h theo thời gian quanh các sự kiện then chốt (suite nhiễu)")
    fig.tight_layout(); fig.savefig(FIG / "F5_event_aligned.png"); plt.close(fig)
    EV["event_aligned"] = {"bins_center": (bins[:-1] + 5).tolist(), **res}


# ------------------------------------------------------------------ F6: does U predict failure, and when
def auroc(p, n):
    if not p or not n:
        return float("nan")
    p, n = np.array(p), np.array(n)
    return float(((p[:, None] > n[None]) + 0.5 * (p[:, None] == n[None])).mean())


def fig_failure_signal():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    out = {}
    for arm, field, color in (("duac", "u_step", COL["duac"]), ("fixed_10", "u_step", COL["fixed_10"])):
        curves = {0: defaultdict(list), 1: defaultdict(list)}
        windows = {w: [] for w in ("đầu (decision 0–4)", "giữa", "5 decision cuối", "cả episode")}
        for s in NOISY:
            eps, _ = episode_info(s, arm)
            per = defaultdict(list)
            for r in rows(A / s / arm / "decisions.csv"):
                per[(int(r["task_id"]), int(r["trial_id"]))].append(np.mean([float(r[f"{field}_{j}"]) for j in range(1, 11)]))
            for k, u in per.items():
                ok = int(eps[k]["success"])
                for i, v in enumerate(u[:60]):
                    curves[ok][i].append(v)
            for t in range(10):
                ks = [k for k in per if k[0] == t]
                f = [k for k in ks if eps[k]["success"] == "0"]; o = [k for k in ks if eps[k]["success"] == "1"]
                if len(f) >= 5 and len(o) >= 5:
                    for w, sl in (("đầu (decision 0–4)", slice(0, 5)), ("giữa", slice(10, 20)), ("5 decision cuối", slice(-5, None)), ("cả episode", slice(None))):
                        windows[w].append(auroc([np.mean(per[k][sl]) for k in f if per[k][sl]], [np.mean(per[k][sl]) for k in o if per[k][sl]]))
        for ok, ls in ((1, "-"), (0, "--")):
            xs = sorted(i for i in curves[ok] if len(curves[ok][i]) >= 20)
            ax[0].plot(xs, [np.mean(curves[ok][i]) for i in xs], ls, color=color, label=f"{LABEL[arm]} {'thành công' if ok else 'thất bại'}")
        out[arm] = {w: float(np.nanmean(v)) for w, v in windows.items()} | {"n_tasks": len(windows["cả episode"])}
    ax[0].set_xlabel("decision thứ k trong episode"); ax[0].set_ylabel("U trung bình của chunk"); ax[0].legend(fontsize=7)
    ax[0].set_title("U theo thời gian: episode thành công vs thất bại")
    w = list(out["duac"].keys())[:4]
    x = np.arange(len(w))
    for i, arm in enumerate(("duac", "fixed_10")):
        ax[1].bar(x + 0.38 * i, [out[arm][k] for k in w], 0.38, color=COL[arm], label=LABEL[arm])
    ax[1].axhline(0.5, color="r", lw=0.8, ls=":"); ax[1].set_xticks(x + 0.19, w, rotation=15); ax[1].set_ylim(0.3, 0.9)
    ax[1].set_title("AUROC(U → thất bại), trong cùng task"); ax[1].legend(fontsize=7)
    fig.suptitle("F6. U của DUAC có báo trước thất bại không? (suite nhiễu)")
    fig.tight_layout(); fig.savefig(FIG / "F6_failure_signal.png"); plt.close(fig)
    EV["failure_signal_auroc_within_task"] = out


# ------------------------------------------------------------------ F7: discordant pairs vs Fixed-10
def fig_discordant():
    res = {}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for i, arm in enumerate(("duac", "aac")):
        stage = {"adaptive thua (A fail, F10 ok)": [0, 0, 0], "adaptive thắng (A ok, F10 fail)": [0, 0, 0]}
        for s in NOISY:
            ea, sa = episode_info(s, arm); ef, sf = episode_info(s, "fixed_10")
            for k in ea:
                a, f = ea[k]["success"] == "1", ef[k]["success"] == "1"
                if a == f:
                    continue
                loser_sub = sa[k] if f else sf[k]
                n_done = sum(x is not None for x in loser_sub)
                lab = "adaptive thua (A fail, F10 ok)" if f else "adaptive thắng (A ok, F10 fail)"
                stage[lab][min(n_done, 2)] += 1
        res[arm] = stage
        x = np.arange(3)
        for j, (lab, v) in enumerate(stage.items()):
            ax[i].bar(x + 0.38 * j, v, 0.38, label=f"{lab} (n={sum(v)})", color=["#d62728", "#2ca02c"][j])
        ax[i].set_xticks(x + 0.19, ["bên thua: 0 bước con", "bên thua: 1 predicate", "bên thua: đủ predicate\n(không cùng lúc)"], fontsize=7)
        ax[i].set_title(f"{LABEL[arm]} vs Fixed-10: các cặp khác kết quả"); ax[i].legend(fontsize=7)
    fig.suptitle("F7. Ở các cặp episode khác kết quả, bên thua dừng ở bước con nào? (suite nhiễu)")
    fig.tight_layout(); fig.savefig(FIG / "F7_discordant.png"); plt.close(fig)
    EV["discordant_vs_fixed10"] = res


# ------------------------------------------------------------------ F8: per task difference
def fig_per_task():
    fig, ax = plt.subplots(figsize=(12, 3.4))
    tasks = range(10); w = 0.13
    comps = [("duac", "fixed_10"), ("duac", "fixed_8"), ("duac", "random_h"), ("aac", "fixed_10"), ("aac", "fixed_8"), ("aac", "random_h")]
    out = {}
    for i, (x, y) in enumerate(comps):
        vals, err = [], []
        for t in tasks:
            d = []
            for s in NOISY:
                ex = {(int(r["task_id"]), int(r["trial_id"])): int(r["success"]) for r in rows(A / s / x / "episodes.csv")}
                ey = {(int(r["task_id"]), int(r["trial_id"])): int(r["success"]) for r in rows(A / s / y / "episodes.csv")}
                d += [ex[k] - ey[k] for k in ex if k[0] == t]
            d = np.array(d); vals.append(100 * d.mean()); err.append(196 * d.std(ddof=1) / math.sqrt(len(d)))
        out[f"{x}-{y}"] = vals
        ax.bar(np.arange(10) + (i - 2.5) * w, vals, w, yerr=err, capsize=1, error_kw={"lw": 0.5},
               color=COL[x], alpha=[1, 0.65, 0.35][i % 3], label=f"{LABEL[x]} − {LABEL[y]}")
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(10), [f"task {t}" for t in tasks]); ax.set_ylabel("chênh success (pp)")
    ax.legend(fontsize=7, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.33))
    ax.set_title("F8. Chênh success theo task (2 suite nhiễu gộp, 100 cặp/task, ±1.96 SE)")
    fig.tight_layout(); fig.savefig(FIG / "F8_per_task.png"); plt.close(fig)
    EV["per_task_pp"] = out


# ------------------------------------------------------------------ F2: h by phase
def fig_phase():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    out = {}
    for i, arm in enumerate(("duac", "aac")):
        ph = defaultdict(list)
        for s in NOISY:
            _, sub = episode_info(s, arm)
            for r in rows(A / s / arm / "decisions.csv"):
                ph[phase_of(int(r["sim_step"]), sub.get((int(r["task_id"]), int(r["trial_id"])), []))].append(int(r["h_executed"]))
        names = ["trước bước con 1", "giữa 2 bước con", "sau bước con 2"]
        for p in range(3):
            c = np.bincount(ph[p], minlength=11)[1:] / max(len(ph[p]), 1)
            ax[i].plot(range(1, 11), 100 * c, "o-", ms=3, label=f"{names[p]} (n={len(ph[p])}, mean {np.mean(ph[p]):.2f})")
        out[arm] = {names[p]: {"mean": float(np.mean(ph[p])), "n": len(ph[p])} for p in range(3)}
        ax[i].set_title(f"{LABEL[arm]}: phân bố h theo giai đoạn"); ax[i].set_xlabel("h"); ax[i].set_ylabel("% decision"); ax[i].legend(fontsize=7)
    fig.suptitle("F2. Phân bố h của AAC và DUAC theo giai đoạn bước con (suite nhiễu)")
    fig.tight_layout(); fig.savefig(FIG / "F2_h_by_phase.png"); plt.close(fig)
    EV["h_by_phase"] = out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig_success, fig_phase, fig_mechanism, fig_profiles, fig_event_aligned, fig_failure_signal, fig_discordant, fig_per_task):
        f(); print("done", f.__name__, flush=True)
    (A / "analysis" / "diagnosis.json").write_text(json.dumps(EV, indent=2, ensure_ascii=False, default=float) + "\n")
    print(json.dumps(EV, indent=1, ensure_ascii=False, default=float)[:6000])


if __name__ == "__main__":
    main()
