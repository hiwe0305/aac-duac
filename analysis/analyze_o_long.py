#!/usr/bin/env python3
"""O-LONG Statistical Analysis & Report Generator (`analysis/analyze_o_long.py`).

Implements all tables, figures data, and decision rules specified in `docs/task3010.md`:
  - T1: Success rate (%) + 95% Wilson CI by suite x arm.
  - T2: Paired difference of AAC and DUAC vs each Fixed arm (Fixed-5, Fixed-8, Fixed-10)
        and vs Random-h: Exact two-sided McNemar (n10, n01), 5,000-iteration cluster
        bootstrap 95% CI, and Holm-Bonferroni adjusted p-values.
  - T3: T2 broken down by task (10 tasks of LIBERO-10), plus T2 after dropping the single
        task with the largest gain ("khi bỏ task có lợi ích lớn nhất").
  - T4: Efficiency & latency: decisions/episode, mean h, distribution of h in {2..10},
        and infer_ms (p50, p95).
  - T5: Where the benefit lies ("Lợi ích nằm ở đâu"):
        (a) Subgoal 1 & Subgoal 2 completion rate (%) and mean first_true_step by arm.
        (b) Mean h and short-horizon fraction (h <= 4) of AAC/DUAC across 3 episode phases
            (Before subgoal 1, Between subgoal 1 & 2, After subgoal 2) and near gripper
            open/close (gripper_toggle == 1).
        (c) Offline alpha sensitivity replay across alpha in {1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0}.
  - T6: Failure taxonomy table for up to 20 failed episodes per arm (auto-labeling
        "hết giờ ở bước con 2" when subgoal 1 succeeded but subgoal 2 failed).
  - Section 5 Decision Rule: Automatic classification into "Adaptive work" / "Không work" /
        "Chưa kết luận được", plus mechanism check vs Fixed-8 and Random-h.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
FLOWBASED_ROOT = WORKSPACE_ROOT / "flowbased_aac"
for p in (WORKSPACE_ROOT, FLOWBASED_ROOT):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from flowbased_aac.duac_policy import find_first_idx, prefix_mean_jump_horizon  # noqa: E402


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Return 95% Wilson score confidence interval (low, high) in [0, 1]."""
    if total <= 0:
        return (0.0, 0.0)
    p = successes / float(total)
    n = float(total)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def exact_mcnemar_pvalue(n10: int, n01: int) -> float:
    """Two-sided exact binomial McNemar p-value for discordant pairs (n10, n01)."""
    b = n10 + n01
    if b == 0:
        return 1.0
    k = min(n10, n01)
    tail_prob = 0.0
    for i in range(k + 1):
        tail_prob += math.comb(b, i) * (0.5 ** b)
    return min(1.0, 2.0 * tail_prob)


def holm_bonferroni(p_values: Dict[str, float]) -> Dict[str, float]:
    """Apply Holm-Bonferroni step-down multiple-testing correction."""
    m = len(p_values)
    if m == 0:
        return {}
    sorted_items = sorted(p_values.items(), key=lambda kv: kv[1])
    adjusted: Dict[str, float] = {}
    running_max = 0.0
    for idx, (name, p_raw) in enumerate(sorted_items):
        mult = m - idx
        p_adj = min(1.0, p_raw * float(mult))
        running_max = max(running_max, p_adj)
        adjusted[name] = running_max
    return adjusted


def percentile(values: Sequence[float], q: float) -> float:
    """Compute percentile q in [0, 100] from a sequence of floats."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(int(round((q / 100.0) * (len(s) - 1))), len(s) - 1))
    return float(s[idx])


def cluster_bootstrap_paired_diff(
    paired_by_task: Dict[int, List[Tuple[int, int]]],
    n_boot: int = 5000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """5,000-iteration two-stage cluster bootstrap (resample tasks, then episodes within task)."""
    task_ids = sorted(paired_by_task.keys())
    n_tasks = len(task_ids)
    if n_tasks == 0:
        return (0.0, 0.0, 0.0)

    all_diffs = [a - b for pairs in paired_by_task.values() for (a, b) in pairs]
    obs_diff = sum(all_diffs) / float(max(len(all_diffs), 1))

    rng = random.Random(seed)
    boot_means: List[float] = []
    for _ in range(n_boot):
        sampled_tasks = [rng.choice(task_ids) for _ in range(n_tasks)]
        total_diff = 0.0
        total_count = 0
        for tid in sampled_tasks:
            pairs = paired_by_task[tid]
            m_ep = len(pairs)
            if m_ep == 0:
                continue
            for _ in range(m_ep):
                a, b = rng.choice(pairs)
                total_diff += float(a - b)
                total_count += 1
        boot_means.append(total_diff / float(max(total_count, 1)))

    boot_means.sort()
    lo_idx = max(0, min(int(0.025 * n_boot), n_boot - 1))
    hi_idx = max(0, min(int(0.975 * n_boot), n_boot - 1))
    return (obs_diff, boot_means[lo_idx], boot_means[hi_idx])


def leave_one_task_out(
    paired_by_task: Dict[int, List[Tuple[int, int]]],
) -> Tuple[Dict[int, float], Optional[int], float]:
    """Return LOTO delta map (pp), the best task_id with largest gain, and delta_pp after dropping best task."""
    out: Dict[int, float] = {}
    best_tid: Optional[int] = None
    best_task_gain = -1e9
    for tid, pairs in sorted(paired_by_task.items()):
        if pairs:
            gain = sum(a - b for a, b in pairs) / float(len(pairs)) * 100.0
            if gain > best_task_gain:
                best_task_gain = gain
                best_tid = tid

    for drop_tid in sorted(paired_by_task.keys()):
        rem = [
            (a - b)
            for tid, pairs in paired_by_task.items()
            if tid != drop_tid
            for (a, b) in pairs
        ]
        out[drop_tid] = (sum(rem) / float(len(rem)) * 100.0) if rem else 0.0

    drop_best_delta = out.get(best_tid, 0.0) if best_tid is not None else 0.0
    return out, best_tid, drop_best_delta


def _ep_trial_id(row: Dict[str, Any]) -> int:
    return int(row.get("trial_id", row.get("episode_id", 0)))


def _ep_n_decisions(row: Dict[str, Any]) -> float:
    return float(row.get("n_decisions", row.get("num_decisions", 0)))


def _ep_end_step(row: Dict[str, Any]) -> float:
    return float(row.get("end_step", row.get("env_steps", 0)))


def _ep_wall_s(row: Dict[str, Any]) -> float:
    return float(row.get("wall_s", row.get("wall_time_s", 0.0)))


def _dec_h_executed(row: Dict[str, Any]) -> int:
    return int(float(row.get("h_executed", row.get("h_selected", 10))))


def _dec_sim_step(row: Dict[str, Any]) -> int:
    return int(float(row.get("sim_step", row.get("env_step", 0))))


def load_episodes(run_dir: Path) -> List[Dict[str, Any]]:
    ep_csv = run_dir / "episodes.csv"
    if not ep_csv.exists():
        return []
    with open(ep_csv, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_decisions(run_dir: Path) -> List[Dict[str, Any]]:
    dec_csv = run_dir / "decisions.csv"
    if not dec_csv.exists():
        return []
    with open(dec_csv, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_subgoals(run_dir: Path) -> List[Dict[str, Any]]:
    sub_csv = run_dir / "subgoals.csv"
    if not sub_csv.exists():
        return []
    with open(sub_csv, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def compute_table_t1_and_t4_efficiency(
    runs_by_method: Dict[str, List[Dict[str, Any]]],
    decisions_by_method: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compute Table T1 (Success Rate + Wilson CI) and Table T4 (Efficiency & h distribution)."""
    t1_rows: List[Dict[str, Any]] = []
    t4_eff_rows: List[Dict[str, Any]] = []

    for method, ep_rows in runs_by_method.items():
        n = len(ep_rows)
        if n == 0:
            continue
        suite_name = ep_rows[0].get("suite", "libero_10")
        succ = sum(int(r["success"]) for r in ep_rows)
        sr = succ / float(n)
        ci_lo, ci_hi = wilson_ci(succ, n)
        t1_rows.append(
            {
                "suite": suite_name,
                "method": method,
                "episodes": n,
                "successes": succ,
                "success_rate_pct": round(sr * 100.0, 2),
                "wilson_95_low_pct": round(ci_lo * 100.0, 2),
                "wilson_95_high_pct": round(ci_hi * 100.0, 2),
            }
        )

        mean_h = sum(float(r["mean_h"]) for r in ep_rows) / float(n)
        mean_dec = sum(_ep_n_decisions(r) for r in ep_rows) / float(n)
        mean_steps = sum(_ep_end_step(r) for r in ep_rows) / float(n)
        mean_wall = sum(_ep_wall_s(r) for r in ep_rows) / float(n)
        mean_fwd = sum(float(r.get("forward_passes", 0.0)) for r in ep_rows) / float(n)
        mean_jvp = sum(float(r.get("jvp_probes", 0.0)) for r in ep_rows) / float(n)

        dec_rows = decisions_by_method.get(method, [])
        infer_vals = [float(d.get("infer_ms", 0.0)) for d in dec_rows if "infer_ms" in d]
        p50_ms = percentile(infer_vals, 50.0) if infer_vals else (mean_wall * 1000.0 / max(mean_dec, 1.0))
        p95_ms = percentile(infer_vals, 95.0) if infer_vals else p50_ms

        h_counts = {h: 0 for h in range(1, 11)}
        for d in dec_rows:
            h_val = _dec_h_executed(d)
            if 1 <= h_val <= 10:
                h_counts[h_val] += 1
        total_decs = max(len(dec_rows), 1)
        h_dist_pct = {f"h_{h}_pct": round(h_counts[h] / float(total_decs) * 100.0, 1) for h in range(2, 11)}

        eff_row: Dict[str, Any] = {
            "method": method,
            "mean_h": round(mean_h, 3),
            "decisions_per_ep": round(mean_dec, 2),
            "env_steps_per_ep": round(mean_steps, 2),
            "forward_passes_per_ep": round(mean_fwd, 1),
            "jvp_probes_per_ep": round(mean_jvp, 1),
            "wall_time_s_per_ep": round(mean_wall, 3),
            "ms_per_decision": round(p50_ms, 2),
            "infer_ms_p50": round(p50_ms, 2),
            "infer_ms_p95": round(p95_ms, 2),
        }
        eff_row.update(h_dist_pct)
        t4_eff_rows.append(eff_row)

    return t1_rows, t4_eff_rows


def compute_paired_comparisons_all_baselines(
    runs_by_method: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compute paired comparisons (T2 & T3 in task3010.md) of adaptive arms against all Fixed & Random baselines."""
    baseline_names = [
        b for b in ("fixed_10", "fixed_8", "fixed_5", "fixed_2", "random_h", "random_matched")
        if b in runs_by_method and runs_by_method[b]
    ]
    if not baseline_names and runs_by_method:
        first_key = next(iter(runs_by_method.keys()))
        baseline_names = [first_key]

    t3_per_task_rows: List[Dict[str, Any]] = []
    t2_paired_rows: List[Dict[str, Any]] = []
    raw_pvals: Dict[str, float] = {}

    for base_name in baseline_names:
        base_rows = runs_by_method[base_name]
        base_map: Dict[Tuple[int, int], int] = {
            (int(r["task_id"]), _ep_trial_id(r)): int(r["success"]) for r in base_rows
        }

        for cand_name, cand_rows in runs_by_method.items():
            if cand_name == base_name or not cand_rows:
                continue
            # Compare adaptive methods against all baselines, plus all methods against fixed_10
            if base_name != "fixed_10" and cand_name not in ("duac", "aac"):
                continue

            paired_by_task: Dict[int, List[Tuple[int, int]]] = {}
            n10 = 0
            n01 = 0
            n11 = 0
            n00 = 0

            for r in cand_rows:
                key = (int(r["task_id"]), _ep_trial_id(r))
                if key not in base_map:
                    continue
                a = int(r["success"])
                b = base_map[key]
                paired_by_task.setdefault(key[0], []).append((a, b))
                if a == 1 and b == 0:
                    n10 += 1
                elif a == 0 and b == 1:
                    n01 += 1
                elif a == 1 and b == 1:
                    n11 += 1
                else:
                    n00 += 1

            if not paired_by_task:
                continue

            loto_map, best_tid, drop_best_delta = leave_one_task_out(paired_by_task)
            for tid, pairs in sorted(paired_by_task.items()):
                sr_a = sum(a for a, _ in pairs) / float(len(pairs)) * 100.0
                sr_b = sum(b for _, b in pairs) / float(len(pairs)) * 100.0
                t3_per_task_rows.append(
                    {
                        "method": cand_name,
                        "baseline": base_name,
                        "task_id": tid,
                        "n_pairs": len(pairs),
                        "sr_adaptive_pct": round(sr_a, 2),
                        "sr_baseline_pct": round(sr_b, 2),
                        "sr_fixed10_pct": round(sr_b, 2),
                        "delta_pp": round(sr_a - sr_b, 2),
                        "loto_overall_delta_pp": round(loto_map.get(tid, 0.0), 2),
                    }
                )

            p_mcnemar = exact_mcnemar_pvalue(n10, n01)
            comp_key = f"{cand_name}__vs__{base_name}"
            raw_pvals[comp_key] = p_mcnemar
            obs_diff, ci_lo, ci_hi = cluster_bootstrap_paired_diff(paired_by_task, n_boot=5000)

            t2_paired_rows.append(
                {
                    "comparison_id": comp_key,
                    "method": cand_name,
                    "baseline": base_name,
                    "n_pairs": n10 + n01 + n11 + n00,
                    "n10_adaptive_only": n10,
                    "n01_fixed10_only": n01,
                    "n01_baseline_only": n01,
                    "delta_pp": round(obs_diff * 100.0, 2),
                    "cluster_boot_95_low_pp": round(ci_lo * 100.0, 2),
                    "cluster_boot_95_high_pp": round(ci_hi * 100.0, 2),
                    "p_mcnemar_raw": round(p_mcnemar, 6),
                    "best_task_id": best_tid,
                    "delta_without_best_task_pp": round(drop_best_delta, 2),
                }
            )

    holm_map = holm_bonferroni(raw_pvals)
    for row in t2_paired_rows:
        cid = row["comparison_id"]
        row["p_holm_adjusted"] = round(holm_map.get(cid, 1.0), 6)

    return t3_per_task_rows, t2_paired_rows


def compute_table_t5_where_benefit_lies(
    subgoals_by_method: Dict[str, List[Dict[str, Any]]],
    decisions_by_method: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Compute Table T5 ('Lợi ích nằm ở đâu') from subgoals.csv and decisions.csv."""
    subgoal_completion: List[Dict[str, Any]] = []
    phase_horizons: List[Dict[str, Any]] = []

    for method, sub_rows in subgoals_by_method.items():
        if not sub_rows:
            continue
        # Group by predicate_idx (0 = subgoal 1, 1 = subgoal 2)
        by_pred: Dict[int, List[Optional[int]]] = {}
        # Also map (task_id, trial_id) -> {pred_idx: first_true_step}
        ep_sub_steps: Dict[Tuple[int, int], Dict[int, Optional[int]]] = {}

        for r in sub_rows:
            if "predicate_idx" not in r:
                continue
            tid = int(r["task_id"])
            tr_id = int(r.get("trial_id", 0))
            p_idx = int(r["predicate_idx"])
            fts_raw = str(r.get("first_true_step", "")).strip()
            fts = int(fts_raw) if fts_raw != "" else None
            by_pred.setdefault(p_idx, []).append(fts)
            ep_sub_steps.setdefault((tid, tr_id), {})[p_idx] = fts

        for p_idx in sorted(by_pred.keys()):
            steps = by_pred[p_idx]
            n_total = len(steps)
            sat_steps = [s for s in steps if s is not None]
            comp_rate = (len(sat_steps) / float(max(n_total, 1))) * 100.0
            mean_step = (sum(sat_steps) / float(len(sat_steps))) if sat_steps else 0.0
            subgoal_completion.append(
                {
                    "method": method,
                    "predicate_idx": p_idx,
                    "subgoal_label": f"subgoal_{p_idx + 1}",
                    "episodes": n_total,
                    "completed": len(sat_steps),
                    "completion_rate_pct": round(comp_rate, 2),
                    "mean_first_true_step": round(mean_step, 1),
                }
            )

        # Compute mean h and fraction(h <= 4) by episode phase and gripper_toggle
        dec_rows = decisions_by_method.get(method, [])
        if not dec_rows:
            continue
        buckets: Dict[str, List[int]] = {
            "before_subgoal_1": [],
            "between_subgoal_1_and_2": [],
            "after_subgoal_2": [],
            "near_gripper_toggle": [],
        }
        for d in dec_rows:
            tid = int(d["task_id"])
            tr_id = _ep_trial_id(d)
            s_step = _dec_sim_step(d)
            h_val = _dec_h_executed(d)
            gtog = int(float(d.get("gripper_toggle", 0)))
            if gtog == 1:
                buckets["near_gripper_toggle"].append(h_val)

            p_map = ep_sub_steps.get((tid, tr_id), {})
            s1 = p_map.get(0)
            s2 = p_map.get(1)
            if s1 is None or s_step < s1:
                buckets["before_subgoal_1"].append(h_val)
            elif s2 is None or s_step < s2:
                buckets["between_subgoal_1_and_2"].append(h_val)
            else:
                buckets["after_subgoal_2"].append(h_val)

        for phase_name, h_list in buckets.items():
            if not h_list:
                continue
            mean_h = sum(h_list) / float(len(h_list))
            short_pct = (sum(1 for h in h_list if h <= 4) / float(len(h_list))) * 100.0
            phase_horizons.append(
                {
                    "method": method,
                    "subgoal_phase": phase_name,
                    "decisions": len(h_list),
                    "mean_h": round(mean_h, 3),
                    "short_horizon_le4_pct": round(short_pct, 2),
                }
            )

    return {
        "subgoal_completion_by_arm": subgoal_completion,
        "horizon_by_phase_and_gripper": phase_horizons,
    }


def compute_table_t5_offline_alpha_replay(
    decisions_rows: List[Dict[str, Any]],
    alphas: Sequence[float] = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0),
) -> List[Dict[str, Any]]:
    """Replay DUAC / AAC decisions offline across alpha grid (Figure F2)."""
    if not decisions_rows:
        return []
    out: List[Dict[str, Any]] = []
    for alpha in alphas:
        for min_h in (1, 2):
            replayed_h: List[int] = []
            xi_dominated = 0
            for r in decisions_rows:
                u_step = [float(r[f"u_step_{j+1}"]) for j in range(10)]
                m_step = [float(r[f"m_step_{j+1}"]) for j in range(10)]
                xi = find_first_idx(m_step, alpha=alpha, min_horizon=min_h)
                h_unc, _, _ = prefix_mean_jump_horizon(u_step, min_horizon=min_h)
                h_sel = max(xi, h_unc)
                replayed_h.append(h_sel)
                if xi > h_unc:
                    xi_dominated += 1
            n_dec = len(replayed_h)
            mean_h = sum(replayed_h) / float(max(n_dec, 1))
            out.append(
                {
                    "alpha": alpha,
                    "min_horizon": min_h,
                    "num_decisions": n_dec,
                    "mean_h": round(mean_h, 3),
                    "xi_dominated_pct": round((xi_dominated / float(max(n_dec, 1))) * 100.0, 2),
                }
            )
    return out


def compute_table_t6_failure_taxonomy(
    runs_by_method: Dict[str, List[Dict[str, Any]]],
    subgoals_by_method: Dict[str, List[Dict[str, Any]]],
    max_per_arm: int = 20,
) -> List[Dict[str, Any]]:
    """Generate Table T6 failure taxonomy up to 20 failed episodes per arm (Section 4.3 T6)."""
    rows: List[Dict[str, Any]] = []
    for method, ep_rows in runs_by_method.items():
        sub_rows = subgoals_by_method.get(method, [])
        ep_sub_map: Dict[Tuple[int, int], Dict[int, Optional[int]]] = {}
        for sr in sub_rows:
            if "predicate_idx" not in sr:
                continue
            tid = int(sr["task_id"])
            tr_id = int(sr.get("trial_id", 0))
            p_idx = int(sr["predicate_idx"])
            fts_raw = str(sr.get("first_true_step", "")).strip()
            ep_sub_map.setdefault((tid, tr_id), {})[p_idx] = int(fts_raw) if fts_raw != "" else None

        failed_eps = [r for r in ep_rows if int(r["success"]) == 0][:max_per_arm]
        for r in failed_eps:
            tid = int(r["task_id"])
            tr_id = _ep_trial_id(r)
            p_map = ep_sub_map.get((tid, tr_id), {})
            s1_done = p_map.get(0) is not None
            s2_done = p_map.get(1) is not None
            if s1_done and not s2_done:
                auto_tag = "hết giờ ở bước con 2"
            else:
                auto_tag = "không với tới vật / gắp trượt / rơi vật / sai vật (cần xem video)"
            rows.append(
                {
                    "arm": method,
                    "task_id": tid,
                    "trial_id": tr_id,
                    "end_step": int(_ep_end_step(r)),
                    "subgoal_1_done": int(s1_done),
                    "subgoal_2_done": int(s2_done),
                    "failure_category": auto_tag,
                }
            )
    return rows


def evaluate_section5_decision_rule(
    t2_paired_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate exact Section 5 Decision Rule from `docs/task3010.md`:
    - 'Adaptive work': AAC or DUAC beats ALL available Fixed arms (Fixed-5, Fixed-8, Fixed-10)
      by >= +3.0 pp, McNemar p_Holm < 0.05, AND still beats by >= +1.0 pp when dropping best task.
    - 'Không work': 95% CI upper bound of diff vs best Fixed arm < +3.0 pp.
    - 'Chưa kết luận được': All other cases.
    """
    by_method: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in t2_paired_rows:
        m = row["method"]
        b = row["baseline"]
        by_method.setdefault(m, {})[b] = row

    verdicts: Dict[str, Any] = {}
    for method, base_dict in by_method.items():
        fixed_candidates = [
            base_dict[b] for b in ("fixed_5", "fixed_8", "fixed_10") if b in base_dict
        ]
        if not fixed_candidates and "fixed_10" in base_dict:
            fixed_candidates = [base_dict["fixed_10"]]

        if not fixed_candidates:
            continue

        # Best Fixed arm has the smallest delta_pp (hardest baseline to beat)
        vs_best_fixed = min(fixed_candidates, key=lambda r: float(r["delta_pp"]))
        beats_all_fixed_3pp = all(float(r["delta_pp"]) >= 3.0 for r in fixed_candidates)
        all_holm_sig = all(float(r["p_holm_adjusted"]) < 0.05 for r in fixed_candidates)
        all_loto_1pp = all(float(r["delta_without_best_task_pp"]) >= 1.0 for r in fixed_candidates)

        ci_hi_vs_best_fixed = float(vs_best_fixed["cluster_boot_95_high_pp"])

        if beats_all_fixed_3pp and all_holm_sig and all_loto_1pp:
            label_vi = "Adaptive work"
            outcome = "SCENARIO_A_ADAPTIVE_WINS"
        elif ci_hi_vs_best_fixed < 3.0:
            label_vi = "Không work"
            outcome = "SCENARIO_B_KHONG_WORK"
        else:
            label_vi = "Chưa kết luận được"
            outcome = "SCENARIO_C_INCONCLUSIVE"

        vs_f8 = base_dict.get("fixed_8")
        vs_rand = base_dict.get("random_h") or base_dict.get("random_matched")

        verdicts[method] = {
            "verdict_vi": label_vi,
            "outcome": outcome,
            "best_fixed_baseline": vs_best_fixed["baseline"],
            "delta_vs_best_fixed_pp": float(vs_best_fixed["delta_pp"]),
            "ci95_high_vs_best_fixed_pp": ci_hi_vs_best_fixed,
            "p_holm_vs_best_fixed": float(vs_best_fixed["p_holm_adjusted"]),
            "delta_without_best_task_pp": float(vs_best_fixed["delta_without_best_task_pp"]),
            "delta_vs_fixed8_pp": float(vs_f8["delta_pp"]) if vs_f8 else None,
            "delta_vs_random_h_pp": float(vs_rand["delta_pp"]) if vs_rand else None,
        }
    return verdicts


def analyze_runs(
    run_dirs: Dict[str, Path],
    output_dir: Path,
) -> Dict[str, Any]:
    """Run full O-LONG analysis across a mapping of `method -> run_dir`."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_by_method: Dict[str, List[Dict[str, Any]]] = {}
    decisions_by_method: Dict[str, List[Dict[str, Any]]] = {}
    subgoals_by_method: Dict[str, List[Dict[str, Any]]] = {}

    for method, rdir in run_dirs.items():
        if not rdir.exists():
            continue
        runs_by_method[method] = load_episodes(rdir)
        decisions_by_method[method] = load_decisions(rdir)
        subgoals_by_method[method] = load_subgoals(rdir)

    t1, t4_eff = compute_table_t1_and_t4_efficiency(runs_by_method, decisions_by_method)
    t3_per_task, t2_paired = compute_paired_comparisons_all_baselines(runs_by_method)

    duac_decs = decisions_by_method.get("duac") or next(iter(decisions_by_method.values()), [])
    t5_alpha = compute_table_t5_offline_alpha_replay(duac_decs)
    t5_where = compute_table_t5_where_benefit_lies(subgoals_by_method, decisions_by_method)
    t6_failures = compute_table_t6_failure_taxonomy(runs_by_method, subgoals_by_method)
    verdicts = evaluate_section5_decision_rule(t2_paired)

    report = {
        "T1_success_rates": t1,
        "T2_per_task_paired": t3_per_task,
        "T3_compute_efficiency": t4_eff,
        "T4_statistical_tests": t2_paired,
        "T5_where_benefit_lies": t5_where,
        "T5_offline_alpha_replay": t5_alpha,
        "T6_subgoal_phases": t5_where["horizon_by_phase_and_gripper"],
        "T6_failure_taxonomy": t6_failures,
        "section5_decision_rule": verdicts,
    }
    (output_dir / "o_long_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Format comprehensive Markdown report matching docs/task3010.md Section 6
    md_lines = [
        "# O-LONG Report: Adaptive Action Chunking (`AAC` vs `DUAC`) trên `pi0.5` (`libero_10`)",
        "",
        "## 1. Kết luận 1 đoạn (`docs/task3010.md` Mục 5 & 6)",
    ]
    if verdicts:
        for m, v in verdicts.items():
            md_lines.append(
                f"- **`{m}`**: **{v['verdict_vi']}** (`{v['outcome']}`) — "
                f"$\\Delta$ so với `{v['best_fixed_baseline']}` = **{v['delta_vs_best_fixed_pp']:+.2f} pp** "
                f"(95% CI upper = {v['ci95_high_vs_best_fixed_pp']:+.2f} pp, "
                f"$p_{{\\text{{Holm}}}}$ = {v['p_holm_vs_best_fixed']:.4f}, "
                f"$\\Delta_{{-\\text{{best task}}}}$ = {v['delta_without_best_task_pp']:+.2f} pp; "
                f"vs `fixed_8` = {v['delta_vs_fixed8_pp']} pp; vs `random_h` = {v['delta_vs_random_h_pp']} pp)."
            )
    else:
        md_lines.append("- Chưa đủ dữ liệu đối chứng để kết luận.")

    md_lines.extend(
        [
            "",
            "## 2. Thiết lập & Bảng T1 — T4 (Kết quả chính & Hiệu năng)",
            "### Bảng T1: Success Rate & 95% Wilson CI theo `suite × arm`",
            "| Suite | Arm | Episodes | Successes | SR (%) | 95% Wilson CI (%) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for r in t1:
        md_lines.append(
            f"| `{r['suite']}` | `{r['method']}` | {r['episodes']} | {r['successes']} | "
            f"{r['success_rate_pct']:.2f}% | [{r['wilson_95_low_pct']:.2f}%, {r['wilson_95_high_pct']:.2f}%] |"
        )

    if t2_paired:
        md_lines.extend(
            [
                "",
                "### Bảng T2: Kiểm định cặp (Exact McNemar + 5,000 Cluster Bootstrap + Holm-Bonferroni)",
                "| Candidate | Baseline | Pairs | $n_{10}$ | $n_{01}$ | $\\Delta$ (pp) | Cluster Boot 95% CI (pp) | $p_{\\text{McNemar}}$ | $p_{\\text{Holm}}$ | $\\Delta_{-\\text{best task}}$ (pp) |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for r in t2_paired:
            md_lines.append(
                f"| `{r['method']}` | `{r['baseline']}` | {r['n_pairs']} | {r['n10_adaptive_only']} | "
                f"{r['n01_baseline_only']} | {r['delta_pp']:+.2f} | "
                f"[{r['cluster_boot_95_low_pp']:+.2f}, {r['cluster_boot_95_high_pp']:+.2f}] | "
                f"{r['p_mcnemar_raw']:.4f} | {r['p_holm_adjusted']:.4f} | "
                f"{r['delta_without_best_task_pp']:+.2f} (drop T{r['best_task_id']}) |"
            )

    md_lines.extend(
        [
            "",
            "### Bảng T4: Hiệu năng (`decisions/episode`, `mean h`, `infer_ms` p50/p95)",
            "| Arm | Mean $h$ | Decisions/Ep | Steps/Ep | Forward Passes/Ep | JVP Probes/Ep | `infer_ms` p50 | `infer_ms` p95 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in t4_eff:
        md_lines.append(
            f"| `{r['method']}` | {r['mean_h']:.2f} | {r['decisions_per_ep']:.1f} | "
            f"{r['env_steps_per_ep']:.1f} | {r['forward_passes_per_ep']:.0f} | "
            f"{r['jvp_probes_per_ep']:.0f} | {r['infer_ms_p50']:.2f} | {r['infer_ms_p95']:.2f} |"
        )

    if t5_where["subgoal_completion_by_arm"]:
        md_lines.extend(
            [
                "",
                "## 3. Bảng T5 — Lợi ích nằm ở đâu (Subgoal 1 & 2, Giai đoạn Episode, Độ nhạy $\\alpha$)",
                "### T5a: Tỉ lệ hoàn thành & bước hoàn thành trung bình (`first_true_step`) của Subgoal 1 & 2",
                "| Arm | Subgoal | Episodes | Completed | Completion (%) | Mean `first_true_step` |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for r in t5_where["subgoal_completion_by_arm"]:
            md_lines.append(
                f"| `{r['method']}` | `{r['subgoal_label']}` | {r['episodes']} | {r['completed']} | "
                f"{r['completion_rate_pct']:.2f}% | {r['mean_first_true_step']:.1f} |"
            )

    if t5_where["horizon_by_phase_and_gripper"]:
        md_lines.extend(
            [
                "",
                "### T5b: Mean $h$ và tỉ lệ $h \\le 4$ theo 3 giai đoạn episode & gần lúc đóng/mở kẹp",
                "| Arm | Giai đoạn | Decisions | Mean $h$ | $h \\le 4$ (%) |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for r in t5_where["horizon_by_phase_and_gripper"]:
            md_lines.append(
                f"| `{r['method']}` | `{r['subgoal_phase']}` | {r['decisions']} | "
                f"{r['mean_h']:.2f} | {r['short_horizon_le4_pct']:.2f}% |"
            )

    if t6_failures:
        md_lines.extend(
            [
                "",
                "## 4. Bảng T6 — Phân loại lỗi (tối đa 20 episode thất bại mỗi arm)",
                "| Arm | Task | Trial | End Step | Subgoal 1 Done | Subgoal 2 Done | Phân loại lỗi |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for r in t6_failures[:40]:
            md_lines.append(
                f"| `{r['arm']}` | {r['task_id']} | {r['trial_id']} | {r['end_step']} | "
                f"{r['subgoal_1_done']} | {r['subgoal_2_done']} | {r['failure_category']} |"
            )

    report_md_text = "\n".join(md_lines) + "\n"
    (output_dir / "o_long_report.md").write_text(report_md_text, encoding="utf-8")
    # Also write outputs/REPORT.md as specified in docs/task3010.md Section 6
    outputs_root = WORKSPACE_ROOT / "outputs"
    if outputs_root.exists():
        (outputs_root / "REPORT.md").write_text(report_md_text, encoding="utf-8")
    return report


def run_self_test() -> int:
    """Self-test statistical functions and end-to-end report generation."""
    from adaptive_action_chunking.aac import run_aac_libero_eval
    from flowbased_aac.eval_libero import run_libero_evaluation

    print("[analysis/analyze_o_long.py --self-test] Running statistical pipeline verification...")
    lo, hi = wilson_ci(450, 500)
    assert 0.86 < lo < 0.90 and 0.90 < hi < 0.93
    p_mc = exact_mcnemar_pvalue(20, 5)
    assert 0.001 < p_mc < 0.01

    with tempfile.TemporaryDirectory(prefix="analyze_test_") as tmpdir:
        tmp = Path(tmpdir)
        f10_dir = tmp / "fixed_10"
        f8_dir = tmp / "fixed_8"
        f5_dir = tmp / "fixed_5"
        duac_dir = tmp / "duac"
        rand_dir = tmp / "random_h"
        aac_dir = tmp / "aac"

        for fh, fdir in ((10, f10_dir), (8, f8_dir), (5, f5_dir)):
            run_libero_evaluation(
                suite="libero_10",
                num_trials=2,
                seed=7,
                fixed_horizon=fh,
                output_dir=fdir,
                task_ids=[0, 1],
                synthetic_dry_run=True,
            )
        run_libero_evaluation(
            suite="libero_10",
            num_trials=2,
            seed=7,
            state_bank=True,
            output_dir=duac_dir,
            task_ids=[0, 1],
            synthetic_dry_run=True,
        )
        run_libero_evaluation(
            suite="libero_10",
            num_trials=2,
            seed=7,
            random_horizon_from=duac_dir / "decisions.csv",
            output_dir=rand_dir,
            task_ids=[0, 1],
            synthetic_dry_run=True,
        )
        run_aac_libero_eval(
            backbone="pi05",
            suite="libero_10",
            trials=2,
            seed=7,
            out_dir=aac_dir,
            task_ids=[0, 1],
            synthetic_dry_run=True,
        )

        rep = analyze_runs(
            {
                "fixed_10": f10_dir,
                "fixed_8": f8_dir,
                "fixed_5": f5_dir,
                "duac": duac_dir,
                "random_h": rand_dir,
                "aac": aac_dir,
            },
            output_dir=tmp / "report",
        )
        assert len(rep["T1_success_rates"]) == 6
        assert len(rep["T4_statistical_tests"]) >= 6
        assert (tmp / "report" / "o_long_report.md").exists()

    print("[analysis/analyze_o_long.py --self-test] ALL STATISTICAL CHECKS PASSED.")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze O-LONG experiments and generate T1-T6 / REPORT.md.")
    parser.add_argument("--manifest", type=Path, default=WORKSPACE_ROOT / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE_ROOT / "outputs" / "analysis_report")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    run_dirs: Dict[str, Path] = {}
    for entry in data.get("runs", []):
        m = entry["method"]
        p = WORKSPACE_ROOT / entry["path"]
        run_dirs[m] = p

    report = analyze_runs(run_dirs, args.output_dir)
    print(json.dumps(report["section5_decision_rule"], indent=2, ensure_ascii=False))
    print(f"Report written to {args.output_dir / 'o_long_report.md'} and outputs/REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
