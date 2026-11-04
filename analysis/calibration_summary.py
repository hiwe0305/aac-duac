#!/usr/bin/env python3
"""Step 1 summary: success rate per calibration run with Wilson 95% CI and the 30-70% selection rule.

    python analysis/calibration_summary.py [outputs/calibration]
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/calibration")
    rows = []
    for ep in sorted(root.glob("*/*/episodes.csv")):
        eps = list(csv.DictReader(open(ep, newline="")))
        if not eps:
            continue
        n = len(eps)
        k = sum(int(r["success"]) for r in eps)
        lo, hi = wilson(k, n)
        per_task = {}
        for r in eps:
            per_task.setdefault(int(r["task_id"]), []).append(int(r["success"]))
        rows.append((ep.parent.parent.name, ep.parent.name, n, k, k / n, lo, hi,
                     " ".join(f"{t}:{sum(v)}/{len(v)}" for t, v in sorted(per_task.items()))))
    print(f"{'suite':30s} {'arm':9s} {'eps':>4s} {'succ':>6s}  {'95% CI':>15s}  30-70%  per task")
    for suite, arm, n, k, p, lo, hi, pt in sorted(rows, key=lambda r: abs(r[4] - 0.5)):
        mark = "yes" if 0.30 <= p <= 0.70 else "no"
        print(f"{suite:30s} {arm:9s} {n:4d} {100 * p:5.1f}%  [{100 * lo:5.1f}, {100 * hi:5.1f}]  {mark:6s}  {pt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
