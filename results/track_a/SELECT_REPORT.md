# Track A: chunk selection (medoid) compared on many axes

*`analysis/track_a_select_report.py`; 1500 matched episodes per arm (3 suites x 10 tasks x 50 trials, env seed 7). π0.5 `pi05_libero`, chunk H = 10.*

Arms: `medoid_h8` / `medoid_h10` = execute the medoid of the AAC server's 20 samples at h = 8 / 10; `sample0_h8` = same server and h, execute sample 0 (the chunk AAC executes): the no-selection control. Earlier arms: Fixed-5/8/10, DUAC, AAC, Random-h.

## 1. Success rate (%, Wilson 95% CI)

| arm | libero_10 (clean) | object | env | **noisy pooled** | all |
|---|---|---|---|---|---|
| fixed_5 | 93.8 [91-96] | 67.6 [63-72] | 69.4 [65-73] | 68.5 [66-71] | 76.9 [75-79] |
| fixed_8 | 94.6 [92-96] | 65.0 [61-69] | 73.6 [70-77] | 69.3 [66-72] | 77.7 [76-80] |
| fixed_10 | 95.4 [93-97] | 68.2 [64-72] | 72.8 [69-77] | 70.5 [68-73] | 78.8 [77-81] |
| duac | 94.2 [92-96] | 67.8 [64-72] | 69.2 [65-73] | 68.5 [66-71] | 77.1 [75-79] |
| aac | 94.0 [92-96] | 67.0 [63-71] | 68.0 [64-72] | 67.5 [65-70] | 76.3 [74-78] |
| random_h | 95.0 [93-97] | 69.0 [65-73] | 70.2 [66-74] | 69.6 [67-72] | 78.1 [76-80] |
| **medoid_h8** | 94.6 [92-96] | 69.8 [66-74] | 72.6 [69-76] | 71.2 [68-74] | 79.0 [77-81] |
| **sample0_h8** | 94.8 [92-96] | 67.0 [63-71] | 71.8 [68-76] | 69.4 [66-72] | 77.9 [76-80] |
| **medoid_h10** | 94.8 [92-96] | 67.8 [64-72] | 70.0 [66-74] | 68.9 [66-72] | 77.5 [75-80] |

## 2. Paired comparisons

Bootstrap clustered by (suite, task); Holm within each block.

**noisy pooled (object + env, n = 1000)**

| comparison | diff (pp) | 95% CI | only A succeeds | only B succeeds | p McNemar | p Holm | verdict |
|---|---|---|---|---|---|---|---|
| medoid_h8 vs fixed_5 | +2.7 | [+0.1, +5.3] | 104 | 77 | 0.053 | 0.689 | better |
| medoid_h8 vs fixed_8 | +1.9 | [-0.3, +4.2] | 79 | 60 | 0.127 | 1 | inconclusive |
| medoid_h8 vs fixed_10 | +0.7 | [-1.6, +3.0] | 75 | 68 | 0.616 | 1 | inconclusive |
| medoid_h8 vs duac | +2.7 | [+0.1, +5.2] | 96 | 69 | 0.0426 | 0.597 | better |
| medoid_h8 vs aac | +3.7 | [+1.4, +6.0] | 92 | 55 | 0.00286 | 0.0429 | better |
| medoid_h8 vs random_h | +1.6 | [-0.7, +3.9] | 76 | 60 | 0.198 | 1 | inconclusive |
| medoid_h8 vs sample0_h8 | +1.8 | [-0.6, +4.2] | 85 | 67 | 0.168 | 1 | inconclusive |
| medoid_h8 vs medoid_h10 | +2.3 | [+0.1, +4.6] | 83 | 60 | 0.0654 | 0.785 | better |
| medoid_h10 vs fixed_5 | +0.4 | [-2.1, +3.0] | 91 | 87 | 0.822 | 1 | inconclusive |
| medoid_h10 vs fixed_8 | -0.4 | [-2.7, +2.0] | 72 | 76 | 0.805 | 1 | inconclusive |
| medoid_h10 vs fixed_10 | -1.6 | [-3.9, +0.7] | 62 | 78 | 0.205 | 1 | inconclusive |
| medoid_h10 vs duac | +0.4 | [-2.0, +2.7] | 77 | 73 | 0.807 | 1 | inconclusive |
| medoid_h10 vs aac | +1.4 | [-0.9, +3.7] | 80 | 66 | 0.282 | 1 | inconclusive |
| medoid_h10 vs random_h | -0.7 | [-3.0, +1.6] | 65 | 72 | 0.608 | 1 | inconclusive |
| medoid_h10 vs sample0_h8 | -0.5 | [-2.8, +1.8] | 69 | 74 | 0.738 | 1 | inconclusive |

**clean libero_10 (n = 500)**

| comparison | diff (pp) | 95% CI | only A succeeds | only B succeeds | p McNemar | p Holm | verdict |
|---|---|---|---|---|---|---|---|
| medoid_h8 vs fixed_5 | +0.8 | [-1.6, +3.2] | 22 | 18 | 0.636 | 1 | inconclusive |
| medoid_h8 vs fixed_8 | +0.0 | [-2.4, +2.4] | 19 | 19 | 1 | 1 | inconclusive |
| medoid_h8 vs fixed_10 | -0.8 | [-3.2, +1.8] | 19 | 23 | 0.644 | 1 | inconclusive |
| medoid_h8 vs duac | +0.4 | [-2.0, +2.8] | 21 | 19 | 0.875 | 1 | inconclusive |
| medoid_h8 vs aac | +0.6 | [-1.8, +3.0] | 20 | 17 | 0.743 | 1 | inconclusive |
| medoid_h8 vs random_h | -0.4 | [-2.6, +1.8] | 15 | 17 | 0.86 | 1 | inconclusive |
| medoid_h8 vs sample0_h8 | -0.2 | [-2.6, +2.2] | 18 | 19 | 1 | 1 | inconclusive |
| medoid_h8 vs medoid_h10 | -0.2 | [-2.6, +2.2] | 18 | 19 | 1 | 1 | inconclusive |
| medoid_h10 vs fixed_5 | +1.0 | [-1.6, +3.4] | 23 | 18 | 0.533 | 1 | inconclusive |
| medoid_h10 vs fixed_8 | +0.2 | [-2.0, +2.6] | 18 | 17 | 1 | 1 | inconclusive |
| medoid_h10 vs fixed_10 | -0.6 | [-3.0, +1.8] | 17 | 20 | 0.743 | 1 | inconclusive |
| medoid_h10 vs duac | +0.6 | [-1.8, +3.0] | 21 | 18 | 0.749 | 1 | inconclusive |
| medoid_h10 vs aac | +0.8 | [-1.8, +3.4] | 24 | 20 | 0.652 | 1 | inconclusive |
| medoid_h10 vs random_h | -0.2 | [-2.4, +2.0] | 17 | 18 | 1 | 1 | inconclusive |
| medoid_h10 vs sample0_h8 | +0.0 | [-2.6, +2.6] | 21 | 21 | 1 | 1 | inconclusive |

**all 3 suites (n = 1500)**

| comparison | diff (pp) | 95% CI | only A succeeds | only B succeeds | p McNemar | p Holm | verdict |
|---|---|---|---|---|---|---|---|
| medoid_h8 vs fixed_5 | +2.1 | [+0.1, +4.0] | 126 | 95 | 0.0433 | 0.607 | better |
| medoid_h8 vs fixed_8 | +1.3 | [-0.5, +2.9] | 98 | 79 | 0.176 | 1 | inconclusive |
| medoid_h8 vs fixed_10 | +0.2 | [-1.6, +1.9] | 94 | 91 | 0.883 | 1 | inconclusive |
| medoid_h8 vs duac | +1.9 | [+0.1, +3.8] | 117 | 88 | 0.0502 | 0.653 | better |
| medoid_h8 vs aac | +2.7 | [+0.9, +4.5] | 112 | 72 | 0.00392 | 0.0588 | better |
| medoid_h8 vs random_h | +0.9 | [-0.7, +2.6] | 91 | 77 | 0.316 | 1 | inconclusive |
| medoid_h8 vs sample0_h8 | +1.1 | [-0.7, +2.9] | 103 | 86 | 0.244 | 1 | inconclusive |
| medoid_h8 vs medoid_h10 | +1.5 | [-0.2, +3.3] | 101 | 79 | 0.117 | 1 | inconclusive |
| medoid_h10 vs fixed_5 | +0.6 | [-1.1, +2.5] | 114 | 105 | 0.589 | 1 | inconclusive |
| medoid_h10 vs fixed_8 | -0.2 | [-1.9, +1.5] | 90 | 93 | 0.883 | 1 | inconclusive |
| medoid_h10 vs fixed_10 | -1.3 | [-3.0, +0.5] | 79 | 98 | 0.176 | 1 | inconclusive |
| medoid_h10 vs duac | +0.5 | [-1.2, +2.2] | 98 | 91 | 0.663 | 1 | inconclusive |
| medoid_h10 vs aac | +1.2 | [-0.5, +3.0] | 104 | 86 | 0.217 | 1 | inconclusive |
| medoid_h10 vs random_h | -0.5 | [-2.2, +1.2] | 82 | 90 | 0.594 | 1 | inconclusive |
| medoid_h10 vs sample0_h8 | -0.3 | [-2.1, +1.5] | 90 | 95 | 0.769 | 1 | inconclusive |

**Track A criterion (manifest, section 5)**: an arm WORKS if on the noisy pool it beats every fixed arm by >= 3 pp with Holm p < 0.05, still beats them by >= 1 pp after dropping its most favourable task, and on clean is not more than 2 pp below the best fixed arm.

| arm | min gain vs Fixed-5/8/10 (noisy) | leave-best-task-out: min gain | clean vs best fixed | verdict |
|---|---|---|---|---|
| medoid_h8 | +0.7 (max Holm p 1) | libero_10_object/9: +0.4 | -0.8 vs fixed_10 | does not meet |
| medoid_h10 | -1.6 (max Holm p 1) | libero_10_object/4: -2.3 | -0.6 vs fixed_10 | does not meet |

## 3. Successes per task (out of 50)

| suite/task | fixed_5 | fixed_8 | fixed_10 | duac | aac | random_h | medoid_h8 | sample0_h8 | medoid_h10 |
|---|---|---|---|---|---|---|---|---|---|
| clean/0 | 47 | 48 | 47 | 47 | 47 | 48 | 49 | 48 | 48 |
| clean/1 | 50 | 50 | 50 | 50 | 50 | 49 | 50 | 49 | 50 |
| clean/2 | 49 | 48 | 49 | 47 | 47 | 48 | 47 | 48 | 50 |
| clean/3 | 50 | 50 | 50 | 49 | 48 | 49 | 49 | 48 | 49 |
| clean/4 | 50 | 49 | 49 | 49 | 49 | 50 | 49 | 49 | 48 |
| clean/5 | 50 | 50 | 50 | 48 | 50 | 49 | 50 | 50 | 49 |
| clean/6 | 46 | 49 | 46 | 46 | 45 | 49 | 47 | 48 | 45 |
| clean/7 | 50 | 50 | 49 | 49 | 50 | 50 | 49 | 49 | 49 |
| clean/8 | 33 | 33 | 39 | 39 | 35 | 35 | 37 | 39 | 40 |
| clean/9 | 44 | 46 | 48 | 47 | 49 | 48 | 46 | 46 | 46 |
| env/0 | 47 | 50 | 50 | 50 | 47 | 49 | 50 | 47 | 49 |
| env/1 | 49 | 47 | 49 | 49 | 49 | 48 | 49 | 48 | 49 |
| env/2 | 27 | 30 | 22 | 22 | 20 | 21 | 27 | 21 | 14 |
| env/3 | 20 | 12 | 13 | 14 | 14 | 16 | 14 | 13 | 12 |
| env/4 | 49 | 50 | 47 | 50 | 48 | 49 | 49 | 49 | 48 |
| env/5 | 31 | 41 | 40 | 39 | 36 | 37 | 39 | 44 | 39 |
| env/6 | 47 | 45 | 47 | 44 | 46 | 46 | 47 | 45 | 46 |
| env/7 | 50 | 49 | 50 | 49 | 50 | 50 | 50 | 50 | 48 |
| env/8 | 2 | 12 | 18 | 9 | 6 | 10 | 13 | 8 | 14 |
| env/9 | 25 | 32 | 28 | 20 | 24 | 25 | 25 | 34 | 31 |
| object/0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| object/1 | 48 | 50 | 50 | 50 | 50 | 50 | 50 | 49 | 50 |
| object/2 | 46 | 43 | 47 | 42 | 45 | 47 | 47 | 45 | 47 |
| object/3 | 42 | 42 | 45 | 43 | 45 | 47 | 45 | 42 | 43 |
| object/4 | 49 | 48 | 44 | 44 | 50 | 48 | 50 | 49 | 50 |
| object/5 | 45 | 36 | 39 | 44 | 38 | 43 | 41 | 43 | 40 |
| object/6 | 48 | 46 | 49 | 49 | 50 | 50 | 49 | 49 | 49 |
| object/7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| object/8 | 40 | 43 | 47 | 43 | 44 | 42 | 44 | 42 | 44 |
| object/9 | 20 | 17 | 20 | 23 | 13 | 18 | 23 | 15 | 16 |

## 4. Sub-goals and failure modes (all 1500 episodes)

A predicate counts as reached if it held at any step. 'all predicates, not together' = every goal predicate held at some point but never simultaneously (e.g. an object placed and then knocked off).

| arm | predicates reached | success | all predicates, not together | partial | none |
|---|---|---|---|---|---|
| fixed_5 | 84.8% | 1154 | 14 | 207 | 125 |
| fixed_8 | 85.4% | 1166 | 8 | 211 | 115 |
| fixed_10 | 85.9% | 1182 | 4 | 201 | 113 |
| duac | 85.4% | 1156 | 11 | 219 | 114 |
| aac | 84.8% | 1145 | 14 | 222 | 119 |
| random_h | 85.8% | 1171 | 14 | 200 | 115 |
| medoid_h8 | 85.8% | 1185 | 7 | 188 | 120 |
| sample0_h8 | 85.3% | 1168 | 11 | 199 | 122 |
| medoid_h10 | 85.1% | 1163 | 8 | 210 | 119 |

## 5. Speed and compute

Steps to success: paired with Fixed-10 on episodes both solve. infer ms = median server-side model time per call as logged during the run (DUAC server for Fixed/DUAC/Random-h; AAC N = 20 server for AAC and the selection arms), so it includes the cost of sampling 20 chunks.

| arm | steps to success (arm vs Fixed-10) | decisions / episode | infer ms / call | policy s / episode |
|---|---|---|---|---|
| fixed_5 | 258 vs 251 (1049 ep) | 64.3 | 129 | 8.30 |
| fixed_8 | 253 vs 252 (1073 ep) | 39.6 | 130 | 5.15 |
| fixed_10 | 257 vs 257 (1182 ep) | 31.6 | 131 | 4.14 |
| duac | 252 vs 252 (1060 ep) | 37.4 | 130 | 4.88 |
| aac | 252 vs 251 (1065 ep) | 43.1 | 165 | 7.11 |
| random_h | 254 vs 253 (1084 ep) | 37.0 | 131 | 4.85 |
| medoid_h8 | 253 vs 252 (1091 ep) | 39.3 | 87 | 3.43 |
| sample0_h8 | 252 vs 252 (1071 ep) | 39.6 | 87 | 3.45 |
| medoid_h10 | 254 vs 254 (1084 ep) | 31.9 | 88 | 2.80 |

## 6. Motion quality (executed actions, robosuite units)

seam = |a_t+1 - a_t| when a_t+1 starts a new chunk; inside = same within a chunk; seam / inside > 1 means jumps at chunk boundaries (mode switching). jerk = mean |second difference|. toggles = gripper open/close switches per episode. commanded path = sum of commanded translation (|dpos| x 0.05 m) over successful episodes.

| arm | episodes | seam jump | inside-chunk change | seam / inside | jerk | gripper toggles | commanded path (m, successes) |
|---|---|---|---|---|---|---|---|
| fixed_5 | 1500 | 0.127 | 0.054 | 2.37 | 0.065 | 7.26 | 6.78 |
| fixed_8 | 1500 | 0.146 | 0.053 | 2.75 | 0.053 | 6.19 | 6.75 |
| fixed_10 | 1500 | 0.160 | 0.053 | 3.00 | 0.049 | 6.20 | 6.79 |
| duac | 1500 | 0.152 | 0.053 | 2.88 | 0.052 | 5.87 | 6.79 |
| aac | 1500 | 0.146 | 0.053 | 2.76 | 0.055 | 5.86 | 6.73 |
| random_h | 1500 | 0.150 | 0.053 | 2.82 | 0.052 | 6.00 | 6.75 |
| medoid_h8 | 1500 | 0.135 | 0.053 | 2.55 | 0.050 | 6.28 | 6.73 |
| sample0_h8 | 1500 | 0.147 | 0.053 | 2.78 | 0.053 | 6.10 | 6.74 |
| medoid_h10 | 1500 | 0.152 | 0.053 | 2.88 | 0.048 | 6.25 | 6.77 |

## 7. What the selector did

| arm | decisions where the executed chunk != sample 0 |
|---|---|
| medoid_h8 | 95.0% |
| sample0_h8 | 0.0% |
| medoid_h10 | 95.0% |

