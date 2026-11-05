## T1. Success (Wilson 95% CI), 500 episodes per cell

| arm | libero_10 | libero_10_object | libero_10_env | 2 noisy pooled |
|---|---|---|---|---|
| duac | 94.2 [91.8, 95.9] | 67.8 [63.6, 71.7] | 69.2 [65.0, 73.1] | 68.5 [65.6, 71.3] |
| aac | 94.0 [91.6, 95.8] | 67.0 [62.8, 71.0] | 68.0 [63.8, 71.9] | 67.5 [64.5, 70.3] |
| fixed_5 | 93.8 [91.3, 95.6] | 67.6 [63.4, 71.6] | 69.4 [65.2, 73.3] | 68.5 [65.6, 71.3] |
| fixed_8 | 94.6 [92.3, 96.3] | 65.0 [60.7, 69.1] | 73.6 [69.6, 77.3] | 69.3 [66.4, 72.1] |
| fixed_10 | 95.4 [93.2, 96.9] | 68.2 [64.0, 72.1] | 72.8 [68.7, 76.5] | 70.5 [67.6, 73.2] |
| random_h | 95.0 [92.7, 96.6] | 69.0 [64.8, 72.9] | 70.2 [66.0, 74.0] | 69.6 [66.7, 72.4] |

## T2. Paired difference on the 2 noisy suites pooled (pp; McNemar exact; bootstrap CI over task x trial)

| comparison | diff | 95% CI | only A / only B | p | p Holm |
|---|---|---|---|---|---|
| aac − fixed_5 | -1.0 | [-3.4, +1.5] | 80 / 90 | 0.490 | 1.000 |
| aac − fixed_8 | -1.8 | [-4.3, +0.6] | 69 / 87 | 0.173 | 0.759 |
| aac − fixed_10 | -3.0 | [-5.4, -0.7] | 63 / 93 | 0.020 | 0.120 |
| aac − random_h | -2.1 | [-4.4, +0.2] | 63 / 84 | 0.099 | — |
| duac − fixed_5 | +0.0 | [-2.5, +2.4] | 83 / 83 | 1.000 | 1.000 |
| duac − fixed_8 | -0.8 | [-3.3, +1.6] | 77 / 85 | 0.582 | 1.000 |
| duac − fixed_10 | -2.0 | [-4.7, +0.7] | 78 / 98 | 0.152 | 0.759 |
| duac − random_h | -1.1 | [-3.6, +1.5] | 73 / 84 | 0.425 | — |
| aac − duac | -1.0 | [-3.4, +1.4] | 80 / 90 | 0.490 | — |

### T2 per suite (pp, McNemar p)

| comparison | libero_10 | libero_10_object | libero_10_env |
|---|---|---|---|
| aac − fixed_5 | +0.2 (p=1.00) | -0.6 (p=0.79) | -1.4 (p=0.57) |
| aac − fixed_8 | -0.6 (p=0.75) | +2.0 (p=0.27) | -5.6 (p=0.00) |
| aac − fixed_10 | -1.4 (p=0.35) | -1.2 (p=0.53) | -4.8 (p=0.02) |
| aac − random_h | -1.0 (p=0.51) | -2.0 (p=0.25) | -2.2 (p=0.28) |
| duac − fixed_5 | +0.4 (p=0.88) | +0.2 (p=1.00) | -0.2 (p=1.00) |
| duac − fixed_8 | -0.4 (p=0.87) | +2.8 (p=0.12) | -4.4 (p=0.03) |
| duac − fixed_10 | -1.2 (p=0.44) | -0.4 (p=0.91) | -3.6 (p=0.09) |
| duac − random_h | -0.8 (p=0.64) | -1.2 (p=0.54) | -1.0 (p=0.68) |
| aac − duac | -0.2 (p=1.00) | -0.8 (p=0.73) | -1.2 (p=0.61) |

## T3. Per task (2 noisy suites pooled; pp) and difference without the task with the largest gain

| comparison | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | without best task |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aac − fixed_5 | +0 | +2 | -8 | -3 | +0 | -2 | +1 | +0 | +8 | -8 | -2.0 (bỏ t8) |
| aac − fixed_8 | -3 | +2 | -8 | +5 | +0 | -3 | +5 | +1 | -5 | -12 | -2.6 (bỏ t3) |
| aac − fixed_10 | -3 | +0 | -4 | +1 | +7 | -5 | +0 | +0 | -15 | -11 | -4.1 (bỏ t4) |
| aac − random_h | -2 | +1 | -3 | -4 | +1 | -6 | +0 | +0 | -2 | -6 | -2.4 (bỏ t1) |
| duac − fixed_5 | +4 | +2 | -9 | -5 | -4 | +7 | -2 | -1 | +10 | -2 | -1.1 (bỏ t8) |
| duac − fixed_8 | +1 | +2 | -9 | +3 | -4 | +6 | +2 | +0 | -3 | -6 | -1.6 (bỏ t5) |
| duac − fixed_10 | +1 | +0 | -5 | -1 | +3 | +4 | -3 | -1 | -13 | -5 | -2.7 (bỏ t5) |
| duac − random_h | +2 | +1 | -4 | -6 | -3 | +3 | -3 | -1 | +0 | +0 | -1.6 (bỏ t5) |
| aac − duac | -4 | +0 | +1 | +2 | +4 | -9 | +3 | +1 | -2 | -6 | -1.6 (bỏ t4) |

## Section 5 decision (2 noisy suites pooled)

- **AAC: KHÔNG WORK**. Vs best Fixed on noisy (fixed_10): -3.0 pp, CI upper -0.7 pp; beats all 3 Fixed by ≥3 pp after Holm: False; ≥1 pp without best task: False; clean vs best Fixed (fixed_10): -1.4 pp; vs Fixed-8 -1.8 pp, vs Random-h -2.1 pp.
- **DUAC: KHÔNG WORK**. Vs best Fixed on noisy (fixed_10): -2.0 pp, CI upper +0.7 pp; beats all 3 Fixed by ≥3 pp after Holm: False; ≥1 pp without best task: False; clean vs best Fixed (fixed_10): -1.2 pp; vs Fixed-8 -0.8 pp, vs Random-h -1.1 pp.
