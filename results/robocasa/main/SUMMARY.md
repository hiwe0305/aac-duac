# RoboCasa summary: `outputs/robocasa/main`

## Success (k/n)

| task | fixed_10 | fixed_25 | fixed_50 | duac | aac |
|---|---|---|---|---|---|
| KettleBoiling | 4/20 | 3/20 | 3/20 | 0/20 | 2/20 |
| LoadDishwasher | 1/20 | 4/20 | 3/20 | 2/20 | 1/20 |
| PreSoakPan | 2/20 | 5/20 | 4/20 | 7/20 | 8/20 |
| ScrubCuttingBoard | 0/20 | 6/20 | 7/20 | 2/20 | 0/20 |
| StackBowlsCabinet | 4/20 | 5/20 | 8/20 | 9/20 | 4/20 |
| **total** | **11/100** (11%, CI 6%–19%) | **23/100** (23%, CI 16%–32%) | **25/100** (25%, CI 18%–34%) | **20/100** (20%, CI 13%–29%) | **15/100** (15%, CI 9%–23%) |

## Paired (task, trial): adaptive wins / fixed wins

- duac vs fixed_10: 18 / 9 (n=100)
- duac vs fixed_25: 14 / 17 (n=100)
- duac vs fixed_50: 10 / 15 (n=100)
- aac vs fixed_10: 12 / 8 (n=100)
- aac vs fixed_25: 11 / 19 (n=100)
- aac vs fixed_50: 10 / 20 (n=100)

## Horizon

| arm | decisions | mean h | median h | mean h_unc | mean ξ | ξ > h_unc | infer ms |
|---|---|---|---|---|---|---|---|
| fixed_10 | 17665 | 10.0 | 10 | – | – | – | 229 |
| fixed_25 | 6744 | 25.0 | 25 | – | – | – | 228 |
| fixed_50 | 3317 | 50.0 | 50 | – | – | – | 229 |
| duac | 5638 | 30.4 | 27 | 23.5 | 18.5 | 49% | 229 |
| aac | 8396 | 20.6 | 16 | 10.8 | 16.8 | 72% | 290 |
