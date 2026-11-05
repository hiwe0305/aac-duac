# RoboCasa summary: `outputs/robocasa/smoke`

## Success (k/n)

| task | fixed_10 | fixed_25 | fixed_50 | duac | aac |
|---|---|---|---|---|---|
| DeliverStraw | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| GetToastedBread | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| KettleBoiling | 1/2 | 0/2 | 1/2 | 0/2 | 0/2 |
| LoadDishwasher | 2/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| PackIdenticalLunches | 0/2 | 0/2 | 0/2 | 1/2 | 0/2 |
| PreSoakPan | 1/2 | 1/2 | 0/2 | 1/2 | 1/2 |
| PrepareCoffee | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| RinseSinkBasin | 0/2 | 0/2 | 1/2 | 0/2 | 0/2 |
| ScrubCuttingBoard | 0/2 | 0/2 | 2/2 | 1/2 | 0/2 |
| SearingMeat | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| SetUpCuttingStation | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| StackBowlsCabinet | 0/2 | 1/2 | 0/2 | 1/2 | 0/2 |
| SteamInMicrowave | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| StirVegetables | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| StoreLeftoversInBowl | 0/2 | 1/2 | 0/2 | 0/2 | 0/2 |
| WashLettuce | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 |
| **total** | **4/32** (12%, CI 5%–28%) | **3/32** (9%, CI 3%–24%) | **4/32** (12%, CI 5%–28%) | **4/32** (12%, CI 5%–28%) | **2/32** (6%, CI 2%–20%) |

## Paired (task, trial): adaptive wins / fixed wins

- duac vs fixed_10: 3 / 3 (n=32)
- duac vs fixed_25: 3 / 2 (n=32)
- duac vs fixed_50: 3 / 3 (n=32)
- aac vs fixed_10: 1 / 3 (n=32)
- aac vs fixed_25: 2 / 3 (n=32)
- aac vs fixed_50: 2 / 4 (n=32)

## Horizon

| arm | decisions | mean h | median h | mean h_unc | mean ξ | ξ > h_unc | infer ms |
|---|---|---|---|---|---|---|---|
| fixed_10 | 7235 | 10.0 | 10 | – | – | – | 240 |
| fixed_25 | 2857 | 25.0 | 25 | – | – | – | 240 |
| fixed_50 | 1448 | 50.0 | 50 | – | – | – | 240 |
| duac | 2396 | 30.2 | 28 | 20.8 | 21.4 | 58% | 241 |
| aac | 3318 | 22.3 | 17 | 10.8 | 18.9 | 73% | 296 |
