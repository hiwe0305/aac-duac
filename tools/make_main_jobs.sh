#!/usr/bin/env bash
# Write the Track A main-run job lists (docs/task3010.md step 3) for clean libero_10 + two noisy suites.
#   bash tools/make_main_jobs.sh libero_10_with_mug libero_10_with_red_stick
# Produces jobs/main_duac_server.txt (DUAC, Fixed-5/8/10, then Random-h) and jobs/main_aac_server.txt (AAC).
# Run only after manifest.json is locked and approved:
#   bash tools/run_jobs.sh duac jobs/main_duac_server.txt outputs/A/logs 2 8
#   bash tools/run_jobs.sh aac  jobs/main_aac_server.txt  outputs/A/logs_aac 2 6
# Random-h needs the DUAC decisions of the same suite, so jobs/main_random_h.txt is run after the DUAC jobs.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 NOISY_SUITE_1 NOISY_SUITE_2"; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p jobs
TRIALS=50
D=jobs/main_duac_server.txt; A=jobs/main_aac_server.txt; R=jobs/main_random_h.txt
: > "$D"; : > "$A"; : > "$R"
for suite in libero_10 "$1" "$2"; do
  out=outputs/A/$suite
  echo "flowbased_aac/duac.py eval-liberopro --suite $suite --trials $TRIALS --port {PORT} --state-bank --save-videos --output-dir $out/duac" >> "$D"
  for k in 5 8 10; do
    echo "flowbased_aac/duac.py eval-liberopro --suite $suite --trials $TRIALS --port {PORT} --fixed-horizon $k --state-bank --save-videos --output-dir $out/fixed_$k" >> "$D"
  done
  echo "flowbased_aac/duac.py eval-liberopro --suite $suite --trials $TRIALS --port {PORT} --random-horizon-from $out/duac/decisions.csv --state-bank --save-videos --output-dir $out/random_h" >> "$R"
  echo "adaptive_action_chunking/aac.py eval-liberopro --backbone pi05 --suite $suite --selector gaussian_bernoulli --trials $TRIALS --seed 7 --move_th 3 --port {PORT} --state-bank --videos --out_dir $out/aac" >> "$A"
done
wc -l "$D" "$A" "$R"
