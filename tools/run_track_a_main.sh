#!/usr/bin/env bash
# Track A main run (docs/task3010.md step 3): phase 1 DUAC + Fixed-5/8/10 on 2 DUAC servers,
# phase 2 Random-h (1 DUAC server) and AAC N=20 (1 AAC server) in parallel. Re-running resumes.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
L=outputs/A/logs; mkdir -p $L
echo "$(date '+%F %T') phase 1: duac + fixed"
bash tools/run_jobs.sh duac jobs/main_duac_server.txt $L 2 ${CONC:-10} > $L/runner_phase1.txt 2>&1
echo "$(date '+%F %T') phase 2: random_h + aac"
bash tools/run_jobs.sh duac jobs/main_random_h.txt $L 1 3 8000 > $L/runner_phase2_random_h.txt 2>&1 &
bash tools/run_jobs.sh aac jobs/main_aac_server.txt $L 1 3 8015 > $L/runner_phase2_aac.txt 2>&1 &
wait
echo "$(date '+%F %T') track A main run finished"
