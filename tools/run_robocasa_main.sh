#!/usr/bin/env bash
# RoboCasa365 main run: 5 composite tasks x 20 trials x {duac, fixed_10, fixed_25, fixed_50, aac} = 500 episodes.
# Run detached so it survives the shell/Claude session:
#   setsid nohup bash tools/run_robocasa_main.sh > /dev/null 2>&1 &
# Progress: tail outputs/robocasa/main/logs/runner_*.txt   (done when "ALL MAIN DONE" appears)
# Re-running the same command resumes: jobs whose output dir already has episodes continue with --resume-dir.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CONC=${CONC:-10}
mkdir -p outputs/robocasa/main/logs
bash tools/run_robocasa_jobs.sh duac jobs/robocasa_main_duac.txt outputs/robocasa/main/logs 1 "$CONC" > outputs/robocasa/main/logs/runner_duac.txt 2>&1
bash tools/run_robocasa_jobs.sh aac  jobs/robocasa_main_aac.txt  outputs/robocasa/main/logs 1 "$CONC" > outputs/robocasa/main/logs/runner_aac.txt 2>&1
.venv-robocasa/bin/python analysis/robocasa_summary.py outputs/robocasa/main > outputs/robocasa/main/SUMMARY.md 2>&1
echo "ALL MAIN DONE $(date '+%F %T')" >> outputs/robocasa/main/logs/runner_aac.txt
