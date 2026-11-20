#!/usr/bin/env bash
# VLABench full evaluation in one command: selftest -> sequential phases -> analysis -> results/.
#
#   cd /mnt/ssd-storage/hieudd/aac-duac
#   setsid nohup bash tools/run_vlabench_full.sh > outputs/vlabench/full/runner.txt 2>&1 &
#   tail -f outputs/vlabench/full/runner.txt                         # done when "VLABENCH FULL DONE" appears
#
# Phases run one after another: for each track, one server at a time (~9 GB VRAM):
#   vanilla server (8700): vanilla  ->  DUAC server (8710): duac, fixed_8, fixed_10 (as in LIBERO/RoboCasa)  ->  AAC server (8720): aac
# Each phase runs CONC clients, stops its server at the end, and is retried (resume) up to RETRIES times if a job failed
# (a server failure never records an episode; the rerun redoes it).
# Settings (environment variables, defaults in brackets):
#   N [50] episodes per task (T2 insert_flower has only 10)   TRACKS ["T1 T2 T3 T4 T6"]   CONC [8] clients per phase
#   ARMS ["vanilla fixed_8 fixed_10 duac aac"]   RETRIES [2]   OUT [outputs/vlabench/full]
#   Ablation without xi (h = h_unc), same episodes, added to the same report:
#   ARMS="duac_noxi aac_noxi" TAG=_noxi setsid nohup bash tools/run_vlabench_full.sh > outputs/vlabench/full/runner_noxi.txt 2>&1 &
#   Control with the same h distribution as duac_noxi, chosen at random (after the duac_noxi runs):
#   ARMS="random_h" TAG=_rh setsid nohup bash tools/run_vlabench_full.sh > outputs/vlabench/full/runner_random_h.txt 2>&1 &
#   PORT_OFFSET [0] (100 -> servers on 88xx) and TAG [""] for a second pipeline in parallel, e.g.
#   TRACKS="T6 T4" CONC=6 PORT_OFFSET=100 TAG=_b setsid nohup bash tools/run_vlabench_full.sh > outputs/vlabench/full/runner_b.txt 2>&1 &
#   (jobs are locked per output dir, so two pipelines never run the same job)
# Re-running the same command resumes: finished episodes are skipped, an interrupted episode is redone.
# Stop: kill $(pgrep -u $USER -f "[r]un_vlabench|[e]val_vlabench|vlabench/[s]erve")
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
N=${N:-50}; TRACKS=${TRACKS:-"T1 T2 T3 T4 T6"}; CONC=${CONC:-8}; ARMS=${ARMS:-"vanilla fixed_8 fixed_10 duac aac"}
RETRIES=${RETRIES:-2}; OUT=${OUT:-outputs/vlabench/full}; PORT_OFFSET=${PORT_OFFSET:-0}; TAG=${TAG:-}
TASKS="add_condiment insert_flower select_book select_chemistry_tube select_drink select_fruit select_mahjong select_painting select_poker select_toy"
mkdir -p "$OUT/logs" "$OUT/jobs"
echo "$(date '+%F %T') VLABench full: N=$N tracks=[$TRACKS] arms=[$ARMS] conc=$CONC out=$OUT"

# 1. checks (CPU, ~10 s)
.venv-vlabench/bin/python flowbased_aac/flowbased_aac/vlabench/selftest.py | grep -E "PASS|FAIL|ALL" || { echo "selftest failed"; exit 1; }

arm_args() {  # $1 arm, $2 track, $3 task
  case $1 in
    vanilla) echo "--port $((8700 + PORT_OFFSET)) --vanilla" ;;
    duac)    echo "--port $((8710 + PORT_OFFSET)) --state-bank --save-videos --video-episodes 2" ;;
    aac)     echo "--port $((8720 + PORT_OFFSET)) --aac-selector gaussian_bernoulli --state-bank --save-videos --video-episodes 2" ;;
    fixed_*) echo "--port $((8710 + PORT_OFFSET)) --fixed-horizon ${1#fixed_}" ;;
    duac_noxi) echo "--port $((8710 + PORT_OFFSET)) --no-xi --state-bank" ;;                      # ablation: h = h_unc
    aac_noxi)  echo "--port $((8720 + PORT_OFFSET)) --aac-selector gaussian_bernoulli --no-xi --state-bank" ;;
    # control: h ~ executed-h distribution of duac_noxi on the same track and task (needs those runs first)
    random_h)  echo "--port $((8710 + PORT_OFFSET)) --random-horizon-from $OUT/$2/${RANDOM_H_FROM:-duac_noxi}/$3/decisions.csv" ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}
server_of() { case $1 in vanilla) echo vanilla ;; aac|aac_*) echo aac ;; *) echo duac ;; esac; }

# 2. sequential phases: track x server
for tr in $TRACKS; do
  for srv in vanilla duac aac; do
    J="$OUT/jobs/${tr}_${srv}${TAG}.txt"
    { echo "# $tr, $srv server, generated $(date '+%F %T')"
      for t in $TASKS; do
        n=$N; [[ $tr == T2 && $t == insert_flower && $n -gt 10 ]] && n=10
        for a in $ARMS; do
          [[ $(server_of "$a") == "$srv" ]] && echo "--track $tr --tasks $t --episodes $n $(arm_args "$a" "$tr" "$t") --output-dir $OUT/$tr/$a/$t"
        done
      done; } > "$J"
    [[ $(grep -vc '^#' "$J") -eq 0 ]] && continue
    for attempt in $(seq 0 "$RETRIES"); do
      echo "$(date '+%F %T') PHASE $tr/$srv attempt $attempt: $(grep -vc '^#' "$J") jobs"
      bash tools/run_vlabench_jobs.sh "$J" "$OUT/logs" "$CONC" > "$OUT/logs/phase_${tr}_${srv}${TAG}_$attempt.txt" 2>&1
      failed=$(grep -c "END rc=[1-9]" "$OUT/logs/phase_${tr}_${srv}${TAG}_$attempt.txt")
      echo "$(date '+%F %T') PHASE $tr/$srv attempt $attempt done: $(grep -c 'END rc=0' "$OUT/logs/phase_${tr}_${srv}${TAG}_$attempt.txt") ok, $failed failed"
      grep -q "died" "$OUT/logs/phase_${tr}_${srv}${TAG}_$attempt.txt" && { echo "server died:"; tail -3 "$OUT/logs/phase_${tr}_${srv}${TAG}_$attempt.txt"; }
      [[ $failed -eq 0 ]] && ! grep -q "died" "$OUT/logs/phase_${tr}_${srv}${TAG}_$attempt.txt" && break
      sleep 20
    done
  done
  # per-track analysis as soon as the track is done (partial report of all finished tracks)
  .venv-vlabench/bin/python analysis/vlabench_report.py "$OUT" --tracks $TRACKS > "$OUT/logs/report.log" 2>&1 \
    && echo "$(date '+%F %T') report updated -> $OUT/REPORT.md"
done

# 3. final analysis
.venv-vlabench/bin/python analysis/vlabench_report.py "$OUT" --tracks $TRACKS > "$OUT/logs/report.log" 2>&1 \
  && echo "$(date '+%F %T') report -> $OUT/REPORT.md" || { echo "report failed:"; tail -20 "$OUT/logs/report.log"; }

# 4. lightweight copy for sharing (CSV, json, report, figures; no state bank / videos / algo logs)
DEST=results/vlabench/full
mkdir -p "$DEST"
rsync -a --prune-empty-dirs --exclude="logs/***" --exclude="state_bank/***" --exclude="videos/***" --include="*/" \
  --include="episodes.csv" --include="decisions.csv" --include="summary.json" --include="args.json" --include="run_info.json" \
  --include="REPORT.md" --include="report.json" --include="*.png" --include="*.pdf" --include="jobs/*.txt" --exclude="*" "$OUT/" "$DEST/"
echo "$(date '+%F %T') results -> $DEST"
echo "$(date '+%F %T') VLABENCH FULL DONE"
