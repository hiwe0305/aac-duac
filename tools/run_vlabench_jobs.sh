#!/usr/bin/env bash
# VLABench job runner: ensure the π0.5 VLABench server(s) are up, run client jobs with bounded concurrency.
#
#   bash tools/run_vlabench_jobs.sh JOBS_FILE LOG_DIR [CONCURRENCY]
#
# JOBS_FILE: one line per job = arguments of flowbased_aac/flowbased_aac/vlabench/eval_vlabench.py. The line's
# --port selects the server: 8700 vanilla, 8710 and 8711 duac (8711 serves the fixed-k arms), 8720 aac (started here if nothing listens on the port;
# servers started by this script are stopped at the end). A job whose --output-dir already has episodes is resumed.
# Ports 8100-8231 are used by other services on this machine. Port mod 100 gives the mode (00 vanilla, 10/11 duac,
# 20 aac), so a second pipeline can use 88xx. Each job holds a lock on <output-dir>/.lock: a job already running in
# another pipeline is skipped.
set -uo pipefail
JOBS=$1; LOGS=$2; CONC=${3:-8}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source adaptive_action_chunking/server141.env
export OPENPI_DATA_HOME="$ROOT/checkpoints/openpi"
SERVER=flowbased_aac/flowbased_aac/vlabench/serve.py
CLIENT=flowbased_aac/flowbased_aac/vlabench/eval_vlabench.py
mkdir -p "$LOGS"
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; }
trap cleanup EXIT
mode_of() { case $(( $1 % 100 )) in 0) echo vanilla ;; 10|11) echo duac ;; 20) echo aac ;; *) echo "" ;; esac; }  # 87xx, 88xx, ...
for port in $(grep -v '^#' "$JOBS" | grep -oE -- '--port +[0-9]+' | awk '{print $2}' | sort -u); do
  if ss -ltn | grep -q ":$port "; then echo "$(date '+%F %T') server $port already listening"; continue; fi
  mode=$(mode_of "$port"); [[ -z $mode ]] && { echo "unknown port $port"; exit 2; }
  .venv-openpi/bin/python $SERVER --mode "$mode" --port "$port" > "$LOGS/server_${mode}_$port.log" 2>&1 &
  PIDS+=($!)
  until grep -q "server listening" "$LOGS/server_${mode}_$port.log"; do
    kill -0 "${PIDS[-1]}" 2>/dev/null || { echo "server $port died:"; tail -5 "$LOGS/server_${mode}_$port.log"; exit 1; }
    sleep 3
  done
  echo "$(date '+%F %T') server $port ($mode) ready"
done
while read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  while (( $(jobs -rp | wc -l) >= CONC + ${#PIDS[@]} )); do wait -n; done
  out=$(sed -E 's/.*--output-dir +([^ ]+).*/\1/' <<< "$line")
  cmd=$line
  if [[ -s "$out/episodes.csv" && $(wc -l < "$out/episodes.csv") -gt 1 ]]; then
    cmd=$(sed -E "s#--output-dir +[^ ]+#--resume-dir $out#" <<< "$cmd")
  fi
  name=$(tr '/ ' '__' <<< "${out#outputs/}")
  mkdir -p "$out"
  ( exec 9>"$out/.lock"
    flock -n 9 || { echo "$(date '+%F %T') SKIP (running in another pipeline) $out"; echo "$(date '+%F %T') END rc=0 $out"; exit 0; }
    echo "$(date '+%F %T') START $cmd"
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0} OMP_NUM_THREADS=2 \
      .venv-vlabench/bin/python $CLIENT $cmd >> "$LOGS/$name.log" 2>&1; rc=$?
    echo "$(date '+%F %T') END rc=$rc $out" ) &
  sleep "${STAGGER:-3}"  # concurrent EGL context creation can fail; spread client start-up
done < "$JOBS"
for p in $(jobs -rp); do [[ " ${PIDS[*]} " == *" $p "* ]] || wait "$p"; done
echo "$(date '+%F %T') all jobs finished"
