#!/usr/bin/env bash
# RoboCasa365 job runner: start π0.5 RoboCasa server(s), run client jobs with bounded concurrency, stop servers.
#
#   bash tools/run_robocasa_jobs.sh duac|aac JOBS_FILE LOG_DIR [N_SERVERS] [CONCURRENCY] [BASE_PORT]
#
# JOBS_FILE: one line per job = arguments of flowbased_aac/flowbased_aac/robocasa/eval_robocasa.py, with {PORT}.
# A job whose --output-dir already holds episodes is resumed (--resume-dir).
# Clients render with Mesa software EGL (llvmpipe): a GPU EGL context costs ~3.8 GB VRAM per RoboCasa client.
set -uo pipefail
KIND=$1; JOBS=$2; LOGS=$3; NSERV=${4:-1}; CONC=${5:-4}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=.venv-robocasa/bin/python
SERVER=flowbased_aac/flowbased_aac/robocasa/serve.py
CLIENT=flowbased_aac/flowbased_aac/robocasa/eval_robocasa.py
case $KIND in
  duac) BASE=${6:-8100}; SERVE_ARGS=(--mode duac --tau 0.9 --num-probes 40 --seed 0) ;;
  aac)  BASE=${6:-8120}; SERVE_ARGS=(--mode aac --n-samples 20 --seed 0) ;;
  *) echo "kind must be duac or aac"; exit 2 ;;
esac
mkdir -p "$LOGS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; }
trap cleanup EXIT
for ((i = 0; i < NSERV; i++)); do
  port=$((BASE + i))
  $PY $SERVER "${SERVE_ARGS[@]}" --port "$port" > "$LOGS/server_${KIND}_$port.log" 2>&1 &
  PIDS+=($!)
done
for ((i = 0; i < NSERV; i++)); do
  port=$((BASE + i))
  until grep -q "server listening" "$LOGS/server_${KIND}_$port.log"; do
    kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "server $port died:"; tail -5 "$LOGS/server_${KIND}_$port.log"; exit 1; }
    sleep 3
  done
  echo "$(date '+%F %T') server $port ready"
done
n=0
while read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  while (( $(jobs -rp | wc -l) >= CONC + NSERV )); do wait -n; done
  port=$((BASE + n % NSERV)); n=$((n + 1))
  cmd=${line//\{PORT\}/$port}
  out=$(sed -E 's/.*--output-dir +([^ ]+).*/\1/' <<< "$cmd")
  if [[ -s "$out/episodes.csv" && $(wc -l < "$out/episodes.csv") -gt 1 ]]; then
    cmd=$(sed -E "s#--output-dir +[^ ]+#--resume-dir $out#" <<< "$cmd")
  fi
  name=$(tr '/ ' '__' <<< "${out#outputs/}")
  ( echo "$(date '+%F %T') START port=$port $cmd"
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=1 LP_NUM_THREADS=${LP_THREADS:-2} OMP_NUM_THREADS=2 \
      __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
      $PY $CLIENT $cmd > "$LOGS/$name.log" 2>&1; rc=$?
    echo "$(date '+%F %T') END rc=$rc $out" ) &
done < "$JOBS"
for p in $(jobs -rp); do [[ " ${PIDS[*]} " == *" $p "* ]] || wait "$p"; done
echo "$(date '+%F %T') all jobs finished"
