#!/usr/bin/env bash
# Start policy servers, run a job list with bounded concurrency (jobs spread over the servers), stop the servers.
#
#   bash tools/run_jobs.sh duac|aac JOBS_FILE LOG_DIR [N_SERVERS] [CONCURRENCY] [BASE_PORT]
#
# JOBS_FILE: one job per line, the arguments after the python interpreter; {PORT} is replaced by the port
# of the server assigned to that job. Lines starting with '#' are skipped. Example (duac):
#   flowbased_aac/duac.py eval-liberopro --suite libero_10_with_mug --trials 10 --port {PORT} --fixed-horizon 10 --output-dir outputs/calibration/libero_10_with_mug/fixed_10
# A job whose output directory already holds episodes is resumed (--resume-dir) instead of restarted.
set -uo pipefail
KIND=$1; JOBS=$2; LOGS=$3; NSERV=${4:-2}; CONC=${5:-6}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source adaptive_action_chunking/server141.env
mkdir -p "$LOGS"
case $KIND in
  duac) BASE=${6:-8000}; SERVE=(flowbased_aac/duac.py serve-pi05 --seed 0) ;;
  aac)  BASE=${6:-8015}; SERVE=(adaptive_action_chunking/aac.py serve-pi05 --n_samples 20 --seed 0 --no_seed_per_step) ;;
  *) echo "kind must be duac or aac"; exit 2 ;;
esac

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; }
trap cleanup EXIT
for ((i = 0; i < NSERV; i++)); do
  port=$((BASE + i))
  .venv-openpi/bin/python "${SERVE[@]}" --port "$port" > "$LOGS/server_$port.log" 2>&1 &
  PIDS+=($!)
done
for ((i = 0; i < NSERV; i++)); do
  port=$((BASE + i))
  until grep -q "server listening" "$LOGS/server_$port.log"; do
    kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "server $port died:"; tail -5 "$LOGS/server_$port.log"; exit 1; }
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
  out=$(sed -E 's/.*--(output-dir|out-dir|out_dir) +([^ ]+).*/\2/' <<< "$cmd")
  if [[ -s "$out/episodes.csv" && $(wc -l < "$out/episodes.csv") -gt 1 ]]; then
    cmd=$(sed -E "s#--(output-dir|out-dir|out_dir) +[^ ]+#--resume-dir $out#" <<< "$cmd")
  fi
  name=$(tr '/ ' '__' <<< "${out#outputs/}")
  ( echo "$(date '+%F %T') START port=$port $cmd"
    OMP_NUM_THREADS=${CLIENT_THREADS:-2} .venv-libero/bin/python $cmd > "$LOGS/$name.log" 2>&1; rc=$?
    echo "$(date '+%F %T') END rc=$rc $out" ) &
done < "$JOBS"
for p in $(jobs -rp); do [[ " ${PIDS[*]} " == *" $p "* ]] || wait "$p"; done
echo "$(date '+%F %T') all jobs finished"
