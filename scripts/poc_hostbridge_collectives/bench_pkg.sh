#!/usr/bin/env bash
# Serve one DS4.1 c32 CSF package on physical GPUs $GPUS (default 0,4,8,12) and run c1/c8/c16/c32 decode cells.
# Usage: bench_pkg.sh LABEL PACKAGE PUBLISHER_JSON TRUST_TOML APPROVE_FINGERPRINT
set -euo pipefail
label=$1 package=$2 publisher=$3 trust=$4 fingerprint=$5
root=/data/ssd2/ds41-c32-csf-test
output=/data/ssd2/claude-ds41/work/poc-phased/bench/$label
[[ ! -e "$output" ]]
gpus=${GPUS:-0,4,8,12}
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $gpus) ]]
[[ -z $(ss -H -ltn 'sport = :8154') ]]
mkdir -p "$output"
cd "$root"
ulimit -n 1048576
ulimit -l unlimited
unset LD_LIBRARY_PATH NCCL_GRAPH_FILE
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpus
./flare --config run.toml run "$package" \
  --offline --publisher-keys "$publisher" --trust-store "$trust" \
  --dangerously-approve-key "$fingerprint" --dangerously-accept-publisher local-development \
  --gpus 0,1,2,3 --max-concurrency 32 --mtp-draft-tokens 7 \
  --mtp-verification-token-budget 128 --embedding-table-storage ram \
  --prefix-cache-memory 64G --host 0.0.0.0 --port 8154 > "$output/flare.log" 2>&1 < /dev/null &
server_pid=$!
echo "$server_pid" > "$output/server.pid"
cleanup() {
  if kill -0 "$server_pid" 2>/dev/null; then kill -TERM "$server_pid"; fi
  for _ in $(seq 120); do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT
for _ in $(seq 600); do
  kill -0 "$server_pid" 2>/dev/null || { tail -20 "$output/flare.log"; exit 1; }
  curl --fail --silent http://127.0.0.1:8154/health > /dev/null && break
  sleep 2
done
curl --fail --silent http://127.0.0.1:8154/health > /dev/null
# Warm the prefill4096 graph and the decode path once before measuring.
./bench-venv/bin/python llm_decode_bench.py --port 8154 --concurrency 1 --contexts 0 --skip-prefill \
  --duration 10 --max-tokens 1024 --temperature 1 --top-p 1 --display-mode plain --no-resume --no-hw-monitor \
  --output "$output/warmup.json" > "$output/warmup.log" 2>&1 || true
for c in 1 8 16 32; do
  for trial in 1 2; do
    cell=c$c-$trial
    curl --fail --silent http://127.0.0.1:8154/metrics > "$output/$cell-before.prom"
    ./bench-venv/bin/python llm_decode_bench.py --port 8154 --concurrency $c --contexts 0 --skip-prefill \
      --duration 30 --max-tokens 8192 --temperature 1 --top-p 1 --display-mode plain --no-resume --no-hw-monitor \
      --output "$output/$cell.json" > "$output/$cell.log" 2>&1
    curl --fail --silent http://127.0.0.1:8154/metrics > "$output/$cell-after.prom"
    ./bench-venv/bin/python -c 'import json,sys; r=json.load(open(sys.argv[1]))["results"][0]; print(sys.argv[2], json.dumps({k:r.get(k) for k in ("aggregate_tps","num_errors","underfilled","request_count")}))' "$output/$cell.json" "$cell" | tee -a "$output/summary.txt"
  done
done
grep -E "startup|collective" "$output/flare.log" | tail -40 > "$output/startup.txt" || true
