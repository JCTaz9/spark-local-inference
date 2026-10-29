#!/bin/bash
# PoC: DS4.1 c32 CSF package with the forced phased collective ladder, physical GPUs 0,4,8,12.
set -euo pipefail
cd /data/ssd2/ds41-c32-csf-test
ulimit -n 1048576
ulimit -l unlimited
unset LD_LIBRARY_PATH NCCL_GRAPH_FILE
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,4,8,12
exec /data/ssd2/ds41-c32-csf-test/flare --config /data/ssd2/ds41-c32-csf-test/run.toml \
  run /data/ssd2/claude-ds41/poc-package/package --offline \
  --publisher-keys /data/ssd2/claude-ds41/poc-package/development-publisher.json \
  --trust-store /data/ssd2/claude-ds41/poc-package/trust.toml \
  --dangerously-approve-key 4926c9ce4898bd4060b63ace4c621b27f96831ac5a093a7afbf57ffa342378da \
  --dangerously-accept-publisher local-development \
  --gpus 0,1,2,3 --max-concurrency 32 --mtp-draft-tokens 7 \
  --mtp-verification-token-budget 128 --embedding-table-storage ram \
  --prefix-cache-memory 64G --host 0.0.0.0 --port 8154
