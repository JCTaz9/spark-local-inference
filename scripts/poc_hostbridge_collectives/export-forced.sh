#!/bin/bash
# PoC export: identical to /data/flare-ds41-c32-csf-20261029/export-no-joint.sh except FI (443d8fad + phased
# joint modes, compile key 211), Flare ladder mode names, and a forced one-profile-per-bucket ladder.
set -euo pipefail
cd /data/claude-poc-phased/flare
ulimit -n 1048576
ulimit -l unlimited
export PATH=/data/flare-reimplementation-20261028/toolchain-bin:/data/flare-reimplementation-20261028/upstream-venv/bin:/root/vllm/flare/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/cuda-13.2/bin:/usr/bin:/bin
export PYTHONPATH=/data/flare-reimplementation-20261028/upstream-native-python:/data/claude-poc-phased/flare:/data/claude-poc-phased/flashinfer
export PYTHONDONTWRITEBYTECODE=1
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export CUDA_VISIBLE_DEVICES=8,9,10,11
unset LD_LIBRARY_PATH NCCL_GRAPH_FILE
exec /data/flare-reimplementation-20261028/upstream-venv/bin/python -m ignition export deepseek-v4.1-flash \
  --config /data/claude-poc-phased/config/serving-config.toml \
  --gpus 0,1,2,3 --max-concurrency 32 --image-concurrency 1 \
  --mtp-draft-tokens 7 --speculative-token-capacity 128 --speculative-graph-budget 64 \
  --prefill-chunk-size 4096 --mla-compute-mode bf16 \
  --expert-scale-compression csf --autotune-effort default --compile-workers 64 \
  --collective-tuning runtime \
  --collective-ladder /data/claude-poc-phased/forced-ladder.json \
  --draft-head-quantization nvfp4 --markov-head-quantization fp8 \
  --vocabulary-parallel-sampling --verbose
