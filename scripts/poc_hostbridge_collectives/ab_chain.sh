#!/bin/bash
# Alternate PoC (forced phased ladder) and original package on 0,4,8,12: poc-1, base-2, poc-2.
set -u
cd /data/ssd2/claude-ds41/work/poc-phased
POC="/data/ssd2/claude-ds41/poc-package/package /data/ssd2/claude-ds41/poc-package/development-publisher.json /data/ssd2/claude-ds41/poc-package/trust.toml 4926c9ce4898bd4060b63ace4c621b27f96831ac5a093a7afbf57ffa342378da"
BASE="/data/ssd2/ds41-c32-csf-test/package /data/ssd2/ds41-c32-csf-test/development-publisher.json /data/ssd2/ds41-c32-csf-test/trust.toml e20d4f6535a011585da2b3856d3afa4702f2c85492608243e5c5d87996ce0770"
for run in "poc-1 $POC" "base-2 $BASE" "poc-2 $POC"; do
  set -- $run
  echo "=== $(date -Is) $1" >> ab_chain.log
  ./bench_pkg.sh "$@" >> ab_chain.log 2>&1 || echo "=== $1 failed" >> ab_chain.log
  sleep 5
done
echo "=== $(date -Is) done" >> ab_chain.log
