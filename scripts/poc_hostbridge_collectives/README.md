# Host-bridge TP4 collectives PoC (not for merge)

Proof of concept for DeepSeek V4.1 TP4 on GPUs that sit behind different CPU IO-die quadrants
(frank2 GPUs 0,4,8,12). Collectives are forced to write-only, one-destination-at-a-time joint modes
through `ignition/preparation/ladders/frank2_hostbridge_tp4_forced.json`. It needs the FlashInfer branch
`claude/poc-hostbridge-phased` (local-inference-lab/flashinfer): FI 443d8fad plus `full_phased`,
`payload_rsag_phased` and `bulk_push_phased`. The phased kernels are Claude-written, so this branch
cannot be merged under design-review principle 14; it exists to show what the fabric allows.

Paths in the scripts are the frank1/frank2 paths used for the measurement.

## Steps

1. Measure write-only profiles on the target GPUs (FI branch, `pull`/`bulk_pull` excluded in the tool):
   ```sh
   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,4,8,12 \
     python -m torch.distributed.run --standalone --nproc-per-node=4 -m b12x.tools.collective_ladder \
     --fresh --output ladder-hb-writeonly.jsonl --min-bytes 16 --budget-seconds 1200 --compile-workers 48
   ```
2. Build the forced ladder (one profile per bucket; lowest mean within 3% of the best median):
   `python forced_ladder.py ladder-hb-writeonly.jsonl frank2_hostbridge_tp4_forced.json`
3. Export with `export-forced.sh` (the same command as the reference c32 CSF package, plus the ladder).
4. Serve with `serve-poc-g0-4-8-12.sh`; compare packages with `ab_chain.sh` / `bench_pkg.sh`
   (c1/c8/c16/c32, two 30 s cells each) and `bench_report.py` (tok/s and GPU time per speculative round
   from Flare metrics).

## Result (frank2, 2026-10-29)

| | reference package on 0,4,8,12 | PoC on 0,4,8,12 | reference package on switches 12-15 |
|---|---|---|---|
| b1w8 GPU time per round | 13.9-14.1 ms | 8.2-8.3 ms | 7.6 ms |
| c16 tok/s | 1001-1102 | 1466-1474 | 1524-1559 |
| c32 tok/s | 1007-1764 | 2155-2204 | 2309-2312 |
