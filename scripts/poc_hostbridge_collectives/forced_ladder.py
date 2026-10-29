"""Build a forced collective ladder (one profile per bucket) from a write-only ladder report.

Usage: forced_ladder.py REPORT.jsonl OUT.json
Per measured size, configs within 3% of the best median are compared by mean over all samples; the
lowest mean wins. Write-only submodes have no stall tails, so mean and median agree closely.
"""
import json
import statistics
import sys
from collections import defaultdict

report, out = sys.argv[1], sys.argv[2]
samples = defaultdict(list)
for line in open(report):
    record = json.loads(line)
    if record.get("kind") != "measurement" or record.get("correctness") != "passed":
        continue
    config = record["config"]
    key = (config["implementation"], config["threads"], config["blocks"], config["submode"], config["active_blocks"])
    samples[record["bytes"], key].extend(record["samples_us"])

by_size = defaultdict(dict)
for (size, key), values in samples.items():
    if len(values) >= 16:
        by_size[size][key] = (statistics.median(values), statistics.fmean(values), max(values), len(values))

buckets = []
for size in sorted(by_size):
    stats = by_size[size]
    best_median = min(v[0] for v in stats.values())
    close = {k: v for k, v in stats.items() if v[0] <= best_median * 1.03}
    key = min(close, key=lambda k: close[k][1])
    median, mean, peak, count = stats[key]
    implementation, threads, blocks, submode, active = key
    buckets.append(dict(max_bytes=size, implementation=implementation, threads=threads, blocks=blocks,
                        profiles={submode: [active]}))
    print(f"{size:>8} {implementation:15s} t{threads:<3} {submode:20s} a{active:<2} "
          f"median {median:7.2f} mean {mean:7.2f} max {peak:7.1f} n {count}")

ladder = dict(version=1, name="frank2_hostbridge_tp4_forced", compute_capability=[12, 0], world_size=4,
              dtype="bfloat16", inplace=False, cache_policy="cold", min_bytes=16, buckets=buckets)
with open(out, "w") as handle:
    json.dump(ladder, handle, indent=1)
print(f"{len(buckets)} buckets -> {out}")
