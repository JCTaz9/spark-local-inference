"""Summarize bench_pkg.sh runs: tok/s per cell and GPU time per speculative round from Flare metrics.

Usage: bench_report.py BENCH_DIR LABEL [LABEL ...]
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

LINE = re.compile(r'^(flare_speculative_(?:rounds|gpu_nanoseconds|accepted_drafts|offered_drafts)_total)\{([^}]*)\} (\S+)')


def prom(path):
    values = defaultdict(float)
    for line in open(path):
        m = LINE.match(line)
        if m:
            labels = dict(kv.split("=") for kv in m.group(2).split(","))
            key = (int(labels["batch"].strip('"')), int(labels["width"].strip('"')))
            values[m.group(1).split("_")[2], key] += float(m.group(3))
    return values


root = Path(sys.argv[1])
for label in sys.argv[2:]:
    d = root / label
    print(f"## {label}")
    for c in (1, 8, 16, 32):
        tps, rounds, gpu, acc = [], defaultdict(float), defaultdict(float), defaultdict(float)
        for trial in (1, 2):
            cell = d / f"c{c}-{trial}"
            try:
                r = json.load(open(f"{cell}.json"))["results"][0]
                before, after = prom(f"{cell}-before.prom"), prom(f"{cell}-after.prom")
            except (OSError, KeyError, ValueError):
                continue
            tps.append(r["aggregate_tps"])
            sums = {"rounds": rounds, "gpu": gpu, "accepted": acc}
            for (name, key), value in after.items():
                if name in sums:
                    sums[name][key] += value - before.get((name, key), 0.0)
        if not tps:
            continue
        top = sorted(((v, k) for k, v in rounds.items() if v > 0), reverse=True)[:3]
        parts = [f"b{k[0]}w{k[1]}: {v:.0f} rounds {gpu[k] / v / 1e6:.3f} ms/round {acc[k] / v:.2f} acc/round"
                 for v, k in top]
        print(f"  c{c:<2} tok/s {' / '.join(f'{x:.1f}' for x in tps)} | " + " | ".join(parts))
