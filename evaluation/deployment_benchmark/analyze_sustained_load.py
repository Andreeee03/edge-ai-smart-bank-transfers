import csv
import json
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs.csv"
OUT = BASE / "sustained_load_analysis.json"

def num(value):
    return float(str(value).strip().replace(",", "."))

def mean(rows, field):
    return statistics.mean(num(r[field]) for r in rows)

def median(rows, field):
    return statistics.median(num(r[field]) for r in rows)

def pct_change(first, last):
    if first == 0:
        return None
    return ((last - first) / first) * 100.0

def slope(values):
    # Simple least-squares slope: metric change per run.
    n = len(values)
    xs = list(range(1, n + 1))
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(values)

    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(xs, values)
    )
    denominator = sum(
        (x - x_mean) ** 2
        for x in xs
    )

    return numerator / denominator if denominator else 0.0

with RUNS.open(
    newline="",
    encoding="utf-8-sig"
) as f:
    rows = list(csv.DictReader(f))

if len(rows) != 102:
    raise RuntimeError(
        f"Expected 102 runs, found {len(rows)}."
    )

# 3 full cycles = 18 runs.
first = rows[:18]
last = rows[-18:]

metrics = [
    "ttft_ms",
    "total_ms",
    "prompt_decode_ms",
    "generation_ms",
    "throughput_tok_s",
]

comparison = {}

for metric in metrics:
    first_mean = mean(first, metric)
    last_mean = mean(last, metric)

    first_median = median(first, metric)
    last_median = median(last, metric)

    comparison[metric] = {
        "first_3_cycles_mean": round(first_mean, 3),
        "last_3_cycles_mean": round(last_mean, 3),
        "mean_change_percent": round(
            pct_change(first_mean, last_mean), 3
        ),
        "first_3_cycles_median": round(first_median, 3),
        "last_3_cycles_median": round(last_median, 3),
        "median_change_percent": round(
            pct_change(first_median, last_median), 3
        ),
        "slope_per_run": round(
            slope([num(r[metric]) for r in rows]),
            5
        ),
    }

# Cycle-level means.
cycles = []

for cycle in range(1, 18):
    subset = [
        r for r in rows
        if int(float(r["cycle"])) == cycle
    ]

    cycles.append({
        "cycle": cycle,
        "total_ms_mean": round(
            mean(subset, "total_ms"), 3
        ),
        "ttft_ms_mean": round(
            mean(subset, "ttft_ms"), 3
        ),
        "throughput_tok_s_mean": round(
            mean(subset, "throughput_tok_s"), 3
        ),
    })

# Compare same prompt at beginning vs end.
per_prompt = {}

for prompt_id in sorted(
    set(r["prompt_id"] for r in rows)
):
    subset = [
        r for r in rows
        if r["prompt_id"] == prompt_id
    ]

    early = subset[:3]
    late = subset[-3:]

    early_total = mean(early, "total_ms")
    late_total = mean(late, "total_ms")

    early_ttft = mean(early, "ttft_ms")
    late_ttft = mean(late, "ttft_ms")

    early_tp = mean(early, "throughput_tok_s")
    late_tp = mean(late, "throughput_tok_s")

    per_prompt[prompt_id] = {
        "task": subset[0]["task"],
        "total_ms_change_percent":
            round(pct_change(early_total, late_total), 3),
        "ttft_ms_change_percent":
            round(pct_change(early_ttft, late_ttft), 3),
        "throughput_change_percent":
            round(pct_change(early_tp, late_tp), 3),
    }

result = {
    "runs": len(rows),
    "comparison": comparison,
    "cycles": cycles,
    "per_prompt": per_prompt,
}

OUT.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8"
)

print()
print("=" * 64)
print("SUSTAINED-LOAD ANALYSIS")
print("=" * 64)

for metric in metrics:
    c = comparison[metric]

    print()
    print(metric)
    print(
        f"  first 3 cycles mean: {c['first_3_cycles_mean']}"
    )
    print(
        f"  last 3 cycles mean:  {c['last_3_cycles_mean']}"
    )
    print(
        f"  mean change:          "
        f"{c['mean_change_percent']:+.2f}%"
    )
    print(
        f"  first 3 median:       "
        f"{c['first_3_cycles_median']}"
    )
    print(
        f"  last 3 median:        "
        f"{c['last_3_cycles_median']}"
    )
    print(
        f"  median change:        "
        f"{c['median_change_percent']:+.2f}%"
    )
    print(
        f"  slope per run:        "
        f"{c['slope_per_run']}"
    )

print()
print("CYCLE MEANS")
print("-" * 64)

for c in cycles:
    print(
        f"Cycle {c['cycle']:02d} | "
        f"total={c['total_ms_mean']:8.3f} ms | "
        f"TTFT={c['ttft_ms_mean']:8.3f} ms | "
        f"throughput={c['throughput_tok_s_mean']:6.3f} tok/s"
    )

print()
print("PER-PROMPT EARLY -> LATE CHANGE")
print("-" * 64)

for prompt_id, values in per_prompt.items():
    print(
        f"{prompt_id} {values['task']}: "
        f"total={values['total_ms_change_percent']:+.2f}% | "
        f"TTFT={values['ttft_ms_change_percent']:+.2f}% | "
        f"throughput={values['throughput_change_percent']:+.2f}%"
    )

print()
print("Saved:")
print(OUT)
print()
