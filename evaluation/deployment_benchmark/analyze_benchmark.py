import csv
import json
import math
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs.csv"
THERMAL = BASE / "thermal_battery.csv"
INITIALIZATION = BASE / "initialization.json"
RAW = BASE / "raw"

def to_float(value):
    if isinstance(value, (int, float)):
        return float(value)

    return float(str(value).strip().replace(",", "."))


def percentile(values, p):
    values = sorted(values)

    if not values:
        return None

    if len(values) == 1:
        return values[0]

    k = (len(values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return values[int(k)]

    return values[f] * (c - k) + values[c] * (k - f)

def stats(values):
    values = [to_float(v) for v in values]

    return {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(percentile(values, 0.95), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "std_dev": round(statistics.stdev(values), 3)
            if len(values) > 1 else 0.0,
    }

def read_proc_memory(path):
    result = {}

    if not path.exists():
        return result

    for line in path.read_text(
        encoding="utf-8",
        errors="ignore"
    ).splitlines():

        if line.startswith("VmRSS:") or line.startswith("VmHWM:"):
            key, rest = line.split(":", 1)

            value_kb = int(
                rest.strip().split()[0]
            )

            result[key] = {
                "kib": value_kb,
                "mib": round(value_kb / 1024, 3),
                "gib": round(value_kb / 1024 / 1024, 3),
            }

    return result

# ---------------------------------------------------------
# Run metrics
# ---------------------------------------------------------

with RUNS.open(
    newline="",
    encoding="utf-8-sig"
) as f:

    runs = list(csv.DictReader(f))

metrics = [
    "ttft_ms",
    "prompt_decode_ms",
    "generation_ms",
    "total_ms",
    "throughput_tok_s",
    "prompt_tokens",
    "generated_tokens",
]

metric_summary = {}

for metric in metrics:
    metric_summary[metric] = stats(
        [to_float(row[metric]) for row in runs]
    )

# ---------------------------------------------------------
# Per-prompt summary
# ---------------------------------------------------------

prompt_summary = {}

for prompt_id in sorted(
    set(row["prompt_id"] for row in runs)
):

    subset = [
        row for row in runs
        if row["prompt_id"] == prompt_id
    ]

    prompt_summary[prompt_id] = {
        "task": subset[0]["task"],
        "runs": len(subset),
        "ttft_ms": stats(
            [to_float(r["ttft_ms"]) for r in subset]
        ),
        "total_ms": stats(
            [to_float(r["total_ms"]) for r in subset]
        ),
        "throughput_tok_s": stats(
            [to_float(r["throughput_tok_s"]) for r in subset]
        ),
    }

# ---------------------------------------------------------
# Thermal / battery
# ---------------------------------------------------------

thermal_summary = {}

if THERMAL.exists():

    with THERMAL.open(
        newline="",
        encoding="utf-8-sig"
    ) as f:

        thermal = list(csv.DictReader(f))

    temperatures = [
        to_float(r["battery_temperature_c"])
        for r in thermal
        if r.get("battery_temperature_c")
    ]

    battery_levels = [
        to_float(r["battery_percent"])
        for r in thermal
        if r.get("battery_percent")
    ]

    thermal_statuses = [
        int(to_float(r["thermal_status"]))
        for r in thermal
        if r.get("thermal_status")
        not in (None, "")
    ]

    thermal_summary = {
        "samples": len(thermal),
        "duration_s": (
            round(to_float(thermal[-1]["elapsed_s"]), 3)
            if thermal else None
        ),
        "battery_temperature_c": {
            "initial": temperatures[0]
                if temperatures else None,
            "final": temperatures[-1]
                if temperatures else None,
            "min": min(temperatures)
                if temperatures else None,
            "max": max(temperatures)
                if temperatures else None,
        },
        "battery_percent": {
            "initial": battery_levels[0]
                if battery_levels else None,
            "final": battery_levels[-1]
                if battery_levels else None,
        },
        "thermal_status": {
            "max": max(thermal_statuses)
                if thermal_statuses else None,
            "observed": sorted(set(thermal_statuses))
                if thermal_statuses else [],
        },
    }

# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------

initialization = {}

if INITIALIZATION.exists():
    initialization = json.loads(
        INITIALIZATION.read_text(
            encoding="utf-8-sig"
        )
    )

# ---------------------------------------------------------
# Memory
# ---------------------------------------------------------

memory_initial = read_proc_memory(
    RAW / "proc_status_initial.txt"
)

memory_final = read_proc_memory(
    RAW / "proc_status_final.txt"
)

# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

eog_success = sum(
    1 for row in runs
    if row["eog"].lower() == "yes"
)

summary = {
    "benchmark": {
        "total_runs": len(runs),
        "successful_eog_runs": eog_success,
        "all_runs_successful":
            len(runs) == 102
            and eog_success == 102,
        "cycles": 17,
        "prompts_per_cycle": 6,
    },
    "initialization": initialization,
    "overall_metrics": metric_summary,
    "per_prompt": prompt_summary,
    "thermal_battery": thermal_summary,
    "memory": {
        "initial_process_status": memory_initial,
        "final_process_status": memory_final,
    },
}

out = BASE / "summary.json"

out.write_text(
    json.dumps(
        summary,
        indent=2
    ),
    encoding="utf-8"
)

print()
print("=" * 60)
print("DEPLOYMENT BENCHMARK SUMMARY")
print("=" * 60)

print(f"Runs: {len(runs)}")
print(f"EOG success: {eog_success}/{len(runs)}")

print()

for metric in [
    "ttft_ms",
    "total_ms",
    "prompt_decode_ms",
    "generation_ms",
    "throughput_tok_s",
    "generated_tokens",
]:
    s = metric_summary[metric]

    print(
        f"{metric}: "
        f"mean={s['mean']} | "
        f"median={s['median']} | "
        f"P95={s['p95']} | "
        f"min={s['min']} | "
        f"max={s['max']} | "
        f"std={s['std_dev']}"
    )

print()

if initialization:
    print(
        "Model load:",
        initialization.get("model_load_ms"),
        "ms"
    )

    print(
        "Context creation:",
        initialization.get("context_creation_ms"),
        "ms"
    )

print()

if thermal_summary:
    t = thermal_summary["battery_temperature_c"]
    b = thermal_summary["battery_percent"]

    print(
        f"Temperature: "
        f"{t['initial']} C -> {t['final']} C "
        f"(max {t['max']} C)"
    )

    print(
        f"Battery: "
        f"{b['initial']}% -> {b['final']}%"
    )

    print(
        "Maximum Android thermal status:",
        thermal_summary["thermal_status"]["max"]
    )

print()

if memory_final:
    print(
        "Final VmRSS:",
        memory_final.get("VmRSS", {}).get("gib"),
        "GiB"
    )

    print(
        "Peak VmHWM:",
        memory_final.get("VmHWM", {}).get("gib"),
        "GiB"
    )

print()
print("Saved:")
print(out)
print()
