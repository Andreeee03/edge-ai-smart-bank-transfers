import json
import re
import time
import urllib.request
from pathlib import Path
from statistics import mean

# ============================================================
# CONFIG
# ============================================================

RAW_PATH = Path("data_GptPlus/splits/test.jsonl")
SFT_PATH = Path("data_GptPlus/processed/test_sft.jsonl")

OUT_DIR = Path("evaluation/q5_40_server_validation")
DETAILS_PATH = OUT_DIR / "q5_40_500_records.jsonl"
SUMMARY_PATH = OUT_DIR / "q5_40_500_summary.json"

SERVER_URL = "http://127.0.0.1:8080/completion"

N_PREDICT = 40
SEED = 42

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_text(text):
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def get_example_id(record):
    return (
        record.get("id_example")
        or record.get("id")
        or record.get("example_id")
    )


def get_activity(record):
    value = (
        record.get("activity_type")
        or record.get("activity")
        or record.get("task")
        or "UNKNOWN"
    )
    return str(value).upper()


def extract_numbered_alternatives(text):
    """
    Estrae SOLO righe numerate:
    1. ...
    2. ...
    Accetta anche 1) / 2) / 1: / 2-
    """
    numbered = []
    extra = []

    for original_line in text.splitlines():
        line = original_line.strip()

        if not line:
            continue

        m = re.match(r"^([12])\s*[\.\)\:\-]\s*(.+?)\s*$", line)

        if m:
            numbered.append({
                "label": int(m.group(1)),
                "text": m.group(2).strip()
            })
        else:
            extra.append(line)

    alternatives = [
        x["text"]
        for x in numbered
        if x["text"].strip()
    ]

    labels = [x["label"] for x in numbered]

    exactly_two_numbered = (
        len(numbered) == 2
        and labels == [1, 2]
    )

    two_distinct = (
        len(alternatives) >= 2
        and normalize_text(alternatives[0])
        != normalize_text(alternatives[1])
    )

    return {
        "numbered_lines": numbered,
        "alternatives": alternatives[:2],
        "extra_lines": extra,
        "exactly_two_numbered": exactly_two_numbered,
        "no_extra_text": len(extra) == 0,
        "two_distinct_alternatives": (
            exactly_two_numbered and two_distinct
        ),
    }


def repetitive_line_loop(text):
    """
    Segnala loop se la stessa riga non vuota compare >= 3 volte.
    """
    lines = [
        normalize_text(line)
        for line in text.splitlines()
        if normalize_text(line)
    ]

    if not lines:
        return False

    counts = {}

    for line in lines:
        counts[line] = counts.get(line, 0) + 1

    return max(counts.values(), default=0) >= 3


def extract_prompt_field(prompt, field):
    pattern = rf"(?im)^{re.escape(field)}:\s*(.+?)\s*$"
    m = re.search(pattern, prompt)
    return m.group(1).strip() if m else None


def completion_prefix_preserved(prompt, alternatives, activity):
    if activity != "COMPLETION":
        return None

    partial = extract_prompt_field(prompt, "Partial description")

    if not partial:
        return None

    partial_norm = normalize_text(partial)

    if not alternatives:
        return False

    # Almeno una alternativa deve iniziare con il testo parziale
    return any(
        normalize_text(alt).startswith(partial_norm)
        for alt in alternatives
    )


def beneficiary_repetition(prompt, alternatives):
    beneficiary = extract_prompt_field(prompt, "Beneficiary")

    if not beneficiary or not alternatives:
        return False

    ben = normalize_text(beneficiary)

    for alt in alternatives:
        alt_n = normalize_text(alt)

        if alt_n.count(ben) > 1:
            return True

    return False


def call_server(prompt):
    payload = {
        "prompt": prompt,
        "n_predict": N_PREDICT,
        "temperature": 0,
        "seed": SEED,
        "stream": False
    }

    req = urllib.request.Request(
        SERVER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = time.perf_counter()

    with urllib.request.urlopen(req, timeout=120) as response:
        result = json.loads(
            response.read().decode("utf-8")
        )

    elapsed = time.perf_counter() - start

    return result, elapsed


# ============================================================
# SERVER CHECK
# ============================================================

try:
    req = urllib.request.Request(
        "http://127.0.0.1:8080/health"
    )

    with urllib.request.urlopen(req, timeout=10) as response:
        health = response.read().decode("utf-8")

    print("Server health:", health, flush=True)

except Exception as exc:
    raise RuntimeError(
        "llama-server non raggiungibile su 127.0.0.1:8080"
    ) from exc


# ============================================================
# LOAD DATA
# ============================================================

raw_records = load_jsonl(RAW_PATH)
sft_records = load_jsonl(SFT_PATH)

assert len(raw_records) == 500, len(raw_records)
assert len(sft_records) == 500, len(sft_records)
assert len(raw_records) == len(sft_records)

print("=" * 80, flush=True)
print("Q5_K_M - 500 RECORD VALIDATION", flush=True)
print("Persistent llama-server", flush=True)
print(f"n_predict = {N_PREDICT}", flush=True)
print(f"seed      = {SEED}", flush=True)
print("prompt    = SFT raw + \\n\\n", flush=True)
print("=" * 80, flush=True)


# ============================================================
# RESUME SUPPORT
# ============================================================

completed_ids = set()

if DETAILS_PATH.exists():
    with DETAILS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                completed_ids.add(item["id_example"])
            except Exception:
                pass

if completed_ids:
    print(
        f"Resume attivo: {len(completed_ids)} record già completati.",
        flush=True
    )


# ============================================================
# RUN
# ============================================================

for index, (raw, sft) in enumerate(
    zip(raw_records, sft_records),
    start=1
):
    example_id = get_example_id(raw)

    if example_id in completed_ids:
        continue

    activity = get_activity(raw)

    prompt = sft["prompt"].rstrip("\r\n") + "\n\n"

    print(
        f"[{index:03d}/500] {activity:<15} {example_id}",
        flush=True
    )

    try:
        response, runtime = call_server(prompt)

        content = response.get("content", "")

        parsed = extract_numbered_alternatives(content)

        prefix_ok = completion_prefix_preserved(
            prompt,
            parsed["alternatives"],
            activity
        )

        ben_rep = beneficiary_repetition(
            prompt,
            parsed["alternatives"]
        )

        record = {
            "id_example": example_id,
            "activity_type": activity,

            "n_predict": N_PREDICT,

            "raw_output": content,

            "application_output": (
                "\n".join(
                    f"{i + 1}. {alt}"
                    for i, alt
                    in enumerate(parsed["alternatives"])
                )
            ),

            "predicted_alternatives":
                parsed["alternatives"],

            "extra_lines":
                parsed["extra_lines"],

            "exactly_two_numbered":
                parsed["exactly_two_numbered"],

            "no_extra_text":
                parsed["no_extra_text"],

            "two_distinct_alternatives":
                parsed["two_distinct_alternatives"],

            "repetitive_line_loop":
                repetitive_line_loop(content),

            "completion_prefix_preserved":
                prefix_ok,

            "beneficiary_repetition":
                ben_rep,

            "tokens_predicted":
                response.get("tokens_predicted"),

            "stop":
                response.get("stop"),

            "stopped_eos":
                response.get("stopped_eos"),

            "stopped_limit":
                response.get("stopped_limit"),

            "runtime_seconds":
                runtime
        }

        # Salvataggio immediato:
        # se SSH/processo cade non perdiamo i record già fatti.
        with DETAILS_PATH.open(
            "a",
            encoding="utf-8"
        ) as f:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

        print(
            f"    -> {runtime:.3f}s | "
            f"tokens={record['tokens_predicted']} | "
            f"2alt={record['exactly_two_numbered']} | "
            f"extra={not record['no_extra_text']} | "
            f"loop={record['repetitive_line_loop']}",
            flush=True
        )

    except Exception as exc:
        print(
            f"    !! ERROR: {type(exc).__name__}: {exc}",
            flush=True
        )
        raise


# ============================================================
# SUMMARY
# ============================================================

records = load_jsonl(DETAILS_PATH)

if len(records) != 500:
    raise RuntimeError(
        f"Attesi 500 record finali, trovati {len(records)}"
    )


def count_true(key, subset):
    return sum(
        1 for r in subset
        if r.get(key) is True
    )


def summarize(subset):
    n = len(subset)

    completion_records = [
        r for r in subset
        if r["activity_type"] == "COMPLETION"
    ]

    prefix_valid = [
        r for r in completion_records
        if r.get("completion_prefix_preserved")
        is not None
    ]

    runtimes = [
        r["runtime_seconds"]
        for r in subset
    ]

    token_counts = [
        r["tokens_predicted"]
        for r in subset
        if isinstance(r.get("tokens_predicted"), int)
    ]

    return {
        "records": n,

        "exactly_two_numbered_count":
            count_true("exactly_two_numbered", subset),

        "exactly_two_numbered_rate":
            count_true("exactly_two_numbered", subset) / n,

        "no_extra_text_count":
            count_true("no_extra_text", subset),

        "no_extra_text_rate":
            count_true("no_extra_text", subset) / n,

        "two_distinct_alternatives_count":
            count_true(
                "two_distinct_alternatives",
                subset
            ),

        "two_distinct_alternatives_rate":
            count_true(
                "two_distinct_alternatives",
                subset
            ) / n,

        "repetitive_line_loop_count":
            count_true(
                "repetitive_line_loop",
                subset
            ),

        "repetitive_line_loop_rate":
            count_true(
                "repetitive_line_loop",
                subset
            ) / n,

        "beneficiary_repetition_count":
            count_true(
                "beneficiary_repetition",
                subset
            ),

        "beneficiary_repetition_rate":
            count_true(
                "beneficiary_repetition",
                subset
            ) / n,

        "completion_prefix_preservation_count":
            count_true(
                "completion_prefix_preserved",
                prefix_valid
            ),

        "completion_prefix_preservation_rate":
            (
                count_true(
                    "completion_prefix_preserved",
                    prefix_valid
                ) / len(prefix_valid)
                if prefix_valid else None
            ),

        "mean_runtime_seconds":
            mean(runtimes),

        "mean_tokens_predicted":
            mean(token_counts)
            if token_counts else None,

        "hit_40_token_limit_count":
            sum(
                1 for r in subset
                if r.get("tokens_predicted")
                == N_PREDICT
            ),

        "hit_40_token_limit_rate":
            sum(
                1 for r in subset
                if r.get("tokens_predicted")
                == N_PREDICT
            ) / n
    }


activities = sorted(
    set(r["activity_type"] for r in records)
)

summary = {
    "configuration": {
        "model": "LFM2-700M_GPTPlus-DS_Q5_K_M",
        "runtime": "llama-server persistent",
        "server_url": SERVER_URL,
        "records": 500,
        "n_predict": N_PREDICT,
        "temperature": 0,
        "seed": SEED,
        "stream": False,
        "prompt_boundary":
            'prompt.rstrip("\\r\\n") + "\\n\\n"'
    },

    "overall": summarize(records),

    "by_activity": {
        activity: summarize([
            r for r in records
            if r["activity_type"] == activity
        ])
        for activity in activities
    },

    "important_ids": {
        "repetitive_line_loop": [
            r["id_example"]
            for r in records
            if r["repetitive_line_loop"]
        ],

        "two_alternative_failure": [
            r["id_example"]
            for r in records
            if not r["exactly_two_numbered"]
        ],

        "format_extra_text": [
            r["id_example"]
            for r in records
            if not r["no_extra_text"]
        ],

        "beneficiary_repetition": [
            r["id_example"]
            for r in records
            if r["beneficiary_repetition"]
        ],

        "hit_40_token_limit": [
            r["id_example"]
            for r in records
            if r.get("tokens_predicted")
            == N_PREDICT
        ]
    }
}

SUMMARY_PATH.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

print("\n" + "=" * 80, flush=True)
print("FINAL SUMMARY", flush=True)
print("=" * 80, flush=True)

print(
    json.dumps(
        summary["overall"],
        indent=2
    ),
    flush=True
)

print("\nBY ACTIVITY", flush=True)

print(
    json.dumps(
        summary["by_activity"],
        indent=2
    ),
    flush=True
)

print("\nDetailed records:", DETAILS_PATH, flush=True)
print("Summary:", SUMMARY_PATH, flush=True)

