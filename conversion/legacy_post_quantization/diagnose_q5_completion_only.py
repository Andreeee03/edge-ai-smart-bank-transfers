import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

OLD_RESULTS = Path(
    "evaluation/full_quantization_validation/"
    "q4_vs_q5_500_records.jsonl"
)

SERVER_RESULTS = Path(
    "evaluation/q5_64_server_validation/"
    "q5_64_500_records.jsonl"
)

RAW_PATH = Path("data_GptPlus/splits/test.jsonl")
SFT_PATH = Path("data_GptPlus/processed/test_sft.jsonl")

MODEL = Path(
    "models/gguf/"
    "LFM2-700M_GPTPlus-DS_Q5_K_M.gguf"
).resolve()

LLAMA_COMPLETION = Path(
    "/home/ubuntu/llama.cpp/build/bin/llama-completion"
)

OUT_DIR = Path(
    "evaluation/q5_completion_only_diagnostic"
)

DETAILS_PATH = OUT_DIR / "q5_completion_only_details.jsonl"
SUMMARY_PATH = OUT_DIR / "q5_completion_only_summary.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def normalize_cli_output(text):
    """
    Rimuove soltanto elementi di visualizzazione del CLI.
    Non altera il contenuto generato dal modello.
    """
    text = text.replace("\r\n", "\n")
    text = text.replace("[end of text]", "")
    return text.strip()


def normalize_for_comparison(text):
    text = str(text).replace("\r\n", "\n").strip()
    return text


def repetitive_line_loop(text):
    lines = [
        re.sub(r"\s+", " ", line.strip().lower())
        for line in text.splitlines()
        if line.strip()
    ]

    counts = {}

    for line in lines:
        counts[line] = counts.get(line, 0) + 1

    maximum = max(counts.values(), default=0)

    return maximum >= 3, maximum


# ============================================================
# LOAD OLD RESULTS
# ============================================================

old_records = load_jsonl(OLD_RESULTS)
server_records = load_jsonl(SERVER_RESULTS)
raw_records = load_jsonl(RAW_PATH)
sft_records = load_jsonl(SFT_PATH)

assert len(old_records) == 500
assert len(server_records) == 500
assert len(raw_records) == 500
assert len(sft_records) == 500

server_by_id = {
    r["id_example"]: r
    for r in server_records
}

# Associazione ID -> prompt SFT attraverso l'allineamento
# raw/test_sft usato anche nelle precedenti valutazioni.
prompt_by_id = {}

for raw, sft in zip(raw_records, sft_records):
    example_id = raw["id_example"]

    prompt_by_id[example_id] = (
        sft["prompt"].rstrip("\r\n") + "\n\n"
    )


# ============================================================
# SELECT ALL OLD Q5 LOOP CASES
# ============================================================

loop_records = [
    r for r in old_records
    if r["Q5_K_M"]["repetitive_line_loop"]
]

print("=" * 90)
print("Q5-ONLY LLAMA-COMPLETION DIAGNOSTIC")
print("=" * 90)
print("Old Q5 repetitive-loop cases:", len(loop_records))
print("Q4 will NOT be executed.")
print("Model:", MODEL)
print()

assert len(loop_records) == 21, (
    f"Attesi 21 loop Q5 dal benchmark precedente, "
    f"trovati {len(loop_records)}"
)

print("IDs:")
print(", ".join(r["id_example"] for r in loop_records))
print()


# ============================================================
# CLEAN OLD DIAGNOSTIC OUTPUTS
# ============================================================

DETAILS_PATH.unlink(missing_ok=True)
SUMMARY_PATH.unlink(missing_ok=True)


# ============================================================
# RUN Q5 ONLY WITH LLAMA-COMPLETION
# ============================================================

results = []

for i, old_record in enumerate(loop_records, start=1):

    example_id = old_record["id_example"]
    activity = old_record["activity_type"]

    prompt = prompt_by_id[example_id]

    old_q5_output = normalize_for_comparison(
        old_record["Q5_K_M"]["clean_output"]
    )

    server_output = normalize_for_comparison(
        server_by_id[example_id]["raw_output"]
    )

    print(
        f"[{i:02d}/{len(loop_records)}] "
        f"{activity:<15} {example_id}",
        flush=True
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".txt",
        delete=False
    ) as tmp:
        tmp.write(prompt)
        prompt_path = Path(tmp.name)

    command = [
        str(LLAMA_COMPLETION),

        "-m",
        str(MODEL),

        "-f",
        str(prompt_path),

        "-n",
        "64",

        "-c",
        "512",

        "--temp",
        "0",

        "--seed",
        "42",

        "-no-cnv",

        "--no-display-prompt",

        "--color",
        "off",
    ]

    start = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )

    finally:
        prompt_path.unlink(missing_ok=True)

    elapsed = time.perf_counter() - start

    if completed.returncode != 0:
        print("ERROR llama-completion")
        print(completed.stderr)
        raise RuntimeError(
            f"llama-completion return code "
            f"{completed.returncode} per {example_id}"
        )

    new_output = normalize_cli_output(
        completed.stdout
    )

    loop_now, max_repeat = repetitive_line_loop(
        new_output
    )

    matches_old_q5 = (
        new_output == old_q5_output
    )

    matches_server = (
        new_output == server_output
    )

    result = {
        "id_example": example_id,
        "activity_type": activity,

        "new_q5_completion_output":
            new_output,

        "old_q5_completion_output":
            old_q5_output,

        "q5_server_output":
            server_output,

        "new_repetitive_line_loop":
            loop_now,

        "new_max_identical_line_count":
            max_repeat,

        "matches_old_q5_completion":
            matches_old_q5,

        "matches_q5_server":
            matches_server,

        "runtime_seconds":
            elapsed,
    }

    results.append(result)

    with DETAILS_PATH.open(
        "a",
        encoding="utf-8"
    ) as f:
        f.write(
            json.dumps(
                result,
                ensure_ascii=False
            ) + "\n"
        )

    print(
        f"    loop={loop_now} | "
        f"same_as_old_Q5={matches_old_q5} | "
        f"same_as_server={matches_server} | "
        f"{elapsed:.2f}s",
        flush=True
    )


# ============================================================
# SUMMARY
# ============================================================

n = len(results)

loop_count = sum(
    r["new_repetitive_line_loop"]
    for r in results
)

old_exact_count = sum(
    r["matches_old_q5_completion"]
    for r in results
)

server_exact_count = sum(
    r["matches_q5_server"]
    for r in results
)

summary = {
    "records": n,

    "configuration": {
        "frontend": "llama-completion",
        "model": str(MODEL),
        "q4_executed": False,
        "n_predict": 64,
        "context": 512,
        "temperature": 0,
        "seed": 42,
        "conversation_mode": False,
        "prompt_boundary":
            'prompt.rstrip("\\r\\n") + "\\n\\n"',
    },

    "old_q5_loop_cases_reproduced_count":
        loop_count,

    "old_q5_loop_cases_reproduced_rate":
        loop_count / n,

    "exact_output_match_old_q5_count":
        old_exact_count,

    "exact_output_match_old_q5_rate":
        old_exact_count / n,

    "exact_output_match_server_count":
        server_exact_count,

    "exact_output_match_server_rate":
        server_exact_count / n,

    "still_looping_ids": [
        r["id_example"]
        for r in results
        if r["new_repetitive_line_loop"]
    ],

    "matches_old_q5_ids": [
        r["id_example"]
        for r in results
        if r["matches_old_q5_completion"]
    ],

    "matches_server_ids": [
        r["id_example"]
        for r in results
        if r["matches_q5_server"]
    ],
}

SUMMARY_PATH.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

print()
print("=" * 90)
print("FINAL SUMMARY")
print("=" * 90)
print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    )
)

print()
print("Details:", DETAILS_PATH)
print("Summary:", SUMMARY_PATH)
