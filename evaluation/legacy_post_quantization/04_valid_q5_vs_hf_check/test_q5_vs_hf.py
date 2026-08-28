import csv
import json
import random
import re
import subprocess
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

HF_MODEL_PATH = (
    "/home/ubuntu/edge-ai-smart-bank-transfers/"
    "models/lfm2_700m_gptplus_merged"
)

GGUF_MODEL_PATH = (
    "/home/ubuntu/edge-ai-smart-bank-transfers/"
    "models/gguf/LFM2-700M_GPTPlus-DS_Q5_K_M.gguf"
)

LLAMA_COMPLETION_PATH = (
    "/home/ubuntu/llama.cpp/build/bin/llama-completion"
)

TEST_FILE = (
    "/home/ubuntu/edge-ai-smart-bank-transfers/"
    "data_GptPlus/processed/test_sft.jsonl"
)

OUTPUT_DIR = Path(
    "/home/ubuntu/edge-ai-smart-bank-transfers/"
    "evaluation/results_q5_validation"
)

OUTPUT_CSV = OUTPUT_DIR / "q5_vs_hf_results.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "q5_vs_hf_summary.json"

N_EXAMPLES = 999999
MAX_NEW_TOKENS = 100
SEED = 42


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


# ============================================================
# PROMPT EXTRACTION
# ============================================================

def split_sft_text(text):
    """
    Divide prompt e reference.

    Cerca prima il boundary:
        \\n\\n1.

    perché l'output target è normalmente:
        1. ...
        2. ...
    """

    matches = list(
        re.finditer(
            r"\n\n(?=1\.\s)",
            text
        )
    )

    if matches:
        boundary = matches[-1].start()

        prompt_base = text[:boundary].rstrip("\r\n")

        reference = text[
            matches[-1].end():
        ].strip()

        return (
            prompt_base + "\n\n",
            reference
        )

    if "\n\n" in text:
        prompt_base, reference = text.rsplit(
            "\n\n",
            1
        )

        return (
            prompt_base.rstrip("\r\n") + "\n\n",
            reference.strip()
        )

    raise ValueError(
        "Impossibile individuare il boundary "
        "prompt/reference."
    )


def extract_prompt_reference(record):

    if "prompt" in record:
        for target_key in [
            "completion",
            "reference",
            "output",
            "target",
            "response",
        ]:
            if target_key in record:
                prompt_base = str(
                    record["prompt"]
                ).rstrip("\r\n")

                reference = str(
                    record[target_key]
                ).strip()

                return (
                    prompt_base + "\n\n",
                    reference
                )

    if "conversation" in record:
        return split_sft_text(
            str(record["conversation"])
        )

    if "text" in record:
        return split_sft_text(
            str(record["text"])
        )

    raise ValueError(
        "Formato record non riconosciuto. "
        f"Chiavi: {list(record.keys())}"
    )


# ============================================================
# METRICS
# ============================================================

def metric_tokens(text):
    return re.findall(
        r"\b\w+\b",
        text.lower(),
        flags=re.UNICODE
    )


def ngrams(tokens, n):
    return [
        tuple(tokens[i:i+n])
        for i in range(
            len(tokens) - n + 1
        )
    ]


def rouge_n_f1(prediction, reference, n):

    pred_tokens = metric_tokens(prediction)
    ref_tokens = metric_tokens(reference)

    pred_ngrams = Counter(
        ngrams(pred_tokens, n)
    )

    ref_ngrams = Counter(
        ngrams(ref_tokens, n)
    )

    if not pred_ngrams or not ref_ngrams:
        return 0.0

    overlap = sum(
        (pred_ngrams & ref_ngrams).values()
    )

    precision = (
        overlap /
        sum(pred_ngrams.values())
    )

    recall = (
        overlap /
        sum(ref_ngrams.values())
    )

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall /
        (precision + recall)
    )


def exact_match(a, b):
    return (
        a.strip().lower()
        ==
        b.strip().lower()
    )


def valid_two_alternatives(text):

    has_1 = bool(
        re.search(
            r"(?m)^\s*1\.\s*\S+",
            text
        )
    )

    has_2 = bool(
        re.search(
            r"(?m)^\s*2\.\s*\S+",
            text
        )
    )

    return has_1 and has_2


def repetition_detected(text):

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if len(lines) < 4:
        return False

    counts = Counter(lines)

    most_common = counts.most_common(1)[0][1]

    return (
        most_common / len(lines)
        >= 0.60
    )


# ============================================================
# PATH CHECK
# ============================================================

print("=" * 100)
print("PATH CHECK")
print("=" * 100)

for path in [
    HF_MODEL_PATH,
    GGUF_MODEL_PATH,
    LLAMA_COMPLETION_PATH,
    TEST_FILE,
]:
    exists = Path(path).exists()

    print(
        "[OK]" if exists else "[MISSING]",
        path
    )

    if not exists:
        raise FileNotFoundError(path)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD TEST SET
# ============================================================

print()
print("=" * 100)
print("LOADING TEST SET")
print("=" * 100)

records = load_jsonl(TEST_FILE)

print(
    "Total test examples:",
    len(records)
)

rng = random.Random(SEED)

if len(records) > N_EXAMPLES:
    selected_indices = rng.sample(
        range(len(records)),
        N_EXAMPLES
    )
else:
    selected_indices = list(
        range(len(records))
    )

selected_records = [
    records[i]
    for i in selected_indices
]

print(
    "Selected examples:",
    len(selected_records)
)


# ============================================================
# FIRST PROMPT CHECK
# ============================================================

first_prompt, first_reference = (
    extract_prompt_reference(
        selected_records[0]
    )
)

print()
print("=" * 100)
print("PROMPT FORMAT CHECK")
print("=" * 100)

print(
    "First record keys:",
    list(selected_records[0].keys())
)

print(
    "Prompt ending:",
    repr(first_prompt[-100:])
)

print(
    "Reference:",
    repr(first_reference[:200])
)

if not first_prompt.endswith("\n\n"):
    raise RuntimeError(
        "Prompt does not end with \\n\\n."
    )

if first_prompt.endswith("\n\n\n"):
    raise RuntimeError(
        "Prompt ends with 3 newlines. "
        "With -p we need exactly 2."
    )

print(
    "Double-newline boundary: OK"
)


# ============================================================
# LOAD HF MODEL
# ============================================================

print()
print("=" * 100)
print("LOADING HF MERGED MODEL")
print("=" * 100)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)

tokenizer = AutoTokenizer.from_pretrained(
    HF_MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_PATH,
    torch_dtype="auto",
    trust_remote_code=True
)

model.to(device)
model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = (
        tokenizer.eos_token_id
    )

print(
    "HF model loaded successfully."
)


# ============================================================
# HF GENERATION
# ============================================================

@torch.inference_mode()
def generate_hf(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    input_length = (
        inputs["input_ids"].shape[1]
    )

    generated = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    new_tokens = generated[
        0,
        input_length:
    ]

    output = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    return output.strip()


# ============================================================
# Q5 GENERATION
# ============================================================

def generate_q5(prompt):

    command = [
        LLAMA_COMPLETION_PATH,

        "-m",
        GGUF_MODEL_PATH,

        # Prompt diretto.
        # NON usiamo -f.
        "-p",
        prompt,

        "-n",
        str(MAX_NEW_TOKENS),

        "--temp",
        "0",

        "--top-k",
        "1",

        "--seed",
        str(SEED),

        "--repeat-penalty",
        "1.0",

        # Fondamentale: il GGUF contiene un chat template,
        # ma noi vogliamo raw text completion con il nostro
        # prompt SFT personalizzato.
        "--no-conversation",

        "--no-display-prompt",

        "--simple-io",

        "--color",
        "off"
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,

        # llama-completion non deve poter rimanere
        # in attesa di input dalla shell.
        stdin=subprocess.DEVNULL,

        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,

        # Evita che un eventuale problema lasci
        # il test bloccato indefinitamente.
        timeout=120
    )

    if result.returncode != 0:
        raise RuntimeError(
            "\nllama-completion failed.\n\n"
            f"RETURN CODE:\n"
            f"{result.returncode}\n\n"
            f"STDERR:\n"
            f"{result.stderr}"
        )

    return result.stdout.strip()


# ============================================================
# EVALUATION
# ============================================================

results = []

for position, (
    original_index,
    record
) in enumerate(
    zip(
        selected_indices,
        selected_records
    ),
    start=1
):

    print()
    print("=" * 100)

    print(
        f"EXAMPLE "
        f"{position}/{len(selected_records)} "
        f"(dataset index {original_index})"
    )

    print("=" * 100)

    prompt, reference = (
        extract_prompt_reference(record)
    )

    if not prompt.endswith("\n\n"):
        raise RuntimeError(
            f"Example {original_index}: "
            "prompt does not end with \\n\\n."
        )

    if prompt.endswith("\n\n\n"):
        raise RuntimeError(
            f"Example {original_index}: "
            "prompt ends with 3 newlines."
        )

    print(
        "Prompt ending:",
        repr(prompt[-80:])
    )

    print()
    print("HF generation...")

    hf_output = generate_hf(
        prompt
    )

    print("HF OUTPUT:")
    print(hf_output)

    print()
    print("Q5 generation...")

    q5_output = generate_q5(
        prompt
    )

    print("Q5 OUTPUT:")
    print(q5_output)

    hf_r1 = rouge_n_f1(
        hf_output,
        reference,
        1
    )

    hf_r2 = rouge_n_f1(
        hf_output,
        reference,
        2
    )

    q5_r1 = rouge_n_f1(
        q5_output,
        reference,
        1
    )

    q5_r2 = rouge_n_f1(
        q5_output,
        reference,
        2
    )

    q5_vs_hf_r1 = rouge_n_f1(
        q5_output,
        hf_output,
        1
    )

    q5_vs_hf_r2 = rouge_n_f1(
        q5_output,
        hf_output,
        2
    )

    row = {
        "dataset_index":
            original_index,

        "prompt":
            prompt,

        "reference":
            reference,

        "hf_output":
            hf_output,

        "q5_output":
            q5_output,

        "hf_rouge1":
            hf_r1,

        "hf_rouge2":
            hf_r2,

        "q5_rouge1":
            q5_r1,

        "q5_rouge2":
            q5_r2,

        "q5_vs_hf_rouge1":
            q5_vs_hf_r1,

        "q5_vs_hf_rouge2":
            q5_vs_hf_r2,

        "hf_exact_reference":
            exact_match(
                hf_output,
                reference
            ),

        "q5_exact_reference":
            exact_match(
                q5_output,
                reference
            ),

        "q5_exact_hf":
            exact_match(
                q5_output,
                hf_output
            ),

        "hf_valid_format":
            valid_two_alternatives(
                hf_output
            ),

        "q5_valid_format":
            valid_two_alternatives(
                q5_output
            ),

        "hf_repetition":
            repetition_detected(
                hf_output
            ),

        "q5_repetition":
            repetition_detected(
                q5_output
            )
    }

    results.append(row)

    print()

    print(
        f"HF  ROUGE-1: "
        f"{hf_r1:.4f}"
    )

    print(
        f"HF  ROUGE-2: "
        f"{hf_r2:.4f}"
    )

    print(
        f"Q5  ROUGE-1: "
        f"{q5_r1:.4f}"
    )

    print(
        f"Q5  ROUGE-2: "
        f"{q5_r2:.4f}"
    )

    print(
        f"Q5 vs HF R1: "
        f"{q5_vs_hf_r1:.4f}"
    )

    print(
        f"Q5 vs HF R2: "
        f"{q5_vs_hf_r2:.4f}"
    )

    print(
        "Q5 format valid:",
        row["q5_valid_format"]
    )

    print(
        "Q5 repetition:",
        row["q5_repetition"]
    )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


# ============================================================
# SUMMARY
# ============================================================

def average(key):

    return (
        sum(
            float(row[key])
            for row in results
        )
        /
        len(results)
    )


summary = {
    "hf_model":
        HF_MODEL_PATH,

    "q5_model":
        GGUF_MODEL_PATH,

    "test_file":
        TEST_FILE,

    "examples_tested":
        len(results),

    "seed":
        SEED,

    "hf_avg_rouge1":
        average(
            "hf_rouge1"
        ),

    "hf_avg_rouge2":
        average(
            "hf_rouge2"
        ),

    "q5_avg_rouge1":
        average(
            "q5_rouge1"
        ),

    "q5_avg_rouge2":
        average(
            "q5_rouge2"
        ),

    "q5_vs_hf_avg_rouge1":
        average(
            "q5_vs_hf_rouge1"
        ),

    "q5_vs_hf_avg_rouge2":
        average(
            "q5_vs_hf_rouge2"
        ),

    "hf_valid_format_percent":
        100
        * sum(
            row["hf_valid_format"]
            for row in results
        )
        / len(results),

    "q5_valid_format_percent":
        100
        * sum(
            row["q5_valid_format"]
            for row in results
        )
        / len(results),

    "hf_repetition_count":
        sum(
            row["hf_repetition"]
            for row in results
        ),

    "q5_repetition_count":
        sum(
            row["q5_repetition"]
            for row in results
        ),

    "q5_exactly_equal_to_hf_count":
        sum(
            row["q5_exact_hf"]
            for row in results
        )
}


with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 100)
print("FINAL SUMMARY")
print("=" * 100)

print(
    "Examples tested:",
    summary["examples_tested"]
)

print()

print(
    "HF average ROUGE-1:",
    f'{summary["hf_avg_rouge1"]:.4f}'
)

print(
    "HF average ROUGE-2:",
    f'{summary["hf_avg_rouge2"]:.4f}'
)

print()

print(
    "Q5 average ROUGE-1:",
    f'{summary["q5_avg_rouge1"]:.4f}'
)

print(
    "Q5 average ROUGE-2:",
    f'{summary["q5_avg_rouge2"]:.4f}'
)

print()

print(
    "Q5 vs HF ROUGE-1:",
    f'{summary["q5_vs_hf_avg_rouge1"]:.4f}'
)

print(
    "Q5 vs HF ROUGE-2:",
    f'{summary["q5_vs_hf_avg_rouge2"]:.4f}'
)

print()

print(
    "HF valid format:",
    f'{summary["hf_valid_format_percent"]:.1f}%'
)

print(
    "Q5 valid format:",
    f'{summary["q5_valid_format_percent"]:.1f}%'
)

print()

print(
    "HF repetition problems:",
    summary["hf_repetition_count"]
)

print(
    "Q5 repetition problems:",
    summary["q5_repetition_count"]
)

print()

print(
    "Q5 exactly equal to HF:",
    summary[
        "q5_exactly_equal_to_hf_count"
    ],
    "/",
    len(results)
)

print()

print(
    "Detailed results:",
    OUTPUT_CSV
)

print(
    "Summary:",
    OUTPUT_SUMMARY
)

print("=" * 100)
