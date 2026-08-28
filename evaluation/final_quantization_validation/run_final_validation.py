import csv
import gc
import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path("/home/ubuntu/edge-ai-smart-bank-transfers")

HF_MODEL_PATH = ROOT / "models/lfm2_700m_gptplus_merged"

GGUF_MODELS = {
    "F16": ROOT / "models/gguf/LFM2-700M_GPTPlus-DS_F16.gguf",
    "Q4_K_M": ROOT / "models/gguf/LFM2-700M_GPTPlus-DS_Q4_K_M.gguf",
    "Q5_K_M": ROOT / "models/gguf/LFM2-700M_GPTPlus-DS_Q5_K_M.gguf",
}

LLAMA_SERVER = Path("/home/ubuntu/llama.cpp/build/bin/llama-server")

TEST_FILE = ROOT / "data_GptPlus/processed/test_sft.jsonl"

OUTPUT_DIR = ROOT / "evaluation/final_quantization_validation"

MAX_NEW_TOKENS = 100
SEED = 42
CTX_SIZE = 512

# Sampling definitivo:
# HF      -> do_sample=False
# llama   -> temperature=0 + top_k=1
#
# Nessuna repetition penalty artificiale.
REPETITION_PENALTY = 1.0


# =============================================================================
# OUTPUT FILES
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILES = {
    "HF": OUTPUT_DIR / "outputs_hf.jsonl",
    "F16": OUTPUT_DIR / "outputs_f16.jsonl",
    "Q4_K_M": OUTPUT_DIR / "outputs_q4_k_m.jsonl",
    "Q5_K_M": OUTPUT_DIR / "outputs_q5_k_m.jsonl",
}

FINAL_RECORDS = OUTPUT_DIR / "final_quantization_500_records.jsonl"
SUMMARY_JSON = OUTPUT_DIR / "final_quantization_500_summary.json"
MODEL_CSV = OUTPUT_DIR / "model_metrics_summary.csv"
PAIRWISE_CSV = OUTPUT_DIR / "pairwise_similarity_summary.csv"


# =============================================================================
# HELPERS
# =============================================================================

def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def prepare_record(record, index):
    if "prompt" not in record or "completion" not in record:
        raise ValueError(
            f"Record {index}: expected keys 'prompt' and 'completion'. "
            f"Found: {list(record.keys())}"
        )

    # IMPORTANTISSIMO:
    # eliminiamo qualsiasi newline finale presente e poi
    # aggiungiamo ESATTAMENTE due newline.
    prompt = str(record["prompt"]).rstrip("\r\n") + "\n\n"

    reference = str(record["completion"]).strip()

    if not prompt.endswith("\n\n"):
        raise RuntimeError(
            f"Record {index}: prompt does not end with \\n\\n"
        )

    if prompt.endswith("\n\n\n"):
        raise RuntimeError(
            f"Record {index}: prompt ends with 3+ newlines"
        )

    return {
        "index": index,
        "prompt": prompt,
        "reference": reference,
    }


def metric_tokens(text):
    return re.findall(
        r"\b\w+\b",
        text.lower(),
        flags=re.UNICODE
    )


def make_ngrams(tokens, n):
    if len(tokens) < n:
        return []

    return [
        tuple(tokens[i:i+n])
        for i in range(len(tokens) - n + 1)
    ]


def rouge_n_f1(prediction, reference, n):
    pred_tokens = metric_tokens(prediction)
    ref_tokens = metric_tokens(reference)

    pred_ngrams = Counter(
        make_ngrams(pred_tokens, n)
    )

    ref_ngrams = Counter(
        make_ngrams(ref_tokens, n)
    )

    if not pred_ngrams or not ref_ngrams:
        return 0.0

    overlap = sum(
        (pred_ngrams & ref_ngrams).values()
    )

    precision = overlap / sum(
        pred_ngrams.values()
    )

    recall = overlap / sum(
        ref_ngrams.values()
    )

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        /
        (precision + recall)
    )


def exact_match(a, b):
    return (
        a.strip().lower()
        ==
        b.strip().lower()
    )


def parse_two_alternatives(text):
    lines = [
        line.strip()
        for line in text.strip().splitlines()
        if line.strip()
    ]

    parsed = []

    for line in lines:
        match = re.match(
            r"^\s*([12])\.\s*(.+?)\s*$",
            line
        )

        if match:
            parsed.append(
                (
                    int(match.group(1)),
                    match.group(2).strip()
                )
            )

    return lines, parsed


def valid_format(text):
    _, parsed = parse_two_alternatives(text)

    return (
        len(parsed) >= 2
        and parsed[0][0] == 1
        and parsed[1][0] == 2
        and bool(parsed[0][1])
        and bool(parsed[1][1])
    )


def exactly_two(text):
    lines, parsed = parse_two_alternatives(text)

    return (
        len(lines) == 2
        and len(parsed) == 2
        and parsed[0][0] == 1
        and parsed[1][0] == 2
    )


def distinct_alternatives(text):
    _, parsed = parse_two_alternatives(text)

    if len(parsed) < 2:
        return False

    return (
        parsed[0][1].strip().lower()
        !=
        parsed[1][1].strip().lower()
    )


def repetition_detected(text):
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if len(lines) < 4:
        return False

    counts = Counter(lines)

    most_common_count = counts.most_common(1)[0][1]

    return (
        most_common_count / len(lines)
        >= 0.60
    )


def average(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def file_size_info(path):
    size = path.stat().st_size

    return {
        "bytes": size,
        "MiB": size / (1024 ** 2),
        "GiB": size / (1024 ** 3),
    }


# =============================================================================
# CACHE / RESUME
# =============================================================================

def load_existing_outputs(path, expected_records):
    """
    Permette di riprendere il test se viene interrotto.
    """

    if not path.exists():
        return []

    rows = load_jsonl(path)

    # Controllo minimo di coerenza.
    for i, row in enumerate(rows):
        if row.get("index") != i:
            raise RuntimeError(
                f"Invalid cache ordering in {path}"
            )

        expected_prompt = expected_records[i]["prompt"]

        if row.get("prompt") != expected_prompt:
            raise RuntimeError(
                f"Prompt mismatch in cached file {path}, "
                f"record {i}"
            )

    if len(rows) > len(expected_records):
        raise RuntimeError(
            f"{path} contains too many records."
        )

    return rows


def append_output(path, row):
    with open(
        path,
        "a",
        encoding="utf-8"
    ) as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False
            )
            +
            "\n"
        )


# =============================================================================
# CHECK PATHS
# =============================================================================

print("=" * 100)
print("FINAL QUANTIZATION VALIDATION")
print("HF vs GGUF F16 vs Q4_K_M vs Q5_K_M")
print("=" * 100)

paths = [
    HF_MODEL_PATH,
    TEST_FILE,
    LLAMA_SERVER,
    *GGUF_MODELS.values(),
]

for path in paths:
    if not Path(path).exists():
        print("[MISSING]", path)
        raise FileNotFoundError(path)

    print("[OK]", path)


# =============================================================================
# LOAD TEST SET
# =============================================================================

print()
print("=" * 100)
print("LOADING TEST SET")
print("=" * 100)

raw_records = load_jsonl(TEST_FILE)

records = [
    prepare_record(record, i)
    for i, record in enumerate(raw_records)
]

print("Examples:", len(records))

if len(records) != 500:
    print(
        "WARNING: expected 500 test examples, "
        f"found {len(records)}."
    )

print()
print("First prompt ending:")
print(repr(records[0]["prompt"][-120:]))

print()
print(
    "Ends with exactly two newlines:",
    records[0]["prompt"].endswith("\n\n")
    and not records[0]["prompt"].endswith("\n\n\n")
)

print()
print("First reference:")
print(records[0]["reference"])


# =============================================================================
# HF GENERATION
# =============================================================================

def run_hf():
    output_path = OUTPUT_FILES["HF"]

    existing = load_existing_outputs(
        output_path,
        records
    )

    start_index = len(existing)

    if start_index == len(records):
        print(
            "\nHF outputs already complete. "
            "Using cached results."
        )
        return existing

    print()
    print("=" * 100)
    print("HF MERGED MODEL")
    print("=" * 100)

    print(
        f"Resuming from record {start_index}/"
        f"{len(records)}"
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_PATH,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_PATH,
        dtype="auto",
        trust_remote_code=True
    )

    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )

    outputs = list(existing)

    for i in range(start_index, len(records)):
        record = records[i]
        prompt = record["prompt"]

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

        start_time = time.perf_counter()

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        elapsed = (
            time.perf_counter()
            -
            start_time
        )

        new_tokens = generated[
            0,
            input_length:
        ]

        text = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        ).strip()

        row = {
            "index": i,
            "prompt": prompt,
            "output": text,
            "elapsed_seconds": elapsed,
        }

        append_output(
            output_path,
            row
        )

        outputs.append(row)

        if i < 3:
            print()
            print(f"HF example {i}:")
            print(text)

        if (
            (i + 1) % 25 == 0
            or
            i + 1 == len(records)
        ):
            print(
                f"HF progress: "
                f"{i + 1}/{len(records)}"
            )

    # Liberiamo GPU/RAM prima di caricare llama.cpp.
    del model
    del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return outputs


# =============================================================================
# LOCAL LLAMA-SERVER
# =============================================================================

def get_free_port():
    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    sock.bind(("127.0.0.1", 0))

    port = sock.getsockname()[1]

    sock.close()

    return port


def http_get(url, timeout=2):
    with urllib.request.urlopen(
        url,
        timeout=timeout
    ) as response:
        return response.read()


def wait_for_server(
    process,
    port,
    log_path,
    timeout=180
):
    deadline = time.time() + timeout

    health_url = (
        f"http://127.0.0.1:{port}/health"
    )

    while time.time() < deadline:
        if process.poll() is not None:
            break

        try:
            http_get(
                health_url,
                timeout=2
            )
            return True
        except Exception:
            time.sleep(1)

    print()
    print("SERVER FAILED TO BECOME READY")

    if log_path.exists():
        print()
        print("--- SERVER LOG TAIL ---")

        lines = log_path.read_text(
            encoding="utf-8",
            errors="replace"
        ).splitlines()

        for line in lines[-80:]:
            print(line)

    return False


def start_server(model_label, model_path):
    """
    Prova prima con GPU offload.
    Se la build non lo supporta, riprova CPU-only.
    """

    port = get_free_port()

    log_path = (
        OUTPUT_DIR
        /
        f"server_{model_label.lower()}_final.log"
    )

    attempts = [
        {
            "name": "GPU",
            "extra": [
                "-ngl",
                "99",
            ],
        },
        {
            "name": "CPU fallback",
            "extra": [],
        },
    ]

    for attempt in attempts:
        print()
        print(
            f"Starting {model_label} server "
            f"({attempt['name']})..."
        )

        command = [
            str(LLAMA_SERVER),

            "--model",
            str(model_path),

            "--host",
            "127.0.0.1",

            "--port",
            str(port),

            "--ctx-size",
            str(CTX_SIZE),

            "--parallel",
            "1",

            *attempt["extra"],
        ]

        log_file = open(
            log_path,
            "w",
            encoding="utf-8"
        )

        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
        )

        ready = wait_for_server(
            process,
            port,
            log_path
        )

        if ready:
            print(
                f"{model_label} server ready "
                f"on 127.0.0.1:{port}"
            )

            return (
                process,
                log_file,
                port,
                log_path,
            )

        try:
            process.terminate()
            process.wait(timeout=10)
        except Exception:
            process.kill()

        log_file.close()

        print(
            f"{attempt['name']} startup failed."
        )

    raise RuntimeError(
        f"Unable to start llama-server "
        f"for {model_label}"
    )


def stop_server(process, log_file):
    if process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    log_file.close()


def generate_server_completion(
    port,
    prompt
):
    """
    RAW completion.

    Nessun chat template.
    Nessun file.
    Nessun -f.

    Il JSON contiene direttamente:
        prompt ... \\n\\n
    """

    payload = {
        "prompt": prompt,

        "n_predict":
            MAX_NEW_TOKENS,

        # Greedy
        "temperature":
            0.0,

        "top_k":
            1,

        # Disabilitiamo gli altri filtri
        # probabilistici.
        "top_p":
            1.0,

        "min_p":
            0.0,

        "typical_p":
            1.0,

        # Nessuna penalty artificiale.
        "repeat_penalty":
            REPETITION_PENALTY,

        "presence_penalty":
            0.0,

        "frequency_penalty":
            0.0,

        "dry_multiplier":
            0.0,

        "seed":
            SEED,

        "stream":
            False,

        "cache_prompt":
            False,

        "ignore_eos":
            False,
    }

    data = json.dumps(
        payload
    ).encode("utf-8")

    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=data,
        headers={
            "Content-Type":
                "application/json"
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=180
    ) as response:
        result = json.loads(
            response.read().decode(
                "utf-8"
            )
        )

    if "content" not in result:
        raise RuntimeError(
            "Invalid llama-server response: "
            +
            json.dumps(
                result,
                ensure_ascii=False
            )[:2000]
        )

    return result


# =============================================================================
# GGUF GENERATION
# =============================================================================

def run_gguf(
    model_label,
    model_path
):
    output_path = (
        OUTPUT_FILES[model_label]
    )

    existing = load_existing_outputs(
        output_path,
        records
    )

    start_index = len(existing)

    if start_index == len(records):
        print(
            f"\n{model_label} outputs already "
            "complete. Using cached results."
        )

        return existing

    print()
    print("=" * 100)
    print(f"GGUF {model_label}")
    print("=" * 100)

    print(
        f"Resuming from record "
        f"{start_index}/{len(records)}"
    )

    (
        process,
        log_file,
        port,
        server_log,
    ) = start_server(
        model_label,
        model_path
    )

    outputs = list(existing)

    try:
        for i in range(
            start_index,
            len(records)
        ):
            record = records[i]

            prompt = record["prompt"]

            # Safety check per ogni record.
            if not prompt.endswith("\n\n"):
                raise RuntimeError(
                    f"{model_label} record {i}: "
                    "missing double newline"
                )

            if prompt.endswith("\n\n\n"):
                raise RuntimeError(
                    f"{model_label} record {i}: "
                    "3+ final newlines"
                )

            start_time = time.perf_counter()

            response = (
                generate_server_completion(
                    port,
                    prompt
                )
            )

            elapsed = (
                time.perf_counter()
                -
                start_time
            )

            text = str(
                response["content"]
            ).strip()

            row = {
                "index": i,
                "prompt": prompt,
                "output": text,
                "elapsed_seconds": elapsed,
                "tokens_predicted":
                    response.get(
                        "tokens_predicted"
                    ),
                "tokens_evaluated":
                    response.get(
                        "tokens_evaluated"
                    ),
                "stop":
                    response.get("stop"),
                "stop_type":
                    response.get(
                        "stop_type"
                    ),
            }

            append_output(
                output_path,
                row
            )

            outputs.append(row)

            if i < 3:
                print()
                print(
                    f"{model_label} "
                    f"example {i}:"
                )
                print(text)

            if (
                (i + 1) % 25 == 0
                or
                i + 1 == len(records)
            ):
                print(
                    f"{model_label} progress: "
                    f"{i + 1}/{len(records)}"
                )

    finally:
        stop_server(
            process,
            log_file
        )

    return outputs


# =============================================================================
# GENERATION
# =============================================================================

hf_outputs = run_hf()

f16_outputs = run_gguf(
    "F16",
    GGUF_MODELS["F16"]
)

q4_outputs = run_gguf(
    "Q4_K_M",
    GGUF_MODELS["Q4_K_M"]
)

q5_outputs = run_gguf(
    "Q5_K_M",
    GGUF_MODELS["Q5_K_M"]
)


# =============================================================================
# SANITY CHECK
# =============================================================================

all_outputs = {
    "HF": hf_outputs,
    "F16": f16_outputs,
    "Q4_K_M": q4_outputs,
    "Q5_K_M": q5_outputs,
}

for label, outputs in all_outputs.items():
    if len(outputs) != len(records):
        raise RuntimeError(
            f"{label}: expected "
            f"{len(records)} outputs, "
            f"found {len(outputs)}"
        )


# =============================================================================
# PER-RECORD METRICS
# =============================================================================

print()
print("=" * 100)
print("COMPUTING METRICS")
print("=" * 100)

model_metric_accumulators = {
    label: {
        "rouge1": [],
        "rouge2": [],
        "valid_format": [],
        "exactly_two": [],
        "distinct": [],
        "repetition": [],
        "exact_reference": [],
    }
    for label in all_outputs
}

pair_names = [
    ("HF", "F16"),
    ("HF", "Q4_K_M"),
    ("HF", "Q5_K_M"),
    ("F16", "Q4_K_M"),
    ("F16", "Q5_K_M"),
    ("Q4_K_M", "Q5_K_M"),
]

pair_accumulators = {
    f"{a}_vs_{b}": {
        "rouge1": [],
        "rouge2": [],
        "exact": [],
    }
    for a, b in pair_names
}

with open(
    FINAL_RECORDS,
    "w",
    encoding="utf-8"
) as output_file:

    for i, record in enumerate(records):
        reference = record["reference"]

        combined = {
            "index": i,
            "prompt": record["prompt"],
            "reference": reference,
            "models": {},
            "pairwise": {},
        }

        # -----------------------------------------------------
        # EACH MODEL VS REFERENCE
        # -----------------------------------------------------

        for label, outputs in all_outputs.items():
            prediction = outputs[i]["output"]

            r1 = rouge_n_f1(
                prediction,
                reference,
                1
            )

            r2 = rouge_n_f1(
                prediction,
                reference,
                2
            )

            vf = valid_format(
                prediction
            )

            e2 = exactly_two(
                prediction
            )

            distinct = distinct_alternatives(
                prediction
            )

            repetition = repetition_detected(
                prediction
            )

            exact_ref = exact_match(
                prediction,
                reference
            )

            model_metric_accumulators[
                label
            ]["rouge1"].append(r1)

            model_metric_accumulators[
                label
            ]["rouge2"].append(r2)

            model_metric_accumulators[
                label
            ]["valid_format"].append(vf)

            model_metric_accumulators[
                label
            ]["exactly_two"].append(e2)

            model_metric_accumulators[
                label
            ]["distinct"].append(distinct)

            model_metric_accumulators[
                label
            ]["repetition"].append(
                repetition
            )

            model_metric_accumulators[
                label
            ]["exact_reference"].append(
                exact_ref
            )

            combined["models"][label] = {
                "output": prediction,
                "rouge1_reference": r1,
                "rouge2_reference": r2,
                "valid_format": vf,
                "exactly_two": e2,
                "distinct_alternatives":
                    distinct,
                "repetition":
                    repetition,
                "exact_reference":
                    exact_ref,
            }

        # -----------------------------------------------------
        # PAIRWISE COMPARISON
        # -----------------------------------------------------

        for a, b in pair_names:
            output_a = (
                all_outputs[a][i]["output"]
            )

            output_b = (
                all_outputs[b][i]["output"]
            )

            r1 = rouge_n_f1(
                output_b,
                output_a,
                1
            )

            r2 = rouge_n_f1(
                output_b,
                output_a,
                2
            )

            exact = exact_match(
                output_a,
                output_b
            )

            key = f"{a}_vs_{b}"

            pair_accumulators[
                key
            ]["rouge1"].append(r1)

            pair_accumulators[
                key
            ]["rouge2"].append(r2)

            pair_accumulators[
                key
            ]["exact"].append(exact)

            combined["pairwise"][key] = {
                "rouge1": r1,
                "rouge2": r2,
                "exact": exact,
            }

        output_file.write(
            json.dumps(
                combined,
                ensure_ascii=False
            )
            +
            "\n"
        )


# =============================================================================
# MODEL SUMMARY
# =============================================================================

model_summary = {}

for label, metrics in (
    model_metric_accumulators.items()
):
    n = len(records)

    model_summary[label] = {
        "average_rouge1_reference":
            average(
                metrics["rouge1"]
            ),

        "average_rouge2_reference":
            average(
                metrics["rouge2"]
            ),

        "valid_format_count":
            sum(
                metrics["valid_format"]
            ),

        "valid_format_percent":
            100
            * sum(
                metrics["valid_format"]
            )
            / n,

        "exactly_two_count":
            sum(
                metrics["exactly_two"]
            ),

        "exactly_two_percent":
            100
            * sum(
                metrics["exactly_two"]
            )
            / n,

        "distinct_alternatives_count":
            sum(
                metrics["distinct"]
            ),

        "distinct_alternatives_percent":
            100
            * sum(
                metrics["distinct"]
            )
            / n,

        "repetition_count":
            sum(
                metrics["repetition"]
            ),

        "exact_reference_count":
            sum(
                metrics["exact_reference"]
            ),
    }


# =============================================================================
# PAIRWISE SUMMARY
# =============================================================================

pairwise_summary = {}

for key, metrics in (
    pair_accumulators.items()
):
    pairwise_summary[key] = {
        "average_rouge1":
            average(
                metrics["rouge1"]
            ),

        "average_rouge2":
            average(
                metrics["rouge2"]
            ),

        "exact_match_count":
            sum(
                metrics["exact"]
            ),

        "exact_match_percent":
            100
            * sum(
                metrics["exact"]
            )
            / len(records),
    }


# =============================================================================
# SIZE SUMMARY
# =============================================================================

sizes = {
    "HF_safetensors":
        file_size_info(
            HF_MODEL_PATH
            /
            "model.safetensors"
        ),

    "F16":
        file_size_info(
            GGUF_MODELS["F16"]
        ),

    "Q4_K_M":
        file_size_info(
            GGUF_MODELS["Q4_K_M"]
        ),

    "Q5_K_M":
        file_size_info(
            GGUF_MODELS["Q5_K_M"]
        ),
}

f16_bytes = sizes["F16"]["bytes"]

sizes["Q4_K_M"][
    "reduction_vs_F16_percent"
] = (
    100
    *
    (
        1
        -
        sizes["Q4_K_M"]["bytes"]
        /
        f16_bytes
    )
)

sizes["Q5_K_M"][
    "reduction_vs_F16_percent"
] = (
    100
    *
    (
        1
        -
        sizes["Q5_K_M"]["bytes"]
        /
        f16_bytes
    )
)


# =============================================================================
# FINAL SUMMARY JSON
# =============================================================================

summary = {
    "experiment":
        "Definitive HF vs F16 vs Q4_K_M vs Q5_K_M validation",

    "test_file":
        str(TEST_FILE),

    "examples_tested":
        len(records),

    "prompt_protocol": {
        "final_boundary":
            "\\n\\n",

        "gguf_transport":
            "raw llama-server /completion JSON prompt; no -f",

        "chat_template":
            False,
    },

    "generation_parameters": {
        "strategy":
            "greedy",

        "max_new_tokens":
            MAX_NEW_TOKENS,

        "temperature":
            0.0,

        "top_k":
            1,

        "top_p":
            1.0,

        "min_p":
            0.0,

        "repetition_penalty":
            REPETITION_PENALTY,

        "seed":
            SEED,
    },

    "model_metrics":
        model_summary,

    "pairwise_similarity":
        pairwise_summary,

    "file_sizes":
        sizes,
}


with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=4,
        ensure_ascii=False
    )


# =============================================================================
# CSV SUMMARY
# =============================================================================

with open(
    MODEL_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "model",
        "avg_rouge1_reference",
        "avg_rouge2_reference",
        "valid_format_percent",
        "exactly_two_percent",
        "distinct_alternatives_percent",
        "repetition_count",
        "exact_reference_count",
    ])

    for label in [
        "HF",
        "F16",
        "Q4_K_M",
        "Q5_K_M",
    ]:
        m = model_summary[label]

        writer.writerow([
            label,
            m["average_rouge1_reference"],
            m["average_rouge2_reference"],
            m["valid_format_percent"],
            m["exactly_two_percent"],
            m["distinct_alternatives_percent"],
            m["repetition_count"],
            m["exact_reference_count"],
        ])


with open(
    PAIRWISE_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "comparison",
        "avg_rouge1",
        "avg_rouge2",
        "exact_match_count",
        "exact_match_percent",
    ])

    for key, m in (
        pairwise_summary.items()
    ):
        writer.writerow([
            key,
            m["average_rouge1"],
            m["average_rouge2"],
            m["exact_match_count"],
            m["exact_match_percent"],
        ])


# =============================================================================
# CONSOLE REPORT
# =============================================================================

print()
print()
print("=" * 100)
print("FINAL DEFINITIVE SUMMARY")
print("=" * 100)

print(
    "Examples tested:",
    len(records)
)

print()
print(
    "MODEL QUALITY VS REFERENCE"
)
print("-" * 100)

for label in [
    "HF",
    "F16",
    "Q4_K_M",
    "Q5_K_M",
]:
    m = model_summary[label]

    print()
    print(label)

    print(
        "  ROUGE-1:",
        f'{m["average_rouge1_reference"]:.4f}'
    )

    print(
        "  ROUGE-2:",
        f'{m["average_rouge2_reference"]:.4f}'
    )

    print(
        "  Valid format:",
        f'{m["valid_format_percent"]:.1f}%'
    )

    print(
        "  Exactly two:",
        f'{m["exactly_two_percent"]:.1f}%'
    )

    print(
        "  Distinct alternatives:",
        f'{m["distinct_alternatives_percent"]:.1f}%'
    )

    print(
        "  Repetition problems:",
        m["repetition_count"]
    )

    print(
        "  Exact reference:",
        m["exact_reference_count"],
        "/",
        len(records)
    )


print()
print(
    "PAIRWISE SIMILARITY"
)
print("-" * 100)

for key, m in pairwise_summary.items():
    print(
        f"{key:20s} "
        f"R1={m['average_rouge1']:.4f} "
        f"R2={m['average_rouge2']:.4f} "
        f"Exact={m['exact_match_count']}/{len(records)}"
    )


print()
print(
    "MODEL FILE SIZES"
)
print("-" * 100)

for label in [
    "HF_safetensors",
    "F16",
    "Q4_K_M",
    "Q5_K_M",
]:
    s = sizes[label]

    print(
        f"{label:16s}: "
        f"{s['MiB']:.2f} MiB"
    )

if (
    "reduction_vs_F16_percent"
    in sizes["Q4_K_M"]
):
    print(
        "Q4 reduction vs F16:",
        f'{sizes["Q4_K_M"]["reduction_vs_F16_percent"]:.2f}%'
    )

    print(
        "Q5 reduction vs F16:",
        f'{sizes["Q5_K_M"]["reduction_vs_F16_percent"]:.2f}%'
    )


print()
print(
    "Outputs:"
)

print(
    " ",
    FINAL_RECORDS
)

print(
    " ",
    SUMMARY_JSON
)

print(
    " ",
    MODEL_CSV
)

print(
    " ",
    PAIRWISE_CSV
)

print("=" * 100)
