import argparse
import json
import math
import re
import subprocess
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# PATHS / CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = ROOT / "data_GptPlus" / "splits" / "test.jsonl"
SFT_PATH = ROOT / "data_GptPlus" / "processed" / "test_sft.jsonl"

HF_PRED_PATH = (
    ROOT
    / "evaluation"
    / "results_sft_prompt_final"
    / "predictions"
    / "predictions_lfm2_700m_gptplus_ds_on_gptplus_test.jsonl"
)

LLAMA_COMPLETION = (
    Path.home()
    / "llama.cpp"
    / "build"
    / "bin"
    / "llama-completion"
)

MODELS = {
    "Q4_K_M": (
        ROOT
        / "models"
        / "gguf"
        / "LFM2-700M_GPTPlus-DS_Q4_K_M.gguf"
    ),
    "Q5_K_M": (
        ROOT
        / "models"
        / "gguf"
        / "LFM2-700M_GPTPlus-DS_Q5_K_M.gguf"
    ),
}

OUTPUT_DIR = (
    ROOT
    / "evaluation"
    / "full_quantization_validation"
)

DETAILS_PATH = OUTPUT_DIR / "q4_vs_q5_500_records.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "q4_vs_q5_500_summary.json"

SEED = 42
MAX_NEW_TOKENS = 64
CTX_SIZE = 512

ACTIVITIES = (
    "GENERATION",
    "COMPLETION",
    "NORMALIZATION",
)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Full 500-record post-quantization validation: "
            "Q4_K_M vs Q5_K_M."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite previous validation outputs.",
    )

    return parser.parse_args()


# ============================================================
# IO
# ============================================================

def read_jsonl(path):
    records = []

    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} "
                    f"at line {line_number}: {exc}"
                ) from exc

    return records


def write_json(path, obj):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    text = unicodedata.normalize(
        "NFKC",
        str(text or ""),
    ).casefold()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    return " ".join(text.split())


def clean_cli_output(text):
    return (
        str(text or "")
        .replace("[end of text]", "")
        .strip()
    )


ALT_LINE_RE = re.compile(
    r"^\s*([12])\s*[\.\):\-]\s*(.+?)\s*$",
    flags=re.IGNORECASE,
)


# ============================================================
# STRICT DIAGNOSTIC PARSER
# ============================================================

def parse_diagnostic_output(raw_text):
    raw_text = str(raw_text or "")

    eos_marker_found = "[end of text]" in raw_text

    text = clean_cli_output(raw_text)

    numbered = []
    extra_lines = []

    nonempty_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    for line in nonempty_lines:
        match = ALT_LINE_RE.match(line)

        if match:
            numbered.append(
                {
                    "number": int(match.group(1)),
                    "text": match.group(2).strip(),
                }
            )
        else:
            extra_lines.append(line)

    alternatives = [
        item["text"]
        for item in numbered
    ]

    numbers = [
        item["number"]
        for item in numbered
    ]

    exactly_two_numbered = (
        len(numbered) == 2
        and numbers == [1, 2]
    )

    normalized_lines = [
        normalize_text(line)
        for line in nonempty_lines
        if normalize_text(line)
    ]

    line_counter = Counter(normalized_lines)

    max_identical_line_count = (
        max(line_counter.values())
        if line_counter
        else 0
    )

    repetitive_line_loop = (
        max_identical_line_count >= 3
    )

    return {
        "clean_output": text,
        "alternatives": alternatives,
        "numbered_labels": numbers,
        "extra_lines": extra_lines,
        "exactly_two_numbered": exactly_two_numbered,
        "has_extra_text": bool(extra_lines),
        "eos_marker_found": eos_marker_found,
        "no_eos_marker": not eos_marker_found,
        "repetitive_line_loop": repetitive_line_loop,
        "max_identical_line_count":
            max_identical_line_count,
    }


# ============================================================
# PAIR / FORMAT CHECKS
# ============================================================

def pair_equal(a, b):
    if len(a) != 2 or len(b) != 2:
        return False

    a_norm = sorted(
        normalize_text(x)
        for x in a
    )

    b_norm = sorted(
        normalize_text(x)
        for x in b
    )

    return a_norm == b_norm


def two_distinct(alternatives):
    return (
        len(alternatives) == 2
        and normalize_text(alternatives[0])
        and normalize_text(alternatives[1])
        and normalize_text(alternatives[0])
        != normalize_text(alternatives[1])
    )


# ============================================================
# TASK-SPECIFIC CHECKS
# ============================================================

def completion_prefix_preserved(
    record,
    alternatives,
):
    if record["activity_type"] != "COMPLETION":
        return None

    partial = normalize_text(
        record.get("input_text")
    )

    if not partial:
        return False

    return any(
        normalize_text(alt).startswith(partial)
        for alt in alternatives
    )


def beneficiary_repeated(
    record,
    alternatives,
):
    beneficiary = normalize_text(
        record.get("beneficiary")
    )

    if not beneficiary:
        return False

    for alternative in alternatives:
        normalized = normalize_text(
            alternative
        )

        if normalized.count(beneficiary) > 1:
            return True

    return False


# ============================================================
# STRUCTURED CONSISTENCY
# Mirrors the checker used in final HF evaluation.
# ============================================================

MONTH_TO_NUMBER = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

MONTH_PATTERN = re.compile(
    r"\b("
    + "|".join(
        sorted(
            MONTH_TO_NUMBER,
            key=len,
            reverse=True,
        )
    )
    + r")\b",
    flags=re.IGNORECASE,
)


def extract_temporal_markers(text):
    text = str(text or "")
    low = text.lower()

    markers = set()

    for year in re.findall(
        r"\b20\d{2}\b",
        low,
    ):
        markers.add(
            f"year:{year}"
        )

    for quarter in re.findall(
        r"\bq([1-4])\b",
        low,
    ):
        markers.add(
            f"quarter:{quarter}"
        )

    for match in MONTH_PATTERN.finditer(low):
        markers.add(
            "month:"
            + str(
                MONTH_TO_NUMBER[
                    match.group(1).lower()
                ]
            )
        )

    installment_patterns = {
        "installment:first":
            r"\b(?:first|1st)\s+installment\b",

        "installment:second":
            r"\b(?:second|2nd)\s+installment\b",

        "installment:third":
            r"\b(?:third|3rd)\s+installment\b",

        "installment:final":
            r"\bfinal\s+installment\b",
    }

    for marker, pattern in installment_patterns.items():
        if re.search(pattern, low):
            markers.add(marker)

    for match in re.finditer(
        r"\b(20\d{2})-"
        r"(0[1-9]|1[0-2])-"
        r"(0[1-9]|[12]\d|3[01])\b",
        low,
    ):
        markers.add(
            f"year:{match.group(1)}"
        )

        markers.add(
            f"month:{int(match.group(2))}"
        )

    return markers


def source_text_for_record(record):
    pieces = [
        record.get("operation_category"),
        record.get("beneficiary"),
        record.get("reference_period"),
        record.get("input_text"),
    ]

    calendar = record.get(
        "calendar_context"
    )

    if isinstance(calendar, dict):
        pieces.extend(
            [
                calendar.get("event_title"),
                calendar.get("event_date"),
                calendar.get("event_category"),
            ]
        )

    return " ".join(
        str(x)
        for x in pieces
        if x is not None
    )


def monetary_mentions(text):
    text = str(text or "")
    mentions = []

    currency_first = re.compile(
        r"\b(EUR|USD|GBP)\s*"
        r"([0-9]+(?:[.,][0-9]{1,2})?)\b",
        flags=re.IGNORECASE,
    )

    amount_first = re.compile(
        r"\b([0-9]+(?:[.,][0-9]{1,2})?)"
        r"\s*(EUR|USD|GBP)\b",
        flags=re.IGNORECASE,
    )

    for currency, amount in currency_first.findall(
        text
    ):
        mentions.append(
            (
                currency.upper(),
                float(
                    amount.replace(",", ".")
                ),
            )
        )

    for amount, currency in amount_first.findall(
        text
    ):
        mentions.append(
            (
                currency.upper(),
                float(
                    amount.replace(",", ".")
                ),
            )
        )

    return mentions


def structured_consistency(
    record,
    alternatives,
):
    prediction_text = "\n".join(
        alternatives
    )

    expected_currency = str(
        record["currency"]
    ).upper()

    expected_amount = float(
        record["amount"]
    )

    currencies = {
        x.upper()
        for x in re.findall(
            r"\b(?:EUR|USD|GBP)\b",
            prediction_text,
            flags=re.IGNORECASE,
        )
    }

    currency_consistent = all(
        currency == expected_currency
        for currency in currencies
    )

    amount_consistent = True

    for currency, amount in monetary_mentions(
        prediction_text
    ):
        if (
            currency != expected_currency
            or not math.isclose(
                amount,
                expected_amount,
                abs_tol=0.011,
            )
        ):
            amount_consistent = False
            break

    allowed_temporal = (
        extract_temporal_markers(
            source_text_for_record(
                record
            )
        )
    )

    predicted_temporal = (
        extract_temporal_markers(
            prediction_text
        )
    )

    temporal_consistent = (
        predicted_temporal.issubset(
            allowed_temporal
        )
    )

    automatic_pass = (
        bool(alternatives)
        and currency_consistent
        and amount_consistent
        and temporal_consistent
    )

    return {
        "currency_consistent":
            currency_consistent,

        "amount_consistent":
            amount_consistent,

        "temporal_consistent":
            temporal_consistent,

        "automatic_structured_consistency_pass":
            automatic_pass,
    }


# ============================================================
# LLAMA.CPP GENERATION
# ============================================================

def generate(
    model_path,
    prompt,
):
    # Exact same SFT boundary used in HF evaluation.
    prompt_text = (
        prompt.rstrip("\r\n")
        + "\n\n"
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        encoding="utf-8",
        delete=False,
    ) as tmp:

        tmp.write(prompt_text)
        tmp_path = Path(tmp.name)

    command = [
        str(LLAMA_COMPLETION),

        "-m",
        str(model_path),

        "-f",
        str(tmp_path),

        "-n",
        str(MAX_NEW_TOKENS),

        "-c",
        str(CTX_SIZE),

        "--temp",
        "0",

        "--seed",
        str(SEED),

        "-no-cnv",

        "--no-display-prompt",

        "--color",
        "off",
    ]

    start = time.perf_counter()

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    finally:
        tmp_path.unlink(
            missing_ok=True
        )

    runtime = (
        time.perf_counter()
        - start
    )

    if result.returncode != 0:
        raise RuntimeError(
            "\nllama-completion failed\n"
            f"Model: {model_path}\n"
            f"Return code: {result.returncode}\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout, runtime


# ============================================================
# EVALUATE ONE OUTPUT
# ============================================================

def evaluate_output(
    record,
    parsed,
    hf_alternatives,
):
    alternatives = parsed[
        "alternatives"
    ]

    result = {
        **parsed,

        "two_distinct_alternatives":
            two_distinct(
                alternatives
            ),

        "hf_pair_match":
            pair_equal(
                alternatives,
                hf_alternatives,
            ),

        "completion_prefix_preserved":
            completion_prefix_preserved(
                record,
                alternatives,
            ),

        "beneficiary_repeated":
            beneficiary_repeated(
                record,
                alternatives,
            ),
    }

    result.update(
        structured_consistency(
            record,
            alternatives,
        )
    )

    return result


# ============================================================
# AGGREGATION
# ============================================================

def rate(values):
    values = list(values)

    if not values:
        return None

    return (
        sum(
            1
            for value in values
            if value
        )
        / len(values)
    )


def summarize_model(
    records,
    model_key,
    activity=None,
):
    subset = [
        record
        for record in records
        if (
            activity is None
            or record["activity_type"]
            == activity
        )
    ]

    outputs = [
        record[model_key]
        for record in subset
    ]

    completion_values = [
        output[
            "completion_prefix_preserved"
        ]
        for output in outputs
        if output[
            "completion_prefix_preserved"
        ] is not None
    ]

    def count_true(key):
        return sum(
            1
            for output in outputs
            if output[key]
        )

    return {
        "records":
            len(outputs),

        "exactly_two_numbered_count":
            count_true(
                "exactly_two_numbered"
            ),

        "exactly_two_numbered_rate":
            rate(
                output[
                    "exactly_two_numbered"
                ]
                for output in outputs
            ),

        "no_extra_text_count":
            sum(
                1
                for output in outputs
                if not output[
                    "has_extra_text"
                ]
            ),

        "no_extra_text_rate":
            rate(
                not output[
                    "has_extra_text"
                ]
                for output in outputs
            ),

        "two_distinct_alternatives_count":
            count_true(
                "two_distinct_alternatives"
            ),

        "two_distinct_alternatives_rate":
            rate(
                output[
                    "two_distinct_alternatives"
                ]
                for output in outputs
            ),

        "hf_pair_match_count":
            count_true(
                "hf_pair_match"
            ),

        "hf_pair_match_rate":
            rate(
                output[
                    "hf_pair_match"
                ]
                for output in outputs
            ),

        "structured_consistency_count":
            count_true(
                "automatic_structured_consistency_pass"
            ),

        "structured_consistency_rate":
            rate(
                output[
                    "automatic_structured_consistency_pass"
                ]
                for output in outputs
            ),

        "completion_prefix_preservation_count":
            (
                sum(
                    1
                    for value
                    in completion_values
                    if value
                )
                if completion_values
                else None
            ),

        "completion_prefix_preservation_rate":
            (
                rate(
                    completion_values
                )
                if completion_values
                else None
            ),

        "beneficiary_repetition_count":
            count_true(
                "beneficiary_repeated"
            ),

        "beneficiary_repetition_rate":
            rate(
                output[
                    "beneficiary_repeated"
                ]
                for output in outputs
            ),

        "repetitive_line_loop_count":
            count_true(
                "repetitive_line_loop"
            ),

        "repetitive_line_loop_rate":
            rate(
                output[
                    "repetitive_line_loop"
                ]
                for output in outputs
            ),

        "eos_marker_count":
            count_true(
                "eos_marker_found"
            ),

        "eos_marker_rate":
            rate(
                output[
                    "eos_marker_found"
                ]
                for output in outputs
            ),

        "mean_runtime_seconds":
            (
                sum(
                    output[
                        "runtime_seconds"
                    ]
                    for output in outputs
                )
                / len(outputs)
                if outputs
                else None
            ),
    }


# ============================================================
# CROSS-QUANTIZATION COMPARISON
# ============================================================

def build_comparison(
    q4,
    q5,
):
    return {
        "q5_pair_matches_q4":
            pair_equal(
                q5["alternatives"],
                q4["alternatives"],
            ),

        # Q4 failed, Q5 recovered.
        "q5_recovers_q4_two_alternative_failure":
            (
                not q4[
                    "exactly_two_numbered"
                ]
                and q5[
                    "exactly_two_numbered"
                ]
            ),

        "q5_recovers_q4_structured_failure":
            (
                not q4[
                    "automatic_structured_consistency_pass"
                ]
                and q5[
                    "automatic_structured_consistency_pass"
                ]
            ),

        "q5_recovers_q4_beneficiary_repetition":
            (
                q4[
                    "beneficiary_repeated"
                ]
                and not q5[
                    "beneficiary_repeated"
                ]
            ),

        "q5_recovers_q4_format_failure":
            (
                q4[
                    "has_extra_text"
                ]
                and not q5[
                    "has_extra_text"
                ]
            ),

        "q5_recovers_q4_repetitive_loop":
            (
                q4[
                    "repetitive_line_loop"
                ]
                and not q5[
                    "repetitive_line_loop"
                ]
            ),

        "q5_recovers_q4_no_eos":
            (
                q4[
                    "no_eos_marker"
                ]
                and q5[
                    "eos_marker_found"
                ]
            ),

        # Q4 passed, Q5 newly failed.
        "q5_new_two_alternative_failure":
            (
                q4[
                    "exactly_two_numbered"
                ]
                and not q5[
                    "exactly_two_numbered"
                ]
            ),

        "q5_new_structured_failure":
            (
                q4[
                    "automatic_structured_consistency_pass"
                ]
                and not q5[
                    "automatic_structured_consistency_pass"
                ]
            ),

        "q5_new_beneficiary_repetition":
            (
                not q4[
                    "beneficiary_repeated"
                ]
                and q5[
                    "beneficiary_repeated"
                ]
            ),

        "q5_new_format_regression":
            (
                not q4[
                    "has_extra_text"
                ]
                and q5[
                    "has_extra_text"
                ]
            ),

        "q5_new_repetitive_loop":
            (
                not q4[
                    "repetitive_line_loop"
                ]
                and q5[
                    "repetitive_line_loop"
                ]
            ),

        "q5_new_no_eos":
            (
                q4[
                    "eos_marker_found"
                ]
                and q5[
                    "no_eos_marker"
                ]
            ),
    }


def ids_for(
    records,
    comparison_key,
):
    return [
        record["id_example"]
        for record in records
        if record["comparison"][
            comparison_key
        ]
    ]


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    required_paths = [
        RAW_PATH,
        SFT_PATH,
        HF_PRED_PATH,
        LLAMA_COMPLETION,
        *MODELS.values(),
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.overwrite:
        existing = [
            path
            for path in (
                DETAILS_PATH,
                SUMMARY_PATH,
            )
            if path.exists()
        ]

        if existing:
            raise FileExistsError(
                "Output already exists. "
                "Use --overwrite.\n"
                + "\n".join(
                    str(path)
                    for path in existing
                )
            )

    raw_records = read_jsonl(
        RAW_PATH
    )

    sft_records = read_jsonl(
        SFT_PATH
    )

    hf_records = read_jsonl(
        HF_PRED_PATH
    )

    if not (
        len(raw_records)
        == len(sft_records)
        == len(hf_records)
        == 500
    ):
        raise ValueError(
            "Expected exactly 500 aligned "
            "raw/SFT/HF test records. "
            f"Found raw={len(raw_records)}, "
            f"sft={len(sft_records)}, "
            f"hf={len(hf_records)}."
        )

    activity_counts = Counter(
        record["activity_type"]
        for record in raw_records
    )

    print("=" * 90)
    print("FULL QUANTIZATION VALIDATION")
    print("=" * 90)
    print("Records:", len(raw_records))
    print(
        "Activities:",
        dict(activity_counts),
    )
    print(
        "Models: Q4_K_M vs Q5_K_M"
    )
    print(
        "HF merged predictions used "
        "as frozen behavioral reference."
    )
    print()

    scored_records = []

    for index, (
        raw,
        sft,
        hf,
    ) in enumerate(
        zip(
            raw_records,
            sft_records,
            hf_records,
        ),
        start=1,
    ):

        example_id = raw[
            "id_example"
        ]

        if hf[
            "id_example"
        ] != example_id:
            raise ValueError(
                "HF/raw alignment mismatch: "
                f"{hf['id_example']} "
                f"!= {example_id}"
            )

        activity = raw[
            "activity_type"
        ]

        print(
            f"[{index:03d}/500] "
            f"{activity:<13} "
            f"{example_id}",
            flush=True,
        )

        hf_alternatives = hf[
            "predicted_alternatives"
        ]

        item = {
            "id_example":
                example_id,

            "activity_type":
                activity,

            "operation_category":
                raw[
                    "operation_category"
                ],

            "prompt_source_record": {
                "beneficiary":
                    raw.get(
                        "beneficiary"
                    ),

                "amount":
                    raw.get(
                        "amount"
                    ),

                "currency":
                    raw.get(
                        "currency"
                    ),

                "reference_period":
                    raw.get(
                        "reference_period"
                    ),

                "input_text":
                    raw.get(
                        "input_text"
                    ),

                "calendar_context":
                    raw.get(
                        "calendar_context"
                    ),
            },

            "hf_merged": {
                "raw_prediction":
                    hf[
                        "raw_prediction"
                    ],

                "alternatives":
                    hf_alternatives,
            },
        }

        for (
            model_key,
            model_path,
        ) in MODELS.items():

            print(
                f"          -> "
                f"{model_key}...",
                end="",
                flush=True,
            )

            raw_output, runtime = generate(
                model_path,
                sft["prompt"],
            )

            parsed = (
                parse_diagnostic_output(
                    raw_output
                )
            )

            evaluated = evaluate_output(
                raw,
                parsed,
                hf_alternatives,
            )

            evaluated[
                "runtime_seconds"
            ] = runtime

            item[
                model_key
            ] = evaluated

            print(
                f" {runtime:.2f}s",
                flush=True,
            )

        item[
            "comparison"
        ] = build_comparison(
            item["Q4_K_M"],
            item["Q5_K_M"],
        )

        scored_records.append(
            item
        )

    # ========================================================
    # SAVE DETAILS
    # ========================================================

    with DETAILS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in scored_records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    comparison_keys = [
        "q5_recovers_q4_two_alternative_failure",
        "q5_recovers_q4_structured_failure",
        "q5_recovers_q4_beneficiary_repetition",
        "q5_recovers_q4_format_failure",
        "q5_recovers_q4_repetitive_loop",
        "q5_recovers_q4_no_eos",

        "q5_new_two_alternative_failure",
        "q5_new_structured_failure",
        "q5_new_beneficiary_repetition",
        "q5_new_format_regression",
        "q5_new_repetitive_loop",
        "q5_new_no_eos",
    ]

    q4_q5_summary = {
        "q5_pair_matches_q4_count":
            sum(
                1
                for record
                in scored_records
                if record[
                    "comparison"
                ][
                    "q5_pair_matches_q4"
                ]
            ),

        "q5_pair_matches_q4_rate":
            rate(
                record[
                    "comparison"
                ][
                    "q5_pair_matches_q4"
                ]
                for record
                in scored_records
            ),
    }

    for key in comparison_keys:
        ids = ids_for(
            scored_records,
            key,
        )

        q4_q5_summary[
            key + "_count"
        ] = len(ids)

        q4_q5_summary[
            key + "_ids"
        ] = ids

    summary = {
        "validation_type":
            "full_500_record_quantization_comparison",

        "records_total":
            len(scored_records),

        "activity_distribution":
            dict(activity_counts),

        "generation": {
            "max_new_tokens":
                MAX_NEW_TOKENS,

            "ctx_size":
                CTX_SIZE,

            "temperature":
                0,

            "seed":
                SEED,

            "conversation_mode":
                False,

            "prompt_boundary":
                'prompt.rstrip("\\r\\n") + "\\n\\n"',
        },

        "model_files": {
            model_key: {
                "path":
                    str(model_path),

                "size_bytes":
                    model_path.stat().st_size,

                "size_mib":
                    (
                        model_path.stat().st_size
                        / 1024
                        / 1024
                    ),
            }
            for (
                model_key,
                model_path,
            ) in MODELS.items()
        },

        "models": {
            model_key: {
                "overall":
                    summarize_model(
                        scored_records,
                        model_key,
                    ),

                "by_activity": {
                    activity:
                        summarize_model(
                            scored_records,
                            model_key,
                            activity,
                        )
                    for activity
                    in ACTIVITIES
                },
            }
            for model_key
            in MODELS
        },

        "q4_vs_q5":
            q4_q5_summary,

        "notes": [
            (
                "HF pair match measures behavioral "
                "stability relative to the frozen "
                "HF prediction, not semantic quality."
            ),
            (
                "Structured consistency checks "
                "currency, amount and temporal "
                "consistency; it is not a complete "
                "semantic correctness metric."
            ),
            (
                "repetitive_line_loop is a "
                "diagnostic heuristic triggered "
                "when an identical non-empty line "
                "appears at least three times."
            ),
            (
                "eos_marker_found is based on the "
                "llama-completion '[end of text]' "
                "marker and is used only as a "
                "runtime diagnostic."
            ),
            (
                "Runtime includes model loading "
                "for every subprocess and must not "
                "be treated as the final Android "
                "inference benchmark."
            ),
        ],
    }

    write_json(
        SUMMARY_PATH,
        summary,
    )

    print(
        "\n"
        + "=" * 90
    )
    print(
        "FULL VALIDATION COMPLETED"
    )
    print(
        "=" * 90
    )

    print(
        "\nQ4_K_M overall:"
    )
    print(
        json.dumps(
            summary[
                "models"
            ][
                "Q4_K_M"
            ][
                "overall"
            ],
            indent=2,
        )
    )

    print(
        "\nQ5_K_M overall:"
    )
    print(
        json.dumps(
            summary[
                "models"
            ][
                "Q5_K_M"
            ][
                "overall"
            ],
            indent=2,
        )
    )

    print(
        "\nQ4 vs Q5:"
    )
    print(
        json.dumps(
            summary[
                "q4_vs_q5"
            ],
            indent=2,
        )
    )

    print(
        "\nDetailed records:"
    )
    print(
        DETAILS_PATH
    )

    print(
        "\nSummary:"
    )
    print(
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()
