import argparse
import json
import math
import random
import re
import subprocess
import tempfile
import time
import unicodedata
from collections import defaultdict
from pathlib import Path


# ============================================================
# PATHS / CONSTANTS
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
    Path.home() / "llama.cpp" / "build" / "bin" / "llama-completion"
)

MODELS = {
    "F16": ROOT / "models" / "gguf" / "LFM2-700M_GPTPlus-DS_F16.gguf",
    "Q4_K_M": ROOT / "models" / "gguf" / "LFM2-700M_GPTPlus-DS_Q4_K_M.gguf",
    "Q5_K_M": ROOT / "models" / "gguf" / "LFM2-700M_GPTPlus-DS_Q5_K_M.gguf",
}

OUTPUT_DIR = ROOT / "evaluation" / "post_quantization_validation"
DETAILS_PATH = OUTPUT_DIR / "post_quantization_30_records.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "post_quantization_summary.json"

SEED = 42
N_PER_ACTIVITY = 10
MAX_NEW_TOKENS = 64
CTX_SIZE = 512

ACTIVITIES = ("GENERATION", "COMPLETION", "NORMALIZATION")


# ============================================================
# IO
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-quantization validation of GGUF F16 vs Q4_K_M vs Q5_K_M."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite previous validation outputs.",
    )
    return parser.parse_args()


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ============================================================
# NORMALIZATION / PARSING
# ============================================================

def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def clean_cli_output(text):
    text = str(text or "")
    text = text.replace("[end of text]", "")
    return text.strip()


ALT_LINE_RE = re.compile(
    r"^\s*([12])\s*[\.\):\-]\s*(.+?)\s*$",
    flags=re.IGNORECASE,
)


def parse_diagnostic_output(text):
    """
    Strict diagnostic parser.

    Numbered alternatives are captured separately from any extra
    generated text such as:
        Category: ...
        Beneficiary: ...

    This deliberately differs from the frozen HF evaluation parser,
    because here extra prefix text is itself something we want to detect.
    """
    text = clean_cli_output(text)

    alternatives = []
    extra_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        match = ALT_LINE_RE.match(line)

        if match:
            alternatives.append(match.group(2).strip())
        else:
            extra_lines.append(line)

    return {
        "clean_output": text,
        "alternatives": alternatives,
        "extra_lines": extra_lines,
        "exactly_two_numbered": len(alternatives) == 2,
        "has_extra_text": bool(extra_lines),
    }


def pair_equal(a, b):
    if len(a) != 2 or len(b) != 2:
        return False

    na = sorted(normalize_text(x) for x in a)
    nb = sorted(normalize_text(x) for x in b)

    return na == nb


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

def completion_prefix_preserved(record, alternatives):
    if record["activity_type"] != "COMPLETION":
        return None

    partial = normalize_text(record.get("input_text"))

    if not partial:
        return False

    return any(
        normalize_text(alt).startswith(partial)
        for alt in alternatives
    )


def beneficiary_repeated(record, alternatives):
    beneficiary = normalize_text(record.get("beneficiary"))

    if not beneficiary:
        return False

    for alt in alternatives:
        normalized = normalize_text(alt)

        if normalized.count(beneficiary) > 1:
            return True

    return False


# ============================================================
# STRUCTURED CONSISTENCY
# Mirrors the logic used by the final HF evaluator.
# ============================================================

MONTH_TO_NUMBER = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

MONTH_PATTERN = re.compile(
    r"\b("
    + "|".join(sorted(MONTH_TO_NUMBER, key=len, reverse=True))
    + r")\b",
    flags=re.IGNORECASE,
)


def extract_temporal_markers(text):
    text = str(text or "")
    low = text.lower()
    markers = set()

    for year in re.findall(r"\b20\d{2}\b", low):
        markers.add(f"year:{year}")

    for quarter in re.findall(r"\bq([1-4])\b", low):
        markers.add(f"quarter:{quarter}")

    for match in MONTH_PATTERN.finditer(low):
        markers.add(
            f"month:{MONTH_TO_NUMBER[match.group(1).lower()]}"
        )

    for match in re.finditer(
        r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b",
        low,
    ):
        markers.add(f"year:{match.group(1)}")
        markers.add(f"month:{int(match.group(2))}")

    return markers


def source_text_for_record(record):
    pieces = [
        record.get("operation_category"),
        record.get("beneficiary"),
        record.get("reference_period"),
        record.get("input_text"),
    ]

    calendar = record.get("calendar_context")

    if isinstance(calendar, dict):
        pieces.extend(
            [
                calendar.get("event_title"),
                calendar.get("event_date"),
                calendar.get("event_category"),
            ]
        )

    return " ".join(str(x) for x in pieces if x is not None)


def monetary_mentions(text):
    text = str(text)
    mentions = []

    currency_first = re.compile(
        r"\b(EUR|USD|GBP)\s*([0-9]+(?:[.,][0-9]{1,2})?)\b",
        flags=re.IGNORECASE,
    )

    amount_first = re.compile(
        r"\b([0-9]+(?:[.,][0-9]{1,2})?)\s*(EUR|USD|GBP)\b",
        flags=re.IGNORECASE,
    )

    for currency, amount in currency_first.findall(text):
        mentions.append(
            (currency.upper(), float(amount.replace(",", ".")))
        )

    for amount, currency in amount_first.findall(text):
        mentions.append(
            (currency.upper(), float(amount.replace(",", ".")))
        )

    return mentions


def structured_consistency(record, alternatives):
    prediction_text = "\n".join(alternatives)

    expected_currency = str(record["currency"]).upper()
    expected_amount = float(record["amount"])

    currencies = {
        x.upper()
        for x in re.findall(
            r"\b(?:EUR|USD|GBP)\b",
            prediction_text,
            flags=re.IGNORECASE,
        )
    }

    currency_consistent = all(
        x == expected_currency for x in currencies
    )

    amount_consistent = True

    for currency, amount in monetary_mentions(prediction_text):
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

    allowed_temporal = extract_temporal_markers(
        source_text_for_record(record)
    )

    predicted_temporal = extract_temporal_markers(
        prediction_text
    )

    temporal_consistent = predicted_temporal.issubset(
        allowed_temporal
    )

    automatic_pass = (
        bool(alternatives)
        and currency_consistent
        and amount_consistent
        and temporal_consistent
    )

    return {
        "currency_consistent": currency_consistent,
        "amount_consistent": amount_consistent,
        "temporal_consistent": temporal_consistent,
        "automatic_structured_consistency_pass": automatic_pass,
    }


# ============================================================
# LLAMA.CPP GENERATION
# ============================================================

def generate(model_path, prompt):
    # Exact prompt interface used by HF evaluation.
    prompt_text = prompt.rstrip("\r\n") + "\n\n"

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
        "-m", str(model_path),
        "-f", str(tmp_path),
        "-n", str(MAX_NEW_TOKENS),
        "-c", str(CTX_SIZE),
        "--temp", "0",
        "--seed", str(SEED),
        "-no-cnv",
        "--no-display-prompt",
        "--color", "off",
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
        tmp_path.unlink(missing_ok=True)

    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        raise RuntimeError(
            f"llama-completion failed for {model_path.name}\n"
            f"Return code: {result.returncode}\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout, elapsed


# ============================================================
# SAMPLE SELECTION
# ============================================================

def select_records(raw, sft, hf):
    if not (len(raw) == len(sft) == len(hf)):
        raise ValueError(
            "Raw, SFT and HF prediction files have different lengths."
        )

    rows = []

    for r, s, p in zip(raw, sft, hf):
        if p["id_example"] != r["id_example"]:
            raise ValueError(
                f"HF/raw alignment mismatch: "
                f"{p['id_example']} != {r['id_example']}"
            )

        rows.append((r, s, p))

    grouped = defaultdict(list)

    for row in rows:
        grouped[row[0]["activity_type"]].append(row)

    rng = random.Random(SEED)
    selected = []

    # GENERATION: deterministic random sample
    selected.extend(
        rng.sample(grouped["GENERATION"], N_PER_ACTIVITY)
    )

    # COMPLETION: force known sensitive case EX3023 + 9 others
    completion_rows = grouped["COMPLETION"]

    ex3023 = [
        row
        for row in completion_rows
        if row[0]["id_example"] == "EX3023"
    ]

    if len(ex3023) != 1:
        raise ValueError("Expected exactly one EX3023 record.")

    completion_others = [
        row
        for row in completion_rows
        if row[0]["id_example"] != "EX3023"
    ]

    selected.extend(
        ex3023
        + rng.sample(
            completion_others,
            N_PER_ACTIVITY - 1,
        )
    )

    # NORMALIZATION: deterministic random sample
    selected.extend(
        rng.sample(grouped["NORMALIZATION"], N_PER_ACTIVITY)
    )

    return selected


# ============================================================
# RECORD EVALUATION
# ============================================================

def evaluate_generation(record, parsed, hf_alternatives):
    alternatives = parsed["alternatives"]

    result = {
        **parsed,
        "two_distinct_alternatives": two_distinct(alternatives),
        "hf_pair_match": pair_equal(
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

    return sum(1 for x in values if x) / len(values)


def summarize_model(records, model_key, activity=None):
    subset = [
        r
        for r in records
        if activity is None or r["activity_type"] == activity
    ]

    outputs = [r[model_key] for r in subset]

    completion_values = [
        x["completion_prefix_preserved"]
        for x in outputs
        if x["completion_prefix_preserved"] is not None
    ]

    return {
        "records": len(outputs),
        "exactly_two_numbered_rate": rate(
            x["exactly_two_numbered"] for x in outputs
        ),
        "no_extra_text_rate": rate(
            not x["has_extra_text"] for x in outputs
        ),
        "two_distinct_alternatives_rate": rate(
            x["two_distinct_alternatives"] for x in outputs
        ),
        "hf_pair_match_rate": rate(
            x["hf_pair_match"] for x in outputs
        ),
        "structured_consistency_rate": rate(
            x["automatic_structured_consistency_pass"]
            for x in outputs
        ),
        "completion_prefix_preservation_rate": (
            rate(completion_values)
            if completion_values
            else None
        ),
        "beneficiary_repetition_rate": rate(
            x["beneficiary_repeated"] for x in outputs
        ),
        "mean_runtime_seconds": (
            sum(x["runtime_seconds"] for x in outputs)
            / len(outputs)
            if outputs
            else None
        ),
    }


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
            raise FileNotFoundError(path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.overwrite:
        existing = [
            p for p in (DETAILS_PATH, SUMMARY_PATH)
            if p.exists()
        ]

        if existing:
            raise FileExistsError(
                "Validation output already exists. "
                "Use --overwrite to replace it:\n"
                + "\n".join(str(x) for x in existing)
            )

    raw = read_jsonl(RAW_PATH)
    sft = read_jsonl(SFT_PATH)
    hf = read_jsonl(HF_PRED_PATH)

    selected = select_records(raw, sft, hf)

    print("=" * 88)
    print("POST-QUANTIZATION VALIDATION")
    print("=" * 88)
    print("30 records: 10 GENERATION + 10 COMPLETION + 10 NORMALIZATION")
    print("Seed:", SEED)
    print("Known sensitive case EX3023: INCLUDED")
    print()

    scored_records = []

    for index, (record, sft_record, hf_record) in enumerate(
        selected,
        start=1,
    ):
        example_id = record["id_example"]
        activity = record["activity_type"]

        print(
            f"[{index:02d}/30] "
            f"{activity:<13} {example_id}"
        )

        hf_alternatives = hf_record["predicted_alternatives"]

        item = {
            "id_example": example_id,
            "activity_type": activity,
            "operation_category": record["operation_category"],
            "prompt": sft_record["prompt"],
            "prompt_source_record": {
                "beneficiary": record["beneficiary"],
                "amount": record["amount"],
                "currency": record["currency"],
                "reference_period": record["reference_period"],
                "input_text": record["input_text"],
                "calendar_context": record["calendar_context"],
            },
            "hf_merged": {
                "raw_prediction": hf_record["raw_prediction"],
                "alternatives": hf_alternatives,
            },
        }

        for model_key, model_path in MODELS.items():
            print(f"         -> {model_key}...", end="", flush=True)

            raw_output, runtime = generate(
                model_path,
                sft_record["prompt"],
            )

            parsed = parse_diagnostic_output(raw_output)

            evaluated = evaluate_generation(
                record,
                parsed,
                hf_alternatives,
            )

            evaluated["runtime_seconds"] = runtime

            item[model_key] = evaluated

            print(f" {runtime:.2f}s")

        # Direct F16 / Q4 / Q5 comparisons
        item["comparison"] = {
            # ------------------------------------------------
            # Q4 vs F16
            # ------------------------------------------------
            "q4_pair_matches_f16": pair_equal(
                item["Q4_K_M"]["alternatives"],
                item["F16"]["alternatives"],
            ),

            "q4_new_beneficiary_repetition": (
                not item["F16"]["beneficiary_repeated"]
                and item["Q4_K_M"]["beneficiary_repeated"]
            ),

            "q4_new_structured_failure": (
                item["F16"]["automatic_structured_consistency_pass"]
                and not item["Q4_K_M"][
                    "automatic_structured_consistency_pass"
                ]
            ),

            "q4_new_two_alternative_failure": (
                item["F16"]["exactly_two_numbered"]
                and not item["Q4_K_M"]["exactly_two_numbered"]
            ),

            "q4_new_completion_prefix_failure": (
                activity == "COMPLETION"
                and item["F16"]["completion_prefix_preserved"]
                and not item["Q4_K_M"][
                    "completion_prefix_preserved"
                ]
            ),

            "q4_format_improvement": (
                item["F16"]["has_extra_text"]
                and not item["Q4_K_M"]["has_extra_text"]
            ),

            "q4_format_regression": (
                not item["F16"]["has_extra_text"]
                and item["Q4_K_M"]["has_extra_text"]
            ),

            # ------------------------------------------------
            # Q5 vs F16
            # ------------------------------------------------
            "q5_pair_matches_f16": pair_equal(
                item["Q5_K_M"]["alternatives"],
                item["F16"]["alternatives"],
            ),

            "q5_new_beneficiary_repetition": (
                not item["F16"]["beneficiary_repeated"]
                and item["Q5_K_M"]["beneficiary_repeated"]
            ),

            "q5_new_structured_failure": (
                item["F16"]["automatic_structured_consistency_pass"]
                and not item["Q5_K_M"][
                    "automatic_structured_consistency_pass"
                ]
            ),

            "q5_new_two_alternative_failure": (
                item["F16"]["exactly_two_numbered"]
                and not item["Q5_K_M"]["exactly_two_numbered"]
            ),

            "q5_new_completion_prefix_failure": (
                activity == "COMPLETION"
                and item["F16"]["completion_prefix_preserved"]
                and not item["Q5_K_M"][
                    "completion_prefix_preserved"
                ]
            ),

            "q5_format_improvement": (
                item["F16"]["has_extra_text"]
                and not item["Q5_K_M"]["has_extra_text"]
            ),

            "q5_format_regression": (
                not item["F16"]["has_extra_text"]
                and item["Q5_K_M"]["has_extra_text"]
            ),

            # ------------------------------------------------
            # Direct Q4 vs Q5
            # ------------------------------------------------
            "q5_pair_matches_q4": pair_equal(
                item["Q5_K_M"]["alternatives"],
                item["Q4_K_M"]["alternatives"],
            ),

            # Q4 introduced beneficiary repetition vs F16,
            # but Q5 removed it.
            "q5_recovers_q4_beneficiary_repetition": (
                not item["F16"]["beneficiary_repeated"]
                and item["Q4_K_M"]["beneficiary_repeated"]
                and not item["Q5_K_M"]["beneficiary_repeated"]
            ),

            # Q4 introduced a structured failure vs F16,
            # but Q5 returns to a valid structured output.
            "q5_recovers_q4_structured_failure": (
                item["F16"]["automatic_structured_consistency_pass"]
                and not item["Q4_K_M"][
                    "automatic_structured_consistency_pass"
                ]
                and item["Q5_K_M"][
                    "automatic_structured_consistency_pass"
                ]
            ),

            # Q4 lost the required two numbered alternatives,
            # but Q5 restores them.
            "q5_recovers_q4_two_alternative_failure": (
                item["F16"]["exactly_two_numbered"]
                and not item["Q4_K_M"]["exactly_two_numbered"]
                and item["Q5_K_M"]["exactly_two_numbered"]
            ),

            # Q4 introduced extra trailing/prefix text where F16
            # was clean, and Q5 removes it again.
            "q5_recovers_q4_format_regression": (
                not item["F16"]["has_extra_text"]
                and item["Q4_K_M"]["has_extra_text"]
                and not item["Q5_K_M"]["has_extra_text"]
            ),
        }

        scored_records.append(item)

    # Save detailed records.
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

    summary = {
        "validation_type": "post_quantization_sample",
        "seed": SEED,
        "records_total": len(scored_records),
        "records_per_activity": N_PER_ACTIVITY,
        "models": {
            "F16": {
                "overall": summarize_model(
                    scored_records,
                    "F16",
                ),
                "by_activity": {
                    activity: summarize_model(
                        scored_records,
                        "F16",
                        activity,
                    )
                    for activity in ACTIVITIES
                },
            },
            "Q4_K_M": {
                "overall": summarize_model(
                    scored_records,
                    "Q4_K_M",
                ),
                "by_activity": {
                    activity: summarize_model(
                        scored_records,
                        "Q4_K_M",
                        activity,
                    )
                    for activity in ACTIVITIES
                },
            },
            "Q5_K_M": {
                "overall": summarize_model(
                    scored_records,
                    "Q5_K_M",
                ),
                "by_activity": {
                    activity: summarize_model(
                        scored_records,
                        "Q5_K_M",
                        activity,
                    )
                    for activity in ACTIVITIES
                },
            },
        },
        "f16_vs_q4": {
            "q4_pair_matches_f16_rate": rate(
                r["comparison"]["q4_pair_matches_f16"]
                for r in scored_records
            ),
            "q4_new_beneficiary_repetition_count": sum(
                r["comparison"]["q4_new_beneficiary_repetition"]
                for r in scored_records
            ),
            "q4_new_structured_failure_count": sum(
                r["comparison"]["q4_new_structured_failure"]
                for r in scored_records
            ),
            "q4_new_two_alternative_failure_count": sum(
                r["comparison"]["q4_new_two_alternative_failure"]
                for r in scored_records
            ),
            "q4_new_completion_prefix_failure_count": sum(
                r["comparison"]["q4_new_completion_prefix_failure"]
                for r in scored_records
            ),
            "q4_format_improvement_count": sum(
                r["comparison"]["q4_format_improvement"]
                for r in scored_records
            ),
            "q4_format_regression_count": sum(
                r["comparison"]["q4_format_regression"]
                for r in scored_records
            ),
            "q4_new_beneficiary_repetition_ids": [
                r["id_example"]
                for r in scored_records
                if r["comparison"][
                    "q4_new_beneficiary_repetition"
                ]
            ],
        },
        "f16_vs_q5": {
            "q5_pair_matches_f16_rate": rate(
                r["comparison"]["q5_pair_matches_f16"]
                for r in scored_records
            ),
            "q5_new_beneficiary_repetition_count": sum(
                r["comparison"]["q5_new_beneficiary_repetition"]
                for r in scored_records
            ),
            "q5_new_structured_failure_count": sum(
                r["comparison"]["q5_new_structured_failure"]
                for r in scored_records
            ),
            "q5_new_two_alternative_failure_count": sum(
                r["comparison"]["q5_new_two_alternative_failure"]
                for r in scored_records
            ),
            "q5_new_completion_prefix_failure_count": sum(
                r["comparison"]["q5_new_completion_prefix_failure"]
                for r in scored_records
            ),
            "q5_format_improvement_count": sum(
                r["comparison"]["q5_format_improvement"]
                for r in scored_records
            ),
            "q5_format_regression_count": sum(
                r["comparison"]["q5_format_regression"]
                for r in scored_records
            ),
            "q5_new_beneficiary_repetition_ids": [
                r["id_example"]
                for r in scored_records
                if r["comparison"][
                    "q5_new_beneficiary_repetition"
                ]
            ],
            "q5_new_structured_failure_ids": [
                r["id_example"]
                for r in scored_records
                if r["comparison"][
                    "q5_new_structured_failure"
                ]
            ],
            "q5_new_two_alternative_failure_ids": [
                r["id_example"]
                for r in scored_records
                if r["comparison"][
                    "q5_new_two_alternative_failure"
                ]
            ],
        },

        "q4_vs_q5": {
            "q5_pair_matches_q4_rate": rate(
                r["comparison"]["q5_pair_matches_q4"]
                for r in scored_records
            ),
            "q5_recovers_q4_beneficiary_repetition_count": sum(
                r["comparison"][
                    "q5_recovers_q4_beneficiary_repetition"
                ]
                for r in scored_records
            ),
            "q5_recovers_q4_structured_failure_count": sum(
                r["comparison"][
                    "q5_recovers_q4_structured_failure"
                ]
                for r in scored_records
            ),
            "q5_recovers_q4_two_alternative_failure_count": sum(
                r["comparison"][
                    "q5_recovers_q4_two_alternative_failure"
                ]
                for r in scored_records
            ),
            "q5_recovers_q4_format_regression_count": sum(
                r["comparison"][
                    "q5_recovers_q4_format_regression"
                ]
                for r in scored_records
            ),
            "q5_recovers_q4_beneficiary_repetition_ids": [
                r["id_example"]
                for r in scored_records
                if r["comparison"][
                    "q5_recovers_q4_beneficiary_repetition"
                ]
            ],
            "q5_recovers_q4_structured_failure_ids": [
                r["id_example"]
                for r in scored_records
                if r["comparison"][
                    "q5_recovers_q4_structured_failure"
                ]
            ],
            "q5_recovers_q4_two_alternative_failure_ids": [
                r["id_example"]
                for r in scored_records
                if r["comparison"][
                    "q5_recovers_q4_two_alternative_failure"
                ]
            ],
        },

        "notes": [
            "HF pair match measures behavioral stability, not semantic quality.",
            "A Q4 output may differ from HF/F16 while remaining semantically valid.",
            "Extra text is evaluated separately from the two numbered alternatives.",
            "The structured consistency checker is diagnostic and should not be treated as the sole quality criterion.",
            "EX3023 is intentionally included as a known sensitive COMPLETION case.",
        ],
    }

    write_json(SUMMARY_PATH, summary)

    print("\n" + "=" * 88)
    print("VALIDATION COMPLETED")
    print("=" * 88)

    print("\nF16 overall:")
    print(
        json.dumps(
            summary["models"]["F16"]["overall"],
            indent=2,
        )
    )

    print("\nQ4_K_M overall:")
    print(
        json.dumps(
            summary["models"]["Q4_K_M"]["overall"],
            indent=2,
        )
    )

    print("\nQ5_K_M overall:")
    print(
        json.dumps(
            summary["models"]["Q5_K_M"]["overall"],
            indent=2,
        )
    )

    print("\nF16 vs Q4:")
    print(
        json.dumps(
            summary["f16_vs_q4"],
            indent=2,
        )
    )

    print("\nF16 vs Q5:")
    print(
        json.dumps(
            summary["f16_vs_q5"],
            indent=2,
        )
    )

    print("\nQ4 vs Q5:")
    print(
        json.dumps(
            summary["q4_vs_q5"],
            indent=2,
        )
    )

    print("\nDetailed records:")
    print(DETAILS_PATH)

    print("\nSummary:")
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
