# ============================================================
# NATIVE-CHAT BASELINE EVALUATION
# Edge AI for Smart Bank Transfers
#
# Purpose:
#   Evaluate the original LiquidAI/LFM2-700M checkpoint on the same
#   GPTPlus and Claude test sets, but using the model's native chat
#   interface via tokenizer.apply_chat_template(...).
#
# Methodological choices:
#   - Original base checkpoint only (no LoRA / merged model)
#   - Official tokenizer chat template
#   - One example at a time (batch_size=1), no padding, no truncation
#   - Greedy decoding, matching the final SFT-prompt evaluation so that
#     prompt formatting is the main experimental difference
#   - Same ROUGE, BERTScore, normalized Exact Match, structured
#     consistency, and output parsing used by evaluate_model.py
#
# Output directory:
#   evaluation/results_base_native_chat/
# ============================================================

import argparse
import csv
import gc
import json
import math
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# PATHS AND SHARED CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"

sys.path.insert(0, str(TRAINING_DIR))
from training_config import MODEL_NAME, MAX_SEQ_LENGTH, SEED  # noqa: E402

def get_base_model_label():
    """Derive a readable base-model name from MODEL_NAME."""
    return str(MODEL_NAME).rstrip("/").split("/")[-1]


def slugify_label(value):
    """Convert a label to a lowercase filesystem-safe slug."""
    value = unicodedata.normalize("NFKC", str(value)).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


BASE_MODEL_LABEL = get_base_model_label()

MODEL_SPECS = {
    "base_native_chat": {
        "label": f"{BASE_MODEL_LABEL}_Native-Chat",
        "path": MODEL_NAME,
    },
}

TESTSET_SPECS = {
    "gptplus": {
        "label": "GPTPlus",
        "display_name": "GPTPlus test set",
        "raw": PROJECT_ROOT / "data_GptPlus" / "splits" / "test.jsonl",
        "sft": PROJECT_ROOT / "data_GptPlus" / "processed" / "test_sft.jsonl",
    },
    "claude": {
        "label": "Claude",
        "display_name": "Claude test set",
        "raw": PROJECT_ROOT / "data_Claude" / "splits" / "test.jsonl",
        "sft": PROJECT_ROOT / "data_Claude" / "processed" / "test_sft.jsonl",
    },
}

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results_base_native_chat"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the original LFM2 base checkpoint on both test sets "
            "using its native Hugging Face chat template."
        )
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        choices=[1],
        help=(
            "Generation batch size. LFM2 evaluation intentionally uses "
            "single-example generation without padding."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--bertscore-batch-size", type=int, default=32)
    parser.add_argument(
        "--bertscore-rescale",
        action="store_true",
        help=(
            "Rescale BERTScore with the model/language baseline. "
            "Default: report raw BERTScore F1."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of records per test set for a smoke test.",
    )
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Skip BERTScore only for quick smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacement of files already present in "
            "evaluation/results_base_native_chat."
        ),
    )
    return parser.parse_args()


# ============================================================
# JSONL UTILITIES
# ============================================================

def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
    return records


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# TEXT NORMALIZATION / GENERATED ALTERNATIVES
# ============================================================

def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def clean_alternative_prefix(text):
    text = str(text).strip()
    patterns = [
        r"^\s*[-*•]\s*",
        r"^\s*\d+\s*[\.\):\-](?:[ \t]+|$)",
        r"^\s*(?:alternative|option)\s*\d+\s*[\.\):\-]\s*",
        r"^\s*(?:normalized\s+description|description)\s*\d+\s*[\.\):\-]\s*",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove a code-fence wrapper when the whole alternative is fenced.
    text = re.sub(r"^\s*```(?:text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


MARKDOWN_LINE_PREFIX = (
    r"^[ \t]*"
    r"(?:>[ \t]*)?"
    r"(?:#{1,6}[ \t]*)?"
    r"(?:[-+*][ \t]+)?"
)

MARKDOWN_EMPHASIS = r"(?:\*\*|__)?"

NAMED_ALTERNATIVE_LABEL = (
    r"(?:"
    r"alternative|"
    r"option|"
    r"normalized[ \t]+description|"
    r"description|"
    r"bank[- \t]+transfer[ \t]+description"
    r")"
)

NAMED_ALTERNATIVE_MARKER_PATTERN = re.compile(
    r"(?im)"
    + MARKDOWN_LINE_PREFIX
    + MARKDOWN_EMPHASIS
    + r"[ \t]*"
    + r"(?P<label>" + NAMED_ALTERNATIVE_LABEL + r")"
    + r"[ \t]*(?P<num>[12])"
    + r"(?:[ \t]*\([^\)\r\n]{1,60}\))?"
    + r"[ \t]*"
    + r"(?:" + MARKDOWN_EMPHASIS + r"[ \t]*)?"
    + r"[\.\):\-]"
    + r"(?:[ \t]*" + MARKDOWN_EMPHASIS + r")?"
    + r"[ \t]*"
)

NUMBERED_ALTERNATIVE_MARKER_PATTERN = re.compile(
    r"(?im)"
    + MARKDOWN_LINE_PREFIX
    + MARKDOWN_EMPHASIS
    + r"[ \t]*(?P<num>[12])"
    + r"[ \t]*(?:" + MARKDOWN_EMPHASIS + r"[ \t]*)?"
    + r"[\.\):\-]"
    + r"(?:[ \t]*" + MARKDOWN_EMPHASIS + r")?"
    + r"[ \t]*"
)

TRAILING_COMMENTARY_PATTERN = re.compile(
    r"(?im)"
    + MARKDOWN_LINE_PREFIX
    + MARKDOWN_EMPHASIS
    + r"[ \t]*(?:"
    r"explanation|reasoning|notes?|"
    r"both[ \t]+(?:descriptions|alternatives)"
    r")"
    r"[ \t]*(?:" + MARKDOWN_EMPHASIS + r"[ \t]*)?"
    r":?"
    r"(?:[ \t]*" + MARKDOWN_EMPHASIS + r")?"
    r".*$"
)

GENERIC_OUTPUT_HEADERS = {
    "alternatives",
    "alternative descriptions",
    "bank transfer descriptions",
    "descriptions",
    "normalized descriptions",
    "output",
    "response",
}

STRUCTURED_FIELD_PREFIX_PATTERN = re.compile(
    r"^\s*(?:"
    r"category|beneficiary|amount|transfer\s+amount|reference(?:\s+period)?|"
    r"calendar\s+context|event|date|event\s+category|partial\s+description|"
    r"deposit|mode|destination|reason|invoice\s+number|service|departure|"
    r"context|purpose|method|time|cost|pickup|recipient|transfer"
    r")\s*:",
    flags=re.IGNORECASE,
)


def trim_trailing_commentary(text):
    text = str(text).strip()
    match = TRAILING_COMMENTARY_PATTERN.search(text)
    if match:
        text = text[:match.start()]
    return text.strip()


def _clean_pair(first, second):
    first = trim_trailing_commentary(clean_alternative_prefix(first))
    second = trim_trailing_commentary(clean_alternative_prefix(second))
    return [x for x in (first, second) if x]


def _extract_marked_alternatives(text, marker_pattern):
    """Extract one or two alternatives from line-anchored markers."""
    matches = list(marker_pattern.finditer(text))
    if not matches:
        return []

    first_marker = next((m for m in matches if m.group("num") == "1"), None)
    if first_marker is None:
        return []

    second_marker = next(
        (
            m for m in matches
            if m.start() > first_marker.start() and m.group("num") == "2"
        ),
        None,
    )

    if second_marker is not None:
        first = text[first_marker.end():second_marker.start()]
        second = text[second_marker.end():]
        return _clean_pair(first, second)[:2]

    first = trim_trailing_commentary(
        clean_alternative_prefix(text[first_marker.end():])
    )
    return [first] if first else []


def extract_alternatives(text):
    """
    Return up to two generated bank-transfer descriptions.

    Supports normal markers and Markdown variants such as:
      **Alternative 1:**
      **Description 1:**
      ### Alternative 1:
      - **Option 1 (Concise):**
      **Bank-Transfer Description 1:**
    """
    text = str(text).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            value = json.loads(text)
            if isinstance(value, list):
                parsed = [
                    trim_trailing_commentary(clean_alternative_prefix(str(x)))
                    for x in value
                    if str(x).strip()
                ]
                return [x for x in parsed if x][:2]
        except json.JSONDecodeError:
            pass

    named = _extract_marked_alternatives(
        text,
        NAMED_ALTERNATIVE_MARKER_PATTERN,
    )
    if named:
        return named[:2]

    numbered = _extract_marked_alternatives(
        text,
        NUMBERED_ALTERNATIVE_MARKER_PATTERN,
    )
    if numbered:
        return numbered[:2]

    lines = []
    for raw_line in text.splitlines():
        line = trim_trailing_commentary(clean_alternative_prefix(raw_line))
        if not line:
            continue
        if re.fullmatch(r"\s*```(?:text)?\s*", line, flags=re.IGNORECASE):
            continue

        header_probe = re.sub(r"[*_`#>]", "", line).strip()
        header_probe = re.sub(r"^\s*[-+]\s+", "", header_probe)

        if normalize_text(header_probe) in GENERIC_OUTPUT_HEADERS:
            continue
        if STRUCTURED_FIELD_PREFIX_PATTERN.match(header_probe):
            continue
        lines.append(line)

    if len(lines) == 2:
        return lines

    if len(lines) == 1:
        pipe_parts = [
            trim_trailing_commentary(clean_alternative_prefix(x))
            for x in re.split(r"[ \t]*\|[ \t]*", lines[0])
            if x.strip()
        ]
        pipe_parts = [x for x in pipe_parts if x]
        if len(pipe_parts) >= 2:
            return pipe_parts[:2]
        return [lines[0]]

    return []


# ============================================================
# DATASET VALIDATION / ALIGNMENT
# ============================================================

RAW_REQUIRED_FIELDS = {
    "id_example",
    "activity_type",
    "operation_category",
    "beneficiary",
    "amount",
    "currency",
    "reference_period",
    "input_text",
    "calendar_context",
    "expected_output",
    "language",
    "split",
}


def validate_raw_record(record, path, index):
    missing = RAW_REQUIRED_FIELDS - set(record.keys())
    if missing:
        raise ValueError(
            f"{path}: record {index} is missing fields: {sorted(missing)}"
        )

    refs = record["expected_output"]
    if (
        not isinstance(refs, list)
        or len(refs) != 2
        or any(not isinstance(x, str) or not x.strip() for x in refs)
    ):
        raise ValueError(
            f"{path}: {record.get('id_example', index)} must contain exactly "
            "two non-empty expected_output strings."
        )


def validate_sft_record(record, path, index):
    if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
        raise ValueError(f"{path}: SFT record {index} has no valid prompt.")
    if (
        not isinstance(record.get("completion"), str)
        or not record["completion"].strip()
    ):
        raise ValueError(f"{path}: SFT record {index} has no valid completion.")


def completion_contains_all_references(completion, references):
    """Require both original references to appear in the processed completion."""
    completion_norm = normalize_text(completion)
    normalized_refs = [normalize_text(ref) for ref in references]

    return (
        len(normalized_refs) == 2
        and all(normalized_refs)
        and all(ref in completion_norm for ref in normalized_refs)
    )


def load_testset(testset_name, limit=None):
    spec = TESTSET_SPECS[testset_name]
    raw_path = spec["raw"]
    sft_path = spec["sft"]

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw test split not found:\n{raw_path}")
    if not sft_path.exists():
        raise FileNotFoundError(f"Processed SFT test file not found:\n{sft_path}")

    raw_records = read_jsonl(raw_path)
    sft_records = read_jsonl(sft_path)

    if len(raw_records) != len(sft_records):
        raise ValueError(
            f"Raw/SFT size mismatch for {testset_name}: "
            f"{len(raw_records)} raw vs {len(sft_records)} processed."
        )

    for i, (raw, sft) in enumerate(zip(raw_records, sft_records), start=1):
        validate_raw_record(raw, raw_path, i)
        validate_sft_record(sft, sft_path, i)
        if not completion_contains_all_references(sft["completion"], raw["expected_output"]):
            raise ValueError(
                f"Raw/SFT files appear misaligned at position {i} "
                f"({raw['id_example']}). The processed completion does not "
                "contain both original references."
            )

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive.")
        raw_records = raw_records[:limit]
        sft_records = sft_records[:limit]

    return raw_records, sft_records


# ============================================================
# MODEL LOADING / GENERATION
# ============================================================

def validate_model_paths():
    # The native-chat baseline uses MODEL_NAME directly from Hugging Face
    # (or its local HF cache), so there are no merged-model directories
    # to validate here.
    if not str(MODEL_NAME).strip():
        raise ValueError("MODEL_NAME is empty in training/training_config.py.")


def load_model_and_tokenizer(model_key, device):
    model_path = MODEL_SPECS[model_key]["path"]

    print(f"\nLoading {MODEL_SPECS[model_key]['label']}...")
    print(f"Source: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


@torch.inference_mode()
def generate_predictions(
    model,
    tokenizer,
    prompts,
    device,
    batch_size,
    max_new_tokens,
):
    """
    Generate exactly one prompt at a time using the tokenizer's native chat
    template.

    This experiment intentionally keeps greedy decoding identical to the
    final SFT-prompt evaluation. The principal experimental variable is the
    prompt interface:

        SFT-prompt evaluation:
            raw prompt + "\\n\\n"

        Native-chat baseline:
            [{"role": "user", "content": prompt}]
            -> tokenizer.apply_chat_template(
                   add_generation_prompt=True,
                   tokenize=True,
                   return_dict=True,
                   return_tensors="pt",
               )

    No padding and no truncation are used.
    """
    if batch_size != 1:
        raise ValueError(
            "LFM2 native-chat evaluation requires batch_size=1. "
            "Padded batched generation is intentionally disabled."
        )

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "The tokenizer does not expose a chat_template. "
            "Cannot run the native-chat baseline."
        )

    predictions = []

    for index, prompt in enumerate(prompts, start=1):
        user_prompt = prompt.rstrip("\r\n")
        messages = [{"role": "user", "content": user_prompt}]

        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        input_width = encoded["input_ids"].shape[1]

        model_context_limit = getattr(
            model.config,
            "max_position_embeddings",
            None,
        )
        if (
            model_context_limit is not None
            and input_width + max_new_tokens > model_context_limit
        ):
            raise ValueError(
                f"Prompt {index} plus max_new_tokens would require "
                f"{input_width + max_new_tokens} tokens, exceeding the "
                f"model context limit of {model_context_limit}. "
                "Evaluation stopped instead of truncating the prompt."
            )

        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        continuation_ids = generated[0, input_width:]

        output = tokenizer.decode(
            continuation_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        predictions.append(output)
        print(f"  Generated {index}/{len(prompts)}", end="\r")

    print()
    return predictions


def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# ROUGE / EXACT MATCH
# ============================================================

def pair_assignment_average(matrix):
    """Best one-to-one assignment between two predictions and two refs."""
    diagonal = matrix[0][0] + matrix[1][1]
    crossed = matrix[0][1] + matrix[1][0]
    return max(diagonal, crossed) / 2.0


def score_rouge_pair(predicted_alternatives, references, scorer, rouge_type):
    preds = list(predicted_alternatives[:2])
    while len(preds) < 2:
        preds.append("")

    matrix = [[0.0, 0.0], [0.0, 0.0]]
    for i, pred in enumerate(preds):
        if not pred.strip():
            continue
        for j, ref in enumerate(references):
            matrix[i][j] = scorer.score(ref, pred)[rouge_type].fmeasure

    return pair_assignment_average(matrix)


def score_exact_match_pair(predicted_alternatives, references):
    preds = list(predicted_alternatives[:2])
    while len(preds) < 2:
        preds.append("")

    pred_norm = [normalize_text(x) for x in preds]
    ref_norm = [normalize_text(x) for x in references]
    matrix = [[0.0, 0.0], [0.0, 0.0]]

    for i, pred in enumerate(pred_norm):
        if not pred:
            continue
        for j, ref in enumerate(ref_norm):
            matrix[i][j] = 1.0 if pred == ref else 0.0

    return pair_assignment_average(matrix)


# ============================================================
# BERTSCORE
# ============================================================

def build_bertscore_pairs(scored_records):
    candidates = []
    references = []
    mapping = []

    for record_index, record in enumerate(scored_records):
        preds = list(record["predicted_alternatives"][:2])
        while len(preds) < 2:
            preds.append("")

        refs = record["references"]
        for pred_index, pred in enumerate(preds):
            if not pred.strip():
                continue
            for ref_index, ref in enumerate(refs):
                mapping.append((record_index, pred_index, ref_index))
                candidates.append(pred)
                references.append(ref)

    return candidates, references, mapping


def add_bertscore(scored_records, bert_scorer, batch_size):
    candidates, references, mapping = build_bertscore_pairs(scored_records)
    matrices = [
        [[0.0, 0.0], [0.0, 0.0]]
        for _ in scored_records
    ]

    if candidates:
        _, _, f1 = bert_scorer.score(
            candidates,
            references,
            batch_size=batch_size,
        )
        values = f1.detach().cpu().tolist()
        for value, (record_index, pred_index, ref_index) in zip(values, mapping):
            matrices[record_index][pred_index][ref_index] = float(value)

    for record, matrix in zip(scored_records, matrices):
        record["bertscore_f1"] = pair_assignment_average(matrix)


# ============================================================
# AUTOMATIC STRUCTURED CONSISTENCY
# ============================================================

CURRENCY_CODES = ("EUR", "USD", "GBP")
CURRENCY_PATTERN = r"(?:EUR|USD|GBP)"

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
    r"\b(" + "|".join(sorted(MONTH_TO_NUMBER, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)

# A year must not be embedded in a decimal/thousands-formatted number.
# Example: 2095.93 EUR must NOT create year:2095.
YEAR_PATTERN = re.compile(
    r"(?<![\d.,])(?P<year>20\d{2})(?![\d.,])",
    flags=re.IGNORECASE,
)

ISO_DATE_PATTERN = re.compile(
    r"(?<!\d)(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])(?!\d)"
)

# Supports ordinary decimals and common thousands separators.
MONEY_NUMBER_PATTERN = (
    r"(?:"
    r"\d{1,3}(?:[ ,]\d{3})+(?:[.,]\d{1,2})?"
    r"|"
    r"\d+(?:[.,]\d{1,2})?"
    r")"
)

CURRENCY_FIRST_MONEY_PATTERN = re.compile(
    rf"\b(?P<currency>{CURRENCY_PATTERN})[ \t]*"
    rf"(?P<amount>{MONEY_NUMBER_PATTERN})(?![\d.,])",
    flags=re.IGNORECASE,
)

AMOUNT_FIRST_MONEY_PATTERN = re.compile(
    rf"(?<![\d.,])(?P<amount>{MONEY_NUMBER_PATTERN})[ \t]*"
    rf"(?P<currency>{CURRENCY_PATTERN})\b",
    flags=re.IGNORECASE,
)


def parse_money_number(value):
    """Parse common English/European money formatting into float."""
    value = str(value).strip().replace(" ", "")

    if "," in value and "." in value:
        # Treat the last separator as decimal and the other as thousands.
        if value.rfind(".") > value.rfind(","):
            value = value.replace(",", "")
        else:
            value = value.replace(".", "").replace(",", ".")
    elif "," in value:
        # 1,200 -> 1200 ; 444,68 -> 444.68
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+", value):
            value = value.replace(",", "")
        else:
            value = value.replace(",", ".")
    elif "." in value:
        # 1.200 is accepted as a thousands form; 444.68 stays decimal.
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", value):
            value = value.replace(".", "")

    return float(value)


def plain_year_value(number_text):
    """
    Return a 2000-2099 year for a plain four-digit token, otherwise None.

    Decimal/thousands forms are deliberately excluded.
    """
    value = str(number_text).strip()
    if re.fullmatch(r"20\d{2}", value):
        return int(value)
    return None


def has_local_temporal_cue(text, start, end):
    """Check whether a year token is locally supported by temporal wording."""
    window = str(text)[max(0, start - 24):min(len(str(text)), end + 24)]
    if MONTH_PATTERN.search(window):
        return True
    if re.search(r"\bq[1-4]\b", window, flags=re.IGNORECASE):
        return True
    if re.search(
        r"\b(?:year|term|semester|installment|instalment)\b",
        window,
        flags=re.IGNORECASE,
    ):
        return True
    return False


def extract_temporal_markers(text, allowed_currency_adjacent_years=None):
    """
    Extract conservative temporal markers.

    Currency-qualified numbers are treated carefully:
    - EUR 2027 is considered an amount, not a year;
    - 2027 EUR is considered a year only when supported by source-allowed
      temporal years or a nearby temporal cue such as a month/quarter/term.
    """
    text = str(text or "")
    low = text.lower()
    markers = set()
    allowed_currency_adjacent_years = {
        int(x) for x in (allowed_currency_adjacent_years or set())
    }

    for quarter in re.findall(r"\bq([1-4])\b", low):
        markers.add(f"quarter:{quarter}")

    for month_match in MONTH_PATTERN.finditer(low):
        markers.add(f"month:{MONTH_TO_NUMBER[month_match.group(1).lower()]}")

    installment_patterns = {
        "installment:first": r"\b(?:first|1st)\s+(?:installment|instalment)\b",
        "installment:second": r"\b(?:second|2nd)\s+(?:installment|instalment)\b",
        "installment:third": r"\b(?:third|3rd)\s+(?:installment|instalment)\b",
        "installment:final": r"\bfinal\s+(?:installment|instalment)\b",
    }
    for marker, pattern in installment_patterns.items():
        if re.search(pattern, low):
            markers.add(marker)

    # ISO dates are unambiguous.
    for match in ISO_DATE_PATTERN.finditer(low):
        markers.add(f"year:{match.group(1)}")
        markers.add(f"month:{int(match.group(2))}")

    for match in YEAR_PATTERN.finditer(low):
        year = int(match.group("year"))
        start, end = match.span("year")

        before = low[max(0, start - 8):start]
        after = low[end:min(len(low), end + 8)]

        # "EUR 2027" -> explicit currency-first monetary amount.
        if re.search(rf"\b{CURRENCY_PATTERN}[ \t]*$", before, flags=re.IGNORECASE):
            continue

        # "2027 EUR" is ambiguous. Keep it as a year only when the source
        # permits that year or the local wording clearly supplies time context.
        followed_by_currency = re.match(
            rf"[ \t]*{CURRENCY_PATTERN}\b",
            after,
            flags=re.IGNORECASE,
        )
        if followed_by_currency:
            if (
                year not in allowed_currency_adjacent_years
                and not has_local_temporal_cue(low, start, end)
            ):
                continue

        markers.add(f"year:{year}")

    return markers


def authoritative_temporal_text_for_record(record):
    """Temporal fields that are explicit in the source record."""
    pieces = [record.get("reference_period")]
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


def source_temporal_markers(record):
    """
    Build the temporal information allowed by the source record.

    Explicit reference_period/calendar fields are parsed first. Their years are
    then used to disambiguate patterns such as "2027 EUR" appearing in input_text.
    """
    authoritative = extract_temporal_markers(
        authoritative_temporal_text_for_record(record)
    )
    authoritative_years = {
        int(marker.split(":", 1)[1])
        for marker in authoritative
        if marker.startswith("year:")
    }

    broader = extract_temporal_markers(
        source_text_for_record(record),
        allowed_currency_adjacent_years=authoritative_years,
    )
    return authoritative | broader


def monetary_mentions(text, temporal_years=None):
    """
    Return currency-qualified monetary values without crossing line boundaries.

    Horizontal whitespace is accepted, but newlines are not. This prevents a
    trailing year on one alternative from being joined to the currency token on
    the next alternative.

    For amount-first forms such as "2027 EUR", a plain 20xx value is ignored
    when it is already a source-supported temporal year or has a local temporal
    cue. This prevents "April 2027 EUR 90" from being interpreted as 2027 EUR.
    """
    mentions = []
    temporal_years = {int(x) for x in (temporal_years or set())}

    for line in str(text).splitlines():
        occupied_spans = []

        # Currency-first is unambiguously monetary: EUR 90, USD 1,200, etc.
        for match in CURRENCY_FIRST_MONEY_PATTERN.finditer(line):
            currency = match.group("currency").upper()
            amount = parse_money_number(match.group("amount"))
            mentions.append((currency, amount))
            occupied_spans.append(match.span())

        # Amount-first: 90 EUR. Avoid year/currency false positives.
        for match in AMOUNT_FIRST_MONEY_PATTERN.finditer(line):
            span = match.span()
            if any(not (span[1] <= a or span[0] >= b) for a, b in occupied_spans):
                continue

            raw_amount = match.group("amount")
            year_value = plain_year_value(raw_amount)
            if year_value is not None:
                if (
                    year_value in temporal_years
                    or has_local_temporal_cue(
                        line,
                        match.start("amount"),
                        match.end("amount"),
                    )
                ):
                    continue

            currency = match.group("currency").upper()
            amount = parse_money_number(raw_amount)
            mentions.append((currency, amount))

    return mentions


def structured_consistency(record, predicted_alternatives, raw_prediction=None):
    # Structured checks inspect the complete generated response so malformed
    # extra fields cannot be hidden by the alternative parser.
    prediction_text = (
        str(raw_prediction).strip()
        if raw_prediction is not None
        else "\n".join(predicted_alternatives)
    )
    expected_currency = str(record["currency"]).upper()
    expected_amount = float(record["amount"])

    currencies = {
        x.upper()
        for x in re.findall(
            rf"\b{CURRENCY_PATTERN}\b",
            prediction_text,
            flags=re.IGNORECASE,
        )
    }
    currency_consistent = all(x == expected_currency for x in currencies)

    allowed_temporal = source_temporal_markers(record)
    allowed_years = {
        int(marker.split(":", 1)[1])
        for marker in allowed_temporal
        if marker.startswith("year:")
    }

    amount_consistent = True
    for currency, amount in monetary_mentions(
        prediction_text,
        temporal_years=allowed_years,
    ):
        if currency != expected_currency or not math.isclose(
            amount,
            expected_amount,
            abs_tol=0.011,
        ):
            amount_consistent = False
            break

    predicted_temporal = extract_temporal_markers(
        prediction_text,
        allowed_currency_adjacent_years=allowed_years,
    )
    temporal_consistent = predicted_temporal.issubset(allowed_temporal)

    beneficiary = normalize_text(record["beneficiary"])
    prediction_norm = normalize_text(prediction_text)
    beneficiary_mentioned = bool(beneficiary) and beneficiary in prediction_norm

    calendar_reflected = None
    calendar = record.get("calendar_context")
    if isinstance(calendar, dict):
        event_title_tokens = {
            token
            for token in normalize_text(calendar.get("event_title", "")).split()
            if len(token) >= 4
        }
        title_overlap = bool(event_title_tokens & set(prediction_norm.split()))

        calendar_temporal = extract_temporal_markers(
            calendar.get("event_date", "")
        )
        prediction_temporal = extract_temporal_markers(
            prediction_text,
            allowed_currency_adjacent_years=allowed_years,
        )
        date_overlap = bool(calendar_temporal & prediction_temporal)
        calendar_reflected = title_overlap or date_overlap

    automatic_pass = (
        bool(predicted_alternatives)
        and currency_consistent
        and amount_consistent
        and temporal_consistent
    )

    return {
        "currency_consistent": currency_consistent,
        "amount_consistent": amount_consistent,
        "temporal_consistent": temporal_consistent,
        "automatic_structured_consistency_pass": automatic_pass,
        "beneficiary_mentioned": beneficiary_mentioned,
        "calendar_context_reflected": calendar_reflected,
    }


def run_internal_sanity_checks():
    """
    Guard against the exact parsing/checker regressions found during evaluation.

    These checks do not use model outputs or GPU and execute instantly at startup.
    """
    # Parser: named normalized descriptions must not be split at the "1:" token.
    parsed = extract_alternatives(
        "Normalized description 1:\n"
        "11.08 EUR from Riverside Transit Co. Services for commuting.\n\n"
        "Normalized description 2:\n"
        "11.08 EUR commuting fare.\n\n"
        "Explanation:\nextra text"
    )
    assert parsed == [
        "11.08 EUR from Riverside Transit Co. Services for commuting.",
        "11.08 EUR commuting fare.",
    ], f"Parser sanity check failed: {parsed!r}"

    # Amount checker: 2027 in 'April 2027 EUR 90' is a year, not EUR 2027.
    mentions = monetary_mentions(
        "water bill April 2027 EUR 90",
        temporal_years={2027},
    )
    assert mentions == [("EUR", 90.0)], (
        f"Money sanity check failed for year/currency adjacency: {mentions!r}"
    )

    # Amount checker: never join a year on one line to a currency on the next.
    mentions = monetary_mentions(
        "Other EUR 444.68 for June 2026\nEUR 444.68 for June 2026",
        temporal_years={2026},
    )
    assert mentions == [("EUR", 444.68), ("EUR", 444.68)], (
        f"Money newline sanity check failed: {mentions!r}"
    )

    # Temporal checker: 2095.93 EUR is an amount, not year 2095.
    markers = extract_temporal_markers("professional service EUR 2095.93")
    assert "year:2095" not in markers, (
        f"Temporal decimal sanity check failed: {markers!r}"
    )


    parsed = extract_alternatives(
        "Here are two descriptions:\n\n"
        "**Alternative 1:**\n"
        "April 2027 water bill EUR 90\n\n"
        "**Alternative 2:**\n"
        "EUR 90 water payment for April 2027"
    )
    assert parsed == [
        "April 2027 water bill EUR 90",
        "EUR 90 water payment for April 2027",
    ], f"Markdown Alternative parser sanity check failed: {parsed!r}"

    parsed = extract_alternatives(
        "**Description 1:**\n"
        "Transfer of 45 EUR to AmberView Mobile.\n\n"
        "**Description 2:**\n"
        "45 EUR mobile service payment."
    )
    assert parsed == [
        "Transfer of 45 EUR to AmberView Mobile.",
        "45 EUR mobile service payment.",
    ], f"Markdown Description parser sanity check failed: {parsed!r}"

    parsed = extract_alternatives(
        "### **Option 1 (Concise):**\n"
        "Consulting fee EUR 100\n\n"
        "- **Option 2 (More Formal):**\n"
        "EUR 100 consulting payment"
    )
    assert parsed == [
        "Consulting fee EUR 100",
        "EUR 100 consulting payment",
    ], f"Markdown Option parser sanity check failed: {parsed!r}"

    parsed = extract_alternatives(
        "**Bank-Transfer Description 1:**\n"
        "Transfer of 100 EUR to Example Ltd.\n\n"
        "**Bank-Transfer Description 2:**\n"
        "100 EUR payment to Example Ltd."
    )
    assert parsed == [
        "Transfer of 100 EUR to Example Ltd.",
        "100 EUR payment to Example Ltd.",
    ], f"Bank-transfer description parser sanity check failed: {parsed!r}"

    parsed = extract_alternatives(
        "**Alternative 1:**\nFirst valid description only"
    )
    assert parsed == [
        "First valid description only"
    ], f"Single marked alternative sanity check failed: {parsed!r}"


# ============================================================
# RECORD SCORING / AGGREGATION
# ============================================================

def prepare_scored_records(
    raw_records,
    predictions,
    model_key,
    testset_name,
    rouge,
):
    if len(raw_records) != len(predictions):
        raise ValueError(
            f"Prediction count mismatch for {model_key} on {testset_name}."
        )

    output = []
    for raw, raw_prediction in zip(raw_records, predictions):
        references = raw["expected_output"]
        predicted_alternatives = extract_alternatives(raw_prediction)

        valid_two = (
            len(predicted_alternatives) == 2
            and all(x.strip() for x in predicted_alternatives)
            and normalize_text(predicted_alternatives[0])
            != normalize_text(predicted_alternatives[1])
        )

        output.append(
            {
                "id_example": raw["id_example"],
                "model": MODEL_SPECS[model_key]["label"],
                "test_set": TESTSET_SPECS[testset_name]["label"],
                "activity_type": raw["activity_type"],
                "operation_category": raw["operation_category"],
                "prompt_source_record": {
                    "beneficiary": raw["beneficiary"],
                    "amount": raw["amount"],
                    "currency": raw["currency"],
                    "reference_period": raw["reference_period"],
                    "input_text": raw["input_text"],
                    "calendar_context": raw["calendar_context"],
                },
                "references": references,
                "raw_prediction": raw_prediction,
                "predicted_alternatives": predicted_alternatives,
                "rouge1_f1": score_rouge_pair(
                    predicted_alternatives, references, rouge, "rouge1"
                ),
                "rouge2_f1": score_rouge_pair(
                    predicted_alternatives, references, rouge, "rouge2"
                ),
                "bertscore_f1": None,
                "normalized_exact_match": score_exact_match_pair(
                    predicted_alternatives, references
                ),
                "two_distinct_alternatives": valid_two,
                **structured_consistency(
                    raw,
                    predicted_alternatives,
                    raw_prediction=raw_prediction,
                ),
            }
        )

    return output


def mean(values):
    values = [float(x) for x in values if x is not None]
    return sum(values) / len(values) if values else None


def aggregate_records(records):
    calendar_values = [
        r["calendar_context_reflected"]
        for r in records
        if r["calendar_context_reflected"] is not None
    ]

    return {
        "records": len(records),
        "rouge1_f1": mean(r["rouge1_f1"] for r in records),
        "rouge2_f1": mean(r["rouge2_f1"] for r in records),
        "bertscore_f1": mean(r["bertscore_f1"] for r in records),
        "normalized_exact_match": mean(
            r["normalized_exact_match"] for r in records
        ),
        "two_distinct_alternatives_rate": mean(
            1.0 if r["two_distinct_alternatives"] else 0.0 for r in records
        ),
        "automatic_structured_consistency_rate": mean(
            1.0 if r["automatic_structured_consistency_pass"] else 0.0
            for r in records
        ),
        "currency_consistency_rate": mean(
            1.0 if r["currency_consistent"] else 0.0 for r in records
        ),
        "amount_consistency_rate": mean(
            1.0 if r["amount_consistent"] else 0.0 for r in records
        ),
        "temporal_consistency_rate": mean(
            1.0 if r["temporal_consistent"] else 0.0 for r in records
        ),
        "beneficiary_mention_rate": mean(
            1.0 if r["beneficiary_mentioned"] else 0.0 for r in records
        ),
        "calendar_context_reflection_rate": (
            mean(1.0 if x else 0.0 for x in calendar_values)
            if calendar_values
            else None
        ),
    }


def build_experiment_summary(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["activity_type"]].append(record)

    return {
        "overall": aggregate_records(records),
        "by_activity_type": {
            activity: aggregate_records(activity_records)
            for activity, activity_records in sorted(groups.items())
        },
    }


# ============================================================
# RESULT FILES
# ============================================================

def ensure_results_directory(overwrite):
    if RESULTS_DIR.exists() and any(RESULTS_DIR.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Evaluation results directory already contains files:\n"
            f"{RESULTS_DIR}\n\nUse --overwrite to replace them."
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def save_summary_csv(summary):
    path = RESULTS_DIR / "model_comparison_summary.csv"
    fieldnames = [
        "model",
        "test_set",
        "records",
        "rouge1_f1",
        "rouge2_f1",
        "bertscore_f1",
        "normalized_exact_match",
        "two_distinct_alternatives_rate",
        "automatic_structured_consistency_rate",
        "currency_consistency_rate",
        "amount_consistency_rate",
        "temporal_consistency_rate",
        "beneficiary_mention_rate",
        "calendar_context_reflection_rate",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for experiment in summary["experiments"].values():
            writer.writerow(
                {
                    "model": experiment["model_label"],
                    "test_set": experiment["test_set_label"],
                    **experiment["metrics"]["overall"],
                }
            )


def save_activity_csv(summary):
    path = RESULTS_DIR / "model_comparison_by_activity.csv"
    fieldnames = [
        "model",
        "test_set",
        "activity_type",
        "records",
        "rouge1_f1",
        "rouge2_f1",
        "bertscore_f1",
        "normalized_exact_match",
        "two_distinct_alternatives_rate",
        "automatic_structured_consistency_rate",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for experiment in summary["experiments"].values():
            for activity_type, metrics in experiment["metrics"]["by_activity_type"].items():
                writer.writerow(
                    {
                        "model": experiment["model_label"],
                        "test_set": experiment["test_set_label"],
                        "activity_type": activity_type,
                        "records": metrics["records"],
                        "rouge1_f1": metrics["rouge1_f1"],
                        "rouge2_f1": metrics["rouge2_f1"],
                        "bertscore_f1": metrics["bertscore_f1"],
                        "normalized_exact_match": metrics["normalized_exact_match"],
                        "two_distinct_alternatives_rate": metrics[
                            "two_distinct_alternatives_rate"
                        ],
                        "automatic_structured_consistency_rate": metrics[
                            "automatic_structured_consistency_rate"
                        ],
                    }
                )


def save_qualitative_review(scored_by_experiment, max_records=60):
    candidates = []

    for experiment_key, records in scored_by_experiment.items():
        for record in records:
            metric_for_sort = (
                record["bertscore_f1"]
                if record["bertscore_f1"] is not None
                else record["rouge1_f1"]
            )
            priority = 0
            if not record["automatic_structured_consistency_pass"]:
                priority += 10
            if not record["two_distinct_alternatives"]:
                priority += 5
            candidates.append((-priority, metric_for_sort, experiment_key, record))

    candidates.sort(key=lambda x: (x[0], x[1]))
    selected = [item[3] for item in candidates[:max_records]]
    write_jsonl(
        RESULTS_DIR / "qualitative_review_cases.jsonl",
        selected,
    )


def prediction_filename(model_key, testset_name):
    """Build the documented record-level prediction filename."""
    model_slug = slugify_label(MODEL_SPECS[model_key]["label"])
    test_slug = f"{slugify_label(TESTSET_SPECS[testset_name]['label'])}_test"
    return f"predictions_{model_slug}_on_{test_slug}.jsonl"


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if args.batch_size != 1:
        raise ValueError("--batch-size must be exactly 1 for LFM2 generation.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if args.bertscore_batch_size <= 0:
        raise ValueError("--bertscore-batch-size must be positive.")

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    run_internal_sanity_checks()

    ensure_results_directory(args.overwrite)
    validate_model_paths()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print(f"{BASE_MODEL_LABEL} NATIVE-CHAT BASELINE EVALUATION")
    print("=" * 72)
    print(f"\nOriginal model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Generation batch size: {args.batch_size}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Results directory:\n{RESULTS_DIR}")

    # Load raw references/metadata and the exact processed prompts used by SFT.
    testsets = {}
    for testset_name in ("gptplus", "claude"):
        print(f"\nLoading {TESTSET_SPECS[testset_name]['display_name']}...")
        raw_records, sft_records = load_testset(testset_name, args.limit)
        testsets[testset_name] = {
            "raw": raw_records,
            "sft": sft_records,
            "prompts": [record["prompt"] for record in sft_records],
        }
        print(f"  Validated records: {len(raw_records)}")

    # Phase 1: two generation experiments with the original model,
    # using its native chat template.
    raw_predictions = {}
    model_key = "base_native_chat"
    model, tokenizer = load_model_and_tokenizer(model_key, device)

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Native chat template is unavailable in the tokenizer.")
    print("Native chat template available: yes")

    for testset_name in ("gptplus", "claude"):
        experiment_key = f"{model_key}_on_{testset_name}"
        print(
            f"\nGenerating: {MODEL_SPECS[model_key]['label']} "
            f"on {TESTSET_SPECS[testset_name]['display_name']}"
        )
        raw_predictions[experiment_key] = generate_predictions(
            model=model,
            tokenizer=tokenizer,
            prompts=testsets[testset_name]["prompts"],
            device=device,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )

    unload_model(model, tokenizer)

    # Phase 2: ROUGE, normalized Exact Match and structured checks.
    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2"],
        use_stemmer=True,
    )
    scored_by_experiment = {}

    model_key = "base_native_chat"
    for testset_name in ("gptplus", "claude"):
        experiment_key = f"{model_key}_on_{testset_name}"
        scored_by_experiment[experiment_key] = prepare_scored_records(
            raw_records=testsets[testset_name]["raw"],
            predictions=raw_predictions[experiment_key],
            model_key=model_key,
            testset_name=testset_name,
            rouge=rouge,
        )

    # Phase 3: BERTScore after the LFM model is unloaded, to reuse GPU memory.
    bert_hash = None
    if not args.skip_bertscore:
        print(f"\nLoading BERTScore model: {args.bertscore_model}...")
        bert_scorer = BERTScorer(
            model_type=args.bertscore_model,
            lang="en",
            rescale_with_baseline=args.bertscore_rescale,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        bert_hash = getattr(bert_scorer, "hash", None)

        for experiment_key, scored_records in scored_by_experiment.items():
            print(f"Computing BERTScore: {experiment_key}...")
            add_bertscore(
                scored_records,
                bert_scorer,
                args.bertscore_batch_size,
            )

        del bert_scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save detailed record-level outputs using readable, reproducible filenames.
    for experiment_key, scored_records in scored_by_experiment.items():
        model_key, _, testset_name = experiment_key.partition("_on_")
        write_jsonl(
            PREDICTIONS_DIR / prediction_filename(model_key, testset_name),
            scored_records,
        )

    # Final summary.
    summary = {
        "evaluation_design": {
            "models": {
                key: {
                    "label": value["label"],
                    "path": str(value["path"]),
                    "checkpoint_type": "original_post_trained_checkpoint",
                    "prompt_interface": "native_chat_template",
                }
                for key, value in MODEL_SPECS.items()
            },
            "test_sets": {
                key: {
                    "label": value["label"],
                    "display_name": value["display_name"],
                    "raw_path": str(value["raw"]),
                    "processed_sft_path": str(value["sft"]),
                }
                for key, value in TESTSET_SPECS.items()
            },
            "comparisons": [
                "base_native_chat_on_gptplus",
                "base_native_chat_on_claude",
            ],
            "generation": {
                "decoding": "greedy",
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
                "padding": False,
                "truncation": False,
                "prompt_interface": "native_chat_template",
                "chat_template_method": "tokenizer.apply_chat_template",
                "messages": '[{"role": "user", "content": prompt.rstrip("\\r\\n")}]',
                "add_generation_prompt": True,
                "single_example_generation": True,
                "audit_note": (
                    "Single-example generation is retained because the "
                    "inference audit showed output instability with padded "
                    "batched generation for LFM2."
                ),
                "comparison_note": (
                    "Greedy decoding is intentionally kept identical to the "
                    "SFT-prompt evaluation so that prompt formatting is the "
                    "primary experimental difference."
                ),
                "seed": SEED,
            },
            "metrics": {
                "rouge": {
                    "types": ["rouge1", "rouge2"],
                    "value": "F1",
                    "use_stemmer": True,
                    "two_reference_strategy": (
                        "Best one-to-one assignment between up to two generated "
                        "alternatives and the two references; a missing alternative "
                        "receives zero."
                    ),
                },
                "bertscore": {
                    "enabled": not args.skip_bertscore,
                    "model": args.bertscore_model if not args.skip_bertscore else None,
                    "value": "F1",
                    "rescale_with_baseline": (
                        args.bertscore_rescale
                        if not args.skip_bertscore
                        else None
                    ),
                    "hash": bert_hash,
                },
                "normalized_exact_match": {
                    "normalization": (
                        "Unicode NFKC, lowercase, punctuation removal, whitespace collapse."
                    ),
                    "two_reference_strategy": (
                        "Maximum one-to-one exact matches divided by 2."
                    ),
                },
                "structured_consistency": {
                    "automatic_checks": [
                        "currency contradiction",
                        "currency-qualified amount contradiction",
                        "temporal contradiction (month/year/quarter/installment)",
                    ],
                    "amount_parser": (
                        "Line-bounded currency/amount parsing; horizontal whitespace only; "
                        "source-supported 20xx tokens adjacent to currency are treated as "
                        "temporal years rather than amounts."
                    ),
                    "temporal_parser": (
                        "20xx years embedded in decimal/thousands numbers are ignored; "
                        "currency-adjacent year-like tokens are disambiguated using source "
                        "temporal context."
                    ),
                    "important_limitation": (
                        "This is a conservative structured check, not a complete "
                        "semantic hallucination detector; qualitative review remains necessary."
                    ),
                },
                "output_parser": {
                    "strategy": (
                        "JSON list, named alternatives/descriptions, numbered alternatives, "
                        "conservative plain-line fallback, pipe-separated fallback."
                    ),
                    "line_anchored_markers": True,
                    "trailing_commentary_removed": True,
                    "structured_field_lines_ignored_in_plain_fallback": True,
                    "structured_checks_use_full_raw_prediction": True,
                },
            },
            "limit_per_test_set": args.limit,
        },
        "experiments": {},
    }

    for experiment_key, scored_records in scored_by_experiment.items():
        model_key, _, testset_name = experiment_key.partition("_on_")
        summary["experiments"][experiment_key] = {
            "model_key": model_key,
            "model_label": MODEL_SPECS[model_key]["label"],
            "test_set_key": testset_name,
            "test_set_label": TESTSET_SPECS[testset_name]["label"],
            "metrics": build_experiment_summary(scored_records),
        }

    summary_path = RESULTS_DIR / "evaluation_run_details.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    save_summary_csv(summary)
    save_activity_csv(summary)
    save_qualitative_review(scored_by_experiment, max_records=60)

    print("\n" + "=" * 72)
    print("EVALUATION COMPLETED")
    print("=" * 72)
    print("\nGenerated files:")
    print(f"  {summary_path}")
    print(f"  {RESULTS_DIR / 'model_comparison_summary.csv'}")
    print(f"  {RESULTS_DIR / 'model_comparison_by_activity.csv'}")
    print(f"  {RESULTS_DIR / 'qualitative_review_cases.jsonl'}")
    print(f"  {PREDICTIONS_DIR}/")


if __name__ == "__main__":
    main()
