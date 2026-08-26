# ============================================================
# FINAL NATIVE-CHAT BASELINE EVALUATION
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
    """
    Remove only genuine list/alternative wrappers.

    Important: a leading '*' is removed only when it is actually used as a
    bullet followed by whitespace. This preserves Markdown such as
    '**Category:**'.
    """
    text = str(text).strip()

    patterns = [
        r"^\s*[-*•]\s+",
        r"^\s*\d+\s*[\.\):\-](?:[ \t]+|$)",
        r"^\s*(?:alternative|option)\s*\d+\s*[\.\):\-]\s*",
        r"^\s*(?:normalized\s+description|description)\s*\d+\s*[\.\):\-]\s*",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove an outer code fence without modifying inner Markdown.
    text = re.sub(
        r"^\s*```(?:text)?[ \t]*(?:\r?\n)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?:\r?\n)?[ \t]*```\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


# Markdown-aware line prefix used by explicit alternative markers.
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
    """Remove an explicit explanation/reasoning tail."""
    text = str(text).strip()
    match = TRAILING_COMMENTARY_PATTERN.search(text)
    if match:
        text = text[:match.start()]
    return text.strip()


def _clean_alternative_body(text):
    return trim_trailing_commentary(clean_alternative_prefix(text))


def _all_explicit_marker_matches(text):
    """
    Return all recognized numbered/named output markers in source order.

    These positions are also used as boundaries, so a later alternative block
    is never accidentally appended to the second alternative of an earlier
    complete pair.
    """
    items = []

    for family, pattern in (
        ("named", NAMED_ALTERNATIVE_MARKER_PATTERN),
        ("numbered", NUMBERED_ALTERNATIVE_MARKER_PATTERN),
    ):
        for match in pattern.finditer(text):
            items.append(
                {
                    "family": family,
                    "match": match,
                    "start": match.start(),
                    "end": match.end(),
                    "num": match.group("num"),
                }
            )

    items.sort(key=lambda item: (item["start"], item["end"]))
    return items


def _next_explicit_boundary(text, position, all_markers):
    """
    Find the next recognized output marker or commentary header.

    This prevents outputs such as:

        1. first description
        2. second description
        Alternative 1: ...

    from contaminating the body of numbered alternative 2.
    """
    candidates = [
        item["start"]
        for item in all_markers
        if item["start"] >= position
    ]

    commentary = TRAILING_COMMENTARY_PATTERN.search(text, pos=position)
    if commentary:
        candidates.append(commentary.start())

    return min(candidates) if candidates else len(text)


def _complete_pair_candidates(text, family, pattern, all_markers):
    """
    Build every valid 1 -> 2 pair for one marker family.

    A repeated marker 1 before marker 2 starts a new candidate instead of
    pairing across two unrelated blocks.
    """
    matches = list(pattern.finditer(text))
    candidates = []

    for index, first in enumerate(matches):
        if first.group("num") != "1":
            continue

        second = None
        for following in matches[index + 1:]:
            if following.group("num") == "1":
                # A new block started before a matching "2".
                break
            if following.group("num") == "2":
                second = following
                break

        if second is None:
            continue

        first_body = _clean_alternative_body(
            text[first.end():second.start()]
        )

        second_end = _next_explicit_boundary(
            text,
            second.end(),
            [
                item
                for item in all_markers
                if not (
                    item["family"] == family
                    and item["start"] == second.start()
                    and item["end"] == second.end()
                )
            ],
        )
        second_body = _clean_alternative_body(
            text[second.end():second_end]
        )

        alternatives = [
            value
            for value in (first_body, second_body)
            if value
        ]

        if len(alternatives) == 2:
            candidates.append(
                {
                    "family": family,
                    "start": first.start(),
                    "end": second_end,
                    "alternatives": alternatives,
                    "complete": True,
                }
            )

    return candidates


def _single_marker_candidates(text, family, pattern, all_markers):
    """
    Build conservative one-alternative candidates.

    This fallback is used only when no complete 1/2 pair exists anywhere in
    the response. It preserves a genuine first alternative when generation is
    truncated before alternative 2.
    """
    candidates = []

    for match in pattern.finditer(text):
        if match.group("num") != "1":
            continue

        boundary = _next_explicit_boundary(
            text,
            match.end(),
            [
                item
                for item in all_markers
                if not (
                    item["family"] == family
                    and item["start"] == match.start()
                    and item["end"] == match.end()
                )
            ],
        )

        body = _clean_alternative_body(
            text[match.end():boundary]
        )

        if body:
            candidates.append(
                {
                    "family": family,
                    "start": match.start(),
                    "end": boundary,
                    "alternatives": [body],
                    "complete": False,
                }
            )

    return candidates


def extract_alternatives(text):
    """
    Return up to two generated bank-transfer descriptions.

    Parsing priority is deliberately conservative:

    1. JSON list;
    2. earliest COMPLETE explicit pair among:
       - named/Markdown markers (Alternative, Option, Description, ...)
       - numbered markers (1., 2.)
    3. earliest single explicit marker only if no complete pair exists;
    4. conservative two-line fallback;
    5. pipe-separated fallback;
    6. one non-empty plain line.

    Choosing the earliest complete pair fixes the native-chat pattern where
    LFM2 first emits:

        1. valid description
        2. valid description

    and then starts an additional "Alternative 1:" block. The complete
    numbered pair must be scored, not the later truncated block.
    """
    text = str(text).strip()
    if not text:
        return []

    # JSON list.
    if text.startswith("[") and text.endswith("]"):
        try:
            value = json.loads(text)
            if isinstance(value, list):
                parsed = [
                    _clean_alternative_body(str(item))
                    for item in value
                    if str(item).strip()
                ]
                return [item for item in parsed if item][:2]
        except json.JSONDecodeError:
            pass

    all_markers = _all_explicit_marker_matches(text)

    complete_candidates = []
    complete_candidates.extend(
        _complete_pair_candidates(
            text,
            "named",
            NAMED_ALTERNATIVE_MARKER_PATTERN,
            all_markers,
        )
    )
    complete_candidates.extend(
        _complete_pair_candidates(
            text,
            "numbered",
            NUMBERED_ALTERNATIVE_MARKER_PATTERN,
            all_markers,
        )
    )

    if complete_candidates:
        chosen = min(
            complete_candidates,
            key=lambda item: (item["start"], item["end"]),
        )
        return chosen["alternatives"][:2]

    # No complete pair exists: preserve the earliest explicit first
    # alternative if generation stopped before the second one.
    single_candidates = []
    single_candidates.extend(
        _single_marker_candidates(
            text,
            "named",
            NAMED_ALTERNATIVE_MARKER_PATTERN,
            all_markers,
        )
    )
    single_candidates.extend(
        _single_marker_candidates(
            text,
            "numbered",
            NUMBERED_ALTERNATIVE_MARKER_PATTERN,
            all_markers,
        )
    )

    if single_candidates:
        chosen = min(
            single_candidates,
            key=lambda item: (item["start"], item["end"]),
        )
        return chosen["alternatives"][:1]

    # Conservative plain-line fallback. Structured field blocks are NOT
    # silently converted into bank-transfer alternatives.
    lines = []
    for raw_line in text.splitlines():
        line = _clean_alternative_body(raw_line)
        if not line:
            continue

        if re.fullmatch(
            r"\s*```(?:text)?\s*",
            line,
            flags=re.IGNORECASE,
        ):
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
            _clean_alternative_body(part)
            for part in re.split(r"[ \t]*\|[ \t]*", lines[0])
            if part.strip()
        ]
        pipe_parts = [part for part in pipe_parts if part]

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
CURRENCY_SYMBOL_TO_CODE = {
    "€": "EUR",
    "$": "USD",
    "£": "GBP",
}
CURRENCY_TOKEN_PATTERN = r"(?:EUR|USD|GBP|€|\$|£)"

CURRENCY_TOKEN_RE = re.compile(
    rf"(?<![A-Za-z])(?P<currency>{CURRENCY_TOKEN_PATTERN})(?![A-Za-z])",
    flags=re.IGNORECASE,
)

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

FULL_MONTH_NAMES = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}

MONTH_PATTERN = re.compile(
    r"\b(" + "|".join(
        sorted(MONTH_TO_NUMBER, key=len, reverse=True)
    ) + r")\b",
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
    rf"(?<![A-Za-z])"
    rf"(?P<currency>{CURRENCY_TOKEN_PATTERN})"
    rf"(?![A-Za-z])[ \t]*"
    rf"(?P<amount>{MONEY_NUMBER_PATTERN})(?![\d.,])",
    flags=re.IGNORECASE,
)

AMOUNT_FIRST_MONEY_PATTERN = re.compile(
    rf"(?<![\d.,])"
    rf"(?P<amount>{MONEY_NUMBER_PATTERN})[ \t]*"
    rf"(?P<currency>{CURRENCY_TOKEN_PATTERN})"
    rf"(?![A-Za-z])",
    flags=re.IGNORECASE,
)

CURRENCY_BEFORE_YEAR_PATTERN = re.compile(
    rf"(?<![A-Za-z])"
    rf"(?:{CURRENCY_TOKEN_PATTERN})"
    rf"(?![A-Za-z])[ \t]*$",
    flags=re.IGNORECASE,
)

CURRENCY_AFTER_YEAR_PATTERN = re.compile(
    rf"^[ \t]*(?:{CURRENCY_TOKEN_PATTERN})(?![A-Za-z])",
    flags=re.IGNORECASE,
)


def normalize_currency_token(token):
    """Map currency codes/symbols to canonical ISO-like codes."""
    token = str(token).strip()
    if token in CURRENCY_SYMBOL_TO_CODE:
        return CURRENCY_SYMBOL_TO_CODE[token]

    token_upper = token.upper()
    if token_upper in CURRENCY_CODES:
        return token_upper

    raise ValueError(f"Unsupported currency token: {token!r}")


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
    text = str(text)
    window = text[max(0, start - 24):min(len(text), end + 24)]

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


def month_has_local_temporal_context(text, start, end):
    """
    Decide whether a month token is being used temporally rather than as a
    name/word fragment.

    A nearby explicit year is a strong temporal signal. Source-authorized
    months are handled separately by extract_temporal_markers().
    """
    text = str(text)
    window = text[max(0, start - 20):min(len(text), end + 20)]

    if YEAR_PATTERN.search(window):
        return True
    if ISO_DATE_PATTERN.search(window):
        return True
    if re.search(
        r"\b(?:month|monthly|period|term|semester|quarter|due)\b",
        window,
        flags=re.IGNORECASE,
    ):
        return True

    return False


def extract_temporal_markers(
    text,
    allowed_currency_adjacent_years=None,
    allowed_months=None,
    assume_month_tokens_temporal=False,
):
    """
    Extract conservative temporal markers.

    Year handling:
    - EUR 2027 / €2027 -> monetary amount, not year;
    - 2027 EUR / 2027 € -> year only when source-supported or locally temporal;
    - years embedded in decimal/thousands amounts are ignored.

    Month handling:
    - authoritative source fields may opt into direct month interpretation;
    - generated/broader text treats a month token as temporal only when it is
      source-supported or has local temporal context.
      This prevents a truncated beneficiary such as "Daniel Nov" from being
      misread automatically as November.
    """
    text = str(text or "")
    low = text.lower()
    markers = set()

    allowed_currency_adjacent_years = {
        int(value)
        for value in (allowed_currency_adjacent_years or set())
    }
    allowed_months = {
        int(value)
        for value in (allowed_months or set())
    }

    for quarter in re.findall(r"\bq([1-4])\b", low):
        markers.add(f"quarter:{quarter}")

    for month_match in MONTH_PATTERN.finditer(low):
        month_token = month_match.group(1).lower()
        month_number = MONTH_TO_NUMBER[month_token]

        if (
            assume_month_tokens_temporal
            or month_number in allowed_months
            or month_has_local_temporal_context(
                low,
                month_match.start(),
                month_match.end(),
            )
        ):
            markers.add(f"month:{month_number}")

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

        before = low[max(0, start - 10):start]
        after = low[end:min(len(low), end + 10)]

        # "EUR 2027", "$2027", "€ 2027" -> explicit monetary amount.
        if CURRENCY_BEFORE_YEAR_PATTERN.search(before):
            continue

        # "2027 EUR" / "2027 €" is ambiguous.
        followed_by_currency = CURRENCY_AFTER_YEAR_PATTERN.match(after)
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

    return " ".join(
        str(value)
        for value in pieces
        if value is not None
    )


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

    return " ".join(
        str(value)
        for value in pieces
        if value is not None
    )


def source_temporal_markers(record):
    """
    Build temporal information allowed by the source record.

    Explicit reference_period/calendar fields are authoritative. Their years
    and months are then used to disambiguate broader source text.
    """
    authoritative = extract_temporal_markers(
        authoritative_temporal_text_for_record(record),
        assume_month_tokens_temporal=True,
    )

    authoritative_years = {
        int(marker.split(":", 1)[1])
        for marker in authoritative
        if marker.startswith("year:")
    }
    authoritative_months = {
        int(marker.split(":", 1)[1])
        for marker in authoritative
        if marker.startswith("month:")
    }

    broader = extract_temporal_markers(
        source_text_for_record(record),
        allowed_currency_adjacent_years=authoritative_years,
        allowed_months=authoritative_months,
    )

    return authoritative | broader


def monetary_mentions(text, temporal_years=None):
    """
    Return currency-qualified monetary values without crossing line boundaries.

    Supported currency forms:
      EUR 90, 90 EUR, €90, 90 €, $100, 100 USD, £50, ...

    For amount-first forms such as "2027 EUR" or "2027 €", a plain 20xx value
    is ignored as money when it is a source-supported temporal year or has a
    local temporal cue.
    """
    mentions = []
    temporal_years = {
        int(value)
        for value in (temporal_years or set())
    }

    for line in str(text).splitlines():
        occupied_spans = []

        # Currency-first.
        for match in CURRENCY_FIRST_MONEY_PATTERN.finditer(line):
            currency = normalize_currency_token(
                match.group("currency")
            )
            amount = parse_money_number(
                match.group("amount")
            )
            mentions.append((currency, amount))
            occupied_spans.append(match.span())

        # Amount-first. Avoid year/currency false positives.
        for match in AMOUNT_FIRST_MONEY_PATTERN.finditer(line):
            span = match.span()

            if any(
                not (span[1] <= start or span[0] >= end)
                for start, end in occupied_spans
            ):
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

            currency = normalize_currency_token(
                match.group("currency")
            )
            amount = parse_money_number(raw_amount)
            mentions.append((currency, amount))

    return mentions


def structured_consistency(record, predicted_alternatives, raw_prediction=None):
    """
    Conservative contradiction checker.

    Checks inspect the FULL raw response so malformed extra fields cannot be
    hidden by output parsing.
    """
    prediction_text = (
        str(raw_prediction).strip()
        if raw_prediction is not None
        else "\n".join(predicted_alternatives)
    )

    expected_currency = str(record["currency"]).upper()
    expected_amount = float(record["amount"])

    currencies = {
        normalize_currency_token(match.group("currency"))
        for match in CURRENCY_TOKEN_RE.finditer(prediction_text)
    }
    currency_consistent = all(
        currency == expected_currency
        for currency in currencies
    )

    allowed_temporal = source_temporal_markers(record)

    allowed_years = {
        int(marker.split(":", 1)[1])
        for marker in allowed_temporal
        if marker.startswith("year:")
    }
    allowed_months = {
        int(marker.split(":", 1)[1])
        for marker in allowed_temporal
        if marker.startswith("month:")
    }

    amount_consistent = True

    for currency, amount in monetary_mentions(
        prediction_text,
        temporal_years=allowed_years,
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

    predicted_temporal = extract_temporal_markers(
        prediction_text,
        allowed_currency_adjacent_years=allowed_years,
        allowed_months=allowed_months,
    )
    temporal_consistent = predicted_temporal.issubset(
        allowed_temporal
    )

    beneficiary = normalize_text(record["beneficiary"])
    prediction_norm = normalize_text(prediction_text)
    beneficiary_mentioned = (
        bool(beneficiary)
        and beneficiary in prediction_norm
    )

    calendar_reflected = None
    calendar = record.get("calendar_context")

    if isinstance(calendar, dict):
        event_title_tokens = {
            token
            for token in normalize_text(
                calendar.get("event_title", "")
            ).split()
            if len(token) >= 4
        }

        title_overlap = bool(
            event_title_tokens
            & set(prediction_norm.split())
        )

        calendar_temporal = extract_temporal_markers(
            calendar.get("event_date", ""),
            assume_month_tokens_temporal=True,
        )

        prediction_temporal = extract_temporal_markers(
            prediction_text,
            allowed_currency_adjacent_years=allowed_years,
            allowed_months=allowed_months,
        )

        date_overlap = bool(
            calendar_temporal
            & prediction_temporal
        )

        calendar_reflected = (
            title_overlap
            or date_overlap
        )

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
    Guard against every parser/checker regression found during evaluation.

    These checks do not use the model or GPU and execute at startup.
    """

    # ---------------------------------------------------------
    # Output parser
    # ---------------------------------------------------------

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
    ], f"Named parser sanity check failed: {parsed!r}"

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

    # Critical regression: a complete numbered pair comes before a later
    # truncated named block. The numbered pair must win.
    parsed = extract_alternatives(
        "1. Transferred 12.68 EUR to Elmwood Energy Services Associates.\n"
        "2. Bank transfer of 12.68 EUR for October 2026.\n\n"
        "Alternative 1:\n"
        "Deposited 12.68 EUR into Elmwood Energy Services Associates"
    )
    assert parsed == [
        "Transferred 12.68 EUR to Elmwood Energy Services Associates.",
        "Bank transfer of 12.68 EUR for October 2026.",
    ], f"Numbered-before-named priority sanity check failed: {parsed!r}"

    # If both complete pairs exist, the earliest complete pair is the primary
    # response and must be scored.
    parsed = extract_alternatives(
        "**Alternative 1:**\n"
        "First named description\n"
        "**Alternative 2:**\n"
        "Second named description\n\n"
        "1. Later numbered description\n"
        "2. Another later numbered description"
    )
    assert parsed == [
        "First named description",
        "Second named description",
    ], f"Earliest-complete-pair sanity check failed: {parsed!r}"

    # A single explicit marker is preserved only when no complete pair exists.
    parsed = extract_alternatives(
        "**Alternative 1:**\n"
        "First valid description only"
    )
    assert parsed == [
        "First valid description only"
    ], f"Single marked alternative sanity check failed: {parsed!r}"

    # Do not reinterpret arbitrary unnumbered structured sections as two
    # causali merely to improve the baseline.
    parsed = extract_alternatives(
        "**EDUCATION Category:**\n"
        "Beneficiary: Example School\n"
        "Amount: 340 EUR\n\n"
        "**School Fee Alternative:**\n"
        "340 EUR"
    )
    assert parsed == [], (
        f"Conservative structured-block sanity check failed: {parsed!r}"
    )

    # Preserve Markdown emphasis inside an extracted body.
    parsed = extract_alternatives(
        "**Alternative 1:**\n"
        "**Category:** RENT\n"
        "**Alternative 2:**\n"
        "Monthly rent EUR 855"
    )
    assert parsed[0].startswith("**Category:**"), (
        f"Markdown body preservation sanity check failed: {parsed!r}"
    )

    # ---------------------------------------------------------
    # Money / currency
    # ---------------------------------------------------------

    mentions = monetary_mentions(
        "water bill April 2027 EUR 90",
        temporal_years={2027},
    )
    assert mentions == [("EUR", 90.0)], (
        f"Money year/currency sanity check failed: {mentions!r}"
    )

    mentions = monetary_mentions(
        "Other EUR 444.68 for June 2026\n"
        "EUR 444.68 for June 2026",
        temporal_years={2026},
    )
    assert mentions == [
        ("EUR", 444.68),
        ("EUR", 444.68),
    ], f"Money newline sanity check failed: {mentions!r}"

    mentions = monetary_mentions(
        "Paid €201.28 and $1420.75 and £50"
    )
    assert mentions == [
        ("EUR", 201.28),
        ("USD", 1420.75),
        ("GBP", 50.0),
    ], f"Currency-symbol sanity check failed: {mentions!r}"

    # 2095.93 EUR is an amount, not year 2095.
    markers = extract_temporal_markers(
        "professional service EUR 2095.93"
    )
    assert "year:2095" not in markers, (
        f"Temporal decimal sanity check failed: {markers!r}"
    )

    # ---------------------------------------------------------
    # Temporal disambiguation
    # ---------------------------------------------------------

    markers = extract_temporal_markers(
        "Family Transfer - 955 EUR to Daniel Nov",
        allowed_months={5},
    )
    assert "month:11" not in markers, (
        f"Month-name false-positive sanity check failed: {markers!r}"
    )

    markers = extract_temporal_markers(
        "Phone payment for Nov 2026",
        allowed_months={11},
    )
    assert {"month:11", "year:2026"}.issubset(markers), (
        f"Month/year sanity check failed: {markers!r}"
    )

    # ---------------------------------------------------------
    # Full structured-consistency regression checks
    # ---------------------------------------------------------

    sample_record = {
        "operation_category": "TAX",
        "beneficiary": "Clearwater Public Services Office Center",
        "amount": 2011.28,
        "currency": "EUR",
        "reference_period": "Q4 2026",
        "input_text": "tax q4 2026 q4 2026",
        "calendar_context": None,
    }

    check = structured_consistency(
        sample_record,
        ["Tax settlement Q4 2026"],
        raw_prediction="Transferred €201.28 on 2026 Q4.",
    )
    assert check["amount_consistent"] is False, (
        "Symbol-qualified wrong amount must be detected."
    )

    check = structured_consistency(
        {
            **sample_record,
            "amount": 90.0,
            "reference_period": "April 2027",
        },
        ["April 2027 water bill EUR 90"],
        raw_prediction="April 2027 water bill EUR 90",
    )
    assert check["automatic_structured_consistency_pass"] is True, (
        f"Valid structured output sanity check failed: {check!r}"
    )


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
                        "EUR/USD/GBP codes plus €/$/£ symbols; source-supported 20xx tokens "
                        "adjacent to currency are treated as temporal years rather than amounts."
                    ),
                    "temporal_parser": (
                        "20xx years embedded in decimal/thousands numbers are ignored; "
                        "currency-adjacent year-like tokens and month tokens are "
                        "disambiguated using authoritative source temporal context."
                    ),
                    "important_limitation": (
                        "This is a conservative structured check, not a complete "
                        "semantic hallucination detector; qualitative review remains necessary."
                    ),
                },
                "output_parser": {
                    "strategy": (
                        "JSON list; earliest complete explicit pair across Markdown-aware "
                        "named markers and numbered 1/2 markers; single explicit first "
                        "alternative only when no complete pair exists; conservative "
                        "plain-line fallback; pipe-separated fallback."
                    ),
                    "line_anchored_markers": True,
                    "markdown_markers_supported": True,
                    "earliest_complete_pair_priority": True,
                    "cross_family_boundary_cutoff": True,
                    "single_marked_alternative_preserved": True,
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
