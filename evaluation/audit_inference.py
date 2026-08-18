#!/usr/bin/env python3

import argparse
import csv
import gc
import json
import re
import sys
import unicodedata
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from training.training_config import MAX_SEQ_LENGTH, SEED
except Exception:
    # Fallback to the values used in the current project configuration.
    MAX_SEQ_LENGTH = 256
    SEED = 42

MODEL_SPECS = {
    "gptplus_ft": {
        "label": "LFM2-700M_GPTPlus-DS",
        "path": PROJECT_ROOT / "models" / "lfm2_700m_gptplus_merged",
    },
    "claude_ft": {
        "label": "LFM2-700M_Claude-DS",
        "path": PROJECT_ROOT / "models" / "lfm2_700m_claude_merged",
    },
    "base": {
        "label": "LFM2-700M",
        "path": "LiquidAI/LFM2-700M",
    },
}

TESTSET_SPECS = {
    "gptplus": {
        "label": "GPTPlus",
        "raw": PROJECT_ROOT / "data_GptPlus" / "splits" / "test.jsonl",
        "sft": PROJECT_ROOT / "data_GptPlus" / "processed" / "test_sft.jsonl",
    },
    "claude": {
        "label": "Claude",
        "raw": PROJECT_ROOT / "data_Claude" / "splits" / "test.jsonl",
        "sft": PROJECT_ROOT / "data_Claude" / "processed" / "test_sft.jsonl",
    },
}

# Known problematic records observed in the previous qualitative review.
# The audit also adds deterministic control records until --cases-per-testset is reached.
KNOWN_FAILURE_IDS = {
    "gptplus": [
        "EX1768",
        "EX0027",
        "EX0032",
        "EX0882",
        "EX0232",
    ],
    "claude": [
        "EX1252",
        "EX0892",
        "EX1301",
        "EX1869",
        "EX1606",
        "EX1538",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic audit for prompt construction, batch-size effects, "
            "raw generation slicing, and alternative parsing."
        )
    )
    parser.add_argument(
        "--cases-per-testset",
        type=int,
        default=10,
        help="Number of records selected from each test set (default: 10).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum new tokens generated per prompt (default: 64).",
    )
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="Also audit the original base model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing audit output directory.",
    )
    return parser.parse_args()


def read_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path}, line {line_number}: {exc}"
                ) from exc
    return records


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def clean_alternative_prefix(text):
    text = str(text).strip()
    patterns = [
        r"^\s*[-*•]\s*",
        r"^\s*\d+\s*[\.\):\-]\s*",
        r"^\s*alternative\s*\d+\s*[\.\):\-]\s*",
        r"^\s*option\s*\d+\s*[\.\):\-]\s*",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_alternatives(text):
    """Mirror the parser used by evaluate_model.py."""
    text = str(text).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            value = json.loads(text)
            if isinstance(value, list):
                return [
                    clean_alternative_prefix(str(x))
                    for x in value
                    if str(x).strip()
                ][:2]
        except json.JSONDecodeError:
            pass

    explicit = re.search(
        r"(?:alternative|option)\s*1\s*[\.\):\-]\s*(.+?)"
        r"\s+(?:alternative|option)\s*2\s*[\.\):\-]\s*(.+)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if explicit:
        return [
            clean_alternative_prefix(explicit.group(1)),
            clean_alternative_prefix(explicit.group(2)),
        ]

    numbered = re.search(
        r"^\s*1\s*[\.\):\-]\s*(.+?)"
        r"\s+2\s*[\.\):\-]\s*(.+)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if numbered:
        return [
            clean_alternative_prefix(numbered.group(1)),
            clean_alternative_prefix(numbered.group(2)),
        ]

    lines = []
    for line in text.splitlines():
        line = clean_alternative_prefix(line)
        if not line:
            continue
        if normalize_text(line) in {
            "alternatives",
            "alternative descriptions",
            "bank transfer descriptions",
            "output",
            "response",
        }:
            continue
        lines.append(line)

    if len(lines) >= 2:
        return lines[:2]

    if len(lines) == 1:
        pipe_parts = [
            clean_alternative_prefix(x)
            for x in re.split(r"\s*\|\s*", lines[0])
            if x.strip()
        ]
        if len(pipe_parts) >= 2:
            return pipe_parts[:2]
        return [lines[0]]

    return []


def two_distinct(alternatives):
    if len(alternatives) != 2:
        return False
    a = normalize_text(alternatives[0])
    b = normalize_text(alternatives[1])
    return bool(a and b and a != b)


def reference_in_completion(reference, completion):
    ref = normalize_text(reference)
    comp = normalize_text(completion)
    return bool(ref) and (ref in comp or comp in ref)


def load_and_validate_testset(testset_name):
    spec = TESTSET_SPECS[testset_name]
    raw_records = read_jsonl(spec["raw"])
    sft_records = read_jsonl(spec["sft"])

    if len(raw_records) != len(sft_records):
        raise ValueError(
            f"{testset_name}: raw/SFT size mismatch: "
            f"{len(raw_records)} vs {len(sft_records)}"
        )

    trailing_newline_count = 0
    prompt_lengths = []

    for index, (raw, sft) in enumerate(zip(raw_records, sft_records), start=1):
        refs = raw.get("expected_output")
        if not isinstance(refs, list) or len(refs) != 2:
            raise ValueError(
                f"{testset_name}: record {index} does not have exactly 2 references."
            )

        prompt = sft.get("prompt")
        completion = sft.get("completion")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"{testset_name}: invalid prompt at position {index}."
            )
        if not isinstance(completion, str) or not completion.strip():
            raise ValueError(
                f"{testset_name}: invalid completion at position {index}."
            )

        # Stronger audit than the old evaluator: BOTH references must be
        # represented in the processed completion.
        missing = [
            ref for ref in refs
            if not reference_in_completion(ref, completion)
        ]
        if missing:
            raise ValueError(
                f"{testset_name}: raw/SFT alignment failure at position {index}, "
                f"id={raw.get('id_example')}. Missing reference(s) from completion: "
                f"{missing}"
            )

        if prompt.endswith(("\n", "\r")):
            trailing_newline_count += 1

        prompt_lengths.append(len(prompt))

    return raw_records, sft_records, {
        "records": len(raw_records),
        "prompts_with_trailing_newline": trailing_newline_count,
        "max_prompt_characters": max(prompt_lengths) if prompt_lengths else 0,
    }


def select_records(testset_name, raw_records, sft_records, n):
    if n <= 0:
        raise ValueError("--cases-per-testset must be positive.")

    by_id = {
        raw["id_example"]: (raw, sft)
        for raw, sft in zip(raw_records, sft_records)
    }

    selected = []
    selected_ids = set()

    for example_id in KNOWN_FAILURE_IDS[testset_name]:
        if example_id in by_id and len(selected) < n:
            raw, sft = by_id[example_id]
            selected.append((raw, sft, True))
            selected_ids.add(example_id)

    if len(selected) < n:
        # Deterministic controls spread across the test set.
        total = len(raw_records)
        needed = n - len(selected)
        if total == 0:
            raise ValueError(f"{testset_name}: empty test set.")

        candidate_indices = []
        if needed == 1:
            candidate_indices = [total // 2]
        else:
            for i in range(needed * 3 + 10):
                idx = round(i * (total - 1) / max(1, (needed * 3 + 9)))
                candidate_indices.append(idx)

        for idx in candidate_indices:
            raw = raw_records[idx]
            example_id = raw["id_example"]
            if example_id in selected_ids:
                continue
            selected.append((raw, sft_records[idx], False))
            selected_ids.add(example_id)
            if len(selected) == n:
                break

    # Final fallback if the evenly-spaced pass encountered too many duplicates.
    if len(selected) < n:
        for raw, sft in zip(raw_records, sft_records):
            if raw["id_example"] in selected_ids:
                continue
            selected.append((raw, sft, False))
            selected_ids.add(raw["id_example"])
            if len(selected) == n:
                break

    return selected


def validate_model_paths(model_keys):
    for model_key in model_keys:
        path = MODEL_SPECS[model_key]["path"]
        if isinstance(path, Path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Merged model not found for {model_key}:\n{path}"
                )
            if not (path / "config.json").exists():
                raise FileNotFoundError(
                    f"config.json not found in:\n{path}"
                )


def load_model_and_tokenizer(model_key, device):
    path = MODEL_SPECS[model_key]["path"]
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # The evaluator uses left padding for decoder-only batched generation.
    tokenizer.padding_side = "left"

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def strip_after_first_eos(token_ids, eos_token_id):
    ids = list(token_ids)
    if eos_token_id is None:
        return ids, False

    if eos_token_id in ids:
        pos = ids.index(eos_token_id)
        return ids[:pos], True

    return ids, False


@torch.inference_mode()
def generate_diagnostic(
    model,
    tokenizer,
    items,
    device,
    batch_size,
    max_new_tokens,
):
    """
    Generate from the exact audited prefix:
        prompt.rstrip("\\r\\n") + "\\n\\n"
    and preserve both the full decoded sequence and continuation-only output.
    """
    results = {}

    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        generation_prompts = [
            item["prompt"].rstrip("\r\n") + "\n\n"
            for item in batch
        ]

        # No selected prompt is expected to exceed MAX_SEQ_LENGTH.
        raw_token_lengths = [
            len(
                tokenizer(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]
            )
            for text in generation_prompts
        ]
        too_long = [
            length > MAX_SEQ_LENGTH
            for length in raw_token_lengths
        ]
        if any(too_long):
            offending = [
                batch[i]["id_example"]
                for i, flag in enumerate(too_long)
                if flag
            ]
            raise RuntimeError(
                f"Prompt(s) exceed MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}: {offending}"
            )

        encoded = tokenizer(
            generation_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        input_width = encoded["input_ids"].shape[1]
        continuation_ids = generated[:, input_width:]

        for row_index, item in enumerate(batch):
            full_ids = generated[row_index].detach().cpu().tolist()
            cont_ids_all = continuation_ids[row_index].detach().cpu().tolist()

            meaningful_cont_ids, eos_seen = strip_after_first_eos(
                cont_ids_all,
                tokenizer.eos_token_id,
            )

            raw_output = tokenizer.decode(
                meaningful_cont_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            full_decoded = tokenizer.decode(
                full_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            results[item["id_example"]] = {
                "raw_prediction": raw_output,
                "predicted_alternatives": extract_alternatives(raw_output),
                "full_decoded_sequence": full_decoded,
                "prompt_token_count_unpadded": raw_token_lengths[row_index],
                "batch_input_width_padded": input_width,
                "generated_token_count_before_eos": len(meaningful_cont_ids),
                "eos_seen": eos_seen,
                "hit_max_new_tokens_without_eos": (
                    not eos_seen
                    and len(cont_ids_all) >= max_new_tokens
                ),
            }

    return results


def make_item(raw, sft, known_failure):
    return {
        "id_example": raw["id_example"],
        "activity_type": raw.get("activity_type"),
        "operation_category": raw.get("operation_category"),
        "prompt": sft["prompt"],
        "training_completion": sft["completion"],
        "references": raw["expected_output"],
        "prompt_source_record": {
            "beneficiary": raw.get("beneficiary"),
            "amount": raw.get("amount"),
            "currency": raw.get("currency"),
            "reference_period": raw.get("reference_period"),
            "input_text": raw.get("input_text"),
            "calendar_context": raw.get("calendar_context"),
        },
        "known_failure_candidate": known_failure,
    }


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["model"], row["test_set"])
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for (model, test_set), records in groups.items():
        n = len(records)
        summary_rows.append({
            "model": model,
            "test_set": test_set,
            "records": n,
            "batch1_vs_batch8_raw_equal": sum(
                1 for r in records if r["raw_prediction_equal"]
            ),
            "batch1_vs_batch8_raw_equal_rate": (
                sum(1 for r in records if r["raw_prediction_equal"]) / n
            ),
            "batch1_empty_outputs": sum(
                1 for r in records if not r["batch1"]["raw_prediction"]
            ),
            "batch8_empty_outputs": sum(
                1 for r in records if not r["batch8"]["raw_prediction"]
            ),
            "batch1_two_distinct": sum(
                1 for r in records
                if two_distinct(r["batch1"]["predicted_alternatives"])
            ),
            "batch8_two_distinct": sum(
                1 for r in records
                if two_distinct(r["batch8"]["predicted_alternatives"])
            ),
            "batch1_hit_max_without_eos": sum(
                1 for r in records
                if r["batch1"]["hit_max_new_tokens_without_eos"]
            ),
            "batch8_hit_max_without_eos": sum(
                1 for r in records
                if r["batch8"]["hit_max_new_tokens_without_eos"]
            ),
        })
    return summary_rows


def write_summary_csv(path, rows):
    fieldnames = [
        "model",
        "test_set",
        "records",
        "batch1_vs_batch8_raw_equal",
        "batch1_vs_batch8_raw_equal_rate",
        "batch1_empty_outputs",
        "batch8_empty_outputs",
        "batch1_two_distinct",
        "batch8_two_distinct",
        "batch1_hit_max_without_eos",
        "batch8_hit_max_without_eos",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text_report(path, dataset_audit, summary_rows):
    with path.open("w", encoding="utf-8") as f:
        f.write("LFM2 INFERENCE AUDIT\n")
        f.write("=" * 72 + "\n\n")

        f.write("DATASET / PROMPT ALIGNMENT\n")
        f.write("-" * 72 + "\n")
        for testset_name, stats in dataset_audit.items():
            f.write(
                f"{testset_name}: records={stats['records']}, "
                f"prompts_with_trailing_newline="
                f"{stats['prompts_with_trailing_newline']}, "
                f"max_prompt_characters={stats['max_prompt_characters']}\n"
            )
        f.write(
            "\nAll test records passed the STRICT audit that BOTH original "
            "references are represented in the processed SFT completion.\n\n"
        )

        f.write("BATCH 1 VS BATCH 8\n")
        f.write("-" * 72 + "\n")
        for row in summary_rows:
            f.write(
                f"{row['model']} on {row['test_set']}: "
                f"raw_equal={row['batch1_vs_batch8_raw_equal']}/"
                f"{row['records']} "
                f"({row['batch1_vs_batch8_raw_equal_rate']:.3f}), "
                f"empty b1/b8={row['batch1_empty_outputs']}/"
                f"{row['batch8_empty_outputs']}, "
                f"two-distinct b1/b8={row['batch1_two_distinct']}/"
                f"{row['batch8_two_distinct']}, "
                f"hit-max-no-EOS b1/b8="
                f"{row['batch1_hit_max_without_eos']}/"
                f"{row['batch8_hit_max_without_eos']}\n"
            )

        f.write("\nINTERPRETATION GUIDE\n")
        f.write("-" * 72 + "\n")
        f.write(
            "1) If raw_equal is 100%, batch padding/batch size is not causing "
            "the observed outputs on these records.\n"
        )
        f.write(
            "2) If raw_prediction itself is degenerate, the parser did not "
            "create the degeneration; it came from model generation.\n"
        )
        f.write(
            "3) If raw_prediction looks correct but predicted_alternatives is "
            "wrong/empty, the parser is the problem for that record.\n"
        )
        f.write(
            "4) full_decoded_sequence lets you verify that continuation slicing "
            "starts after the complete prompt.\n"
        )
        f.write(
            "5) hit_max_new_tokens_without_eos=True on repetitive outputs is "
            "evidence that the model entered a loop and never emitted EOS.\n"
        )


def main():
    args = parse_args()

    if args.cases_per_testset <= 0:
        raise ValueError("--cases-per-testset must be positive.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")

    output_dir = PROJECT_ROOT / "evaluation" / "audit_results"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise RuntimeError(
                f"Audit output directory is not empty:\n{output_dir}\n"
                "Use --overwrite to replace its contents."
            )
        for child in output_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                import shutil
                shutil.rmtree(child)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("LFM2-700M INFERENCE AUDIT")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Cases per test set: {args.cases_per_testset}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Output directory: {output_dir}")

    dataset_audit = {}
    selected_by_testset = {}

    for testset_name in ("gptplus", "claude"):
        raw_records, sft_records, stats = load_and_validate_testset(testset_name)
        dataset_audit[testset_name] = stats
        selected = select_records(
            testset_name,
            raw_records,
            sft_records,
            args.cases_per_testset,
        )
        selected_by_testset[testset_name] = [
            make_item(raw, sft, known_failure)
            for raw, sft, known_failure in selected
        ]

        print(
            f"\n{TESTSET_SPECS[testset_name]['label']} alignment OK: "
            f"{stats['records']} records; "
            f"both references found in every processed completion."
        )
        print(
            "Selected IDs: "
            + ", ".join(
                item["id_example"]
                for item in selected_by_testset[testset_name]
            )
        )

    model_keys = ["gptplus_ft", "claude_ft"]
    if args.include_base:
        model_keys.insert(0, "base")

    validate_model_paths(model_keys)

    audit_rows = []

    for model_key in model_keys:
        print(f"\nLoading {MODEL_SPECS[model_key]['label']}...")
        model, tokenizer = load_model_and_tokenizer(model_key, device)

        for testset_name in ("gptplus", "claude"):
            items = selected_by_testset[testset_name]

            print(
                f"\n{MODEL_SPECS[model_key]['label']} "
                f"on {TESTSET_SPECS[testset_name]['label']} — batch_size=1"
            )
            batch1 = generate_diagnostic(
                model,
                tokenizer,
                items,
                device,
                batch_size=1,
                max_new_tokens=args.max_new_tokens,
            )

            print(
                f"{MODEL_SPECS[model_key]['label']} "
                f"on {TESTSET_SPECS[testset_name]['label']} — batch_size=8"
            )
            batch8 = generate_diagnostic(
                model,
                tokenizer,
                items,
                device,
                batch_size=8,
                max_new_tokens=args.max_new_tokens,
            )

            for item in items:
                example_id = item["id_example"]
                b1 = batch1[example_id]
                b8 = batch8[example_id]

                row = {
                    "model_key": model_key,
                    "model": MODEL_SPECS[model_key]["label"],
                    "test_set": TESTSET_SPECS[testset_name]["label"],
                    **item,
                    "generation_prompt": (
                        item["prompt"].rstrip("\r\n") + "\n\n"
                    ),
                    "prompt_had_trailing_newline": item["prompt"].endswith(
                        ("\n", "\r")
                    ),
                    "batch1": b1,
                    "batch8": b8,
                    "raw_prediction_equal": (
                        b1["raw_prediction"] == b8["raw_prediction"]
                    ),
                    "parsed_alternatives_equal": (
                        b1["predicted_alternatives"]
                        == b8["predicted_alternatives"]
                    ),
                }
                audit_rows.append(row)

        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_rows = summarize(audit_rows)

    details_path = output_dir / "inference_audit_details.jsonl"
    summary_path = output_dir / "inference_audit_summary.csv"
    report_path = output_dir / "inference_audit_report.txt"

    write_jsonl(details_path, audit_rows)
    write_summary_csv(summary_path, summary_rows)
    write_text_report(report_path, dataset_audit, summary_rows)

    print("\n" + "=" * 72)
    print("AUDIT COMPLETED")
    print("=" * 72)
    print("\nSummary:")
    for row in summary_rows:
        print(
            f"  {row['model']} on {row['test_set']}: "
            f"raw_equal "
            f"{row['batch1_vs_batch8_raw_equal']}/{row['records']}; "
            f"empty b1/b8 "
            f"{row['batch1_empty_outputs']}/{row['batch8_empty_outputs']}; "
            f"two-distinct b1/b8 "
            f"{row['batch1_two_distinct']}/{row['batch8_two_distinct']}; "
            f"hit-max-no-EOS b1/b8 "
            f"{row['batch1_hit_max_without_eos']}/"
            f"{row['batch8_hit_max_without_eos']}"
        )

    print("\nGenerated files:")
    print(f"  {details_path}")
    print(f"  {summary_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
