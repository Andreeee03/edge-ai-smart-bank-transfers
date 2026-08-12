import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILES = {
    "train": SPLITS_DIR / "train_5000.jsonl",
    "validation": SPLITS_DIR / "validation_5000.jsonl",
    "test": SPLITS_DIR / "test_5000.jsonl",
}

OUTPUT_FILES = {
    "train": PROCESSED_DIR / "train_sft_5000.jsonl",
    "validation": PROCESSED_DIR / "validation_sft_5000.jsonl",
    "test": PROCESSED_DIR / "test_sft_5000.jsonl",
}


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(file_path):
    records = []

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                records.append(record)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {file_path.name} "
                    f"at line {line_number}: {exc}"
                )

    return records


# ============================================================
# FORMAT HELPERS
# ============================================================

def format_amount(amount):
    """
    Format amounts without unnecessary trailing zeros.

    Examples:
    750.0   -> 750
    35.50   -> 35.5
    123.45  -> 123.45
    """

    if isinstance(amount, int):
        return str(amount)

    if isinstance(amount, float):
        return f"{amount:.2f}".rstrip("0").rstrip(".")

    return str(amount)


def add_common_fields(parts, record):
    """
    Add structured transaction fields to the prompt.
    """

    parts.append(
        f"Category: {record['operation_category']}"
    )

    parts.append(
        f"Beneficiary: {record['beneficiary']}"
    )

    amount = format_amount(record["amount"])
    currency = record["currency"]

    parts.append(
        f"Amount: {amount} {currency}"
    )

    reference_period = record.get("reference_period")

    if reference_period is not None:
        parts.append(
            f"Reference period: {reference_period}"
        )


def add_calendar_context(parts, calendar_context):
    """
    Add calendar information when available.
    """

    if calendar_context is None:
        return

    parts.append("Calendar context:")

    parts.append(
        f"- Event: {calendar_context['event_title']}"
    )

    parts.append(
        f"- Date: {calendar_context['event_date']}"
    )

    parts.append(
        f"- Event category: "
        f"{calendar_context['event_category']}"
    )


# ============================================================
# PROMPT BUILDERS
# ============================================================

def build_generation_prompt(record):
    parts = [
        (
            "Generate exactly two concise and natural "
            "bank-transfer descriptions using only the "
            "information provided."
        ),
        (
            "Return two alternative descriptions without "
            "adding unsupported information."
        ),
    ]

    add_common_fields(parts, record)

    add_calendar_context(
        parts,
        record.get("calendar_context"),
    )

    return "\n".join(parts)


def build_completion_prompt(record):
    parts = [
        (
            "Complete the following partially written "
            "bank-transfer description."
        ),
        (
            "Generate exactly two concise and natural "
            "completed alternatives using only the "
            "information provided."
        ),
    ]

    add_common_fields(parts, record)

    parts.append(
        f"Partial description: {record['input_text']}"
    )

    return "\n".join(parts)


def build_normalization_prompt(record):
    parts = [
        (
            "Normalize the following bank-transfer "
            "description by making it clear, concise "
            "and natural."
        ),
        (
            "Generate exactly two alternative normalized "
            "descriptions while preserving the original "
            "meaning and without adding unsupported "
            "information."
        ),
    ]

    add_common_fields(parts, record)

    parts.append(
        f"Original description: {record['input_text']}"
    )

    return "\n".join(parts)


def build_prompt(record):
    activity = record["activity_type"]

    if activity == "GENERATION":
        return build_generation_prompt(record)

    if activity == "COMPLETION":
        return build_completion_prompt(record)

    if activity == "NORMALIZATION":
        return build_normalization_prompt(record)

    raise ValueError(
        f"Unsupported activity_type: {activity}"
    )


# ============================================================
# COMPLETION BUILDER
# ============================================================

def build_completion(record):
    expected_output = record["expected_output"]

    if not isinstance(expected_output, list):
        raise ValueError(
            f"{record['id_example']}: "
            "expected_output must be a list."
        )

    if len(expected_output) != 2:
        raise ValueError(
            f"{record['id_example']}: "
            "expected_output must contain exactly 2 items."
        )

    first = expected_output[0].strip()
    second = expected_output[1].strip()

    return f"1. {first}\n2. {second}"


# ============================================================
# CONVERT RECORD
# ============================================================

def convert_record(record):
    """
    Convert one master-dataset record into SFT
    prompt-completion format.
    """

    return {
        "prompt": build_prompt(record),
        "completion": build_completion(record),
    }


# ============================================================
# SAVE JSONL
# ============================================================

def save_jsonl(records, file_path):
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open("w", encoding="utf-8") as file:

        for record in records:

            json.dump(
                record,
                file,
                ensure_ascii=False,
            )

            file.write("\n")


# ============================================================
# VALIDATION
# ============================================================

def validate_processed_dataset(
    original_records,
    processed_records,
    split_name,
):
    errors = []

    # Same number of records before and after conversion
    if len(original_records) != len(processed_records):
        errors.append(
            f"{split_name}: record count changed "
            "during preprocessing."
        )

    for index, record in enumerate(processed_records):

        # Exactly two fields
        if set(record.keys()) != {
            "prompt",
            "completion",
        }:
            errors.append(
                f"{split_name}, record {index + 1}: "
                "invalid output fields."
            )

        prompt = record.get("prompt")
        completion = record.get("completion")

        if not isinstance(prompt, str) or not prompt.strip():
            errors.append(
                f"{split_name}, record {index + 1}: "
                "empty prompt."
            )

        if (
            not isinstance(completion, str)
            or not completion.strip()
        ):
            errors.append(
                f"{split_name}, record {index + 1}: "
                "empty completion."
            )

        # The completion should contain the two
        # numbered alternatives.
        if not completion.startswith("1. "):
            errors.append(
                f"{split_name}, record {index + 1}: "
                "completion does not start with '1. '."
            )

        if "\n2. " not in completion:
            errors.append(
                f"{split_name}, record {index + 1}: "
                "completion does not contain second output."
            )

    return errors


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split_name):
    input_file = INPUT_FILES[split_name]
    output_file = OUTPUT_FILES[split_name]

    print(f"\nProcessing {split_name}...")
    print(f"Input:  {input_file}")

    original_records = load_jsonl(input_file)

    processed_records = [
        convert_record(record)
        for record in original_records
    ]

    errors = validate_processed_dataset(
        original_records,
        processed_records,
        split_name,
    )

    if errors:
        print(
            f"\nValidation failed for {split_name}."
        )

        for error in errors[:20]:
            print(f"- {error}")

        raise RuntimeError(
            f"SFT preprocessing failed for {split_name}."
        )

    save_jsonl(
        processed_records,
        output_file,
    )

    print(
        f"Records processed: "
        f"{len(processed_records)}"
    )

    print(
        f"Output: {output_file}"
    )

    return (
        len(original_records),
        len(processed_records),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("SFT DATASET PREPROCESSING")
    print("=" * 60)

    results = {}

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        original_count, processed_count = (
            process_split(split_name)
        )

        results[split_name] = {
            "original": original_count,
            "processed": processed_count,
        }

    # --------------------------------------------------------
    # Final dataset-size validation
    # --------------------------------------------------------

    expected_sizes = {
        "train": 4000,
        "validation": 500,
        "test": 500,
    }

    final_errors = []

    for split_name, expected_size in expected_sizes.items():

        actual_size = results[split_name]["processed"]

        if actual_size != expected_size:
            final_errors.append(
                f"{split_name}: expected "
                f"{expected_size}, found {actual_size}"
            )

    if final_errors:

        print("\n" + "=" * 60)
        print("FINAL VALIDATION FAILED")
        print("=" * 60)

        for error in final_errors:
            print(f"- {error}")

        raise RuntimeError(
            "Processed dataset sizes are incorrect."
        )

    print("\n" + "=" * 60)
    print("SFT PREPROCESSING PASSED")
    print("=" * 60)

    print("\nFinal processed datasets:")

    print(
        f"Train:      "
        f"{results['train']['processed']}"
    )

    print(
        f"Validation: "
        f"{results['validation']['processed']}"
    )

    print(
        f"Test:       "
        f"{results['test']['processed']}"
    )

    print("\nFiles saved in:")
    print(PROCESSED_DIR)


if __name__ == "__main__":
    main()