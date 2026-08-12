import json
import random
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data_GptPlus"
    / "cleaned"
    / "dataset_master_cleaned.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "data_GptPlus" / "splits"

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VALIDATION_FILE = OUTPUT_DIR / "validation.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"

RANDOM_SEED = 42

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    records = []

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                records.append(record)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                )

    return records


# ============================================================
# BUILD STRATIFICATION GROUP
# ============================================================

def get_stratification_key(record):
    """
    Stratify by activity type.

    GENERATION is additionally divided into:
    - with calendar
    - without calendar

    This preserves both activity distribution and
    calendar-context distribution across all splits.
    """

    activity = record["activity_type"]

    if activity == "GENERATION":
        if record.get("calendar_context") is None:
            return "GENERATION_NO_CALENDAR"

        return "GENERATION_WITH_CALENDAR"

    return activity


# ============================================================
# SPLIT ONE GROUP
# ============================================================

def split_group(records, rng):
    """
    Shuffle and split one homogeneous group into
    train / validation / test.
    """

    records = records.copy()
    rng.shuffle(records)

    total = len(records)

    train_count = round(total * TRAIN_RATIO)
    validation_count = round(total * VALIDATION_RATIO)

    # Test receives the remaining records so that
    # no record can be lost because of rounding.
    test_count = total - train_count - validation_count

    train_records = records[:train_count]

    validation_records = records[
        train_count:
        train_count + validation_count
    ]

    test_records = records[
        train_count + validation_count:
    ]

    assert len(train_records) == train_count
    assert len(validation_records) == validation_count
    assert len(test_records) == test_count

    return (
        train_records,
        validation_records,
        test_records,
    )


# ============================================================
# STRATIFIED RANDOM SPLIT
# ============================================================

def stratified_split(records):
    rng = random.Random(RANDOM_SEED)

    groups = defaultdict(list)

    for record in records:
        key = get_stratification_key(record)
        groups[key].append(record)

    train_records = []
    validation_records = []
    test_records = []

    print("\nStratification groups:")

    for group_name, group_records in sorted(groups.items()):
        (
            group_train,
            group_validation,
            group_test,
        ) = split_group(group_records, rng)

        train_records.extend(group_train)
        validation_records.extend(group_validation)
        test_records.extend(group_test)

        print(
            f"  {group_name}: "
            f"{len(group_records)} total -> "
            f"{len(group_train)} train, "
            f"{len(group_validation)} validation, "
            f"{len(group_test)} test"
        )

    # Shuffle each final split so records from the same
    # activity are not stored consecutively.
    rng.shuffle(train_records)
    rng.shuffle(validation_records)
    rng.shuffle(test_records)

    return (
        train_records,
        validation_records,
        test_records,
    )


# ============================================================
# ASSIGN SPLIT FIELD
# ============================================================

def assign_split(records, split_name):
    output = []

    for record in records:
        updated_record = record.copy()
        updated_record["split"] = split_name
        output.append(updated_record)

    return output


# ============================================================
# SAVE JSONL
# ============================================================

def save_jsonl(records, output_file):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
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

def validate_splits(
    original_records,
    train_records,
    validation_records,
    test_records,
):
    errors = []

    # --------------------------------------------------------
    # Total size
    # --------------------------------------------------------

    total_after_split = (
        len(train_records)
        + len(validation_records)
        + len(test_records)
    )

    if total_after_split != len(original_records):
        errors.append(
            "Total number of records changed after split."
        )

    # --------------------------------------------------------
    # Expected sizes
    # --------------------------------------------------------

    if len(train_records) != 4000:
        errors.append(
            f"Train must contain 4000 records, "
            f"found {len(train_records)}."
        )

    if len(validation_records) != 500:
        errors.append(
            f"Validation must contain 500 records, "
            f"found {len(validation_records)}."
        )

    if len(test_records) != 500:
        errors.append(
            f"Test must contain 500 records, "
            f"found {len(test_records)}."
        )

    # --------------------------------------------------------
    # Check IDs
    # --------------------------------------------------------

    original_ids = {
        record["id_example"]
        for record in original_records
    }

    train_ids = {
        record["id_example"]
        for record in train_records
    }

    validation_ids = {
        record["id_example"]
        for record in validation_records
    }

    test_ids = {
        record["id_example"]
        for record in test_records
    }

    if train_ids & validation_ids:
        errors.append(
            "Some records occur in both train and validation."
        )

    if train_ids & test_ids:
        errors.append(
            "Some records occur in both train and test."
        )

    if validation_ids & test_ids:
        errors.append(
            "Some records occur in both validation and test."
        )

    all_split_ids = (
        train_ids
        | validation_ids
        | test_ids
    )

    if all_split_ids != original_ids:
        errors.append(
            "Split IDs do not exactly match original dataset IDs."
        )

    # --------------------------------------------------------
    # Check split field
    # --------------------------------------------------------

    for record in train_records:
        if record.get("split") != "train":
            errors.append(
                f"{record['id_example']} has incorrect split value."
            )

    for record in validation_records:
        if record.get("split") != "validation":
            errors.append(
                f"{record['id_example']} has incorrect split value."
            )

    for record in test_records:
        if record.get("split") != "test":
            errors.append(
                f"{record['id_example']} has incorrect split value."
            )

    return errors


# ============================================================
# STATISTICS
# ============================================================

def print_split_statistics(name, records):
    activity_counts = Counter(
        record["activity_type"]
        for record in records
    )

    generation_records = [
        record
        for record in records
        if record["activity_type"] == "GENERATION"
    ]

    generation_with_calendar = sum(
        record.get("calendar_context") is not None
        for record in generation_records
    )

    generation_without_calendar = sum(
        record.get("calendar_context") is None
        for record in generation_records
    )

    print(f"\n{name.upper()}")
    print("-" * 40)
    print(f"Total: {len(records)}")

    for activity in [
        "GENERATION",
        "COMPLETION",
        "NORMALIZATION",
    ]:
        print(
            f"{activity}: "
            f"{activity_counts.get(activity, 0)}"
        )

    print(
        "GENERATION with calendar: "
        f"{generation_with_calendar}"
    )

    print(
        "GENERATION without calendar: "
        f"{generation_without_calendar}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("DATASET RANDOM STRATIFIED SPLIT")
    print("=" * 60)

    print(f"\nInput file:\n{INPUT_FILE}")

    records = load_dataset()

    print(f"\nRecords loaded: {len(records)}")

    if len(records) != 5000:
        raise ValueError(
            f"Expected 5000 records, found {len(records)}."
        )

    (
        train_records,
        validation_records,
        test_records,
    ) = stratified_split(records)

    train_records = assign_split(
        train_records,
        "train",
    )

    validation_records = assign_split(
        validation_records,
        "validation",
    )

    test_records = assign_split(
        test_records,
        "test",
    )

    errors = validate_splits(
        records,
        train_records,
        validation_records,
        test_records,
    )

    print_split_statistics(
        "train",
        train_records,
    )

    print_split_statistics(
        "validation",
        validation_records,
    )

    print_split_statistics(
        "test",
        test_records,
    )

    if errors:
        print("\n" + "=" * 60)
        print("SPLIT VALIDATION FAILED")
        print("=" * 60)

        for error in errors:
            print(f"- {error}")

        raise RuntimeError(
            "Split validation failed. Files were not saved."
        )

    save_jsonl(
        train_records,
        TRAIN_FILE,
    )

    save_jsonl(
        validation_records,
        VALIDATION_FILE,
    )

    save_jsonl(
        test_records,
        TEST_FILE,
    )

    print("\n" + "=" * 60)
    print("SPLIT VALIDATION PASSED")
    print("=" * 60)

    print("\nFiles saved to:")

    print(TRAIN_FILE)
    print(VALIDATION_FILE)
    print(TEST_FILE)

    print(
        f"\nRandom seed used: {RANDOM_SEED}"
    )


if __name__ == "__main__":
    main()