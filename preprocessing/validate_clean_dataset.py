import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "dataset_master_GptPlus.jsonl"

CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

OUTPUT_FILE = CLEANED_DIR / "dataset_master_cleaned.jsonl"
REPORT_FILE = METADATA_DIR / "preprocessing_validation_report.json"


# ============================================================
# EXPECTED DATASET STRUCTURE
# ============================================================

EXPECTED_FIELDS = {
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

EXPECTED_ACTIVITY_COUNTS = {
    "GENERATION": 2000,
    "COMPLETION": 1500,
    "NORMALIZATION": 1500,
}

EXPECTED_TOTAL_RECORDS = 5000

EXPECTED_GENERATION_WITH_CALENDAR = 1000
EXPECTED_GENERATION_WITHOUT_CALENDAR = 1000

VALID_ACTIVITIES = {
    "GENERATION",
    "COMPLETION",
    "NORMALIZATION",
}

VALID_CURRENCIES = {
    "EUR",
    "USD",
    "GBP",
}

CALENDAR_FIELDS = {
    "event_title",
    "event_date",
    "event_category",
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_text(text):
    """
    Normalize text for duplicate / near-duplicate comparison.
    """
    if text is None:
        return ""

    text = str(text).lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize monetary values / numbers
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", text)

    # Normalize years
    text = re.sub(r"\b20\d{2}\b", "<year>", text)

    months = (
        "january|february|march|april|may|june|july|august|"
        "september|october|november|december"
    )

    text = re.sub(
        rf"\b({months})\b",
        "<month>",
        text,
        flags=re.IGNORECASE,
    )

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_similarity_signature(record):
    """
    Build a normalized textual signature for near-duplicate detection.
    """

    expected_output = record.get("expected_output", [])

    if isinstance(expected_output, list):
        expected_text = " ".join(str(x) for x in expected_output)
    else:
        expected_text = str(expected_output)

    parts = [
        record.get("input_text"),
        record.get("beneficiary"),
        record.get("reference_period"),
        expected_text,
    ]

    return normalize_text(" ".join(str(x or "") for x in parts))


def is_valid_date(date_string):
    try:
        datetime.strptime(date_string, "%Y-%m-%d")
        return True
    except (ValueError, TypeError):
        return False


def add_error(errors, record_id, message):
    errors.append({
        "id_example": record_id,
        "error": message,
    })


# ============================================================
# LOAD JSONL
# ============================================================

def load_dataset():
    records = []
    json_errors = []

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                json_errors.append({
                    "line": line_number,
                    "error": "Empty line",
                })
                continue

            try:
                record = json.loads(line)
                records.append(record)

            except json.JSONDecodeError as exc:
                json_errors.append({
                    "line": line_number,
                    "error": str(exc),
                })

    return records, json_errors


# ============================================================
# RECORD VALIDATION
# ============================================================

def validate_record(record, errors):
    record_id = record.get("id_example", "<missing-id>")

    # --------------------------------------------------------
    # Fields
    # --------------------------------------------------------

    actual_fields = set(record.keys())

    missing_fields = EXPECTED_FIELDS - actual_fields
    extra_fields = actual_fields - EXPECTED_FIELDS

    if missing_fields:
        add_error(
            errors,
            record_id,
            f"Missing fields: {sorted(missing_fields)}",
        )

    if extra_fields:
        add_error(
            errors,
            record_id,
            f"Unexpected fields: {sorted(extra_fields)}",
        )

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    if not isinstance(record.get("id_example"), str):
        add_error(errors, record_id, "id_example must be a string")

    # --------------------------------------------------------
    # Activity
    # --------------------------------------------------------

    activity = record.get("activity_type")

    if activity not in VALID_ACTIVITIES:
        add_error(
            errors,
            record_id,
            f"Invalid activity_type: {activity}",
        )

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    category = record.get("operation_category")

    if not isinstance(category, str) or not category.strip():
        add_error(
            errors,
            record_id,
            "operation_category must be a non-empty string",
        )

    # --------------------------------------------------------
    # Beneficiary
    # --------------------------------------------------------

    beneficiary = record.get("beneficiary")

    if not isinstance(beneficiary, str) or not beneficiary.strip():
        add_error(
            errors,
            record_id,
            "beneficiary must be a non-empty string",
        )

    # --------------------------------------------------------
    # Amount
    # --------------------------------------------------------

    amount = record.get("amount")

    if (
        not isinstance(amount, (int, float))
        or isinstance(amount, bool)
        or amount <= 0
    ):
        add_error(
            errors,
            record_id,
            "amount must be a positive numeric value",
        )

    # --------------------------------------------------------
    # Currency
    # --------------------------------------------------------

    currency = record.get("currency")

    if currency not in VALID_CURRENCIES:
        add_error(
            errors,
            record_id,
            f"Invalid currency: {currency}",
        )

    # --------------------------------------------------------
    # Reference period
    # --------------------------------------------------------

    reference_period = record.get("reference_period")

    if reference_period is not None:
        if (
            not isinstance(reference_period, str)
            or not reference_period.strip()
        ):
            add_error(
                errors,
                record_id,
                "reference_period must be null or a non-empty string",
            )

    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    if record.get("language") != "en":
        add_error(
            errors,
            record_id,
            'language must always be "en"',
        )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    if record.get("split") is not None:
        add_error(
            errors,
            record_id,
            "split must be null before dataset splitting",
        )

    # --------------------------------------------------------
    # Expected output
    # --------------------------------------------------------

    expected_output = record.get("expected_output")

    if not isinstance(expected_output, list):
        add_error(
            errors,
            record_id,
            "expected_output must be a list",
        )

    else:
        if len(expected_output) != 2:
            add_error(
                errors,
                record_id,
                "expected_output must contain exactly 2 strings",
            )

        for output in expected_output:
            if not isinstance(output, str) or not output.strip():
                add_error(
                    errors,
                    record_id,
                    "expected_output items must be non-empty strings",
                )

        if len(expected_output) == 2:
            first = normalize_text(expected_output[0])
            second = normalize_text(expected_output[1])

            if first == second:
                add_error(
                    errors,
                    record_id,
                    "The two expected outputs are identical after normalization",
                )

    # --------------------------------------------------------
    # Activity-specific rules
    # --------------------------------------------------------

    input_text = record.get("input_text")
    calendar_context = record.get("calendar_context")

    if activity == "GENERATION":

        if input_text is not None:
            add_error(
                errors,
                record_id,
                "GENERATION must have input_text = null",
            )

    elif activity in {"COMPLETION", "NORMALIZATION"}:

        if not isinstance(input_text, str) or not input_text.strip():
            add_error(
                errors,
                record_id,
                f"{activity} must have non-empty input_text",
            )

        if calendar_context is not None:
            add_error(
                errors,
                record_id,
                f"{activity} must have calendar_context = null",
            )

    # --------------------------------------------------------
    # Calendar context
    # --------------------------------------------------------

    if calendar_context is not None:

        if activity != "GENERATION":
            add_error(
                errors,
                record_id,
                "calendar_context is only allowed for GENERATION",
            )

        if not isinstance(calendar_context, dict):
            add_error(
                errors,
                record_id,
                "calendar_context must be an object or null",
            )

        else:
            calendar_keys = set(calendar_context.keys())

            if calendar_keys != CALENDAR_FIELDS:
                add_error(
                    errors,
                    record_id,
                    (
                        "calendar_context must contain exactly: "
                        "event_title, event_date, event_category"
                    ),
                )

            event_title = calendar_context.get("event_title")
            event_date = calendar_context.get("event_date")
            event_category = calendar_context.get("event_category")

            if not isinstance(event_title, str) or not event_title.strip():
                add_error(
                    errors,
                    record_id,
                    "calendar event_title must be non-empty",
                )

            if not isinstance(event_category, str) or not event_category.strip():
                add_error(
                    errors,
                    record_id,
                    "calendar event_category must be non-empty",
                )

            if not is_valid_date(event_date):
                add_error(
                    errors,
                    record_id,
                    f"Invalid event_date: {event_date}",
                )


# ============================================================
# DATASET-LEVEL VALIDATION
# ============================================================

def validate_dataset(records):
    errors = []

    # Validate each individual record
    for record in records:
        validate_record(record, errors)

    # --------------------------------------------------------
    # Total number of records
    # --------------------------------------------------------

    if len(records) != EXPECTED_TOTAL_RECORDS:
        errors.append({
            "id_example": None,
            "error": (
                f"Expected {EXPECTED_TOTAL_RECORDS} records, "
                f"found {len(records)}"
            ),
        })

    # --------------------------------------------------------
    # ID checks
    # --------------------------------------------------------

    ids = [record.get("id_example") for record in records]

    id_counts = Counter(ids)

    duplicate_ids = [
        record_id
        for record_id, count in id_counts.items()
        if count > 1
    ]

    if duplicate_ids:
        errors.append({
            "id_example": None,
            "error": f"Duplicate IDs: {duplicate_ids[:20]}",
        })

    expected_ids = {
        f"EX{i:04d}"
        for i in range(1, EXPECTED_TOTAL_RECORDS + 1)
    }

    actual_ids = set(ids)

    missing_ids = sorted(expected_ids - actual_ids)

    if missing_ids:
        errors.append({
            "id_example": None,
            "error": f"Missing IDs: {missing_ids[:20]}",
        })

    # --------------------------------------------------------
    # Activity distribution
    # --------------------------------------------------------

    activity_counts = Counter(
        record.get("activity_type")
        for record in records
    )

    for activity, expected_count in EXPECTED_ACTIVITY_COUNTS.items():
        actual_count = activity_counts.get(activity, 0)

        if actual_count != expected_count:
            errors.append({
                "id_example": None,
                "error": (
                    f"{activity}: expected {expected_count}, "
                    f"found {actual_count}"
                ),
            })

    # --------------------------------------------------------
    # Calendar distribution
    # --------------------------------------------------------

    generation_records = [
        record
        for record in records
        if record.get("activity_type") == "GENERATION"
    ]

    generation_with_calendar = sum(
        record.get("calendar_context") is not None
        for record in generation_records
    )

    generation_without_calendar = sum(
        record.get("calendar_context") is None
        for record in generation_records
    )

    if (
        generation_with_calendar
        != EXPECTED_GENERATION_WITH_CALENDAR
    ):
        errors.append({
            "id_example": None,
            "error": (
                "GENERATION records with calendar_context: "
                f"expected {EXPECTED_GENERATION_WITH_CALENDAR}, "
                f"found {generation_with_calendar}"
            ),
        })

    if (
        generation_without_calendar
        != EXPECTED_GENERATION_WITHOUT_CALENDAR
    ):
        errors.append({
            "id_example": None,
            "error": (
                "GENERATION records without calendar_context: "
                f"expected {EXPECTED_GENERATION_WITHOUT_CALENDAR}, "
                f"found {generation_without_calendar}"
            ),
        })

    # --------------------------------------------------------
    # Category distribution
    # --------------------------------------------------------

    category_counts = Counter(
        record.get("operation_category")
        for record in records
    )

    if len(category_counts) < 25:
        errors.append({
            "id_example": None,
            "error": (
                "Dataset must contain at least 25 distinct "
                f"operation categories; found {len(category_counts)}"
            ),
        })

    max_category_count = EXPECTED_TOTAL_RECORDS * 0.10

    for category, count in category_counts.items():
        if count > max_category_count:
            errors.append({
                "id_example": None,
                "error": (
                    f"Category {category} exceeds 10% limit: "
                    f"{count} records"
                ),
            })

    return (
        errors,
        activity_counts,
        category_counts,
        generation_with_calendar,
        generation_without_calendar,
    )


# ============================================================
# EXACT DUPLICATES
# ============================================================

def find_exact_duplicates(records):
    seen = {}
    duplicates = []

    for record in records:

        # Ignore ID when comparing record content
        comparable = {
            key: value
            for key, value in record.items()
            if key != "id_example"
        }

        serialized = json.dumps(
            comparable,
            sort_keys=True,
            ensure_ascii=False,
        )

        if serialized in seen:
            duplicates.append({
                "first_id": seen[serialized],
                "duplicate_id": record.get("id_example"),
            })
        else:
            seen[serialized] = record.get("id_example")

    return duplicates


# ============================================================
# NEAR-DUPLICATES
# ============================================================

def find_near_duplicates(records, threshold=0.93):
    """
    Search near-duplicates only inside the same
    activity_type + operation_category group.

    To avoid unnecessary O(n²) comparisons across 5,000 records,
    records are first grouped by activity and category.
    """

    groups = defaultdict(list)

    for record in records:
        key = (
            record.get("activity_type"),
            record.get("operation_category"),
        )

        groups[key].append(record)

    near_duplicates = []

    for group_records in groups.values():

        signatures = [
            (
                record.get("id_example"),
                build_similarity_signature(record),
            )
            for record in group_records
        ]

        for i in range(len(signatures)):

            id_a, signature_a = signatures[i]

            if not signature_a:
                continue

            for j in range(i + 1, len(signatures)):

                id_b, signature_b = signatures[j]

                if not signature_b:
                    continue

                similarity = SequenceMatcher(
                    None,
                    signature_a,
                    signature_b,
                ).ratio()

                if similarity >= threshold:
                    near_duplicates.append({
                        "id_1": id_a,
                        "id_2": id_b,
                        "similarity": round(similarity, 4),
                    })

    return near_duplicates


# ============================================================
# BASIC CLEANING
# ============================================================

def clean_record(record):
    """
    Apply only deterministic and meaning-preserving cleaning.
    Do not rewrite semantic content.
    """

    cleaned = dict(record)

    # Trim whitespace from simple string fields
    string_fields = [
        "id_example",
        "activity_type",
        "operation_category",
        "beneficiary",
        "currency",
        "reference_period",
        "input_text",
        "language",
    ]

    for field in string_fields:
        value = cleaned.get(field)

        if isinstance(value, str):
            cleaned[field] = value.strip()

    # Normalize known categorical values
    if isinstance(cleaned.get("activity_type"), str):
        cleaned["activity_type"] = cleaned["activity_type"].upper()

    if isinstance(cleaned.get("operation_category"), str):
        cleaned["operation_category"] = (
            cleaned["operation_category"]
            .strip()
            .upper()
            .replace(" ", "_")
        )

    if isinstance(cleaned.get("currency"), str):
        cleaned["currency"] = cleaned["currency"].upper().strip()

    if isinstance(cleaned.get("language"), str):
        cleaned["language"] = cleaned["language"].lower().strip()

    # Trim expected outputs
    if isinstance(cleaned.get("expected_output"), list):
        cleaned["expected_output"] = [
            output.strip() if isinstance(output, str) else output
            for output in cleaned["expected_output"]
        ]

    # Clean calendar strings
    calendar_context = cleaned.get("calendar_context")

    if isinstance(calendar_context, dict):
        cleaned["calendar_context"] = {
            key: value.strip() if isinstance(value, str) else value
            for key, value in calendar_context.items()
        }

    return cleaned


# ============================================================
# SAVE FILES
# ============================================================

def save_cleaned_dataset(records):
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(
                record,
                file,
                ensure_ascii=False,
            )
            file.write("\n")


def save_report(report):
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DATASET VALIDATION AND CLEANING")
    print("=" * 60)

    print(f"\nInput file:\n{INPUT_FILE}")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    records, json_errors = load_dataset()

    print(f"\nRecords loaded: {len(records)}")
    print(f"JSON parsing errors: {len(json_errors)}")

    # --------------------------------------------------------
    # Clean deterministic formatting
    # --------------------------------------------------------

    cleaned_records = [
        clean_record(record)
        for record in records
    ]

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    (
        validation_errors,
        activity_counts,
        category_counts,
        generation_with_calendar,
        generation_without_calendar,
    ) = validate_dataset(cleaned_records)

    # --------------------------------------------------------
    # Exact duplicates
    # --------------------------------------------------------

    exact_duplicates = find_exact_duplicates(cleaned_records)

    # --------------------------------------------------------
    # Near duplicates
    # --------------------------------------------------------

    print("\nSearching for near-duplicates...")

    near_duplicates = find_near_duplicates(
        cleaned_records,
        threshold=0.93,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    currency_counts = Counter(
        record.get("currency")
        for record in cleaned_records
    )

    amounts = [
        record.get("amount")
        for record in cleaned_records
        if isinstance(record.get("amount"), (int, float))
        and not isinstance(record.get("amount"), bool)
    ]

    amount_stats = {}

    if amounts:
        amount_stats = {
            "min": min(amounts),
            "max": max(amounts),
            "average": round(sum(amounts) / len(amounts), 2),
        }

    # --------------------------------------------------------
    # Final validation status
    # --------------------------------------------------------

    validation_passed = (
        len(json_errors) == 0
        and len(validation_errors) == 0
        and len(exact_duplicates) == 0
    )

    # Near-duplicates are reported as candidates,
    # not automatically treated as invalid.

    report = {
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),

        "total_records": len(cleaned_records),

        "json_errors": json_errors,

        "validation_errors": validation_errors,

        "activity_distribution": dict(activity_counts),

        "generation_with_calendar": generation_with_calendar,
        "generation_without_calendar": generation_without_calendar,

        "category_count": len(category_counts),
        "category_distribution": dict(category_counts),

        "currency_distribution": dict(currency_counts),

        "amount_statistics": amount_stats,

        "exact_duplicate_count": len(exact_duplicates),
        "exact_duplicates": exact_duplicates,

        "near_duplicate_threshold": 0.93,
        "near_duplicate_candidate_count": len(near_duplicates),
        "near_duplicate_candidates": near_duplicates,

        "validation_passed": validation_passed,
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_cleaned_dataset(cleaned_records)
    save_report(report)

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nTotal records: {len(cleaned_records)}")

    print("\nActivity distribution:")
    for activity, count in activity_counts.items():
        print(f"  {activity}: {count}")

    print(
        f"\nGENERATION with calendar: "
        f"{generation_with_calendar}"
    )

    print(
        f"GENERATION without calendar: "
        f"{generation_without_calendar}"
    )

    print(
        f"\nDistinct categories: "
        f"{len(category_counts)}"
    )

    print(
        f"Exact duplicates: "
        f"{len(exact_duplicates)}"
    )

    print(
        f"Near-duplicate candidates: "
        f"{len(near_duplicates)}"
    )

    print(
        f"Validation errors: "
        f"{len(validation_errors)}"
    )

    print(
        f"JSON errors: "
        f"{len(json_errors)}"
    )

    print("\nCleaned dataset saved to:")
    print(OUTPUT_FILE)

    print("\nValidation report saved to:")
    print(REPORT_FILE)

    if validation_passed:
        print("\nVALIDATION PASSED")
    else:
        print("\nVALIDATION FAILED")
        print(
            "Inspect preprocessing_validation_report.json "
            "before continuing."
        )


if __name__ == "__main__":
    main()