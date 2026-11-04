import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


RAW_PATH = Path("data/raw/ConnectionsFinalDataset.json")
OUTPUT_PATH = Path("data/processed/nyt_connections.jsonl")
REPORT_PATH = Path("data/processed/validation_report.json")


def normalize_text(value: str) -> str:
    """Normalize casing and whitespace while preserving punctuation."""
    return " ".join(str(value).strip().upper().split())


def normalize_date(raw_date: str) -> str:
    """Convert supported source date formats to YYYY-MM-DD."""
    raw_date = str(raw_date).strip()

    supported_formats = (
        "%Y/%m/%d",
        "%Y-%m-%d",
        "%Y/%d/%m",
    )

    for date_format in supported_formats:
        try:
            parsed = datetime.strptime(raw_date, date_format)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue

    raise ValueError(f"Unsupported date format: {raw_date!r}")


def canonical_puzzle_hash(words: list[str]) -> str:
    """
    Generate a content-based hash from the 16 normalized words.

    This helps detect duplicate puzzles even if their dates or contest names
    differ.
    """
    canonical = "|".join(sorted(words))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_and_normalize_record(
    record: dict[str, Any],
    source_index: int,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("Record is not a JSON object")

    required_fields = {"date", "contest", "words", "answers"}
    missing_fields = required_fields - record.keys()

    if missing_fields:
        raise ValueError(
            f"Missing required fields: {sorted(missing_fields)}"
        )

    normalized_date = normalize_date(record["date"])
    puzzle_id = f"nyt_{normalized_date}"

    source_words = record["words"]
    source_answers = record["answers"]

    if not isinstance(source_words, list):
        raise ValueError("'words' must be a list")

    if not isinstance(source_answers, list):
        raise ValueError("'answers' must be a list")

    normalized_words = [
        normalize_text(word)
        for word in source_words
    ]

    if len(normalized_words) != 16:
        raise ValueError(
            f"Expected 16 puzzle words, found {len(normalized_words)}"
        )

    word_counts = Counter(normalized_words)
    duplicate_words = sorted(
        word
        for word, count in word_counts.items()
        if count > 1
    )

    if duplicate_words:
        raise ValueError(
            f"Duplicate puzzle words: {duplicate_words}"
        )

    if len(source_answers) != 4:
        raise ValueError(
            f"Expected 4 answer groups, found {len(source_answers)}"
        )

    normalized_groups: list[dict[str, Any]] = []
    all_group_words: list[str] = []

    for group_index, answer in enumerate(source_answers):
        if not isinstance(answer, dict):
            raise ValueError(
                f"Answer group {group_index} is not an object"
            )

        if "words" not in answer:
            raise ValueError(
                f"Answer group {group_index} has no 'words' field"
            )

        group_words_raw = answer["words"]

        if not isinstance(group_words_raw, list):
            raise ValueError(
                f"Answer group {group_index} words are not a list"
            )

        group_words = [
            normalize_text(word)
            for word in group_words_raw
        ]

        if len(group_words) != 4:
            raise ValueError(
                f"Answer group {group_index} has "
                f"{len(group_words)} words instead of 4"
            )

        if len(set(group_words)) != 4:
            raise ValueError(
                f"Answer group {group_index} contains duplicate words"
            )

        category = normalize_text(
            answer.get("answerDescription", "")
        )

        normalized_groups.append(
            {
                "group_index": group_index,
                "category": category,
                "words": group_words,
            }
        )

        all_group_words.extend(group_words)

    group_word_counts = Counter(all_group_words)

    repeated_across_groups = sorted(
        word
        for word, count in group_word_counts.items()
        if count > 1
    )

    if repeated_across_groups:
        raise ValueError(
            "Words appear in multiple ground-truth groups: "
            f"{repeated_across_groups}"
        )

    puzzle_word_set = set(normalized_words)
    grouped_word_set = set(all_group_words)

    missing_from_groups = sorted(
        puzzle_word_set - grouped_word_set
    )

    extra_in_groups = sorted(
        grouped_word_set - puzzle_word_set
    )

    if missing_from_groups:
        raise ValueError(
            "Puzzle words missing from ground truth: "
            f"{missing_from_groups}"
        )

    if extra_in_groups:
        raise ValueError(
            "Ground-truth words not found in puzzle words: "
            f"{extra_in_groups}"
        )

    if len(all_group_words) != 16:
        raise ValueError(
            "Ground-truth groups do not collectively contain 16 words"
        )

    difficulty = record.get("difficulty")

    if difficulty is not None:
        try:
            difficulty = float(difficulty)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid difficulty value: {difficulty!r}"
            ) from error

    return {
        "puzzle_id": puzzle_id,
        "date": normalized_date,
        "contest": str(record["contest"]).strip(),
        "source": "tm21cy/NYT-Connections",
        "source_index": source_index,
        "words": normalized_words,
        "groups": normalized_groups,
        "difficulty": difficulty,
        "puzzle_hash": canonical_puzzle_hash(normalized_words),
    }


def main() -> None:
    if not RAW_PATH.exists():
        print(
            f"Error: raw dataset not found at {RAW_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    with RAW_PATH.open("r", encoding="utf-8") as file:
        raw_records = json.load(file)

    if not isinstance(raw_records, list):
        print(
            "Error: expected the raw dataset to contain a JSON list",
            file=sys.stderr,
        )
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    normalized_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    seen_puzzle_ids: dict[str, int] = {}
    seen_hashes: dict[str, int] = {}

    duplicate_ids: list[dict[str, Any]] = []
    duplicate_hashes: list[dict[str, Any]] = []

    for source_index, record in enumerate(raw_records):
        try:
            normalized = validate_and_normalize_record(
                record=record,
                source_index=source_index,
            )

            puzzle_id = normalized["puzzle_id"]
            puzzle_hash = normalized["puzzle_hash"]

            if puzzle_id in seen_puzzle_ids:
                duplicate_ids.append(
                    {
                        "puzzle_id": puzzle_id,
                        "first_source_index": seen_puzzle_ids[puzzle_id],
                        "duplicate_source_index": source_index,
                    }
                )
                continue

            if puzzle_hash in seen_hashes:
                duplicate_hashes.append(
                    {
                        "puzzle_hash": puzzle_hash,
                        "first_source_index": seen_hashes[puzzle_hash],
                        "duplicate_source_index": source_index,
                    }
                )
                continue

            seen_puzzle_ids[puzzle_id] = source_index
            seen_hashes[puzzle_hash] = source_index
            normalized_records.append(normalized)

        except Exception as error:
            failures.append(
                {
                    "source_index": source_index,
                    "date": (
                        record.get("date")
                        if isinstance(record, dict)
                        else None
                    ),
                    "contest": (
                        record.get("contest")
                        if isinstance(record, dict)
                        else None
                    ),
                    "error": str(error),
                }
            )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for record in normalized_records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    report = {
        "raw_path": str(RAW_PATH),
        "output_path": str(OUTPUT_PATH),
        "total_source_records": len(raw_records),
        "valid_records_written": len(normalized_records),
        "validation_failures": len(failures),
        "duplicate_puzzle_ids_skipped": len(duplicate_ids),
        "duplicate_word_sets_skipped": len(duplicate_hashes),
        "failures": failures,
        "duplicate_ids": duplicate_ids,
        "duplicate_hashes": duplicate_hashes,
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

    print("Normalization complete.")
    print(f"Source records:       {len(raw_records)}")
    print(f"Valid records:        {len(normalized_records)}")
    print(f"Validation failures:  {len(failures)}")
    print(f"Duplicate IDs:        {len(duplicate_ids)}")
    print(f"Duplicate word sets:  {len(duplicate_hashes)}")
    print(f"\nNormalized dataset:   {OUTPUT_PATH}")
    print(f"Validation report:    {REPORT_PATH}")

    if failures:
        print("\nFirst validation failures:")
        for failure in failures[:5]:
            print(
                f"- index={failure['source_index']} "
                f"date={failure['date']!r}: "
                f"{failure['error']}"
            )


if __name__ == "__main__":
    main()
