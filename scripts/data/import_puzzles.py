import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


INPUT_PATH = Path("data/processed/nyt_connections.jsonl")
DATABASE_PATH = Path("data/db/connections.db")


CREATE_PUZZLES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS puzzles (
    puzzle_id TEXT PRIMARY KEY,
    publication_date TEXT NOT NULL,
    contest TEXT,
    source_dataset TEXT NOT NULL,

    words_json TEXT NOT NULL,
    ground_truth_json TEXT NOT NULL,

    difficulty REAL,
    puzzle_hash TEXT NOT NULL UNIQUE,
    source_index INTEGER,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


CREATE_DATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_puzzles_publication_date
ON puzzles(publication_date);
"""


CREATE_SOURCE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_puzzles_source_dataset
ON puzzles(source_dataset);
"""


INSERT_PUZZLE_SQL = """
INSERT OR IGNORE INTO puzzles (
    puzzle_id,
    publication_date,
    contest,
    source_dataset,
    words_json,
    ground_truth_json,
    difficulty,
    puzzle_hash,
    source_index
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Read the normalized JSONL dataset.

    Each non-empty line should contain one complete puzzle as JSON.
    """
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} is not a JSON object"
                )

            records.append(record)

    return records


def validate_import_record(record: dict[str, Any]) -> None:
    """
    Perform lightweight validation before inserting into the database.

    Full validation was already done during normalization. These checks protect
    the database from incomplete or accidentally modified processed files.
    """
    required_fields = {
        "puzzle_id",
        "date",
        "source",
        "words",
        "groups",
        "puzzle_hash",
    }

    missing_fields = required_fields - record.keys()

    if missing_fields:
        raise ValueError(
            f"Record {record.get('puzzle_id', '<unknown>')} is missing "
            f"fields: {sorted(missing_fields)}"
        )

    if len(record["words"]) != 16:
        raise ValueError(
            f"{record['puzzle_id']} does not contain 16 words"
        )

    if len(record["groups"]) != 4:
        raise ValueError(
            f"{record['puzzle_id']} does not contain 4 groups"
        )


def create_database(connection: sqlite3.Connection) -> None:
    """
    Create the puzzles table and useful indexes if they do not exist.
    """
    connection.execute(CREATE_PUZZLES_TABLE_SQL)
    connection.execute(CREATE_DATE_INDEX_SQL)
    connection.execute(CREATE_SOURCE_INDEX_SQL)
    connection.commit()


def insert_puzzle(
    connection: sqlite3.Connection,
    record: dict[str, Any],
) -> bool:
    """
    Insert one puzzle.

    Returns True if a new row was inserted.
    Returns False if the puzzle was skipped because its puzzle_id or
    puzzle_hash already exists.
    """
    before_changes = connection.total_changes

    connection.execute(
        INSERT_PUZZLE_SQL,
        (
            record["puzzle_id"],
            record["date"],
            record.get("contest"),
            record["source"],
            json.dumps(
                record["words"],
                ensure_ascii=False,
            ),
            json.dumps(
                record["groups"],
                ensure_ascii=False,
            ),
            record.get("difficulty"),
            record["puzzle_hash"],
            record.get("source_index"),
        ),
    )

    return connection.total_changes > before_changes


def main() -> None:
    if not INPUT_PATH.exists():
        print(
            f"Error: normalized dataset not found at {INPUT_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(INPUT_PATH)

    print(f"Loaded {len(records)} normalized puzzles.")
    print(f"Database path: {DATABASE_PATH.resolve()}")

    inserted = 0
    skipped = 0
    failed = 0

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON;")
        create_database(connection)

        for record in records:
            try:
                validate_import_record(record)

                was_inserted = insert_puzzle(
                    connection=connection,
                    record=record,
                )

                if was_inserted:
                    inserted += 1
                else:
                    skipped += 1

            except Exception as error:
                failed += 1
                print(
                    f"Failed to import "
                    f"{record.get('puzzle_id', '<unknown>')}: {error}",
                    file=sys.stderr,
                )

        connection.commit()

        total_rows = connection.execute(
            "SELECT COUNT(*) FROM puzzles"
        ).fetchone()[0]

    print("\nImport complete.")
    print(f"New puzzles inserted: {inserted}")
    print(f"Duplicates skipped:   {skipped}")
    print(f"Import failures:      {failed}")
    print(f"Total database rows:  {total_rows}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
