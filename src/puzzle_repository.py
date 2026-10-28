import json
import sqlite3
from pathlib import Path

from src.connections_game import (
    ConnectionsGame,
    GroundTruthGroup,
)


DATABASE_PATH = Path("data/db/connections.db")


def load_game_from_database(
    puzzle_id: str,
    shuffle_seed: int,
    *,
    auto_complete_final_group: bool = True,
) -> ConnectionsGame:
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_PATH}"
        )

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            """
            SELECT
                puzzle_id,
                words_json,
                ground_truth_json
            FROM puzzles
            WHERE puzzle_id = ?
            """,
            (puzzle_id,),
        ).fetchone()

    if row is None:
        raise ValueError(
            f"Puzzle not found in database: {puzzle_id}"
        )

    words = tuple(json.loads(row["words_json"]))
    raw_groups = json.loads(row["ground_truth_json"])

    groups = tuple(
        GroundTruthGroup(
            group_index=int(group["group_index"]),
            category=str(group["category"]),
            words=frozenset(group["words"]),
        )
        for group in raw_groups
    )

    return ConnectionsGame(
        puzzle_id=row["puzzle_id"],
        original_words=words,
        groups=groups,
        shuffle_seed=shuffle_seed,
        auto_complete_final_group=auto_complete_final_group,
    )


def load_latest_puzzle_id() -> str:
    with sqlite3.connect(DATABASE_PATH) as connection:
        row = connection.execute(
            """
            SELECT puzzle_id
            FROM puzzles
            ORDER BY publication_date DESC
            LIMIT 1
            """
        ).fetchone()

    if row is None:
        raise RuntimeError("The puzzles table is empty.")

    return str(row[0])

