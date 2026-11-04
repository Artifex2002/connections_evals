from __future__ import annotations

import argparse
import json
import signal
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from run_one_puzzle import run_one_puzzle


DEFAULT_DATABASE = "data/db/connections.db"
DEFAULT_MODEL = "nvidia/Nemotron-H-8B-Reasoning-128K"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_CHECKPOINT = "checkpoints/nemotron_all.json"


STOP_REQUESTED = False


def request_stop(
    signum: int,
    frame: Any,
) -> None:
    del frame

    global STOP_REQUESTED
    STOP_REQUESTED = True

    print(
        f"\nReceived signal {signum}. "
        "The batch will stop after the current puzzle.",
        flush=True,
    )


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "attempts": {},
            "completed": [],
            "failed": [],
            "last_puzzle_id": None,
        }

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_checkpoint(
    path: Path,
    checkpoint: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            checkpoint,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        file.flush()

    temporary_path.replace(path)


def load_puzzle_ids(
    database_path: str,
) -> list[str]:
    """
    Load all puzzle IDs.

    This assumes the source puzzle table is named `puzzles` and contains
    a `puzzle_id` column. Change this query only if your schema uses a
    different source-table name.
    """

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT puzzle_id
            FROM puzzles
            ORDER BY puzzle_id
            """
        ).fetchall()

    return [
        str(row[0])
        for row in rows
    ]


def completed_puzzle_ids(
    *,
    database_path: str,
    model_name: str,
    shuffle_seed: int,
) -> set[str]:
    """
    Return puzzles that already have a completed run for this model and
    shuffle seed.

    A failed or interrupted run is not considered completed and can be
    retried.
    """

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT puzzle_id
            FROM runs
            WHERE status = 'completed'
              AND model_name = ?
              AND shuffle_seed = ?
            """,
            (
                model_name,
                shuffle_seed,
            ),
        ).fetchall()

    return {
        str(row[0])
        for row in rows
    }


def mark_stale_running_runs(
    *,
    database_path: str,
    model_name: str,
) -> int:
    """
    Mark abandoned `running` rows from previous batch processes as failed.

    This is safe when only one batch for this model is active at a time.
    """

    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE runs
            SET
                status = 'failed',
                completed_at = datetime('now'),
                failure_message = (
                    'Interrupted before completion; '
                    'marked stale by resumable batch runner'
                )
            WHERE status = 'running'
              AND model_name = ?
            """,
            (model_name,),
        )

        connection.commit()
        return cursor.rowcount


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an OpenAI-compatible reasoning model over "
            "all Connections puzzles."
        )
    )

    parser.add_argument(
        "--database",
        default=DEFAULT_DATABASE,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-model-turns",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--max-attempts-per-puzzle",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--start-after",
        default=None,
        help=(
            "Skip IDs through this puzzle ID. Useful for "
            "manual range splitting."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of puzzles for testing.",
    )

    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=5.0,
    )

    return parser.parse_args()


def main() -> int:
    arguments = parse_args()

    signal.signal(
        signal.SIGTERM,
        request_stop,
    )
    signal.signal(
        signal.SIGINT,
        request_stop,
    )
    signal.signal(
        signal.SIGUSR1,
        request_stop,
    )

    checkpoint_path = Path(
        arguments.checkpoint
    )

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    stale_count = mark_stale_running_runs(
        database_path=arguments.database,
        model_name=arguments.model,
    )

    if stale_count:
        print(
            f"Marked {stale_count} stale running "
            "run(s) as failed.",
            flush=True,
        )

    puzzle_ids = load_puzzle_ids(
        arguments.database
    )

    if arguments.start_after is not None:
        puzzle_ids = [
            puzzle_id
            for puzzle_id in puzzle_ids
            if puzzle_id > arguments.start_after
        ]

    completed_in_database = completed_puzzle_ids(
        database_path=arguments.database,
        model_name=arguments.model,
        shuffle_seed=arguments.shuffle_seed,
    )

    pending_puzzle_ids = [
        puzzle_id
        for puzzle_id in puzzle_ids
        if puzzle_id not in completed_in_database
    ]

    if arguments.limit is not None:
        pending_puzzle_ids = pending_puzzle_ids[
            : arguments.limit
        ]

    print(
        json.dumps(
            {
                "total_puzzles": len(puzzle_ids),
                "already_completed": len(
                    completed_in_database
                ),
                "pending_this_job": len(
                    pending_puzzle_ids
                ),
                "model": arguments.model,
                "shuffle_seed": (
                    arguments.shuffle_seed
                ),
                "checkpoint": str(
                    checkpoint_path
                ),
            },
            indent=2,
        ),
        flush=True,
    )

    attempts = checkpoint.setdefault(
        "attempts",
        {},
    )

    completed_checkpoint = set(
        checkpoint.setdefault(
            "completed",
            [],
        )
    )

    failed_checkpoint = set(
        checkpoint.setdefault(
            "failed",
            [],
        )
    )

    for puzzle_number, puzzle_id in enumerate(
        pending_puzzle_ids,
        start=1,
    ):
        if STOP_REQUESTED:
            print(
                "Stop requested before next puzzle.",
                flush=True,
            )
            break

        attempt_count = int(
            attempts.get(puzzle_id, 0)
        )

        if (
            attempt_count
            >= arguments.max_attempts_per_puzzle
        ):
            print(
                f"Skipping {puzzle_id}: "
                f"{attempt_count} prior attempts.",
                flush=True,
            )
            continue

        attempt_count += 1
        attempts[puzzle_id] = attempt_count
        checkpoint["last_puzzle_id"] = puzzle_id

        save_checkpoint(
            checkpoint_path,
            checkpoint,
        )

        print()
        print("#" * 88)
        print(
            f"PUZZLE {puzzle_number}/"
            f"{len(pending_puzzle_ids)}: "
            f"{puzzle_id}"
        )
        print(
            f"ATTEMPT {attempt_count}/"
            f"{arguments.max_attempts_per_puzzle}"
        )
        print("#" * 88)
        sys.stdout.flush()

        try:
            run_id = run_one_puzzle(
                puzzle_id=puzzle_id,
                database_path=arguments.database,
                model_name=arguments.model,
                base_url=arguments.base_url,
                shuffle_seed=arguments.shuffle_seed,
                max_model_turns=(
                    arguments.max_model_turns
                ),
                temperature=arguments.temperature,
                top_p=arguments.top_p,
                max_tokens=arguments.max_tokens,
            )

            completed_checkpoint.add(
                puzzle_id
            )
            failed_checkpoint.discard(
                puzzle_id
            )

            checkpoint["completed"] = sorted(
                completed_checkpoint
            )
            checkpoint["failed"] = sorted(
                failed_checkpoint
            )
            checkpoint["last_run_id"] = run_id

            print(
                f"Completed puzzle {puzzle_id}; "
                f"run_id={run_id}",
                flush=True,
            )

        except Exception as error:
            failed_checkpoint.add(
                puzzle_id
            )

            checkpoint["failed"] = sorted(
                failed_checkpoint
            )
            checkpoint["last_error"] = repr(
                error
            )

            print(
                f"Puzzle {puzzle_id} failed: "
                f"{error!r}",
                file=sys.stderr,
                flush=True,
            )

            if (
                attempt_count
                < arguments.max_attempts_per_puzzle
                and not STOP_REQUESTED
            ):
                time.sleep(
                    arguments.retry_delay_seconds
                )

        finally:
            save_checkpoint(
                checkpoint_path,
                checkpoint,
            )

    completed_now = completed_puzzle_ids(
        database_path=arguments.database,
        model_name=arguments.model,
        shuffle_seed=arguments.shuffle_seed,
    )

    print()
    print(
        json.dumps(
            {
                "completed_in_database": len(
                    completed_now
                ),
                "checkpoint_completed": len(
                    checkpoint["completed"]
                ),
                "checkpoint_failed": len(
                    checkpoint["failed"]
                ),
                "stop_requested": STOP_REQUESTED,
            },
            indent=2,
        ),
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
