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
DEFAULT_CHECKPOINT = "checkpoints/nemotron_all_seed42.json"


STOP_REQUESTED = False


def request_stop(
    signum: int,
    frame: Any,
) -> None:
    """
    Ask the batch runner to stop after the current puzzle.

    Puzzle-level checkpointing means a puzzle interrupted midway may be
    restarted, while previously completed puzzles remain completed.
    """

    del frame

    global STOP_REQUESTED
    STOP_REQUESTED = True

    print(
        f"\nReceived signal {signum}. "
        "Stopping after the current puzzle.",
        flush=True,
    )


def load_checkpoint(
    path: Path,
) -> dict[str, Any]:
    """
    Load the lightweight puzzle-level checkpoint.
    """

    if not path.exists():
        return {
            "attempts": {},
            "completed": [],
            "failed": [],
            "last_puzzle_id": None,
            "last_run_id": None,
            "last_error": None,
        }

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            checkpoint = json.load(file)

    except (
        json.JSONDecodeError,
        OSError,
    ) as error:
        raise RuntimeError(
            f"Could not load checkpoint {path}: {error}"
        ) from error

    checkpoint.setdefault(
        "attempts",
        {},
    )
    checkpoint.setdefault(
        "completed",
        [],
    )
    checkpoint.setdefault(
        "failed",
        [],
    )
    checkpoint.setdefault(
        "last_puzzle_id",
        None,
    )
    checkpoint.setdefault(
        "last_run_id",
        None,
    )
    checkpoint.setdefault(
        "last_error",
        None,
    )

    return checkpoint


def save_checkpoint(
    path: Path,
    checkpoint: dict[str, Any],
) -> None:
    """
    Atomically write the checkpoint.

    The temporary-file replacement prevents a partially written JSON file
    if the process is interrupted during the write.
    """

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
        file.write("\n")
        file.flush()

    temporary_path.replace(path)


def load_puzzle_ids(
    database_path: str,
) -> list[str]:
    """
    Load all puzzle IDs in deterministic order.
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
    Return puzzle IDs with an already completed run for this model and seed.

    Failed or interrupted runs are intentionally excluded so they can be
    attempted again.
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
    Mark abandoned running rows from previous processes as failed.

    Run only one batch for a given model/database at a time. Otherwise this
    could mark another active batch's run as stale.
    """

    failure_message = (
        "Interrupted before completion; "
        "marked stale by resumable batch runner"
    )

    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE runs
            SET
                status = 'failed',
                completed_at = datetime('now'),
                failure_message = ?
            WHERE status = 'running'
              AND model_name = ?
            """,
            (
                failure_message,
                model_name,
            ),
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
        "--system-prompt-prefix",
        default=None,
        help=(
            "Optional text prepended to the model system prompt."
        ),
    )

    parser.add_argument(
        "--max-attempts-per-puzzle",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--start-after",
        default=None,
        help=(
            "Skip puzzle IDs up to and including this ID. "
            "Useful for manually splitting the dataset."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optionally process only the first N pending puzzles. "
            "Useful for smoke tests."
        ),
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

    if hasattr(
        signal,
        "SIGUSR1",
    ):
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

    all_puzzle_ids = load_puzzle_ids(
        arguments.database
    )

    selected_puzzle_ids = all_puzzle_ids

    if arguments.start_after is not None:
        selected_puzzle_ids = [
            puzzle_id
            for puzzle_id in selected_puzzle_ids
            if puzzle_id > arguments.start_after
        ]

    already_completed = completed_puzzle_ids(
        database_path=arguments.database,
        model_name=arguments.model,
        shuffle_seed=arguments.shuffle_seed,
    )

    pending_puzzle_ids = [
        puzzle_id
        for puzzle_id in selected_puzzle_ids
        if puzzle_id not in already_completed
    ]

    if arguments.limit is not None:
        if arguments.limit < 0:
            raise ValueError(
                "--limit must be non-negative."
            )

        pending_puzzle_ids = pending_puzzle_ids[
            : arguments.limit
        ]

    print(
        json.dumps(
            {
                "database": arguments.database,
                "model": arguments.model,
                "base_url": arguments.base_url,
                "shuffle_seed": arguments.shuffle_seed,
                "total_puzzles_in_database": len(
                    all_puzzle_ids
                ),
                "selected_puzzles": len(
                    selected_puzzle_ids
                ),
                "already_completed": len(
                    already_completed
                ),
                "pending_this_process": len(
                    pending_puzzle_ids
                ),
                "checkpoint": str(
                    checkpoint_path
                ),
            },
            indent=2,
        ),
        flush=True,
    )

    attempts: dict[str, int] = {
        str(puzzle_id): int(attempt_count)
        for puzzle_id, attempt_count
        in checkpoint.get(
            "attempts",
            {},
        ).items()
    }

    completed_checkpoint = {
        str(puzzle_id)
        for puzzle_id in checkpoint.get(
            "completed",
            [],
        )
    }

    failed_checkpoint = {
        str(puzzle_id)
        for puzzle_id in checkpoint.get(
            "failed",
            [],
        )
    }

    checkpoint["attempts"] = attempts
    checkpoint["completed"] = sorted(
        completed_checkpoint
    )
    checkpoint["failed"] = sorted(
        failed_checkpoint
    )

    save_checkpoint(
        checkpoint_path,
        checkpoint,
    )

    processed_count = 0

    for puzzle_number, puzzle_id in enumerate(
        pending_puzzle_ids,
        start=1,
    ):
        if STOP_REQUESTED:
            print(
                "Stop requested before starting "
                "the next puzzle.",
                flush=True,
            )
            break

        current_attempts = int(
            attempts.get(
                puzzle_id,
                0,
            )
        )

        if (
            current_attempts
            >= arguments.max_attempts_per_puzzle
        ):
            print(
                f"Skipping {puzzle_id}: "
                f"{current_attempts} attempt(s) already recorded, "
                f"maximum is "
                f"{arguments.max_attempts_per_puzzle}.",
                flush=True,
            )
            continue

        current_attempts += 1
        attempts[puzzle_id] = current_attempts

        checkpoint["attempts"] = attempts
        checkpoint["last_puzzle_id"] = puzzle_id
        checkpoint["last_error"] = None

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
            f"ATTEMPT {current_attempts}/"
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
                system_prompt_prefix=(
                    arguments.system_prompt_prefix
                ),
            )

            processed_count += 1

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
            checkpoint["last_error"] = None

            print(
                f"Completed puzzle {puzzle_id}; "
                f"run_id={run_id}",
                flush=True,
            )

        except Exception as error:
            processed_count += 1

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
                current_attempts
                < arguments.max_attempts_per_puzzle
                and not STOP_REQUESTED
            ):
                print(
                    "The puzzle remains eligible for a future "
                    "retry.",
                    flush=True,
                )

                if (
                    arguments.retry_delay_seconds
                    > 0
                ):
                    time.sleep(
                        arguments.retry_delay_seconds
                    )

        finally:
            save_checkpoint(
                checkpoint_path,
                checkpoint,
            )

    completed_after_run = completed_puzzle_ids(
        database_path=arguments.database,
        model_name=arguments.model,
        shuffle_seed=arguments.shuffle_seed,
    )

    remaining_after_run = [
        puzzle_id
        for puzzle_id in selected_puzzle_ids
        if puzzle_id not in completed_after_run
    ]

    print()
    print("=" * 88)
    print("BATCH SUMMARY")
    print("=" * 88)

    print(
        json.dumps(
            {
                "processed_this_process": (
                    processed_count
                ),
                "completed_in_database": len(
                    completed_after_run
                ),
                "remaining_selected_puzzles": len(
                    remaining_after_run
                ),
                "checkpoint_completed": len(
                    checkpoint.get(
                        "completed",
                        [],
                    )
                ),
                "checkpoint_failed": len(
                    checkpoint.get(
                        "failed",
                        [],
                    )
                ),
                "last_puzzle_id": checkpoint.get(
                    "last_puzzle_id"
                ),
                "last_run_id": checkpoint.get(
                    "last_run_id"
                ),
                "last_error": checkpoint.get(
                    "last_error"
                ),
                "system_prompt_prefix": (
                    arguments.system_prompt_prefix
                ),
                "stop_requested": STOP_REQUESTED,
            },
            indent=2,
        ),
        flush=True,
    )

    # A signal-triggered stop is considered a clean, resumable exit.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
