from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from connections_research.game.environment import (
    ConnectionsGame,
    GuessResult,
)


DATABASE_PATH = Path("data/db/connections.db")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
    )


@dataclass
class TurnLog:
    turn_index: int

    prompt_shown: str
    raw_generation: str

    remaining_words_before: list[str]
    solved_group_indices_before: list[int]
    mistakes_used_before: int

    # Exact reasoning tokens exposed by the model/backend, when available.
    internal_reasoning: str | None = None

    # User-facing portion after separating any reasoning block.
    visible_response: str | None = None

    reasoning_extraction_method: str | None = None
    reasoning_start_token: str | None = None
    reasoning_end_token: str | None = None

    parsed_guess: list[str] | None = None
    proposed_category: str | None = None

    parse_success: bool = False
    parse_error: str | None = None

    guess_result: GuessResult | None = None
    feedback_returned: str | None = None

    latency_ms: float | None = None

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    provider_request_id: str | None = None
    model_metadata: dict[str, Any] | None = None

    # Optional cheap recovery call used when the main generation
    # does not yield a parseable final answer.
    recovery_used: bool = False
    recovery_type: str | None = None

    recovery_raw_output: str | None = None

    recovery_parse_success: bool | None = None
    recovery_parse_error: str | None = None

    recovery_latency_ms: float | None = None

    recovery_prompt_tokens: int | None = None
    recovery_completion_tokens: int | None = None
    recovery_total_tokens: int | None = None

    recovery_provider_request_id: str | None = None
    recovery_metadata: dict[str, Any] | None = None


class TrajectoryLogger:
    def __init__(
        self,
        database_path: Path = DATABASE_PATH,
    ) -> None:
        self.database_path = database_path

    def start_run(
        self,
        *,
        puzzle_id: str,
        model_provider: str,
        model_name: str,
        shuffle_seed: int,
        max_mistakes: int,
	auto_complete_final_group: bool,
        decoding_settings: dict[str, Any],
        system_prompt: str | None = None,
        run_metadata: dict[str, Any] | None = None,
    ) -> str:
        run_id = str(uuid.uuid4())

        with sqlite3.connect(self.database_path) as connection:
            connection.execute("PRAGMA foreign_keys = ON;")

            connection.execute(
                """
                INSERT INTO runs (
                    run_id,
                    puzzle_id,
                    model_provider,
                    model_name,
                    shuffle_seed,
                    max_mistakes,
		    auto_complete_final_group,
                    decoding_settings_json,
                    system_prompt,
                    run_metadata_json,
                    started_at,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running')
                """,
                (
                    run_id,
                    puzzle_id,
                    model_provider,
                    model_name,
                    shuffle_seed,
                    max_mistakes,
		    int(auto_complete_final_group),
                    json_dumps(decoding_settings),
                    system_prompt,
                    json_dumps(run_metadata or {}),
                    utc_now_iso(),
                ),
            )

            connection.commit()

        return run_id

    def log_turn(
        self,
        *,
        run_id: str,
        turn: TurnLog,
    ) -> str:
        turn_id = str(uuid.uuid4())
        result = turn.guess_result

        with sqlite3.connect(self.database_path) as connection:
            connection.execute("PRAGMA foreign_keys = ON;")

            connection.execute(
                """
                INSERT INTO turns (
                    turn_id,
                    run_id,
                    turn_index,

                    prompt_shown,
                    feedback_returned,

                    reasoning_trace,
		    visible_response,
		    reasoning_extraction_method,
		    reasoning_start_token,
		    reasoning_end_token,
                    raw_output,

                    parsed_guess_json,
                    proposed_category,

                    parse_success,
                    parse_error,

                    guess_status,
                    matched_group_index,
                    matched_category,

                    remaining_words_before_json,
                    remaining_words_after_json,

                    solved_group_indices_before_json,
                    solved_group_indices_after_json,

                    mistakes_used_before,
                    mistakes_used_after,
                    mistakes_remaining_after,

                    game_over_after,
                    solved_after,

                    latency_ms,

                    prompt_tokens,
                    completion_tokens,
                    total_tokens,

                    provider_request_id,
                    model_metadata_json,

                    recovery_used,
                    recovery_type,
                    recovery_raw_output,
                    recovery_parse_success,
                    recovery_parse_error,

                    recovery_latency_ms,

                    recovery_prompt_tokens,
                    recovery_completion_tokens,
                    recovery_total_tokens,

                    recovery_provider_request_id,
                    recovery_metadata_json
                )
                VALUES (
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?,
                    ?, ?, ?,
                    ?, ?,

                    ?, ?, ?, ?, ?,
                    ?,
                    ?, ?, ?,
                    ?, ?
                )
                """,
                (
                    turn_id,
                    run_id,
                    turn.turn_index,

                    turn.prompt_shown,
                    turn.feedback_returned,

                    turn.internal_reasoning,
    		    turn.visible_response,
    		    turn.reasoning_extraction_method,
    		    turn.reasoning_start_token,
    		    turn.reasoning_end_token,
    		    turn.raw_generation,

                    (
                        json_dumps(turn.parsed_guess)
                        if turn.parsed_guess is not None
                        else None
                    ),
                    turn.proposed_category,

                    int(turn.parse_success),
                    turn.parse_error,

                    (
                        result.status.value
                        if result is not None
                        else None
                    ),
                    (
                        result.matched_group_index
                        if result is not None
                        else None
                    ),
                    (
                        result.matched_category
                        if result is not None
                        else None
                    ),

                    json_dumps(
                        turn.remaining_words_before
                    ),
                    (
                        json_dumps(result.remaining_words)
                        if result is not None
                        else None
                    ),

                    json_dumps(
                        turn.solved_group_indices_before
                    ),
                    (
                        json_dumps(
                            result.solved_group_indices
                        )
                        if result is not None
                        else None
                    ),

                    turn.mistakes_used_before,
                    (
                        result.mistakes_used
                        if result is not None
                        else None
                    ),
                    (
                        result.mistakes_remaining
                        if result is not None
                        else None
                    ),

                    (
                        int(result.game_over)
                        if result is not None
                        else None
                    ),
                    (
                        int(result.solved)
                        if result is not None
                        else None
                    ),

                    turn.latency_ms,

                    turn.prompt_tokens,
                    turn.completion_tokens,
                    turn.total_tokens,

                    turn.provider_request_id,
                    json_dumps(
                        turn.model_metadata or {}
                    ),

                    int(turn.recovery_used),
                    turn.recovery_type,
                    turn.recovery_raw_output,

                    (
                        int(turn.recovery_parse_success)
                        if turn.recovery_parse_success is not None
                        else None
                    ),
                    turn.recovery_parse_error,

                    turn.recovery_latency_ms,

                    turn.recovery_prompt_tokens,
                    turn.recovery_completion_tokens,
                    turn.recovery_total_tokens,

                    turn.recovery_provider_request_id,

                    json_dumps(
                        turn.recovery_metadata or {}
                    ),
                ),
            )

            connection.commit()

        return turn_id

    def complete_run(
        self,
        *,
        run_id: str,
        game: ConnectionsGame,
        final_predicted_groups: list[dict[str, Any]],
    ) -> None:
        token_totals = self._calculate_run_totals(run_id)

        with sqlite3.connect(self.database_path) as connection:
            connection.execute("PRAGMA foreign_keys = ON;")

            connection.execute(
                """
                UPDATE runs
                SET
                    completed_at = ?,
                    status = 'completed',
                    solved = ?,
                    binary_score = ?,
                    guesses_used = ?,
                    mistakes_used = ?,
                    final_predicted_groups_json = ?,
                    final_remaining_words_json = ?,
                    total_latency_ms = ?,
                    total_prompt_tokens = ?,
                    total_completion_tokens = ?,
                    total_tokens = ?
                WHERE run_id = ?
                """,
                (
                    utc_now_iso(),
                    int(game.solved),
                    int(game.solved),
                    game.guesses_used,
                    game.mistakes_used,
                    json_dumps(final_predicted_groups),
                    json_dumps(
                        game.get_remaining_words_in_display_order()
                    ),
                    token_totals["latency_ms"],
                    token_totals["prompt_tokens"],
                    token_totals["completion_tokens"],
                    token_totals["total_tokens"],
                    run_id,
                ),
            )

            connection.commit()

    def fail_run(
        self,
        *,
        run_id: str,
        error: Exception | str,
        game: ConnectionsGame | None = None,
    ) -> None:
        token_totals = self._calculate_run_totals(run_id)

        failure_message = (
            str(error)
            if isinstance(error, Exception)
            else error
        )

        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                UPDATE runs
                SET
                    completed_at = ?,
                    status = 'failed',
                    solved = ?,
                    binary_score = ?,
                    guesses_used = ?,
                    mistakes_used = ?,
                    final_remaining_words_json = ?,
                    failure_message = ?,
                    total_latency_ms = ?,
                    total_prompt_tokens = ?,
                    total_completion_tokens = ?,
                    total_tokens = ?
                WHERE run_id = ?
                """,
                (
                    utc_now_iso(),
                    (
                        int(game.solved)
                        if game is not None
                        else None
                    ),
                    (
                        int(game.solved)
                        if game is not None
                        else None
                    ),
                    (
                        game.guesses_used
                        if game is not None
                        else None
                    ),
                    (
                        game.mistakes_used
                        if game is not None
                        else None
                    ),
                    (
                        json_dumps(
                            game.get_remaining_words_in_display_order()
                        )
                        if game is not None
                        else None
                    ),
                    failure_message,
                    token_totals["latency_ms"],
                    token_totals["prompt_tokens"],
                    token_totals["completion_tokens"],
                    token_totals["total_tokens"],
                    run_id,
                ),
            )

            connection.commit()

    def _calculate_run_totals(
        self,
        run_id: str,
    ) -> dict[str, float | int]:
        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT
                    COALESCE(SUM(latency_ms), 0),
                    COALESCE(SUM(prompt_tokens), 0),
                    COALESCE(SUM(completion_tokens), 0),
                    COALESCE(SUM(total_tokens), 0)
                FROM turns
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

        return {
            "latency_ms": float(row[0]),
            "prompt_tokens": int(row[1]),
            "completion_tokens": int(row[2]),
            "total_tokens": int(row[3]),
        }
