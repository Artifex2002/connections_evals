from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.model_output_parser import (
    parse_final_guess,
    split_generation,
)
from src.prompt_builder import (
    SYSTEM_PROMPT,
    build_turn_prompt,
)
from src.puzzle_repository import load_game_from_database
from src.trajectory_logger import (
    TrajectoryLogger,
    TurnLog,
)


DATABASE_PATH = "data/db/connections.db"


@dataclass
class ModelResponse:
    raw_output: str
    reasoning_content: str | None = None
    visible_content: str | None = None

    latency_ms: float | None = None

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    provider_request_id: str | None = None
    metadata: dict[str, Any] | None = None


def call_model(
    *,
    system_prompt: str,
    user_prompt: str,
) -> ModelResponse:
    """
    Replace only this function with your actual model backend.
    """

    raise NotImplementedError(
        "Connect this function to your vLLM, Hugging Face, "
        "or OpenAI-compatible model server."
    )


def run_puzzle(
    *,
    puzzle_id: str,
    model_provider: str,
    model_name: str,
    shuffle_seed: int = 42,
    max_model_turns: int = 12,
) -> str:
    game = load_game_from_database(
        puzzle_id=puzzle_id,
        shuffle_seed=shuffle_seed,
        auto_complete_final_group=False,
    )

    logger = TrajectoryLogger(DATABASE_PATH)

    run_id = logger.start_run(
        puzzle_id=game.puzzle_id,
        model_provider=model_provider,
        model_name=model_name,
        shuffle_seed=shuffle_seed,
        max_mistakes=game.max_mistakes,
        auto_complete_final_group=False,
        decoding_settings={
            "temperature": 0.0,
        },
        system_prompt=SYSTEM_PROMPT,
        run_metadata={},
    )

    previous_turns: list[dict[str, Any]] = []

    try:
        for turn_index in range(max_model_turns):
            if game.game_over:
                break

            state_before = game.get_state()

            prompt = build_turn_prompt(
                game_state=state_before,
                previous_turns=previous_turns,
            )

            start_time = time.perf_counter()

            response = call_model(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
            )

            if response.latency_ms is None:
                response.latency_ms = (
                    time.perf_counter() - start_time
                ) * 1000

            split = split_generation(
                response.raw_output,
                reasoning_content=response.reasoning_content,
                visible_content=response.visible_content,
            )

            parsed = parse_final_guess(
                split.visible_response
            )

            result = None

            if parsed.success:
                result = game.submit_guess(
                    parsed.guess,
                    proposed_category=parsed.category,
                )

                feedback = result.message
            else:
                feedback = (
                    "Your response could not be parsed. "
                    f"Error: {parsed.error}"
                )

            logger.log_turn(
                run_id=run_id,
                turn=TurnLog(
                    turn_index=turn_index,
                    prompt_shown=prompt,

                    raw_output=split.raw_output,
                    internal_reasoning=split.internal_reasoning,
                    visible_response=split.visible_response,

                    reasoning_extraction_method=(
                        split.extraction_method
                    ),

                    parsed_guess=parsed.guess,
                    proposed_category=parsed.category,

                    parse_success=parsed.success,
                    parse_error=parsed.error,

                    guess_result=result,
                    feedback_returned=feedback,

                    remaining_words_before=state_before[
                        "remaining_words"
                    ],
                    solved_group_indices_before=state_before[
                        "solved_group_indices"
                    ],
                    mistakes_used_before=state_before[
                        "mistakes_used"
                    ],

                    latency_ms=response.latency_ms,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    total_tokens=response.total_tokens,

                    provider_request_id=(
                        response.provider_request_id
                    ),
                    model_metadata=(
                        response.metadata or {}
                    ),
                ),
            )

            previous_turns.append(
                {
                    "turn_index": turn_index,
                    "raw_response": split.raw_output,
                    "visible_response": split.visible_response,
                    "guess": parsed.guess,
                    "category": parsed.category,
                    "parse_success": parsed.success,
                    "parse_error": parsed.error,
                    "game_status": (
                        result.status.value
                        if result is not None
                        else None
                    ),
                    "feedback": feedback,
                    "mistakes_remaining": (
                        game.mistakes_remaining
                    ),
                    "remaining_words": list(
                        game.remaining_words
                    ),
                }
            )

        if game.game_over:
            logger.complete_run(
                run_id=run_id,
                game=game,
            )
        else:
            logger.fail_run(
                run_id=run_id,
                failure_message=(
                    "Maximum model-turn limit reached."
                ),
            )

        return run_id

    except Exception as error:
        logger.fail_run(
            run_id=run_id,
            failure_message=repr(error),
        )
        raise


if __name__ == "__main__":
    run_id = run_puzzle(
        puzzle_id="nyt_2024-06-03",
        model_provider="your-provider",
        model_name="your-model-name",
        shuffle_seed=42,
    )

    print(f"Completed run: {run_id}")
