from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from src.model_client import OpenAICompatibleModel
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


DEFAULT_DATABASE = "data/db/connections.db"
DEFAULT_MODEL = "nvidia/Nemotron-H-8B-Reasoning-128K"


def run_one_puzzle(
    *,
    puzzle_id: str,
    database_path: str,
    model_name: str,
    base_url: str,
    shuffle_seed: int,
    max_model_turns: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    """
    Run one complete multi-turn Connections game.

    The model receives the full history of its previous generations,
    submitted guesses, and game feedback after every turn.
    """

    model = OpenAICompatibleModel(
        model_name=model_name,
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    available_models = model.health_check()

    print(
        "vLLM models:",
        json.dumps(available_models, indent=2),
    )

    game = load_game_from_database(
        puzzle_id=puzzle_id,
        shuffle_seed=shuffle_seed,
        auto_complete_final_group=False,
    )

    logger = TrajectoryLogger(database_path)

    run_id = logger.start_run(
        puzzle_id=game.puzzle_id,
        model_provider=model.provider_name,
        model_name=model.model_name,
        shuffle_seed=shuffle_seed,
        max_mistakes=game.max_mistakes,
        auto_complete_final_group=False,
        decoding_settings={
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
        system_prompt=SYSTEM_PROMPT,
        run_metadata={
            "runner": "run_one_puzzle.py",
            "full_history_enabled": True,
            "reasoning_capture": "raw_think_tags",
        },
    )

    # Full history supplied back to the model on subsequent turns.
    previous_turns: list[dict[str, Any]] = []

    # Correct groups explicitly submitted by the model.
    # Because auto_complete_final_group=False, the model must submit
    # all four groups itself.
    predicted_groups: list[dict[str, Any]] = []

    print(f"Run ID: {run_id}")
    print(f"Puzzle: {game.puzzle_id}")
    print(
        "Initial words:",
        list(game.get_remaining_words_in_display_order()),
    )

    try:
        for turn_index in range(max_model_turns):
            if game.game_over:
                break

            # Authoritative pre-turn state.
            state_before = game.get_state()

            # Include details of solved groups in the prompt.
            state_before["solved_groups"] = (
                game.get_solved_groups()
            )

            prompt = build_turn_prompt(
                game_state=state_before,
                previous_turns=previous_turns,
            )

            print()
            print("=" * 80)
            print(f"TURN {turn_index}")
            print("=" * 80)

            response = model.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
            )

            finish_reason = response.metadata.get(
                "finish_reason"
            )

            generation_truncated = (
                finish_reason == "length"
            )

            print("RAW MODEL OUTPUT")
            print(response.raw_output)

            # Preserve the complete raw generation while separating any
            # exposed reasoning from the user-facing response.
            split = split_generation(
                response.raw_output
            )

            # Parse the final JSON guess from the visible response.
            parsed = parse_final_guess(
                split.visible_response
            )

        if generation_truncated and not parsed.success:
            parsed.error = (
                "Generation was truncated before a valid final JSON "
                "guess was produced."
            )

            result = None

            if parsed.success:
                result = game.submit_guess(
                    parsed.guess,
                    proposed_category=parsed.category,
                )

                feedback = result.message

                # Store every correctly submitted model group.
                if result.status.value == "correct":
                    predicted_groups.append(
                        {
                            "turn_index": turn_index,
                            "words": list(
                                result.guessed_words
                            ),
                            "proposed_category": (
                                parsed.category
                            ),
                            "matched_group_index": (
                                result.matched_group_index
                            ),
                            "ground_truth_category": (
                                result.matched_category
                            ),
                        }
                    )

            else:
                # A formatting failure is recorded as a model turn,
                # but it does not consume a game mistake.
                feedback = (
                    "FORMAT ERROR: "
                    f"{parsed.error} "
                    "No game mistake was consumed."
                )

            print()
            print("PARSED TURN")
            print(
                json.dumps(
                    {
                        "reasoning_extraction_method": (
                            split.extraction_method
                        ),
                        "has_internal_reasoning": (
                            split.internal_reasoning
                            is not None
                        ),
            "finish_reason": finish_reason,
            "generation_truncated": (
                generation_truncated
            ),
                        "guess": parsed.guess,
                        "category": parsed.category,
                        "parse_success": parsed.success,
                        "game_status": (
                            result.status.value
                            if result is not None
                            else "parse_failure"
                        ),
                        "feedback": feedback,
                    },
                    indent=2,
                )
            )

            logger.log_turn(
                run_id=run_id,
                turn=TurnLog(
                    turn_index=turn_index,
                    prompt_shown=prompt,
                    raw_generation=split.raw_output,

                    remaining_words_before=list(
                        state_before["remaining_words"]
                    ),
                    solved_group_indices_before=list(
                        state_before[
                            "solved_group_indices"
                        ]
                    ),
                    mistakes_used_before=state_before[
                        "mistakes_used"
                    ],

                    internal_reasoning=(
                        split.internal_reasoning
                    ),
                    visible_response=(
                        split.visible_response
                    ),
                    reasoning_extraction_method=(
                        split.extraction_method
                    ),
                    reasoning_start_token=(
                        "<think>"
                        if split.extraction_method
                        == "think_tags"
                        else None
                    ),
                    reasoning_end_token=(
                        "</think>"
                        if split.extraction_method
                        in {
                            "think_tags",
                            "closing_think_tag",
                        }
                        else None
                    ),

                    parsed_guess=parsed.guess,
                    proposed_category=parsed.category,
                    parse_success=parsed.success,
                    parse_error=parsed.error,

                    guess_result=result,
                    feedback_returned=feedback,

                    latency_ms=response.latency_ms,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=(
                        response.completion_tokens
                    ),
                    total_tokens=response.total_tokens,

                    provider_request_id=(
                        response.provider_request_id
                    ),
                    model_metadata=response.metadata,
                ),
            )

            # Use GuessResult as the authoritative post-turn state.
            # For parse failures, no game transition occurred, so retain
            # the pre-turn state.
            previous_turns.append(
                {
                    "turn_index": turn_index,
                    "raw_generation": split.raw_output,
                    "guess": parsed.guess,
                    "category": parsed.category,
                    "parse_success": parsed.success,
                    "parse_error": parsed.error,
            "finish_reason": finish_reason,
            "generation_truncated": (
            generation_truncated
            ),
                    "game_status": (
                        result.status.value
                        if result is not None
                        else "parse_failure"
                    ),
                    "feedback": feedback,
                    "mistakes_remaining": (
                        result.mistakes_remaining
                        if result is not None
                        else state_before[
                            "mistakes_remaining"
                        ]
                    ),
                    "remaining_words": (
                        list(result.remaining_words)
                        if result is not None
                        else list(
                            state_before[
                                "remaining_words"
                            ]
                        )
                    ),
                    "solved_group_indices": (
                        list(
                            result.solved_group_indices
                        )
                        if result is not None
                        else list(
                            state_before[
                                "solved_group_indices"
                            ]
                        )
                    ),
                }
            )

        if game.game_over:
            logger.complete_run(
                run_id=run_id,
                game=game,
                final_predicted_groups=predicted_groups,
            )

            print()
            print("=" * 80)
            print("RUN COMPLETED")
            print("=" * 80)
            print(f"Solved: {game.solved}")
            print(
                f"Guesses used: {game.guesses_used}"
            )
            print(
                f"Mistakes used: {game.mistakes_used}"
            )
            print(
                "Solved groups:",
                json.dumps(
                    game.get_solved_groups(),
                    indent=2,
                ),
            )
            print(
                "Model-predicted groups:",
                json.dumps(
                    predicted_groups,
                    indent=2,
                ),
            )

        else:
            raise RuntimeError(
                "Maximum model-turn limit reached "
                f"({max_model_turns})."
            )

        return run_id

    except Exception as error:
        logger.fail_run(
            run_id=run_id,
            error=error,
            game=game,
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one multi-turn Connections puzzle "
            "using an OpenAI-compatible model server."
        )
    )

    parser.add_argument(
        "--puzzle-id",
        required=True,
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
        default="http://127.0.0.1:8000/v1",
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

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()

    try:
        completed_run_id = run_one_puzzle(
            puzzle_id=arguments.puzzle_id,
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

        print(f"Stored run: {completed_run_id}")

    except Exception as error:
        print(
            f"Run failed: {error}",
            file=sys.stderr,
        )
        raise
