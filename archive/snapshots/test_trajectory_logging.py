from src.puzzle_repository import load_game_from_database
from src.trajectory_logger import (
    TrajectoryLogger,
    TurnLog,
)


def main() -> None:
    game = load_game_from_database(
        puzzle_id="nyt_2024-06-03",
        shuffle_seed=42,
    )

    logger = TrajectoryLogger()

    run_id = logger.start_run(
        puzzle_id=game.puzzle_id,
        model_provider="test",
        model_name="manual-fake-model",
        shuffle_seed=game.shuffle_seed,
        max_mistakes=game.max_mistakes,
        decoding_settings={
            "temperature": 0.0,
            "max_tokens": 1000,
        },
        system_prompt=(
            "Solve the NYT Connections puzzle."
        ),
        run_metadata={
            "purpose": "trajectory logger smoke test"
        },
    )

    print("Run ID:", run_id)

    fake_turns = [
        {
            "guess": [
                "LASER",
                "PLUCK",
                "THREAD",
                "WAX",
            ],
            "category": "Hair removal methods",
            "reasoning": (
                "These four are methods for removing body hair."
            ),
        },
        {
            "guess": [
                "COIL",
                "SPOOL",
                "WIND",
                "WRAP",
            ],
            "category": "Twist around",
            "reasoning": (
                "All four can describe winding something."
            ),
        },
        {
            "guess": [
                "HONEYCOMB",
                "ORGANISM",
                "SOLAR PANEL",
                "SPREADSHEET",
            ],
            "category": "Things made of cells",
            "reasoning": (
                "Each consists of structures called cells."
            ),
        },
	{
   	   "guess": [
           	"BALL",
        	"MOVIE",
        	"SCHOOL",
        	"VITAMIN",
    	],
    	"category": "B-___",
    	"reasoning": (
        	"Each word forms a familiar term when preceded by B: "
        	"B-ball, B-movie, B-school, and B-vitamin."
    	)

    ]

    try:
        feedback = "No previous feedback."

        for turn_index, fake in enumerate(
            fake_turns,
            start=1,
        ):
            state_before = game.get_state()

            prompt = f"""
You are playing NYT Connections.

Remaining words:
{state_before["remaining_words"]}

Mistakes remaining:
{state_before["mistakes_remaining"]}

Previous feedback:
{feedback}
""".strip()

            raw_output = (
                f"Reasoning: {fake['reasoning']}\n"
                f"Guess: {fake['guess']}\n"
                f"Category: {fake['category']}"
            )

            result = game.submit_guess(
                guessed_words=fake["guess"],
                proposed_category=fake["category"],
            )

            feedback = result.message

            logger.log_turn(
                run_id=run_id,
                turn=TurnLog(
                    turn_index=turn_index,
                    prompt_shown=prompt,
                    raw_output=raw_output,
                    reasoning_trace=fake["reasoning"],
                    parsed_guess=fake["guess"],
                    proposed_category=fake["category"],
                    parse_success=True,
                    guess_result=result,
                    feedback_returned=feedback,
                    remaining_words_before=list(
                        state_before["remaining_words"]
                    ),
                    solved_group_indices_before=list(
                        state_before["solved_group_indices"]
                    ),
                    mistakes_used_before=state_before[
                        "mistakes_used"
                    ],
                    latency_ms=250.0,
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    model_metadata={
                        "finish_reason": "stop"
                    },
                ),
            )

        logger.complete_run(
            run_id=run_id,
            game=game,
            final_predicted_groups=game.get_solved_groups(),
        )

        print("Run completed.")
        print("Solved:", game.solved)
        print("Score:", int(game.solved))

    except Exception as error:
        logger.fail_run(
            run_id=run_id,
            error=error,
            game=game,
        )
        raise


if __name__ == "__main__":
    main()
