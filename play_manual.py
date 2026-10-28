import argparse

from src.connections_game import GuessStatus
from src.puzzle_repository import (
    load_game_from_database,
    load_latest_puzzle_id,
)


def print_words(words: tuple[str, ...]) -> None:
    for index, word in enumerate(words, start=1):
        print(f"{word:<20}", end="")

        if index % 4 == 0:
            print()

    print()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--puzzle-id",
        default=None,
        help="Puzzle ID such as nyt_2024-06-03",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed controlling word presentation order",
    )

    args = parser.parse_args()

    puzzle_id = args.puzzle_id or load_latest_puzzle_id()

    game = load_game_from_database(
        puzzle_id=puzzle_id,
        shuffle_seed=args.seed,
    )

    print(f"\nPuzzle: {game.puzzle_id}")
    print(f"Shuffle seed: {game.shuffle_seed}")
    print("Enter four comma-separated words.")
    print("Type 'quit' to stop.\n")

    while not game.game_over:
        print("Remaining words:")
        print_words(
            game.get_remaining_words_in_display_order()
        )

        print(
            f"Mistakes remaining: "
            f"{game.max_mistakes - game.mistakes_used}"
        )

        raw_guess = input("\nGuess: ").strip()

        if raw_guess.lower() == "quit":
            break

        guessed_words = [
            word.strip()
            for word in raw_guess.split(",")
            if word.strip()
        ]

        category = input(
            "Proposed category, optional: "
        ).strip()

        result = game.submit_guess(
            guessed_words=guessed_words,
            proposed_category=category or None,
        )

        print(f"\nResult: {result.status.value}")
        print(result.message)

        if result.status in {
            GuessStatus.INVALID,
            GuessStatus.REPEATED,
        }:
            print("This did not consume a mistake.")

        print()

    print("\nFinal state:")

    state = game.get_state()

    for key, value in state.items():
        print(f"{key}: {value}")

    print("\nBinary score:", int(game.solved))


if __name__ == "__main__":
    main()
