from connections_research.game.environment import (
    ConnectionsGame,
    GroundTruthGroup,
    GuessStatus,
)


WORDS = (
    "LASER",
    "PLUCK",
    "THREAD",
    "WAX",
    "COIL",
    "SPOOL",
    "WIND",
    "WRAP",
    "HONEYCOMB",
    "ORGANISM",
    "SOLAR PANEL",
    "SPREADSHEET",
    "BALL",
    "MOVIE",
    "SCHOOL",
    "VITAMIN",
)


GROUPS = (
    GroundTruthGroup(
        0,
        "REMOVE, AS BODY HAIR",
        frozenset(
            {"LASER", "PLUCK", "THREAD", "WAX"}
        ),
    ),
    GroundTruthGroup(
        1,
        "TWIST AROUND",
        frozenset(
            {"COIL", "SPOOL", "WIND", "WRAP"}
        ),
    ),
    GroundTruthGroup(
        2,
        "THINGS MADE OF CELLS",
        frozenset(
            {
                "HONEYCOMB",
                "ORGANISM",
                "SOLAR PANEL",
                "SPREADSHEET",
            }
        ),
    ),
    GroundTruthGroup(
        3,
        "B-___",
        frozenset(
            {"BALL", "MOVIE", "SCHOOL", "VITAMIN"}
        ),
    ),
)


def make_game() -> ConnectionsGame:
    return ConnectionsGame(
        puzzle_id="test",
        original_words=WORDS,
        groups=GROUPS,
        shuffle_seed=42,
    )


def test_correct_guess_removes_words() -> None:
    game = make_game()

    result = game.submit_guess(
        ["WAX", "LASER", "THREAD", "PLUCK"]
    )

    assert result.status == GuessStatus.CORRECT
    assert game.mistakes_used == 0
    assert "LASER" not in game.remaining_words


def test_one_away_consumes_mistake() -> None:
    game = make_game()

    result = game.submit_guess(
        ["LASER", "PLUCK", "THREAD", "COIL"]
    )

    assert result.status == GuessStatus.ONE_AWAY
    assert game.mistakes_used == 1


def test_invalid_guess_does_not_consume_mistake() -> None:
    game = make_game()

    result = game.submit_guess(
        ["LASER", "LASER", "THREAD", "WAX"]
    )

    assert result.status == GuessStatus.INVALID
    assert game.mistakes_used == 0


def test_repeated_guess_does_not_consume_extra_mistake() -> None:
    game = make_game()

    first = game.submit_guess(
        ["LASER", "PLUCK", "THREAD", "COIL"]
    )

    second = game.submit_guess(
        ["COIL", "THREAD", "PLUCK", "LASER"]
    )

    assert first.status == GuessStatus.ONE_AWAY
    assert second.status == GuessStatus.REPEATED
    assert game.mistakes_used == 1


def test_game_ends_after_four_mistakes() -> None:
    game = make_game()

    guesses = [
        ["LASER", "COIL", "HONEYCOMB", "BALL"],
        ["PLUCK", "SPOOL", "ORGANISM", "MOVIE"],
        ["THREAD", "WIND", "SOLAR PANEL", "SCHOOL"],
        ["WAX", "WRAP", "SPREADSHEET", "VITAMIN"],
    ]

    for guess in guesses:
        game.submit_guess(guess)

    assert game.game_over
    assert not game.solved
    assert game.mistakes_used == 4


def test_final_group_is_auto_completed() -> None:
    game = make_game()

    game.submit_guess(
        ["LASER", "PLUCK", "THREAD", "WAX"]
    )
    game.submit_guess(
        ["COIL", "SPOOL", "WIND", "WRAP"]
    )
    result = game.submit_guess(
        [
            "HONEYCOMB",
            "ORGANISM",
            "SOLAR PANEL",
            "SPREADSHEET",
        ]
    )

    assert result.game_over
    assert result.solved
    assert len(game.solved_group_indices) == 4
