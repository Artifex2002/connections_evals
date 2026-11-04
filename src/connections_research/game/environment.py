from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


def normalize_word(word: str) -> str:
    """Normalize a word before comparison."""
    return " ".join(str(word).strip().upper().split())


class GuessStatus(str, Enum):
    CORRECT = "correct"
    ONE_AWAY = "one_away"
    INCORRECT = "incorrect"
    INVALID = "invalid"
    REPEATED = "repeated"
    ALREADY_FINISHED = "already_finished"


@dataclass(frozen=True)
class GroundTruthGroup:
    group_index: int
    category: str
    words: frozenset[str]


@dataclass
class GuessResult:
    guess_number: int
    guessed_words: tuple[str, ...]
    status: GuessStatus

    category: str | None = None
    matched_group_index: int | None = None
    matched_category: str | None = None

    mistakes_used: int = 0
    mistakes_remaining: int = 4

    remaining_words: tuple[str, ...] = ()
    solved_group_indices: tuple[int, ...] = ()

    game_over: bool = False
    solved: bool = False
    message: str = ""


@dataclass
class ConnectionsGame:
    puzzle_id: str
    original_words: tuple[str, ...]
    groups: tuple[GroundTruthGroup, ...]
    shuffle_seed: int

    max_mistakes: int = 4
    auto_complete_final_group: bool = True

    displayed_words: list[str] = field(init=False)
    remaining_words: set[str] = field(init=False)
    unsolved_groups: dict[int, GroundTruthGroup] = field(init=False)

    solved_group_indices: list[int] = field(default_factory=list)
    submitted_guesses: set[frozenset[str]] = field(default_factory=set)

    mistakes_used: int = 0
    guesses_used: int = 0
    game_over: bool = False
    solved: bool = False

    def __post_init__(self) -> None:
        normalized_words = tuple(
            normalize_word(word)
            for word in self.original_words
        )

        if len(normalized_words) != 16:
            raise ValueError(
                f"Expected 16 words, found {len(normalized_words)}"
            )

        if len(set(normalized_words)) != 16:
            raise ValueError("Puzzle contains duplicate words")

        if len(self.groups) != 4:
            raise ValueError(
                f"Expected 4 groups, found {len(self.groups)}"
            )

        normalized_groups: list[GroundTruthGroup] = []

        for group in self.groups:
            normalized_group_words = frozenset(
                normalize_word(word)
                for word in group.words
            )

            if len(normalized_group_words) != 4:
                raise ValueError(
                    f"Group {group.group_index} does not contain "
                    "4 unique words"
                )

            normalized_groups.append(
                GroundTruthGroup(
                    group_index=group.group_index,
                    category=group.category,
                    words=normalized_group_words,
                )
            )

        grouped_words = set().union(
            *(group.words for group in normalized_groups)
        )

        if grouped_words != set(normalized_words):
            raise ValueError(
                "Ground-truth group words do not exactly match puzzle words"
            )

        self.original_words = normalized_words
        self.groups = tuple(normalized_groups)

        self.displayed_words = list(normalized_words)
        random.Random(self.shuffle_seed).shuffle(self.displayed_words)

        self.remaining_words = set(normalized_words)

        self.unsolved_groups = {
            group.group_index: group
            for group in normalized_groups
        }

    def submit_guess(
        self,
        guessed_words: Iterable[str],
        proposed_category: str | None = None,
    ) -> GuessResult:
        """
        Submit one four-word guess and update the game state.

        Invalid and repeated guesses do not consume a mistake.
        Valid incorrect guesses consume one mistake.
        """
        if self.game_over:
            return self._make_result(
                guessed_words=tuple(
                    normalize_word(word)
                    for word in guessed_words
                ),
                status=GuessStatus.ALREADY_FINISHED,
                proposed_category=proposed_category,
                message="The game has already finished.",
            )

        normalized_guess = tuple(
            normalize_word(word)
            for word in guessed_words
        )

        validation_error = self._validate_guess(normalized_guess)

        if validation_error is not None:
            return self._make_result(
                guessed_words=normalized_guess,
                status=GuessStatus.INVALID,
                proposed_category=proposed_category,
                message=validation_error,
            )

        guess_set = frozenset(normalized_guess)

        if guess_set in self.submitted_guesses:
            return self._make_result(
                guessed_words=normalized_guess,
                status=GuessStatus.REPEATED,
                proposed_category=proposed_category,
                message="This exact group has already been submitted.",
            )

        self.guesses_used += 1
        self.submitted_guesses.add(guess_set)

        matched_group = self._find_exact_group(guess_set)

        if matched_group is not None:
            self._solve_group(matched_group)

            result = self._make_result(
                guessed_words=normalized_guess,
                status=GuessStatus.CORRECT,
                proposed_category=proposed_category,
                matched_group=matched_group,
                message=f"Correct: {matched_group.category}",
            )

            self._auto_complete_if_needed()
            self._update_game_over()

            return self._refresh_result_state(result)

        one_away = self._is_one_away(guess_set)

        self.mistakes_used += 1
        self._update_game_over()

        if one_away:
            status = GuessStatus.ONE_AWAY
            message = "Incorrect. One word away."
        else:
            status = GuessStatus.INCORRECT
            message = "Incorrect."

        return self._make_result(
            guessed_words=normalized_guess,
            status=status,
            proposed_category=proposed_category,
            message=message,
        )

    def _validate_guess(
        self,
        guessed_words: tuple[str, ...],
    ) -> str | None:
        if len(guessed_words) != 4:
            return (
                f"A guess must contain exactly 4 words; "
                f"received {len(guessed_words)}."
            )

        if len(set(guessed_words)) != 4:
            return "A guess must contain 4 distinct words."

        unknown_words = sorted(
            set(guessed_words) - self.remaining_words
        )

        if unknown_words:
            return (
                "Guess contains words that are unavailable or unknown: "
                + ", ".join(unknown_words)
            )

        return None

    def _find_exact_group(
        self,
        guess_set: frozenset[str],
    ) -> GroundTruthGroup | None:
        for group in self.unsolved_groups.values():
            if guess_set == group.words:
                return group

        return None

    def _is_one_away(
        self,
        guess_set: frozenset[str],
    ) -> bool:
        return any(
            len(guess_set.intersection(group.words)) == 3
            for group in self.unsolved_groups.values()
        )

    def _solve_group(
        self,
        group: GroundTruthGroup,
    ) -> None:
        self.remaining_words -= set(group.words)
        self.solved_group_indices.append(group.group_index)
        del self.unsolved_groups[group.group_index]

    def _auto_complete_if_needed(self) -> None:
        """
        Mirror the NYT game: after three groups are found, the final four
        remaining words are automatically the final group.
        """
        if not self.auto_complete_final_group:
            return

        if len(self.unsolved_groups) != 1:
            return

        final_group = next(iter(self.unsolved_groups.values()))

        if self.remaining_words == set(final_group.words):
            self._solve_group(final_group)

    def _update_game_over(self) -> None:
        if not self.unsolved_groups:
            self.game_over = True
            self.solved = True
            return

        if self.mistakes_used >= self.max_mistakes:
            self.game_over = True
            self.solved = False

    def _make_result(
        self,
        guessed_words: tuple[str, ...],
        status: GuessStatus,
        proposed_category: str | None,
        message: str,
        matched_group: GroundTruthGroup | None = None,
    ) -> GuessResult:
        return GuessResult(
            guess_number=self.guesses_used,
            guessed_words=guessed_words,
            status=status,
            category=proposed_category,
            matched_group_index=(
                matched_group.group_index
                if matched_group
                else None
            ),
            matched_category=(
                matched_group.category
                if matched_group
                else None
            ),
            mistakes_used=self.mistakes_used,
            mistakes_remaining=max(
                0,
                self.max_mistakes - self.mistakes_used,
            ),
            remaining_words=self.get_remaining_words_in_display_order(),
            solved_group_indices=tuple(self.solved_group_indices),
            game_over=self.game_over,
            solved=self.solved,
            message=message,
        )

    def _refresh_result_state(
        self,
        result: GuessResult,
    ) -> GuessResult:
        """
        Refresh fields that may have changed because of final-group
        auto-completion.
        """
        result.mistakes_used = self.mistakes_used
        result.mistakes_remaining = max(
            0,
            self.max_mistakes - self.mistakes_used,
        )
        result.remaining_words = (
            self.get_remaining_words_in_display_order()
        )
        result.solved_group_indices = tuple(
            self.solved_group_indices
        )
        result.game_over = self.game_over
        result.solved = self.solved

        return result

    def get_remaining_words_in_display_order(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            word
            for word in self.displayed_words
            if word in self.remaining_words
        )

    def get_solved_groups(self) -> list[dict]:
    	solved_indices = set(self.solved_group_indices)

    	return [
        	{
            	"group_index": group.group_index,
            	"category": group.category,
            	"words": sorted(group.words),
        	}
        	for group in self.groups
   	     if group.group_index in solved_indices
    	]

    def get_state(self) -> dict:
        return {
            "puzzle_id": self.puzzle_id,
            "shuffle_seed": self.shuffle_seed,
            "displayed_words": list(self.displayed_words),
            "remaining_words": list(
                self.get_remaining_words_in_display_order()
            ),
            "solved_group_indices": list(
                self.solved_group_indices
            ),
            "mistakes_used": self.mistakes_used,
            "mistakes_remaining": max(
                0,
                self.max_mistakes - self.mistakes_used,
            ),
            "guesses_used": self.guesses_used,
            "game_over": self.game_over,
            "solved": self.solved,
        }
