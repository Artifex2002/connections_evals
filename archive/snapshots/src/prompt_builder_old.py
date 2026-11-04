from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """
{'reasoning': True}

You are the sole player of a New York Times Connections-style game.

You are directly playing the game. You are not advising, assisting, or
describing the actions of a user. Every previous guess in the history was
submitted by you.

GAME OBJECTIVE

There are sixteen words divided into exactly four hidden groups of four.
Every word belongs to exactly one group.
The four words in each group share a specific connection.

On each turn, submit exactly one set of four currently available words.

GAME RULES

- You begin with four mistakes available.
- You may continue making guesses until either:
  1. all four groups are solved, or
  2. you accumulate four mistakes.
- A correct guess does not consume a mistake.
- An incorrect guess consumes one mistake.
- A one-away guess is incorrect and consumes one mistake.
- "One away" means exactly three of the four submitted words belong to
  one currently unsolved correct group.
- A repeated guess does not consume a mistake.
- An invalid or malformed guess does not consume a mistake.
- Only words in a CORRECT group are removed from the board.
- Words from incorrect, one-away, repeated, invalid, or malformed guesses
  remain available.
- The CURRENT AVAILABLE WORDS list is authoritative.
- Never mentally remove words unless they appear in a solved group.
- Never use words from a solved group.
- Never submit the same set of four words more than once.
- Changing the category or word order does not make it a new guess.
- Guess evaluation is based on the submitted set of four words, not on the
  proposed category wording.
- Use all previous guesses and feedback as evidence.
- If only four words remain, reason about them and explicitly submit them.

FEEDBACK MEANINGS

CORRECT:
The submitted words form an exact hidden group. The words are removed.

ONE_AWAY:
Exactly three of the four submitted words belong to the same currently
unsolved correct group.
One of the submitted words does not belong to that group and must be replaced
with another currently available word.
The proposed category label is not used to determine whether a guess is
correct, incorrect, or one away. Changing only the category cannot fix a
one-away guess.
The guess consumes one mistake. All submitted words remain available.

INCORRECT:
The guess is not correct and is not one away.
The guess consumes one mistake. All submitted words remain available.

REPEATED:
The exact same four-word set was previously submitted.
No mistake is consumed. All submitted words remain available.

INVALID:
The guess was malformed or used unavailable words.
No mistake is consumed.

PARSE_FAILURE:
No valid final JSON guess was produced, so no official guess was submitted
or evaluated.

A parse failure is not an incorrect guess and consumes no mistake.

Ideas, candidate groups, or word sets mentioned inside an unfinished or
truncated response do not count as previous guesses. Only a successfully
parsed final JSON object counts as an official submitted guess.

TURN GUIDANCE

You only need to submit one group on each turn.

You may reason about the broader puzzle if that helps, but once you have a
sufficiently strong candidate group, consider committing to it rather than
continuing to explore every possible full partition.

A complete turn must include the required final JSON guess, so preserve enough
response space to provide it.

OUTPUT FORMAT

Reason naturally before answering.

At the very end of the response, output exactly one JSON object:

{
  "guess": ["WORD 1", "WORD 2", "WORD 3", "WORD 4"],
  "category": "PROPOSED CONNECTION"
}

The JSON object must be the final content in the response.

Use words exactly as displayed.
""".strip()


def _canonical_guess(words: list[str] | None) -> tuple[str, ...] | None:
    if not words or len(words) != 4:
        return None

    return tuple(sorted(word.strip().upper() for word in words))


def _collect_rejected_guesses(
    previous_turns: list[dict[str, Any]],
) -> list[list[str]]:
    """
    Return unique previously submitted word sets that must not be
    submitted again.

    Correct guesses are excluded because their words are already removed.
    Parse failures are excluded because no valid group was submitted.
    """

    rejected: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    for turn in previous_turns:
        status = turn.get("game_status")
        guess = turn.get("guess")

        if status in {
            None,
            "correct",
            "parse_failure",
            "invalid",
        }:
            continue

        canonical = _canonical_guess(guess)

        if canonical is None or canonical in seen:
            continue

        seen.add(canonical)
        rejected.append(list(canonical))

    return rejected


def build_turn_prompt(
    *,
    game_state: dict[str, Any],
    previous_turns: list[dict[str, Any]],
) -> str:
    rejected_guesses = _collect_rejected_guesses(
        previous_turns
    )

    sections: list[str] = [
        "CURRENT AUTHORITATIVE GAME STATE",
        "",
        "You are the player. Submit your own next guess.",
        "",
        "Available words:",
        json.dumps(
            game_state["remaining_words"],
            ensure_ascii=False,
        ),
        "",
        (
            "Mistakes used: "
            f"{game_state['mistakes_used']}"
        ),
        (
            "Mistakes remaining: "
            f"{game_state['mistakes_remaining']}"
        ),
        "",
        "Solved groups:",
        json.dumps(
            game_state.get("solved_groups", []),
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "PREVIOUSLY REJECTED WORD SETS",
        "Never submit any of these exact four-word sets again,",
        "even with a different category or ordering:",
        json.dumps(
            rejected_guesses,
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "IMPORTANT STATE RULES",
        "- Only words from correct groups are removed.",
        "- Words from incorrect, one-away, repeated, invalid, or",
        "  malformed guesses remain available.",
        "- Trust the available-word list above.",
        "- Do not infer that words disappeared merely because they",
        "  appeared in an earlier guess.",
    ]

    if previous_turns:
        sections.extend(
            [
                "",
                "COMPACT TURN SUMMARY",
                "",
            ]
        )

        for turn in previous_turns:
            officially_submitted = (
                turn.get("parse_success") is True
            )
            sections.extend(
                [
                    (
                        f"Turn {turn['turn_index']}: "
            f"official_submission={officially_submitted}, "
                        f"status={turn.get('game_status')}, "
                        f"guess={json.dumps(turn.get('guess'))}, "
                        f"category={turn.get('category')}, "
                        f"feedback={turn.get('feedback')}"
                    ),
                ]
            )

        sections.extend(
            [
                "",
                "FULL RAW PREVIOUS-TURN HISTORY",
                "",
                (
            "The following responses were generated by you, not by a user. "
                    "Some responses may have been truncated or may not have produced "
            "an official submitted guess. Trust the parsed status, official "
            "submission flag, and feedback shown after each response."
                ),
                "",
            ]
        )

        for turn in previous_turns:
            sections.extend(
                [
                    f"--- YOUR TURN {turn['turn_index']} ---",
            (
            "Official submission: "
            f"{turn.get('parse_success') is True}"
            ),
                    "",
                    "Your complete previous generation:",
                    turn["raw_generation"],
                    "",
                    "Parsed guess:",
                    json.dumps(
                        turn.get("guess"),
                        ensure_ascii=False,
                    ),
                    "",
                    (
                        "Proposed category: "
                        f"{turn.get('category')}"
                    ),
                    (
                        "Result: "
                        f"{turn.get('game_status')}"
                    ),
                    (
                        "Feedback: "
                        f"{turn.get('feedback')}"
                    ),
                    (
                        "Mistakes remaining afterward: "
                        f"{turn.get('mistakes_remaining')}"
                    ),
                    "Remaining words afterward:",
                    json.dumps(
                        turn.get("remaining_words"),
                        ensure_ascii=False,
                    ),
                    "",
                ]
            )
    else:
        sections.extend(
            [
                "",
                "No previous turns have been submitted.",
            ]
        )

    sections.extend(
        [
            "",
            "NEXT ACTION",
            "",
            "Analyze the puzzle and submit one new four-word group.",
            "Do not repeat a previously rejected set.",
            "",
            "End your response with exactly:",
            "",
            "{",
            '  "guess": ["WORD 1", "WORD 2", '
            '"WORD 3", "WORD 4"],',
            '  "category": "PROPOSED CONNECTION"',
            "}",
        ]
    )

    return "\n".join(sections).strip()
