from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class SplitGeneration:
    raw_output: str
    internal_reasoning: str | None
    visible_response: str
    extraction_method: str


@dataclass
class ParsedGuess:
    success: bool
    guess: list[str] | None = None
    category: str | None = None
    error: str | None = None
    parsed_object: dict[str, Any] | None = None


def split_generation(raw_output: str) -> SplitGeneration:
    """
    Separate reasoning from the visible response while preserving the exact
    raw generation.

    Supported forms:

    1. <think>reasoning</think>visible
    2. reasoning</think>visible
       Some chat templates insert the opening reasoning marker internally,
       so it does not appear in message.content.
    3. visible only
    """

    start_token = "<think>"
    end_token = "</think>"

    start = raw_output.find(start_token)
    end = raw_output.find(end_token)

    # Normal form:
    # <think>...</think> visible response
    if start != -1 and end != -1 and end > start:
        reasoning_start = start + len(start_token)

        internal_reasoning = raw_output[
            reasoning_start:end
        ].strip()

        before_reasoning = raw_output[:start].strip()
        after_reasoning = raw_output[
            end + len(end_token):
        ].strip()

        visible_parts = [
            part
            for part in (
                before_reasoning,
                after_reasoning,
            )
            if part
        ]

        return SplitGeneration(
            raw_output=raw_output,
            internal_reasoning=internal_reasoning or None,
            visible_response="\n\n".join(visible_parts),
            extraction_method="think_tags",
        )

    # Nemotron/vLLM form observed in the smoke test:
    # reasoning text </think> visible response
    if start == -1 and end != -1:
        internal_reasoning = raw_output[:end].strip()
        visible_response = raw_output[
            end + len(end_token):
        ].strip()

        return SplitGeneration(
            raw_output=raw_output,
            internal_reasoning=internal_reasoning or None,
            visible_response=visible_response,
            extraction_method="closing_think_tag",
        )

    # Malformed form:
    # <think> reasoning without a closing tag
    if start != -1 and end == -1:
        internal_reasoning = raw_output[
            start + len(start_token):
        ].strip()

        visible_response = raw_output[:start].strip()

        return SplitGeneration(
            raw_output=raw_output,
            internal_reasoning=internal_reasoning or None,
            visible_response=visible_response,
            extraction_method="unclosed_think_tag",
        )

    # No reasoning markers exposed.
    return SplitGeneration(
        raw_output=raw_output,
        internal_reasoning=None,
        visible_response=raw_output.strip(),
        extraction_method="not_available",
    )


def find_json_objects(text: str) -> list[str]:
    """
    Find complete top-level JSON objects without relying on regex.
    """

    objects: list[str] = []

    depth = 0
    object_start: int | None = None
    in_string = False
    escaped = False

    for index, character in enumerate(text):
        if escaped:
            escaped = False
            continue

        if character == "\\" and in_string:
            escaped = True
            continue

        if character == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if character == "{":
            if depth == 0:
                object_start = index

            depth += 1

        elif character == "}" and depth > 0:
            depth -= 1

            if depth == 0 and object_start is not None:
                objects.append(
                    text[object_start:index + 1]
                )
                object_start = None

    return objects


def parse_final_guess(
    visible_response: str,
) -> ParsedGuess:
    """
    Search backward for the final valid guess object.
    """

    json_objects = find_json_objects(visible_response)

    for candidate in reversed(json_objects):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if not isinstance(parsed, dict):
            continue

        guess = parsed.get("guess")
        category = parsed.get("category")

        if not isinstance(guess, list):
            continue

        if len(guess) != 4:
            continue

        if not all(
            isinstance(word, str)
            for word in guess
        ):
            continue

        if (
            not isinstance(category, str)
            or not category.strip()
        ):
            continue

        normalized_guess = [
            " ".join(
                word.strip().upper().split()
            )
            for word in guess
        ]

        return ParsedGuess(
            success=True,
            guess=normalized_guess,
            category=category.strip(),
            parsed_object=parsed,
        )

    return ParsedGuess(
        success=False,
        error=(
            "No valid final JSON object containing exactly "
            "four words and a category was found."
        ),
    )
