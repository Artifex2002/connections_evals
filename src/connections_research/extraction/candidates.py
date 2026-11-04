#!/usr/bin/env python3

"""
tier1_scale_extract.py
======================

Deterministic Tier-1 hypothesis-candidate extraction for the current
multiturn Connections SQLite databases.

SOURCE SCHEMA
-------------
puzzles
runs
turns

UNIT OF ANALYSIS
----------------
One TURN.

For every turn from a completed run:

    1. If reasoning_trace exists:
           analyze reasoning_trace

    2. Otherwise, if visible_response exists:
           analyze visible_response

    3. Otherwise:
           record the turn as having no analyzable text

Tier 1 ONLY:
    - segments the selected text
    - detects literal puzzle-word mentions
    - logs candidate word sets

Tier 1 DOES NOT:
    - classify stance
    - reconstruct hypotheses
    - union refine_add candidates
    - compute F1
    - score against gold groups

Tier 2 performs stance classification later.

The script is incremental/resumable:
    done_turns stores (run_id, turn_id).

Because only COMPLETED source runs are processed, rerunning this script
while raw collection continues will skip old turns and process newly
completed runs.

The LABELS/build_prompt definitions remain here so Tier 2 can import the
exact same stance rubric used by the original pipeline.
"""

import argparse
import json
import re
import sqlite3
import time


# =============================================================================
# 1. STANCE RUBRIC
#
# Tier 1 does NOT use this itself.
# Tier 2 imports LABELS + build_prompt from this file.
# =============================================================================

LABELS = [
    "propose",
    "refine_add",
    "refine_pivot",
    "reject",
    "commit",
    "enumerate",
    "mention",
]


FEWSHOT = """Examples:

PREVIOUS GROUP: THREAD, SPOOL, WIND, WRAP
PUZZLE WORDS IN LINE: PLUCK, THREAD
LINE: But there's also 'pluck', which might fit here too.
LABEL: refine_add

PREVIOUS GROUP: FAMILY, FLEA, FLYING, MEDIA
PUZZLE WORDS IN LINE: LOCK, PRESS
LINE: Maybe lock and press go together instead.
LABEL: refine_pivot

PREVIOUS GROUP: (none)
PUZZLE WORDS IN LINE: BALL, MOVIE, SCHOOL
LINE: Maybe 'ball' as in sports, but then 'movie' and 'school' don't fit.
LABEL: reject

PREVIOUS GROUP: (none)
PUZZLE WORDS IN LINE: LASER, SOLAR PANEL, VITAMIN, ORGANISM, PLUCK, WRAP, SPREADSHEET
LINE: Remaining words: laser, solar panel, vitamin, organism, pluck, wrap, spreadsheet.
LABEL: enumerate

PREVIOUS GROUP: (none)
PUZZLE WORDS IN LINE: COIL, SPOOL, WIND, THREAD
LINE: Group1: COIL, SPOOL, WIND, THREAD -> "Coiling and Wrapping".
LABEL: commit

PREVIOUS GROUP: COIL, SPOOL, WIND
PUZZLE WORDS IN LINE: WIND, THREAD
LINE: 'WIND' is in the first group, so it's already placed.
LABEL: mention
"""


RUBRIC = """You label one line from a model solving a Connections puzzle
(group 16 words into 4 groups of 4).

Output ONE label for the model's stance toward grouping the mentioned words:

propose = newly suggests they might form a group

refine_add = keeps the previous group idea and adds/drops words while
persisting it ("...X also fits")

refine_pivot = abandons the previous group idea and pivots to a different
overlapping candidate

reject = lists them only to dismiss ("don't fit", "stretch", "no")

commit = states a chosen/final/locked group ("definitely", "Final",
"Group1: ...")

enumerate = merely lists leftover/remaining words, not proposing a group

mention = refers to words already placed elsewhere, not grouping them now
"""


def build_prompt(line, ws, prev):
    """
    Used by Tier 2.

    Kept here deliberately so Tier 1 and Tier 2 share one authoritative
    prompt/rubric definition.
    """
    return (
        RUBRIC
        + FEWSHOT
        + "\nNow label this one. Output only the label.\n"
        + f"PREVIOUS GROUP: {', '.join(prev) if prev else '(none)'}\n"
        + f"PUZZLE WORDS IN LINE: {', '.join(ws)}\n"
        + f"LINE: {line[:500]}\n"
        + "LABEL:"
    )


# =============================================================================
# 2. PUZZLE WORDS
# =============================================================================

def parse_puzzle_words(words_json, expected=16):
    """
    Parse puzzles.words_json.

    Example:
        ["LASER", "PLUCK", ..., "VITAMIN"]
    """

    if not words_json:
        return None

    try:
        parsed = json.loads(words_json)
    except (TypeError, json.JSONDecodeError):
        return None

    if not isinstance(parsed, list):
        return None

    words = [
        str(x).strip().upper()
        for x in parsed
        if str(x).strip()
    ]

    # Preserve original puzzle order while removing duplicates.
    seen = set()
    out = []

    for word in words:
        if word not in seen:
            seen.add(word)
            out.append(word)

    if len(out) != expected:
        return None

    return out


# =============================================================================
# 3. TIER-1 LITERAL WORD MATCHING
# =============================================================================

def compile_matchers(words):
    """
    Build one regex matcher per canonical puzzle word.

    Multiword puzzle entries tolerate arbitrary whitespace.

    Example:
        "RACE CAR"
            ->
        r"\\bRACE\\s+CARs?\\b"
    """

    out = []

    for word in words:

        body = r"\s+".join(
            re.escape(token)
            for token in word.split()
        )

        rx = re.compile(
            r"\b" + body + r"s?\b",
            re.IGNORECASE,
        )

        out.append(
            (
                word,
                rx,
            )
        )

    return out


def words_in_span(span, matchers):
    """
    Return puzzle words literally mentioned in `span`.

    Ordering follows textual occurrence, not puzzle ordering.
    """

    found = {}

    for word, rx in matchers:

        match = rx.search(span)

        if match:
            found[word] = match.start()

    return [
        word
        for word, _position
        in sorted(
            found.items(),
            key=lambda kv: kv[1],
        )
    ]


# =============================================================================
# 4. SEGMENTATION
# =============================================================================

def segment(text):
    """
    Preserve the segmentation behavior of the original single-turn
    extractor as closely as possible.

    Split:
        newline first
        then sentence boundaries [.?!]
    """

    spans = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        parts = re.split(
            r"(?<=[.?!])\s+",
            line,
        )

        for part in parts:

            part = part.strip()

            if part:
                spans.append(part)

    return spans


# =============================================================================
# 5. CANDIDATE EXTRACTION
# =============================================================================

def tier1_candidates(
    text,
    words,
    min_size=2,
):
    """
    Every span containing >= min_size puzzle words becomes a Tier-1
    candidate.

    No claim is made here about whether the model proposes/rejects/refines
    that set. Tier 2 determines the stance.
    """

    matchers = compile_matchers(
        words
    )

    candidates = []

    for span_index, span in enumerate(
        segment(text)
    ):

        wordset = words_in_span(
            span,
            matchers,
        )

        if len(wordset) < min_size:
            continue

        candidates.append(
            {
                "span_index": span_index,
                "span": span,
                "wordset": wordset,
            }
        )

    return candidates


# =============================================================================
# 6. OUTPUT DATABASE
# =============================================================================

def ensure_out(conn):

    # -------------------------------------------------------------------------
    # Raw Tier-1 candidates
    # -------------------------------------------------------------------------

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tier1_candidates(

            run_id TEXT,
            puzzle_id TEXT,

            turn_id TEXT,
            turn_index INTEGER,

            model TEXT,

            candidate_index INTEGER,
            span_index INTEGER,

            span_text TEXT,
            wordset_json TEXT,

            text_source TEXT,

            PRIMARY KEY(
                run_id,
                turn_id,
                candidate_index
            )
        )
        """
    )

    # -------------------------------------------------------------------------
    # Coverage table
    #
    # IMPORTANT:
    #
    # This records every turn from completed runs, INCLUDING turns where:
    #
    #   - there was no reasoning
    #   - visible_response had to be used
    #   - Tier 1 found zero candidate groups
    #
    # This avoids ambiguity later.
    # -------------------------------------------------------------------------

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tier1_turns(

            run_id TEXT,
            puzzle_id TEXT,

            turn_id TEXT,
            turn_index INTEGER,

            model TEXT,

            text_source TEXT,

            reasoning_extraction_method TEXT,

            source_char_count INTEGER,
            candidate_count INTEGER,

            guess_status TEXT,
            parse_success INTEGER,

            PRIMARY KEY(
                run_id,
                turn_id
            )
        )
        """
    )

    # -------------------------------------------------------------------------
    # Resume state
    # -------------------------------------------------------------------------

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS done_turns(

            run_id TEXT,
            turn_id TEXT,

            PRIMARY KEY(
                run_id,
                turn_id
            )
        )
        """
    )

    # -------------------------------------------------------------------------
    # Run metadata
    # -------------------------------------------------------------------------

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tier1_run_meta(

            started_at TEXT,
            source_db TEXT,
            min_size INTEGER

        )
        """
    )

    # -------------------------------------------------------------------------
    # Useful indexes
    # -------------------------------------------------------------------------

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_tier1_candidates_run_turn

        ON tier1_candidates(
            run_id,
            turn_index
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_tier1_candidates_puzzle

        ON tier1_candidates(
            puzzle_id,
            turn_index
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_tier1_turns_model

        ON tier1_turns(
            model
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_tier1_turns_source

        ON tier1_turns(
            text_source
        )
        """
    )

    conn.commit()


# =============================================================================
# 7. RESUME
# =============================================================================

def load_done(conn):

    return {
        (
            run_id,
            turn_id,
        )
        for (
            run_id,
            turn_id,
        )
        in conn.execute(
            """
            SELECT
                run_id,
                turn_id
            FROM done_turns
            """
        )
    }


# =============================================================================
# 8. SELECT TEXT SOURCE
# =============================================================================

def choose_turn_text(
    reasoning_trace,
    visible_response,
):
    """
    Priority requested for the experiment:

        reasoning_trace
            ↓ if absent
        visible_response
            ↓ if absent
        none

    Returns:
        text_source, text
    """

    if (
        reasoning_trace is not None
        and reasoning_trace.strip()
    ):
        return (
            "reasoning",
            reasoning_trace,
        )

    if (
        visible_response is not None
        and visible_response.strip()
    ):
        return (
            "visible_response",
            visible_response,
        )

    return (
        "none",
        "",
    )


# =============================================================================
# 9. MAIN
# =============================================================================

def run(args):

    # -------------------------------------------------------------------------
    # SOURCE: immutable/read-only from Tier-1's perspective
    # -------------------------------------------------------------------------

    src = sqlite3.connect(
        f"file:{args.src}?mode=ro",
        uri=True,
    )

    # -------------------------------------------------------------------------
    # OUTPUT: may be reused across multiple source model DBs
    # -------------------------------------------------------------------------

    out = sqlite3.connect(
        args.out
    )

    out.execute(
        "PRAGMA journal_mode=WAL;"
    )

    ensure_out(out)

    out.execute(
        """
        INSERT INTO tier1_run_meta
        VALUES (?, ?, ?)
        """,
        (
            time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            args.src,
            args.min_size,
        ),
    )

    out.commit()

    done = load_done(out)

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # ONLY completed runs are processed.
    #
    # Therefore:
    #
    #   - actively changing puzzle runs are ignored
    #   - newly completed runs become eligible next time this script is run
    #
    # This is what makes periodic incremental extraction safe.
    # -------------------------------------------------------------------------

    query = """
        SELECT

            r.run_id,
            r.puzzle_id,
            r.model_name,

            t.turn_id,
            t.turn_index,

            t.reasoning_trace,
            t.visible_response,

            t.reasoning_extraction_method,

            t.guess_status,
            t.parse_success,

            p.words_json

        FROM runs r

        JOIN turns t
            ON t.run_id = r.run_id

        JOIN puzzles p
            ON p.puzzle_id = r.puzzle_id

        WHERE r.status = 'completed'
    """

    params = []

    if args.model_filter:

        query += """
            AND r.model_name = ?
        """

        params.append(
            args.model_filter
        )

    if args.run_id:

        query += """
            AND r.run_id = ?
        """

        params.append(
            args.run_id
        )

    if args.puzzle_id:

        query += """
            AND r.puzzle_id = ?
        """

        params.append(
            args.puzzle_id
        )

    if args.turn_index is not None:

        query += """
            AND t.turn_index = ?
        """

        params.append(
            args.turn_index
        )

    query += """
        ORDER BY
            r.run_id,
            t.turn_index
    """

    cursor = src.execute(
        query,
        params,
    )

    started_at = time.time()

    processed_turns = 0
    extracted_candidates = 0

    reasoning_turns = 0
    visible_turns = 0
    empty_turns = 0

    for (
        run_id,
        puzzle_id,
        model,
        turn_id,
        turn_index,
        reasoning_trace,
        visible_response,
        reasoning_extraction_method,
        guess_status,
        parse_success,
        words_json,
    ) in cursor:

        if (
            args.max_turns is not None
            and processed_turns
            >= args.max_turns
        ):
            break

        key = (
            run_id,
            turn_id,
        )

        if key in done:
            continue

        words = parse_puzzle_words(
            words_json
        )

        if not words:

            print(
                "WARNING: invalid words_json:",
                puzzle_id,
                run_id,
                turn_id,
                flush=True,
            )

            continue

        text_source, text = (
            choose_turn_text(
                reasoning_trace,
                visible_response,
            )
        )

        if text_source == "reasoning":

            reasoning_turns += 1

        elif text_source == "visible_response":

            visible_turns += 1

        else:

            empty_turns += 1

        candidates = tier1_candidates(
            text,
            words,
            min_size=args.min_size,
        )

        # ---------------------------------------------------------------------
        # Raw Tier-1 candidate rows
        # ---------------------------------------------------------------------

        candidate_rows = []

        for candidate_index, candidate in enumerate(
            candidates
        ):

            candidate_rows.append(
                (
                    run_id,
                    puzzle_id,
                    turn_id,
                    turn_index,
                    model,
                    candidate_index,
                    candidate[
                        "span_index"
                    ],
                    candidate[
                        "span"
                    ],
                    json.dumps(
                        candidate[
                            "wordset"
                        ]
                    ),
                    text_source,
                )
            )

        if candidate_rows:

            out.executemany(
                """
                INSERT OR REPLACE
                INTO tier1_candidates

                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                candidate_rows,
            )

        # ---------------------------------------------------------------------
        # Coverage/status row
        # ---------------------------------------------------------------------

        out.execute(
            """
            INSERT OR REPLACE
            INTO tier1_turns

            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                run_id,
                puzzle_id,
                turn_id,
                turn_index,
                model,
                text_source,
                reasoning_extraction_method,
                len(text),
                len(candidates),
                guess_status,
                parse_success,
            ),
        )

        # ---------------------------------------------------------------------
        # Mark processed even when:
        #
        #     candidate_count == 0
        #
        # This is essential for proper resume semantics.
        # ---------------------------------------------------------------------

        out.execute(
            """
            INSERT OR REPLACE
            INTO done_turns

            VALUES (?, ?)
            """,
            (
                run_id,
                turn_id,
            ),
        )

        out.commit()

        processed_turns += 1
        extracted_candidates += len(
            candidates
        )

        if (
            processed_turns % 100
            == 0
        ):

            elapsed = max(
                time.time()
                - started_at,
                1e-9,
            )

            print(
                f"{processed_turns} turns | "
                f"{extracted_candidates} candidates | "
                f"{processed_turns / elapsed:.1f} turns/s | "
                f"reasoning={reasoning_turns} | "
                f"visible={visible_turns} | "
                f"empty={empty_turns}",
                flush=True,
            )

    elapsed = max(
        time.time()
        - started_at,
        1e-9,
    )

    print()
    print(
        "DONE"
    )
    print(
        f"processed turns:      {processed_turns}"
    )
    print(
        f"candidates extracted: {extracted_candidates}"
    )
    print(
        f"reasoning turns:      {reasoning_turns}"
    )
    print(
        f"visible turns:        {visible_turns}"
    )
    print(
        f"empty turns:          {empty_turns}"
    )
    print(
        f"elapsed:              {elapsed:.1f}s"
    )
    print(
        f"rate:                 "
        f"{processed_turns / elapsed:.1f} turns/s"
    )

    src.close()
    out.close()


# =============================================================================
# 10. CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src",
        required=True,
        help=(
            "Raw multiturn Connections SQLite DB"
        ),
    )

    parser.add_argument(
        "--out",
        required=True,
        help=(
            "Tier-1 output SQLite DB"
        ),
    )

    parser.add_argument(
        "--min-size",
        type=int,
        default=2,
        help=(
            "Minimum number of puzzle words "
            "that must appear in a span."
        ),
    )

    parser.add_argument(
        "--model-filter",
        default=None,
    )

    parser.add_argument(
        "--run-id",
        default=None,
    )

    parser.add_argument(
        "--puzzle-id",
        default=None,
    )

    parser.add_argument(
        "--turn-index",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help=(
            "Smoke test: process at most N new turns."
        ),
    )

    return parser.parse_args()


def main():

    run(
        parse_args()
    )


if __name__ == "__main__":
    main()
