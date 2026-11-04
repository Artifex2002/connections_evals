#!/usr/bin/env python3

"""
tier2_scale_hf_multiturn.py
===========================

Tier-2 stance classification for multiturn Connections candidate spans.

INPUT
-----
tier1_candidates

OUTPUT LABELS
-------------
propose
refine_add
refine_pivot
reject
commit
enumerate
mention

Tier 2 ONLY classifies stance.

It does NOT:
    - re-run Tier 1
    - extract puzzle words
    - modify word sets
    - reconstruct hypotheses
    - union refine_add candidates
    - compute F1
    - calculate trajectory metrics

Each turn is handled independently.

V5 improvements
---------------
1. Uses Qwen's chat template.
2. Distinguishes historical recall from active current hypotheses.
3. Gives the classifier deterministic set-relation information:
       same_set
       overlap
       added words
       removed words
4. Tightens commit vs propose:
       tentative belief != commitment
5. Tightens refine_add vs refine_pivot.
6. Applies one structural post-check:
       same exact candidate set cannot be refine_pivot.
7. Supports RANDOM turn sampling for broader validation before full scale.

Resume key:
    (run_id, turn_id)
"""

import argparse
import json
import random
import re
import shutil
import sqlite3
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from connections_research.extraction.candidates import LABELS

import hashlib
# =============================================================================
# 1. RUBRIC
# =============================================================================

RUBRIC = r"""
You are classifying ONE span from a language model solving a
NYT Connections puzzle.

You are NOT solving the puzzle.

You are ONLY identifying the model's CURRENT STANCE toward the puzzle
words mentioned in the span.

Output EXACTLY ONE label:

propose
refine_add
refine_pivot
reject
commit
enumerate
mention


======================================================================
CORE PRINCIPLE
======================================================================

Classify what the model is DOING NOW.

Historical context may appear in the sentence, but historical wording
does not automatically make a span "mention".

Example:

    "My previous guess A B C D was one-away."

This only reports history.

    -> mention


But:

    "My previous guess A B C D was one-away,
     so maybe A B C E is correct."

This uses history to generate a new current hypothesis.

    -> refine_pivot


======================================================================
LABELS
======================================================================


propose
-------

The model CURRENTLY entertains, suggests, develops, or argues for a
candidate grouping.

Examples:

    "Maybe A, B, C, D form a group."

    "A, B, C, and D all seem related to weather."

    "Perhaps the correct group is A, B, C, D."

    "I think these four fit together."

Tentative language such as:

    maybe
    perhaps
    could
    might
    seems
    likely

normally indicates PROPOSE, not COMMIT.

Explaining why a candidate fits is also normally PROPOSE.

IMPORTANT:

    "So maybe the correct group is A B C D"

is PROPOSE.

The word "so" does not imply commitment.


refine_add
----------

The model continues the SAME underlying candidate hypothesis while
locally adding, dropping, incorporating, or narrowing members.

Examples:

    Previous candidate:
        A B C

    Current span:
        "D also fits with these."

    -> refine_add


    Previous candidate:
        A B C D E

    Current span:
        "E is probably the odd one out; A B C D are the core."

    -> refine_add


Typical semantic behavior:

    preserve the existing hypothesis
    +
    make a local membership modification


refine_pivot
------------

The model CURRENTLY switches to a different candidate, explicitly
substitutes members, or moves to an alternative overlapping hypothesis.

Examples:

    "Replace D with E."

    "Maybe A B C E instead."

    "Alternatively, what about W X Y Z?"

    "Let's try a different group."

    "If I swap FACILITY for COMPLEX..."

IMPORTANT:

A repeated statement of the EXACT SAME candidate is NOT a pivot.

Example:

    Previous candidate:
        A B C D

    Current span:
        "Maybe A B C D are correct."

This is reconsideration/persistence, normally:

    -> propose

NOT refine_pivot.


reject
------

The model CURRENTLY dismisses a grouping or connection.

Examples:

    "That doesn't fit."

    "This seems forced."

    "Probably not."

    "The connection is too weak."

    "A and B fit, but C and D really don't."


commit
------

The model CURRENTLY makes a decision to submit, choose, lock, try,
or proceed with a candidate.

Strong commitment examples:

    "My guess is A B C D."

    "I'll submit A B C D."

    "I'll go with A B C D."

    "My next guess will be A B C D."

    "I'll proceed with this group."

    "I think I'll try A B C D."

    "Final answer: A B C D."


IMPORTANT:

Tentative belief is NOT commitment.

    "Maybe A B C D are correct."

        -> propose

    "Perhaps the group is A B C D."

        -> propose

    "A B C D would form the group."

        -> propose

The model must actually express a CURRENT DECISION or intended action.


enumerate
---------

The span primarily lists puzzle state rather than advancing one candidate.

Examples:

    "Available words are..."

    "Remaining words..."

    "That leaves..."

    "Words still on the board..."

    "The group must contain three of A B C D and one other word."

Enumeration includes constraint summaries that do not yet specify a
concrete new candidate group.


mention
-------

The span ONLY reports historical/contextual information and makes no
new current grouping decision.

Examples:

    "My previous guess was A B C D."

    "A B C D was one-away."

    "In Attempt 1 I tried A B C D."

    '["A","B","C","D"] with category "ACTION" resulting in one-away.'

    "A B C D are already solved."

    "Those words were removed."

    "A, B, and C were correct and D remained."

    "Earlier I considered A B C D."

Game-state descriptions about already submitted/removed/solved words
are MENTION unless they actively construct a new candidate.


======================================================================
STRUCTURAL INFORMATION
======================================================================

You will also receive deterministic information comparing the CURRENT
candidate word set with the immediately PREVIOUS Tier-1 candidate.

Use it as supporting evidence.

SAME_SET = YES
means the exact same puzzle-word set is being discussed again.

Such a span cannot represent a membership pivot.

SAME_SET = NO
means membership differs.

ADDED and REMOVED show literal set differences.

These features DO NOT determine the label by themselves.

For example:

    same set + "previous guess was..." -> mention
    same set + "maybe these are right" -> propose
    same set + "I'll submit these"     -> commit

A changed set may be:
    refine_add
    refine_pivot
    propose
    reject
    etc.

Interpret the sentence's CURRENT ACTION.
"""


# =============================================================================
# 2. FEW-SHOT EXAMPLES
# =============================================================================

FEWSHOT = r"""
======================================================================
EXAMPLES
======================================================================

EXAMPLE 1

PREVIOUS CANDIDATE:
(none)

CURRENT WORD SET:
RAIN, SLEET, HAIL, SNOW

SET RELATION:
No previous candidate.

SPAN:
RAIN, SLEET, HAIL, and SNOW all seem like forms of precipitation.

LABEL:
propose


EXAMPLE 2

PREVIOUS CANDIDATE:
BLOCK, COMPOUND, DEVELOPMENT, FACILITY

CURRENT WORD SET:
BLOCK, COMPOUND, DEVELOPMENT, FACILITY

SET RELATION:
SAME_SET: YES
OVERLAP: 4
ADDED: (none)
REMOVED: (none)

SPAN:
In my first attempt I tried BLOCK, COMPOUND, DEVELOPMENT, and FACILITY,
and it was one-away.

LABEL:
mention


EXAMPLE 3

PREVIOUS CANDIDATE:
BLOCK, COMPOUND, DEVELOPMENT, FACILITY

CURRENT WORD SET:
BLOCK, COMPOUND, DEVELOPMENT, FACILITY

SET RELATION:
SAME_SET: YES
OVERLAP: 4
ADDED: (none)
REMOVED: (none)

SPAN:
Maybe BLOCK, COMPOUND, DEVELOPMENT, and FACILITY are actually correct.

LABEL:
propose


EXAMPLE 4

PREVIOUS CANDIDATE:
BLOCK, COMPOUND, DEVELOPMENT, FACILITY

CURRENT WORD SET:
BLOCK, COMPOUND, COMPLEX, FACILITY

SET RELATION:
SAME_SET: NO
OVERLAP: 3
ADDED: COMPLEX
REMOVED: DEVELOPMENT

SPAN:
If I replace DEVELOPMENT with COMPLEX, maybe this group works.

LABEL:
refine_pivot


EXAMPLE 5

PREVIOUS CANDIDATE:
COMPOUND, COMPLEX, DEVELOPMENT

CURRENT WORD SET:
COMPOUND, COMPLEX, DEVELOPMENT, FACILITY

SET RELATION:
SAME_SET: NO
OVERLAP: 3
ADDED: FACILITY
REMOVED: (none)

SPAN:
FACILITY also seems to fit with those three.

LABEL:
refine_add


EXAMPLE 6

PREVIOUS CANDIDATE:
COMPOUND, COMPLEX, DEVELOPMENT, FACILITY, BLOCK

CURRENT WORD SET:
COMPOUND, COMPLEX, DEVELOPMENT, FACILITY

SET RELATION:
SAME_SET: NO
OVERLAP: 4
ADDED: (none)
REMOVED: BLOCK

SPAN:
BLOCK seems like the odd one out, so I'd keep the other four together.

LABEL:
refine_add


EXAMPLE 7

PREVIOUS CANDIDATE:
FLAIR, GIFT, KNACK, NEAT

CURRENT WORD SET:
FLAIR, GIFT, KNACK, NEAT

SET RELATION:
SAME_SET: YES
OVERLAP: 4
ADDED: (none)
REMOVED: (none)

SPAN:
So maybe FLAIR, GIFT, KNACK, and NEAT are the correct group.

LABEL:
propose


EXAMPLE 8

PREVIOUS CANDIDATE:
FLAIR, GIFT, KNACK, NEAT

CURRENT WORD SET:
FLAIR, GIFT, KNACK, NEAT

SET RELATION:
SAME_SET: YES
OVERLAP: 4
ADDED: (none)
REMOVED: (none)

SPAN:
I'll submit FLAIR, GIFT, KNACK, and NEAT.

LABEL:
commit


EXAMPLE 9

PREVIOUS CANDIDATE:
INSULT, INSPIRE, PROMPT, GENERATE

CURRENT WORD SET:
INSULT, INSPIRE, PROMPT, GENERATE

SET RELATION:
SAME_SET: YES
OVERLAP: 4
ADDED: (none)
REMOVED: (none)

SPAN:
["INSULT", "INSPIRE", "PROMPT", "GENERATE"] with category "ACTION"
resulting in one-away.

LABEL:
mention


EXAMPLE 10

PREVIOUS CANDIDATE:
INSULT, INSPIRE, PROMPT, GENERATE

CURRENT WORD SET:
INSULT, INSPIRE, PROMPT, PROVOKE

SET RELATION:
SAME_SET: NO
OVERLAP: 3
ADDED: PROVOKE
REMOVED: GENERATE

SPAN:
Since the previous guess was one-away, maybe PROVOKE belongs instead
of GENERATE.

LABEL:
refine_pivot


EXAMPLE 11

PREVIOUS CANDIDATE:
(none)

CURRENT WORD SET:
COMPOUND, COMPLEX

SET RELATION:
No previous candidate.

SPAN:
COMPOUND and COMPLEX might both be chemistry terms, but that feels
too weak and I don't see enough others.

LABEL:
reject


EXAMPLE 12

PREVIOUS CANDIDATE:
(none)

CURRENT WORD SET:
FLAIR, GIFT, BLOCK, FACILITY, NIFTY, SUPER, COMPLEX

SET RELATION:
No previous candidate.

SPAN:
The remaining words are FLAIR, GIFT, BLOCK, FACILITY, NIFTY, SUPER,
and COMPLEX.

LABEL:
enumerate


EXAMPLE 13

PREVIOUS CANDIDATE:
BLOCK, FACILITY, DEVELOPMENT, COMPOUND

CURRENT WORD SET:
DEVELOPMENT, FACILITY

SET RELATION:
SAME_SET: NO
OVERLAP: 2
ADDED: (none)
REMOVED: BLOCK, COMPOUND

SPAN:
The first guess was one-away, so maybe DEVELOPMENT and FACILITY are
part of the correct group with two others.

LABEL:
propose


EXAMPLE 14

PREVIOUS CANDIDATE:
INSULT, INSPIRE, PROMPT, GENERATE

CURRENT WORD SET:
INSULT, INSPIRE, PROMPT, GENERATE

SET RELATION:
SAME_SET: YES
OVERLAP: 4
ADDED: (none)
REMOVED: (none)

SPAN:
So the three correct words INSULT, INSPIRE, and PROMPT were removed,
while GENERATE remains.

LABEL:
mention
"""


# =============================================================================
# 3. SET RELATION
# =============================================================================

def set_relation(current, previous):
    """
    Deterministically describe membership changes.

    This is NOT itself the stance classifier.
    It gives Tier 2 structural context.
    """

    if not previous:
        return {
            "has_previous": False,
            "same_set": False,
            "overlap": 0,
            "added": list(current),
            "removed": [],
        }

    current_set = set(current)
    previous_set = set(previous)

    return {
        "has_previous": True,
        "same_set": current_set == previous_set,
        "overlap": len(
            current_set & previous_set
        ),
        "added": [
            w
            for w in current
            if w not in previous_set
        ],
        "removed": [
            w
            for w in previous
            if w not in current_set
        ],
    }


def format_relation(current, previous):

    rel = set_relation(
        current,
        previous,
    )

    if not rel["has_previous"]:
        return "No previous candidate."

    added = (
        ", ".join(rel["added"])
        if rel["added"]
        else "(none)"
    )

    removed = (
        ", ".join(rel["removed"])
        if rel["removed"]
        else "(none)"
    )

    return (
        f"SAME_SET: "
        f"{'YES' if rel['same_set'] else 'NO'}\n"
        f"OVERLAP: {rel['overlap']}\n"
        f"ADDED: {added}\n"
        f"REMOVED: {removed}"
    )


# =============================================================================
# 4. PROMPT
# =============================================================================

def build_tier2_prompt(
    line,
    wordset,
    prev,
):

    prev_text = (
        ", ".join(prev)
        if prev
        else "(none)"
    )

    current_text = ", ".join(
        wordset
    )

    relation_text = (
        format_relation(
            wordset,
            prev,
        )
    )

    return (
        RUBRIC
        + "\n"
        + FEWSHOT
        + "\n\n"
        + "======================================================================\n"
        + "CLASSIFY THIS NEW SPAN\n"
        + "======================================================================\n\n"
        + f"PREVIOUS CANDIDATE:\n{prev_text}\n\n"
        + f"CURRENT WORD SET:\n{current_text}\n\n"
        + f"SET RELATION:\n{relation_text}\n\n"
        + f"SPAN:\n{line[:700]}\n\n"
        + "OUTPUT EXACTLY ONE LABEL AND NOTHING ELSE."
    )


# =============================================================================
# 5. LABEL NORMALIZATION
# =============================================================================

def normalize_label(text):

    raw = (
        text or ""
    ).strip()

    if not raw:
        return "propose"

    normalized = (
        raw
        .lower()
        .replace("-", "_")
        .replace("`", "")
        .strip()
    )

    lines = [
        line.strip()
        for line in normalized.splitlines()
        if line.strip()
    ]

    if lines:

        first = re.sub(
            r"[^a-z_]",
            "",
            lines[0],
        )

        if first in LABELS:
            return first

        for label in sorted(
            LABELS,
            key=len,
            reverse=True,
        ):
            if lines[0].startswith(label):
                return label

    for label in sorted(
        LABELS,
        key=len,
        reverse=True,
    ):

        pattern = (
            r"(?<![a-z_])"
            + re.escape(label)
            + r"(?![a-z_])"
        )

        if re.search(
            pattern,
            normalized,
        ):
            return label

    fallback = {
        "pivot": "refine_pivot",
        "add": "refine_add",
        "reject": "reject",
        "commit": "commit",
        "enumerate": "enumerate",
        "mention": "mention",
        "propose": "propose",
    }

    for token, label in (
        fallback.items()
    ):

        if re.search(
            r"\b"
            + re.escape(token)
            + r"\b",
            normalized,
        ):
            return label

    return "propose"


def structural_postcheck(
    label,
    current,
    previous,
):
    """
    Minimal ontology-level correction.

    We deliberately avoid keyword-based hard rules here.

    The only enforced rule is:

        exact same word-set cannot be a membership pivot.

    The classifier still decides between:
        propose
        mention
        commit
        reject
        enumerate
        refine_add
        etc.
    """

    if not previous:
        return label

    rel = set_relation(
        current,
        previous,
    )

    if (
        label == "refine_pivot"
        and rel["same_set"]
    ):
        return "propose"

    return label


# =============================================================================
# 6. HF CLASSIFIER
# =============================================================================

class HFClassifier:

    def __init__(
        self,
        model_name,
        seed=7,
        temperature=0.0,
        max_new_tokens=6,
    ):

        self.model_name = (
            model_name
        )

        self.seed = seed
        self.temperature = (
            temperature
        )

        self.max_new_tokens = (
            max_new_tokens
        )

        random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                seed
            )

        print(
            f"Loading Tier-2 model: "
            f"{model_name}",
            flush=True,
        )

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                model_name,
                local_files_only=True,
            )
        )

        self.tokenizer.padding_side = (
            "left"
        )

        if (
            self.tokenizer.pad_token_id
            is None
        ):
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available()
            else torch.float32
        )

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(
                model_name,
                torch_dtype=dtype,
                local_files_only=True,
            )
        )

        if torch.cuda.is_available():

            self.model = (
                self.model.to(
                    "cuda"
                )
            )

        self.model.eval()

        print(
            "Tier-2 model loaded on:",
            next(
                self.model.parameters()
            ).device,
            flush=True,
        )

    def render_chat_prompt(
        self,
        prompt,
    ):

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise classification model. "
                    "Follow the supplied rubric exactly. "
                    "Return exactly one allowed label "
                    "and nothing else."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        return (
            self.tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    @torch.inference_mode()
    def classify_batch(
        self,
        prompts,
    ):

        if not prompts:
            return []

        rendered = [
            self.render_chat_prompt(
                prompt
            )
            for prompt
            in prompts
        ]

        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
        )

        device = next(
            self.model.parameters()
        ).device

        inputs = {
            key: value.to(device)
            for key, value
            in inputs.items()
        }

        do_sample = (
            self.temperature > 0.0
        )

        generation_kwargs = {
            "max_new_tokens":
                self.max_new_tokens,

            "do_sample":
                do_sample,

            "pad_token_id":
                self.tokenizer.pad_token_id,

            "eos_token_id":
                self.tokenizer.eos_token_id,
        }

        if do_sample:
            generation_kwargs[
                "temperature"
            ] = self.temperature

        outputs = (
            self.model.generate(
                **inputs,
                **generation_kwargs,
            )
        )

        input_len = (
            inputs["input_ids"]
            .shape[1]
        )

        results = []

        for i in range(
            len(prompts)
        ):

            new_tokens = outputs[i][
                input_len:
            ]

            raw = (
                self.tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                )
            )

            label = normalize_label(
                raw
            )

            results.append(
                (
                    label,
                    raw,
                )
            )

        return results


# =============================================================================
# 7. STUB
# =============================================================================

def stub_classify(
    prompt,
    *_,
    **__,
):

    return (
        "propose",
        "stub",
    )


# =============================================================================
# 8. OUTPUT DB
# =============================================================================

def ensure_out(conn):

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tier2_labels(

            run_id TEXT,
            puzzle_id TEXT,
            turn_id TEXT,
            turn_index INTEGER,
            model TEXT,

            candidate_index INTEGER,

            text_source TEXT,

            label TEXT,
            raw_response TEXT,

            PRIMARY KEY(
                run_id,
                turn_id,
                candidate_index
            )
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tier2_done(

            run_id TEXT,
            turn_id TEXT,

            PRIMARY KEY(
                run_id,
                turn_id
            )
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tier2_run_meta(

            started_at TEXT,
            hf_model TEXT,
            seed INTEGER,
            temperature REAL,
            prompt_version TEXT,
            tier1_db TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_tier2_turn
        ON tier2_labels(
            run_id,
            turn_index
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_tier2_puzzle
        ON tier2_labels(
            puzzle_id,
            turn_index
        )
        """
    )

    conn.commit()


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
            FROM tier2_done
            """
        )
    }


# =============================================================================
# 9. TIER-1 DATA
# =============================================================================

def tier1_has_text_source(
    src,
):

    columns = {
        row[1]
        for row
        in src.execute(
            """
            PRAGMA table_info(
                tier1_candidates
            )
            """
        )
    }

    return (
        "text_source"
        in columns
    )


def load_turn_queue(
    src,
    done,
):

    rows = src.execute(
        """
        SELECT DISTINCT
            run_id,
            puzzle_id,
            turn_id,
            turn_index,
            model

        FROM tier1_candidates

        ORDER BY
            run_id,
            turn_index
        """
    ).fetchall()

    return [
        row
        for row in rows
        if (
            row[0],
            row[2],
        )
        not in done
    ]


def load_candidates(
    src,
    run_id,
    turn_id,
    has_text_source,
):

    if has_text_source:

        query = """
            SELECT
                candidate_index,
                span_text,
                wordset_json,
                text_source

            FROM tier1_candidates

            WHERE
                run_id = ?
                AND turn_id = ?

            ORDER BY
                candidate_index
        """

    else:

        query = """
            SELECT
                candidate_index,
                span_text,
                wordset_json,
                'unknown' AS text_source

            FROM tier1_candidates

            WHERE
                run_id = ?
                AND turn_id = ?

            ORDER BY
                candidate_index
        """

    rows = src.execute(
        query,
        (
            run_id,
            turn_id,
        ),
    ).fetchall()

    return [
        (
            candidate_index,
            span_text,
            json.loads(
                wordset_json
            ),
            text_source,
        )
        for (
            candidate_index,
            span_text,
            wordset_json,
            text_source,
        )
        in rows
    ]


# =============================================================================
# 10. PROGRESS
# =============================================================================

def format_bar(
    done,
    total,
    width=22,
):

    if total <= 0:

        return (
            "["
            + ("-" * width)
            + "]"
        )

    filled = int(
        width
        * done
        / total
    )

    return (
        "["
        + ("#" * filled)
        + (
            "-"
            * (
                width
                - filled
            )
        )
        + "]"
    )


def eta_str(
    started_at,
    done,
    total,
):

    if done <= 0:
        return "ETA --:--:--"

    elapsed = (
        time.time()
        - started_at
    )

    rate = (
        done / elapsed
        if elapsed > 0
        else 0.0
    )

    if rate <= 0:
        return "ETA --:--:--"

    remain = (
        max(
            total - done,
            0,
        )
        / rate
    )

    h, rem = divmod(
        int(remain),
        3600,
    )

    m, s = divmod(
        rem,
        60,
    )

    return (
        f"ETA "
        f"{h:02d}:"
        f"{m:02d}:"
        f"{s:02d}"
    )


def render_progress(
    started_at,
    done,
    total,
    model,
    n_labels,
):

    cols = (
        shutil
        .get_terminal_size(
            (120, 20)
        )
        .columns
    )

    pct = (
        100.0
        * done
        / max(
            total,
            1,
        )
    )

    rate = (
        done
        / max(
            time.time()
            - started_at,
            1e-9,
        )
    )

    msg = (
        f"{done}/{total} "
        f"{format_bar(done, total)} "
        f"{pct:5.1f}% | "
        f"{model} | "
        f"labels {n_labels} | "
        f"{rate:5.1f} turns/s | "
        f"{eta_str(started_at, done, total)}"
    )

    if len(msg) > (
        cols - 1
    ):
        msg = msg[
            : cols - 2
        ]

    print(
        "\r"
        + msg.ljust(
            cols - 1
        ),
        end="",
        flush=True,
    )


# =============================================================================
# 11. MAIN
# =============================================================================

def run(args):

    src = sqlite3.connect(
        f"file:{args.tier1_db}"
        "?mode=ro",
        uri=True,
    )

    out = sqlite3.connect(
        args.out
    )

    out.execute(
        "PRAGMA journal_mode=WAL;"
    )

    ensure_out(out)

    out.execute(
        """
        INSERT INTO tier2_run_meta
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            args.model,
            args.seed,
            args.temperature,
            args.prompt_version,
            args.tier1_db,
        ),
    )

    out.commit()

    done = load_done(
        out
    )

    has_text_source = (
        tier1_has_text_source(
            src
        )
    )

    print(
        "Tier-1 text_source column:",
        has_text_source,
        flush=True,
    )

    queue = load_turn_queue(
        src,
        done,
    )

    # -------------------------------------------------------------------------
    # FILTERS
    # -------------------------------------------------------------------------

    if args.model_filter:

        queue = [
            row
            for row in queue
            if (
                row[4]
                == args.model_filter
            )
        ]

    if args.run_id:

        queue = [
            row
            for row in queue
            if (
                row[0]
                == args.run_id
            )
        ]

    # -------------------------------------------------------------------------
    # DETERMINISTIC SHARDING
    #
    # A stable SHA-1 hash of (run_id, turn_id) assigns each whole turn
    # to exactly one shard. Python's built-in hash() is deliberately NOT
    # used because it is randomized across processes.
    # -------------------------------------------------------------------------

    if args.num_shards < 1:
        raise ValueError(
            "--num-shards must be >= 1"
        )

    if not (
        0 <= args.shard_index < args.num_shards
    ):
        raise ValueError(
            "--shard-index must satisfy "
            "0 <= shard-index < num-shards"
        )

    if args.num_shards > 1:

        before_sharding = len(queue)

        def belongs_to_shard(row):

            run_id = row[0]
            turn_id = row[2]

            key = (
                f"{run_id}:{turn_id}"
                .encode("utf-8")
            )

            digest = hashlib.sha1(
                key
            ).digest()

            value = int.from_bytes(
                digest[:8],
                byteorder="big",
                signed=False,
            )

            return (
                value % args.num_shards
                == args.shard_index
            )

        queue = [
            row
            for row in queue
            if belongs_to_shard(row)
        ]

        print(
            f"Shard "
            f"{args.shard_index}/"
            f"{args.num_shards}: "
            f"{len(queue)} of "
            f"{before_sharding} eligible turns",
            flush=True,
        )

    if args.puzzle_id:

        queue = [
            row
            for row in queue
            if (
                row[1]
                == args.puzzle_id
            )
        ]

    if (
        args.turn_index
        is not None
    ):

        queue = [
            row
            for row in queue
            if (
                row[3]
                == args.turn_index
            )
        ]

    # -------------------------------------------------------------------------
    # RANDOM SAMPLE MODE
    #
    # Useful for validation across many puzzles/models rather than always
    # inspecting the lexicographically first turns.
    # -------------------------------------------------------------------------

    if (
        args.sample_turns
        is not None
    ):

        rng = random.Random(
            args.sample_seed
        )

        sample_n = min(
            args.sample_turns,
            len(queue),
        )

        queue = rng.sample(
            queue,
            sample_n,
        )

    elif (
        args.limit
        is not None
    ):

        queue = queue[
            : args.limit
        ]

    # -------------------------------------------------------------------------
    # CLASSIFIER
    # -------------------------------------------------------------------------

    if args.classifier == "stub":

        classify_batch = (
            lambda prompts: [
                stub_classify(
                    prompt
                )
                for prompt in prompts
            ]
        )

    else:

        hf = HFClassifier(
            model_name=args.model,
            seed=args.seed,
            temperature=args.temperature,
            max_new_tokens=(
                args.max_new_tokens
            ),
        )

        classify_batch = (
            hf.classify_batch
        )

    started_at = (
        time.time()
    )

    total_turns = len(
        queue
    )

    n_turns = 0
    n_labels = 0

    # -------------------------------------------------------------------------
    # PROCESS
    # -------------------------------------------------------------------------

    for (
        run_id,
        puzzle_id,
        turn_id,
        turn_index,
        trace_model,
    ) in queue:

        candidates = (
            load_candidates(
                src,
                run_id,
                turn_id,
                has_text_source,
            )
        )

        prompts = []
        meta = []

        for i, (
            candidate_index,
            span_text,
            wordset,
            text_source,
        ) in enumerate(
            candidates
        ):

            prev = (
                candidates[
                    i - 1
                ][2]
                if i > 0
                else None
            )

            prompts.append(
                build_tier2_prompt(
                    span_text,
                    wordset,
                    prev,
                )
            )

            meta.append(
                (
                    candidate_index,
                    text_source,
                    wordset,
                    prev,
                )
            )

        rows = []

        for start in range(
            0,
            len(prompts),
            args.span_batch,
        ):

            batch_prompts = (
                prompts[
                    start:
                    start
                    + args.span_batch
                ]
            )

            batch_meta = (
                meta[
                    start:
                    start
                    + args.span_batch
                ]
            )

            batch_results = (
                classify_batch(
                    batch_prompts
                )
            )

            for (
                candidate_index,
                text_source,
                wordset,
                prev,
            ), (
                label,
                raw,
            ) in zip(
                batch_meta,
                batch_results,
            ):

                final_label = (
                    structural_postcheck(
                        label,
                        wordset,
                        prev,
                    )
                )

                rows.append(
                    (
                        run_id,
                        puzzle_id,
                        turn_id,
                        turn_index,
                        trace_model,
                        candidate_index,
                        text_source,
                        final_label,
                        raw,
                    )
                )

        if rows:

            out.executemany(
                """
                INSERT OR REPLACE
                INTO tier2_labels
                VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?
                )
                """,
                rows,
            )

        out.execute(
            """
            INSERT OR REPLACE
            INTO tier2_done
            VALUES (?, ?)
            """,
            (
                run_id,
                turn_id,
            ),
        )

        out.commit()

        n_turns += 1
        n_labels += len(
            rows
        )

        render_progress(
            started_at,
            n_turns,
            total_turns,
            trace_model,
            n_labels,
        )

    if total_turns:
        print()

    elapsed = (
        time.time()
        - started_at
    )

    print(
        f"DONE: "
        f"{n_turns} turns, "
        f"{n_labels} labels "
        f"in {elapsed:.1f}s"
    )

    src.close()
    out.close()


# =============================================================================
# 12. CLI
# =============================================================================

def parse_args():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--tier1-db",
        required=True,
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    parser.add_argument(
        "--classifier",
        choices=[
            "hf",
            "stub",
        ],
        default="hf",
    )

    parser.add_argument(
        "--model",
        default=(
            "/data/user_data/"
            "apanda2/.hf_cache/hub/"
            "models--Qwen--"
            "Qwen2.5-7B-Instruct/"
            "snapshots/"
            "a09a35458c702b33eeacc393d103063234e8bc28"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--span-batch",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--prompt-version",
        default=(
            "v5_set_relation_generalized"
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
        "--limit",
        type=int,
        default=None,
        help=(
            "Take the first N eligible turns."
        ),
    )

    parser.add_argument(
        "--sample-turns",
        type=int,
        default=None,
        help=(
            "Randomly sample N eligible turns. "
            "Preferred for validation."
        ),
    )

    parser.add_argument(
        "--sample-seed",
        type=int,
        default=20260825,
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of deterministic processing shards.",
    )

    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index.",
    )

    return (
        parser.parse_args()
    )


def main():

    run(
        parse_args()
    )


if __name__ == "__main__":
    main()
