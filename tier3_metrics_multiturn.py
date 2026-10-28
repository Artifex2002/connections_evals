#!/usr/bin/env python3

"""
metrics_multiturn.py
====================

Deterministic offline metrics pass for multiturn NYT Connections.

INPUTS
------
Tier-1 DB:
    tier1_candidates

Tier-2 DB:
    tier2_labels

Raw multiturn DB(s):
    puzzles
    runs
    turns

OUTPUT
------
step_cache
    Every turn boundary, reasoning hypothesis, and submitted action.

turn_metrics
    Per-turn reasoning/action/feedback-boundary quantities.

run_metrics
    Whole-game M1/M2/M3/M4 metrics.

processed_runs
    Resume/incremental-processing metadata.

skipped_runs
    Runs that could not yet be scored, e.g. incomplete Tier-2 coverage.


======================================================================
TEMPORAL SEMANTICS
======================================================================

Within one turn:
    hypotheses are reconstructed and accumulated exactly as in the
    original single-shot pipeline.

Across turns:
    speculative/unsubmitted hypotheses DO NOT persist.

At the beginning of turn t:
    state = only groups explicitly recorded by the game as solved
            before turn t.

This is reconstructed from:

    solved_group_indices_before_json

Therefore hidden reasoning from turn t-1 never leaks into turn t.

Incorrect / one-away / repeated submitted guesses are historical
information in the next prompt, but NOT active belief-state groups.

Correct solved groups persist because they are authoritative game state.


======================================================================
RESUME / INCREMENTAL BEHAVIOR
======================================================================

For every successfully processed run we store:

    raw_turn_count
    tier1_candidate_count
    tier2_label_count

On rerun:

    same counts
        -> skip

    changed counts
        -> delete old metrics for that run and recompute

    new run
        -> process

Runs skipped because Tier-2 is incomplete are NOT marked processed and
will automatically be tried again next time.


======================================================================
METRICS
======================================================================

M4
    accumulated partition pairwise F1 trajectory:
        peak
        final
        regret
        peak location

Also stores:
    hypothesis_pair_f1

which scores EACH current hypothesis against its best matching gold group.

M1
    churn
    correct-group abandonment
    final-group lock-in latency

Cross-turn context resets are NOT counted as churn.

M2
    original blast-duration metric plus interpretable components.

M3
    original dstruct final structural-invalidity diagnostic.
    This remains a diagnostic, not yet the final conceptual drift metric.

ARI
    only computed for a legal 4 x 4 partition of all 16 words.
"""

import argparse
import json
import sqlite3
import time

from collections import Counter, defaultdict
from itertools import combinations
from math import comb


# =============================================================================
# CONSTANTS
# =============================================================================

HELD = {
    "propose",
    "refine_add",
    "refine_pivot",
    "commit",
}

TURN_START_LABEL = "turn_start"
SUBMIT_LABEL = "submit"


# =============================================================================
# BASIC PARSING
# =============================================================================

def load_json(value):

    if value is None:
        return None

    if isinstance(
        value,
        (list, dict),
    ):
        return value

    try:
        return json.loads(
            value
        )

    except Exception:
        return None


def parse_gold_groups(value):
    """
    Current puzzle schema:

    [
        {
            "category": "...",
            "group_index": 0,
            "words": [...]
        },
        ...
    ]

    Also accepts list-of-lists.
    """

    obj = load_json(
        value
    )

    if obj is None:
        return None

    groups = []

    if isinstance(
        obj,
        dict,
    ):

        iterable = list(
            obj.values()
        )

    elif isinstance(
        obj,
        list,
    ):

        iterable = obj

    else:
        return None

    for item in iterable:

        if isinstance(
            item,
            dict,
        ):

            words = item.get(
                "words"
            )

            if not isinstance(
                words,
                list,
            ):
                continue

        elif isinstance(
            item,
            (list, tuple),
        ):

            words = item

        else:
            continue

        group = {
            str(word)
            .strip()
            .upper()

            for word in words

            if str(word).strip()
        }

        groups.append(
            group
        )

    if len(groups) != 4:
        return None

    if any(
        len(group) != 4
        for group in groups
    ):
        return None

    vocabulary = set().union(
        *groups
    )

    if len(vocabulary) != 16:
        return None

    return groups


def parse_guess(value):

    obj = load_json(
        value
    )

    if not isinstance(
        obj,
        list,
    ):
        return None

    words = [
        str(word)
        .strip()
        .upper()

        for word in obj

        if str(word).strip()
    ]

    if len(words) != 4:
        return None

    group = set(
        words
    )

    if len(group) != 4:
        return None

    return group


def parse_solved_indices(value):

    obj = load_json(
        value
    )

    if not isinstance(
        obj,
        list,
    ):
        return []

    result = []

    for item in obj:

        try:
            index = int(
                item
            )

        except Exception:
            continue

        if 0 <= index < 4:
            result.append(
                index
            )

    return sorted(
        set(result)
    )


# =============================================================================
# HYPOTHESIS RECONSTRUCTION
# =============================================================================

def reconstruct_turn(candidates):
    """
    Reconstruct hypotheses WITHIN ONE TURN.

    Previous-candidate context resets each turn, matching the Tier-2 V5
    classifier.
    """

    previous_candidate = set()

    output = []

    for candidate in candidates:

        raw = set(
            candidate[
                "raw"
            ]
        )

        label = (
            candidate[
                "label"
            ]
        )

        if label == "refine_pivot":

            resolved = raw

        elif label == "refine_add":

            # Guard against a clearly erroneous disjoint refine_add.
            if (
                previous_candidate
                and not (
                    raw
                    & previous_candidate
                )
            ):

                resolved = raw

            # Narrowing/subsetting refinement.
            elif (
                previous_candidate
                and raw < previous_candidate
            ):

                resolved = raw

            else:

                resolved = (
                    previous_candidate
                    | raw
                )

        else:

            resolved = raw

        kept = (
            label in HELD
        )

        # Connections hypotheses larger than 4 are not admitted into the
        # accumulated partition.
        well_formed = (
            0 < len(resolved) <= 4
        )

        output.append(
            {
                **candidate,

                "resolved":
                    resolved,

                "kept":
                    kept,

                "well_formed":
                    well_formed,
            }
        )

        # IMPORTANT:
        # Tier-2 saw immediately preceding Tier-1 candidate, not preceding
        # held hypothesis.
        previous_candidate = (
            resolved
        )

    return output


# =============================================================================
# PAIRWISE QUALITY
# =============================================================================

def pairs_for_group(group):

    if len(group) < 2:
        return set()

    return {
        tuple(
            sorted(pair)
        )

        for pair in combinations(
            sorted(group),
            2,
        )
    }


def gold_pairs_for_partition(
    gold,
):

    pairs = set()

    for group in gold:

        pairs |= pairs_for_group(
            group
        )

    return pairs


def pairwise_prf(
    partition,
    gold_pairs,
):

    predicted_pairs = set()

    for group in partition:

        predicted_pairs |= (
            pairs_for_group(
                group
            )
        )

    if not predicted_pairs:

        return (
            0.0,
            0.0,
            0.0,
        )

    intersection = len(
        predicted_pairs
        & gold_pairs
    )

    precision = (
        intersection
        / len(predicted_pairs)
    )

    recall = (
        intersection
        / len(gold_pairs)
    )

    if (
        precision
        + recall
    ):

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    else:

        f1 = 0.0

    return (
        precision,
        recall,
        f1,
    )


def hypothesis_best_gold_score(
    hypothesis,
    gold,
):
    """
    Pairwise F1 between THIS current hypothesis and the best matching
    gold group.
    """

    predicted_pairs = (
        pairs_for_group(
            hypothesis
        )
    )

    if not predicted_pairs:

        return (
            0.0,
            None,
        )

    best_f1 = 0.0
    best_index = None

    for gold_index, gold_group in enumerate(
        gold
    ):

        target_pairs = (
            pairs_for_group(
                gold_group
            )
        )

        intersection = len(
            predicted_pairs
            & target_pairs
        )

        precision = (
            intersection
            / len(predicted_pairs)
        )

        recall = (
            intersection
            / len(target_pairs)
        )

        if (
            precision
            + recall
        ):

            f1 = (
                2
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )

        else:

            f1 = 0.0

        if f1 > best_f1:

            best_f1 = f1
            best_index = (
                gold_index
            )

    return (
        round(
            best_f1,
            4,
        ),

        best_index,
    )


# =============================================================================
# PARTITION QUALITY
# =============================================================================

def exact_hits(
    partition,
    gold,
):

    gold_sets = {
        frozenset(
            group
        )

        for group in gold
    }

    return sum(
        1

        for group in partition

        if (
            frozenset(
                group
            )
            in gold_sets
        )
    )


def adjusted_rand(
    partition,
    gold,
):
    """
    ARI only for a legal complete Connections partition:
        4 groups
        4 words each
        all 16 unique gold words exactly once.
    """

    if len(partition) != 4:
        return None

    if any(
        len(group) != 4
        for group in partition
    ):
        return None

    gold_words = set().union(
        *gold
    )

    predicted_words = set().union(
        *partition
    )

    if (
        predicted_words
        != gold_words
    ):
        return None

    if sum(
        len(group)
        for group in partition
    ) != 16:
        return None

    gold_id = {
        word: group_index

        for group_index, group
        in enumerate(gold)

        for word in group
    }

    predicted_id = {
        word: group_index

        for group_index, group
        in enumerate(partition)

        for word in group
    }

    words = sorted(
        gold_words
    )

    gold_labels = [
        gold_id[
            word
        ]

        for word in words
    ]

    predicted_labels = [
        predicted_id[
            word
        ]

        for word in words
    ]

    n = len(
        words
    )

    contingency_index = sum(
        comb(
            count,
            2,
        )

        for count in Counter(
            zip(
                gold_labels,
                predicted_labels,
            )
        ).values()
    )

    gold_sum = sum(
        comb(
            count,
            2,
        )

        for count in Counter(
            gold_labels
        ).values()
    )

    pred_sum = sum(
        comb(
            count,
            2,
        )

        for count in Counter(
            predicted_labels
        ).values()
    )

    expected = (
        gold_sum
        * pred_sum
        / comb(
            n,
            2,
        )
    )

    maximum = (
        gold_sum
        + pred_sum
    ) / 2

    if maximum == expected:
        return 1.0

    return (
        contingency_index
        - expected
    ) / (
        maximum
        - expected
    )


# =============================================================================
# BELIEF STATE
# =============================================================================

class BeliefState:
    """
    Last-writer-wins accumulated partition WITHIN a turn.

    A fresh instance is created at every turn boundary.

    Authoritative solved groups are inserted before processing that turn.
    """

    def __init__(self):

        self.word_to_slot = {}
        self.slots = []

    def apply_group(
        self,
        group,
    ):

        group = set(
            group
        )

        detached = []

        for word in group:

            old_slot = (
                self.word_to_slot.get(
                    word
                )
            )

            if (
                old_slot is not None
                and word in old_slot
            ):

                old_slot.discard(
                    word
                )

                detached.append(
                    word
                )

        new_slot = set(
            group
        )

        self.slots.append(
            new_slot
        )

        for word in group:

            self.word_to_slot[
                word
            ] = new_slot

        self.slots = [
            slot

            for slot in self.slots

            if slot
        ]

        return detached

    def partition(self):

        return [
            set(slot)

            for slot in self.slots

            if len(slot) >= 2
        ]

    def n_assigned(self):

        return sum(
            len(slot)

            for slot in self.slots
        )

    def max_group_size(self):

        return max(
            (
                len(slot)

                for slot in self.slots
            ),
            default=0,
        )


# =============================================================================
# STEP SCORING
# =============================================================================

def score_state(
    *,
    state,
    gold,
    gold_pairs,
    hypothesis,
):

    partition = (
        state.partition()
    )

    precision, recall, f1 = (
        pairwise_prf(
            partition,
            gold_pairs,
        )
    )

    hypothesis_f1, best_gold_index = (
        hypothesis_best_gold_score(
            hypothesis,
            gold,
        )
    )

    return {
        "partition":
            partition,

        "pair_p":
            round(
                precision,
                4,
            ),

        "pair_r":
            round(
                recall,
                4,
            ),

        "pair_f1":
            round(
                f1,
                4,
            ),

        "hypothesis_pair_f1":
            hypothesis_f1,

        "hypothesis_best_gold_index":
            best_gold_index,

        "exact_hits":
            exact_hits(
                partition,
                gold,
            ),

        "ari":
            adjusted_rand(
                partition,
                gold,
            ),

        "n_assigned":
            state.n_assigned(),

        "n_groups":
            len(
                partition
            ),

        "max_group_size":
            state.max_group_size(),
    }


# =============================================================================
# BUILD MULTITURN TRAJECTORY
# =============================================================================

def build_run_series(
    *,
    raw_turns,
    candidates_by_turn,
    labels_by_turn,
    gold,
):
    """
    Build the complete run trajectory.

    IMPORTANT:
        belief state resets every turn.

    At turn start:
        load ONLY authoritative solved groups recorded before that turn.

    Then:
        reasoning hypotheses accumulate within the turn.

    Finally:
        actual submitted guess becomes an action anchor.

    Next turn:
        speculative state is discarded and rebuilt from solved groups only.
    """

    gold_pairs = (
        gold_pairs_for_partition(
            gold
        )
    )

    vocabulary = set().union(
        *gold
    )

    series = []

    global_step = 0

    for turn in raw_turns:

        turn_id = (
            turn[
                "turn_id"
            ]
        )

        turn_index = (
            turn[
                "turn_index"
            ]
        )

        # ---------------------------------------------------------------------
        # FRESH STATE AT EVERY TURN
        # ---------------------------------------------------------------------

        state = BeliefState()

        solved_before = (
            parse_solved_indices(
                turn[
                    "solved_group_indices_before_json"
                ]
            )
        )

        for group_index in (
            solved_before
        ):

            state.apply_group(
                gold[
                    group_index
                ]
            )

        # ---------------------------------------------------------------------
        # EXPLICIT TURN-BOUNDARY MARKER
        # ---------------------------------------------------------------------

        scored = score_state(
            state=state,
            gold=gold,
            gold_pairs=gold_pairs,
            hypothesis=set(),
        )

        series.append(
            {
                "global_step":
                    global_step,

                "turn_id":
                    turn_id,

                "turn_index":
                    turn_index,

                "candidate_index":
                    None,

                "step_type":
                    "turn_start",

                "text_source":
                    "game_state",

                "label":
                    TURN_START_LABEL,

                "guess_status":
                    None,

                "kept":
                    0,

                "well_formed":
                    1,

                "raw":
                    [],

                "resolved":
                    [],

                "detached":
                    [],

                "solved_before":
                    solved_before,

                **scored,
            }
        )

        global_step += 1

        # ---------------------------------------------------------------------
        # LOAD TIER-1 / TIER-2 FOR THIS TURN
        # ---------------------------------------------------------------------

        tier1_candidates = (
            candidates_by_turn.get(
                turn_id,
                [],
            )
        )

        tier2_labels = (
            labels_by_turn.get(
                turn_id,
                {},
            )
        )

        if tier1_candidates:

            candidate_indexes = {
                candidate[
                    "candidate_index"
                ]

                for candidate
                in tier1_candidates
            }

            label_indexes = set(
                tier2_labels
            )

            if (
                candidate_indexes
                != label_indexes
            ):

                return (
                    None,
                    (
                        "Tier-2 coverage mismatch "
                        f"for turn {turn_id}: "
                        f"{len(candidate_indexes)} candidates "
                        f"vs {len(label_indexes)} labels"
                    ),
                )

        reconstruction_input = []

        for candidate in (
            tier1_candidates
        ):

            candidate_index = (
                candidate[
                    "candidate_index"
                ]
            )

            reconstruction_input.append(
                {
                    "candidate_index":
                        candidate_index,

                    "raw":
                        set(
                            candidate[
                                "raw"
                            ]
                        ),

                    "label":
                        tier2_labels[
                            candidate_index
                        ],

                    "text_source":
                        candidate[
                            "text_source"
                        ],
                }
            )

        reconstructed = (
            reconstruct_turn(
                reconstruction_input
            )
        )

        # ---------------------------------------------------------------------
        # REASONING STEPS
        # ---------------------------------------------------------------------

        for candidate in (
            reconstructed
        ):

            detached = []

            if (
                candidate[
                    "kept"
                ]
                and candidate[
                    "well_formed"
                ]
            ):

                detached = (
                    state.apply_group(
                        candidate[
                            "resolved"
                        ]
                    )
                )

            scored = score_state(
                state=state,
                gold=gold,
                gold_pairs=gold_pairs,
                hypothesis=(
                    candidate[
                        "resolved"
                    ]
                ),
            )

            series.append(
                {
                    "global_step":
                        global_step,

                    "turn_id":
                        turn_id,

                    "turn_index":
                        turn_index,

                    "candidate_index":
                        candidate[
                            "candidate_index"
                        ],

                    "step_type":
                        "reasoning",

                    "text_source":
                        candidate[
                            "text_source"
                        ],

                    "label":
                        candidate[
                            "label"
                        ],

                    "guess_status":
                        None,

                    "kept":
                        int(
                            candidate[
                                "kept"
                            ]
                        ),

                    "well_formed":
                        int(
                            candidate[
                                "well_formed"
                            ]
                        ),

                    "raw":
                        sorted(
                            candidate[
                                "raw"
                            ]
                        ),

                    "resolved":
                        sorted(
                            candidate[
                                "resolved"
                            ]
                        ),

                    "detached":
                        sorted(
                            detached
                        ),

                    "solved_before":
                        solved_before,

                    **scored,
                }
            )

            global_step += 1

        # ---------------------------------------------------------------------
        # ACTUAL SUBMITTED GUESS
        # ---------------------------------------------------------------------

        submitted_group = (
            parse_guess(
                turn[
                    "parsed_guess_json"
                ]
            )
        )

        if (
            turn[
                "parse_success"
            ]
            and submitted_group
            and submitted_group
            <= vocabulary
        ):

            detached = (
                state.apply_group(
                    submitted_group
                )
            )

            scored = score_state(
                state=state,
                gold=gold,
                gold_pairs=gold_pairs,
                hypothesis=(
                    submitted_group
                ),
            )

            series.append(
                {
                    "global_step":
                        global_step,

                    "turn_id":
                        turn_id,

                    "turn_index":
                        turn_index,

                    "candidate_index":
                        None,

                    "step_type":
                        "submit",

                    "text_source":
                        "action",

                    "label":
                        SUBMIT_LABEL,

                    "guess_status":
                        turn[
                            "guess_status"
                        ],

                    "kept":
                        1,

                    "well_formed":
                        1,

                    "raw":
                        sorted(
                            submitted_group
                        ),

                    "resolved":
                        sorted(
                            submitted_group
                        ),

                    "detached":
                        sorted(
                            detached
                        ),

                    "solved_before":
                        solved_before,

                    **scored,
                }
            )

            global_step += 1

    return (
        series,
        None,
    )


# =============================================================================
# STATE HELPERS
# =============================================================================

def word_to_group(
    partition,
):

    mapping = {}

    for group in partition:

        frozen = frozenset(
            group
        )

        if len(frozen) < 2:
            continue

        for word in frozen:

            mapping[
                word
            ] = frozen

    return mapping


def final_partition(
    series,
):

    if not series:
        return []

    return [
        frozenset(
            group
        )

        for group in series[-1][
            "partition"
        ]
    ]


# =============================================================================
# M1
# =============================================================================

def churn_for_series(
    series,
):
    """
    Original churn logic for ONE continuous local trajectory.
    """

    signature_maps = [
        word_to_group(
            step[
                "partition"
            ]
        )

        for step in series
    ]

    if not signature_maps:

        return (
            0,
            0,
        )

    words = set().union(
        *[
            set(
                mapping
            )

            for mapping
            in signature_maps
        ]
    )

    total = 0

    for word in words:

        previous = None

        for mapping in (
            signature_maps
        ):

            current = (
                mapping.get(
                    word
                )
            )

            if current != previous:

                if (
                    previous
                    is not None
                    and (
                        current
                        is None
                        or not (
                            current >= previous
                            or current <= previous
                        )
                    )
                ):

                    total += 1

                previous = current

    transitions = max(
        len(series) - 1,
        0,
    )

    return (
        total,
        transitions,
    )


def m1_run_churn_components(
    series,
):
    """
    Sum churn WITHIN turns.

    Crucially:
        cross-turn reset transitions are NOT counted as model churn.
    """

    by_turn = defaultdict(
        list
    )

    for step in series:

        by_turn[
            step[
                "turn_index"
            ]
        ].append(
            step
        )

    total = 0
    transitions = 0

    for turn_index in sorted(
        by_turn
    ):

        local_total, local_transitions = (
            churn_for_series(
                by_turn[
                    turn_index
                ]
            )
        )

        total += local_total
        transitions += (
            local_transitions
        )

    per_word = (
        total / 16.0
    )

    per_word_step = (
        total
        / (
            16.0
            * transitions
        )

        if transitions > 0

        else 0.0
    )

    return {
        "total":
            total,

        "per_word":
            round(
                per_word,
                4,
            ),

        "per_word_step":
            round(
                per_word_step,
                6,
            ),

        "transitions":
            transitions,
    }


def m1_correct_abandonment(
    series,
    gold,
):
    """
    Did a gold-exact group appear in inferred state at some point but
    disappear from the final inferred state?

    Because turn resets are real context boundaries, losing an
    unsubmitted correct hypothesis across turns counts as abandonment.
    """

    if not series:
        return 0.0

    final = {
        frozenset(
            group
        )

        for group in series[-1][
            "partition"
        ]
    }

    appeared_before = set()

    for step in series[:-1]:

        for group in step[
            "partition"
        ]:

            appeared_before.add(
                frozenset(
                    group
                )
            )

    abandoned = 0

    for gold_group in gold:

        frozen_gold = (
            frozenset(
                gold_group
            )
        )

        if (
            frozen_gold
            in appeared_before
            and frozen_gold
            not in final
        ):

            abandoned += 1

    return round(
        abandoned
        / len(gold),
        4,
    )


def m1_lockin_latencies(
    series,
):
    """
    Stable lock-in of FINAL inferred groups.

    Solved groups can legitimately remain locked across turns because
    they are restored at every turn_start.
    """

    if not series:
        return []

    final_groups = (
        final_partition(
            series
        )
    )

    if not final_groups:
        return []

    if len(series) <= 1:

        return [
            0.0
            for _ in final_groups
        ]

    per_step = [
        {
            frozenset(
                group
            )

            for group in step[
                "partition"
            ]
        }

        for step in series
    ]

    latencies = []

    for final_group in (
        final_groups
    ):

        index = len(
            series
        ) - 1

        lock_index = index

        while (
            index >= 0
            and final_group
            in per_step[
                index
            ]
        ):

            lock_index = index
            index -= 1

        latencies.append(
            round(
                lock_index
                / (
                    len(series)
                    - 1
                ),
                4,
            )
        )

    return latencies


# =============================================================================
# M2
# =============================================================================

def m2_components(
    series,
    gold,
):
    """
    Original blast-duration definition plus interpretable components.

    turn_start states are included because they represent the real
    persistent game state at that moment.
    """

    gold_index = {
        word: group_index

        for group_index, group
        in enumerate(gold)

        for word in group
    }

    blocked_blast_sum = 0

    wrong_word_steps = 0
    blocked_word_steps = 0

    total_blast_on_wrong = 0
    max_blast = 0

    for step in series:

        mapping = word_to_group(
            step[
                "partition"
            ]
        )

        for word, slot in (
            mapping.items()
        ):

            group_index = (
                gold_index.get(
                    word
                )
            )

            if group_index is None:
                continue

            blast = sum(
                1

                for other in slot

                if (
                    gold_index.get(
                        other
                    )
                    != group_index
                )
            )

            if blast == 0:
                continue

            wrong_word_steps += 1

            total_blast_on_wrong += (
                blast
            )

            max_blast = max(
                max_blast,
                blast,
            )

            gold_mates = [
                mate

                for mate in gold[
                    group_index
                ]

                if mate != word
            ]

            blocked = any(
                (
                    mate in mapping
                    and mate not in slot
                )

                for mate in gold_mates
            )

            if blocked:

                blocked_word_steps += 1

                blocked_blast_sum += (
                    blast
                )

    n_steps = len(
        series
    )

    blast_duration = (
        blocked_blast_sum
        / (
            16
            * n_steps
        )

        if n_steps

        else 0.0
    )

    mean_blast = (
        total_blast_on_wrong
        / wrong_word_steps

        if wrong_word_steps

        else 0.0
    )

    return {
        "m2_blast_duration":
            round(
                blast_duration,
                6,
            ),

        "m2_wrong_word_steps":
            wrong_word_steps,

        "m2_blocked_word_steps":
            blocked_word_steps,

        "m2_blocked_blast_sum":
            blocked_blast_sum,

        "m2_mean_blast_radius":
            round(
                mean_blast,
                4,
            ),

        "m2_max_blast_radius":
            max_blast,
    }


# =============================================================================
# M3 STRUCTURAL DIAGNOSTIC
# =============================================================================

def m3_dstruct(
    series,
    gold,
):

    if not series:

        return (
            16,
            16,
            4,
            16,
        )

    final = [
        set(
            group
        )

        for group in series[-1][
            "partition"
        ]
    ]

    assigned = (
        set().union(
            *final
        )

        if final

        else set()
    )

    all_words = set().union(
        *gold
    )

    word_violations = len(
        all_words
        - assigned
    )

    group_violations = abs(
        len(final)
        - 4
    )

    size_violations = sum(
        abs(
            len(group)
            - 4
        )

        for group in final
    )

    total = (
        word_violations
        + group_violations
        + size_violations
    )

    return (
        total,
        word_violations,
        group_violations,
        size_violations,
    )


# =============================================================================
# RUN-LEVEL METRICS
# =============================================================================

def compute_run_metrics(
    *,
    series,
    gold,
    source_run,
):

    if not series:
        return None

    qualities = [
        step[
            "pair_f1"
        ]

        for step in series
    ]

    peak = max(
        qualities
    )

    final = (
        qualities[-1]
    )

    peak_index = (
        qualities.index(
            peak
        )
    )

    if len(
        qualities
    ) <= 1:

        peak_fraction = 0.0

    else:

        peak_fraction = (
            peak_index
            / (
                len(
                    qualities
                )
                - 1
            )
        )

    churn = (
        m1_run_churn_components(
            series
        )
    )

    lock_latencies = (
        m1_lockin_latencies(
            series
        )
    )

    m2 = (
        m2_components(
            series,
            gold,
        )
    )

    (
        m3_total,
        m3_word,
        m3_group,
        m3_size,
    ) = m3_dstruct(
        series,
        gold,
    )

    return {
        "n_steps":
            len(series),

        "n_turn_starts":
            sum(
                step[
                    "step_type"
                ]
                == "turn_start"

                for step in series
            ),

        "n_candidates":
            sum(
                step[
                    "step_type"
                ]
                == "reasoning"

                for step in series
            ),

        "n_kept":
            sum(
                (
                    step[
                        "step_type"
                    ]
                    == "reasoning"
                )
                and step[
                    "kept"
                ]
                and step[
                    "well_formed"
                ]

                for step in series
            ),

        "n_action_anchors":
            sum(
                step[
                    "step_type"
                ]
                == "submit"

                for step in series
            ),

        # M4
        "m4_regret":
            round(
                peak - final,
                4,
            ),

        "m4_peak":
            round(
                peak,
                4,
            ),

        "m4_final":
            round(
                final,
                4,
            ),

        "m4_peak_frac":
            round(
                peak_fraction,
                4,
            ),

        "m4_peak_global_step":
            peak_index,

        "m4_peak_turn_index":
            series[
                peak_index
            ][
                "turn_index"
            ],

        # M1
        "m1_churn_total":
            churn[
                "total"
            ],

        "m1_churn_per_word":
            churn[
                "per_word"
            ],

        "m1_churn_per_word_step":
            churn[
                "per_word_step"
            ],

        "m1_churn_transitions":
            churn[
                "transitions"
            ],

        "m1_correct_abandonment":
            m1_correct_abandonment(
                series,
                gold,
            ),

        "m1_lockin_json":
            json.dumps(
                lock_latencies
            ),

        "m1_lockin_mean":
            (
                round(
                    sum(
                        lock_latencies
                    )
                    / len(
                        lock_latencies
                    ),
                    4,
                )

                if lock_latencies

                else None
            ),

        # M2
        **m2,

        # M3
        "m3_dstruct":
            m3_total,

        "m3_word_violations":
            m3_word,

        "m3_group_violations":
            m3_group,

        "m3_size_violations":
            m3_size,

        # Final inferred state
        "final_exact_groups":
            series[-1][
                "exact_hits"
            ],

        "final_ari":
            series[-1][
                "ari"
            ],

        "trajectory_solved":
            int(
                series[-1][
                    "exact_hits"
                ]
                == 4
            ),

        # Actual game outcome
        "db_solved":
            source_run[
                "solved"
            ],

        "db_binary_score":
            source_run[
                "binary_score"
            ],

        "guesses_used":
            source_run[
                "guesses_used"
            ],

        "mistakes_used":
            source_run[
                "mistakes_used"
            ],
    }


# =============================================================================
# TURN-LEVEL METRICS
# =============================================================================

def compute_turn_metrics(
    *,
    series,
    raw_turns,
):
    """
    q_start:
        quality at explicit turn_start, containing solved groups only.

    q_first:
        quality after first reasoning step.
        If no reasoning exists, first submit anchor; otherwise q_start.

    q_pre_submit:
        quality immediately before submitted action.

    q_submit:
        quality after the submitted guess is applied.

    q_end:
        final inferred state in this turn.

    within_turn_gain:
        q_pre_submit - q_start

    action_gain:
        q_submit - q_pre_submit

    context_reset_delta:
        q_start(next turn) - q_end(current turn)

        Measures what survives the context/game-state boundary.

    next_turn_initial_gain:
        q_first(next turn) - q_start(next turn)

        Measures the first observed reasoning movement after the reset.

    response_delta:
        q_first(next turn) - q_end(current turn)

        Combined cross-turn response.
    """

    steps_by_turn = defaultdict(
        list
    )

    for step in series:

        steps_by_turn[
            step[
                "turn_index"
            ]
        ].append(
            step
        )

    raw_by_turn = {
        turn[
            "turn_index"
        ]: turn

        for turn in raw_turns
    }

    summaries = []

    for turn_index in sorted(
        raw_by_turn
    ):

        raw_turn = (
            raw_by_turn[
                turn_index
            ]
        )

        local = (
            steps_by_turn.get(
                turn_index,
                [],
            )
        )

        turn_start_steps = [
            step

            for step in local

            if (
                step[
                    "step_type"
                ]
                == "turn_start"
            )
        ]

        reasoning_steps = [
            step

            for step in local

            if (
                step[
                    "step_type"
                ]
                == "reasoning"
            )
        ]

        submit_steps = [
            step

            for step in local

            if (
                step[
                    "step_type"
                ]
                == "submit"
            )
        ]

        q_start = (
            turn_start_steps[0][
                "pair_f1"
            ]

            if turn_start_steps

            else 0.0
        )

        if reasoning_steps:

            q_first = (
                reasoning_steps[0][
                    "pair_f1"
                ]
            )

        elif submit_steps:

            q_first = (
                submit_steps[0][
                    "pair_f1"
                ]
            )

        else:

            q_first = q_start

        q_pre_submit = (
            reasoning_steps[-1][
                "pair_f1"
            ]

            if reasoning_steps

            else q_start
        )

        q_submit = (
            submit_steps[-1][
                "pair_f1"
            ]

            if submit_steps

            else None
        )

        q_end = (
            local[-1][
                "pair_f1"
            ]

            if local

            else q_start
        )

        q_peak = max(
            [
                step[
                    "pair_f1"
                ]

                for step in local
            ]
            or [
                q_start
            ]
        )

        churn_total, churn_transitions = (
            churn_for_series(
                local
            )
        )

        summaries.append(
            {
                "run_id":
                    raw_turn[
                        "run_id"
                    ],

                "puzzle_id":
                    raw_turn[
                        "puzzle_id"
                    ],

                "model":
                    raw_turn[
                        "model"
                    ],

                "turn_id":
                    raw_turn[
                        "turn_id"
                    ],

                "turn_index":
                    turn_index,

                "guess_status":
                    raw_turn[
                        "guess_status"
                    ],

                "feedback_returned":
                    raw_turn[
                        "feedback_returned"
                    ],

                "parse_success":
                    raw_turn[
                        "parse_success"
                    ],

                "solved_before_json":
                    raw_turn[
                        "solved_group_indices_before_json"
                    ],

                "solved_after_json":
                    raw_turn[
                        "solved_group_indices_after_json"
                    ],

                "n_candidates":
                    len(
                        reasoning_steps
                    ),

                "n_kept":
                    sum(
                        step[
                            "kept"
                        ]
                        and step[
                            "well_formed"
                        ]

                        for step in (
                            reasoning_steps
                        )
                    ),

                "has_submit_anchor":
                    int(
                        bool(
                            submit_steps
                        )
                    ),

                "q_start":
                    round(
                        q_start,
                        4,
                    ),

                "q_first":
                    round(
                        q_first,
                        4,
                    ),

                "q_pre_submit":
                    round(
                        q_pre_submit,
                        4,
                    ),

                "q_submit":
                    (
                        round(
                            q_submit,
                            4,
                        )

                        if q_submit
                        is not None

                        else None
                    ),

                "q_end":
                    round(
                        q_end,
                        4,
                    ),

                "q_peak":
                    round(
                        q_peak,
                        4,
                    ),

                "q_regret":
                    round(
                        q_peak
                        - q_end,
                        4,
                    ),

                "within_turn_gain":
                    round(
                        q_pre_submit
                        - q_start,
                        4,
                    ),

                "action_gain":
                    (
                        round(
                            q_submit
                            - q_pre_submit,
                            4,
                        )

                        if q_submit
                        is not None

                        else None
                    ),

                "context_reset_delta":
                    None,

                "next_turn_initial_gain":
                    None,

                "response_delta":
                    None,

                "m1_turn_churn_total":
                    churn_total,

                "m1_turn_churn_per_word":
                    round(
                        churn_total
                        / 16.0,
                        4,
                    ),

                "m1_turn_churn_per_word_step":
                    (
                        round(
                            churn_total
                            / (
                                16.0
                                * churn_transitions
                            ),
                            6,
                        )

                        if churn_transitions

                        else 0.0
                    ),
            }
        )

    # -------------------------------------------------------------------------
    # Cross-turn quantities.
    # -------------------------------------------------------------------------

    for index in range(
        len(summaries) - 1
    ):

        current = (
            summaries[
                index
            ]
        )

        nxt = (
            summaries[
                index + 1
            ]
        )

        current[
            "context_reset_delta"
        ] = round(
            nxt[
                "q_start"
            ]
            - current[
                "q_end"
            ],
            4,
        )

        current[
            "next_turn_initial_gain"
        ] = round(
            nxt[
                "q_first"
            ]
            - nxt[
                "q_start"
            ],
            4,
        )

        current[
            "response_delta"
        ] = round(
            nxt[
                "q_first"
            ]
            - current[
                "q_end"
            ],
            4,
        )

    return summaries


# =============================================================================
# OUTPUT DATABASE
# =============================================================================

def ensure_out(conn):

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_metrics(

            run_id TEXT PRIMARY KEY,

            puzzle_id TEXT,
            model TEXT,

            n_turns INTEGER,
            n_steps INTEGER,
            n_turn_starts INTEGER,
            n_candidates INTEGER,
            n_kept INTEGER,
            n_action_anchors INTEGER,

            m4_regret REAL,
            m4_peak REAL,
            m4_final REAL,
            m4_peak_frac REAL,
            m4_peak_global_step INTEGER,
            m4_peak_turn_index INTEGER,

            m1_churn_total INTEGER,
            m1_churn_per_word REAL,
            m1_churn_per_word_step REAL,
            m1_churn_transitions INTEGER,
            m1_correct_abandonment REAL,
            m1_lockin_mean REAL,
            m1_lockin_json TEXT,

            m2_blast_duration REAL,
            m2_wrong_word_steps INTEGER,
            m2_blocked_word_steps INTEGER,
            m2_blocked_blast_sum INTEGER,
            m2_mean_blast_radius REAL,
            m2_max_blast_radius INTEGER,

            m3_dstruct INTEGER,
            m3_word_violations INTEGER,
            m3_group_violations INTEGER,
            m3_size_violations INTEGER,

            final_exact_groups INTEGER,
            final_ari REAL,

            trajectory_solved INTEGER,

            db_solved INTEGER,
            db_binary_score INTEGER,

            guesses_used INTEGER,
            mistakes_used INTEGER
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS turn_metrics(

            run_id TEXT,
            puzzle_id TEXT,
            model TEXT,

            turn_id TEXT,
            turn_index INTEGER,

            guess_status TEXT,
            feedback_returned TEXT,
            parse_success INTEGER,

            solved_before_json TEXT,
            solved_after_json TEXT,

            n_candidates INTEGER,
            n_kept INTEGER,
            has_submit_anchor INTEGER,

            q_start REAL,
            q_first REAL,
            q_pre_submit REAL,
            q_submit REAL,
            q_end REAL,
            q_peak REAL,
            q_regret REAL,

            within_turn_gain REAL,
            action_gain REAL,

            context_reset_delta REAL,
            next_turn_initial_gain REAL,
            response_delta REAL,

            m1_turn_churn_total INTEGER,
            m1_turn_churn_per_word REAL,
            m1_turn_churn_per_word_step REAL,

            PRIMARY KEY(
                run_id,
                turn_id
            )
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS step_cache(

            run_id TEXT,
            puzzle_id TEXT,
            model TEXT,

            turn_id TEXT,
            turn_index INTEGER,

            global_step INTEGER,
            candidate_index INTEGER,

            step_type TEXT,
            text_source TEXT,

            label TEXT,
            guess_status TEXT,

            kept INTEGER,
            well_formed INTEGER,

            solved_before_json TEXT,

            raw_json TEXT,
            resolved_json TEXT,
            partition_json TEXT,
            detached_json TEXT,

            hypothesis_pair_f1 REAL,
            hypothesis_best_gold_index INTEGER,

            pair_p REAL,
            pair_r REAL,
            pair_f1 REAL,

            exact_hits INTEGER,
            ari REAL,

            n_assigned INTEGER,
            n_groups INTEGER,
            max_group_size INTEGER,

            PRIMARY KEY(
                run_id,
                global_step
            )
        )
        """
    )

    # Successfully completed inputs.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS processed_runs(

            run_id TEXT PRIMARY KEY,

            source_db TEXT,

            raw_turn_count INTEGER,
            tier1_candidate_count INTEGER,
            tier2_label_count INTEGER,

            processed_at TEXT
        )
        """
    )

    # These are retryable. A skipped run is NOT considered processed.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS skipped_runs(

            run_id TEXT PRIMARY KEY,

            source_db TEXT,
            reason TEXT,
            last_attempted_at TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_step_run_turn
        ON step_cache(
            run_id,
            turn_index,
            global_step
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_turn_model
        ON turn_metrics(
            model,
            turn_index
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_run_model
        ON run_metrics(
            model
        )
        """
    )

    conn.commit()


# =============================================================================
# INCREMENTAL / RESUME HELPERS
# =============================================================================

def get_processed_signature(
    out,
    run_id,
):

    row = out.execute(
        """
        SELECT
            raw_turn_count,
            tier1_candidate_count,
            tier2_label_count

        FROM processed_runs

        WHERE run_id = ?
        """,
        (
            run_id,
        ),
    ).fetchone()

    if not row:
        return None

    return tuple(
        row
    )


def current_signature(
    *,
    source,
    tier1,
    tier2,
    run_id,
):

    raw_turn_count = source.execute(
        """
        SELECT COUNT(*)
        FROM turns
        WHERE run_id = ?
        """,
        (
            run_id,
        ),
    ).fetchone()[0]

    tier1_candidate_count = (
        tier1.execute(
            """
            SELECT COUNT(*)
            FROM tier1_candidates
            WHERE run_id = ?
            """,
            (
                run_id,
            ),
        ).fetchone()[0]
    )

    tier2_label_count = (
        tier2.execute(
            """
            SELECT COUNT(*)
            FROM tier2_labels
            WHERE run_id = ?
            """,
            (
                run_id,
            ),
        ).fetchone()[0]
    )

    return (
        raw_turn_count,
        tier1_candidate_count,
        tier2_label_count,
    )


def clear_run_output(
    out,
    run_id,
):
    """
    Delete previous derived data before recomputing a changed run.
    """

    for table in [
        "step_cache",
        "turn_metrics",
        "run_metrics",
        "processed_runs",
        "skipped_runs",
    ]:

        out.execute(
            f"""
            DELETE FROM {table}
            WHERE run_id = ?
            """,
            (
                run_id,
            ),
        )

    out.commit()


def record_skip(
    out,
    *,
    run_id,
    source_db,
    reason,
):

    out.execute(
        """
        INSERT OR REPLACE
        INTO skipped_runs
        VALUES (?, ?, ?, ?)
        """,
        (
            run_id,
            source_db,
            reason,
            time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
        ),
    )

    out.commit()


# =============================================================================
# TIER-1 / TIER-2 LOADERS
# =============================================================================

def load_tier1_candidates(
    tier1,
    run_id,
):

    rows = tier1.execute(
        """
        SELECT
            turn_id,
            candidate_index,
            wordset_json,
            text_source

        FROM tier1_candidates

        WHERE run_id = ?

        ORDER BY
            turn_index,
            candidate_index
        """,
        (
            run_id,
        ),
    ).fetchall()

    by_turn = defaultdict(
        list
    )

    for (
        turn_id,
        candidate_index,
        wordset_json,
        text_source,
    ) in rows:

        raw = load_json(
            wordset_json
        )

        if not isinstance(
            raw,
            list,
        ):
            raw = []

        by_turn[
            turn_id
        ].append(
            {
                "candidate_index":
                    candidate_index,

                "raw":
                    {
                        str(word)
                        .strip()
                        .upper()

                        for word in raw

                        if str(word).strip()
                    },

                "text_source":
                    text_source,
            }
        )

    return by_turn


def load_tier2_labels(
    tier2,
    run_id,
):

    rows = tier2.execute(
        """
        SELECT
            turn_id,
            candidate_index,
            label

        FROM tier2_labels

        WHERE run_id = ?

        ORDER BY
            turn_index,
            candidate_index
        """,
        (
            run_id,
        ),
    ).fetchall()

    by_turn = defaultdict(
        dict
    )

    for (
        turn_id,
        candidate_index,
        label,
    ) in rows:

        by_turn[
            turn_id
        ][
            candidate_index
        ] = label

    return by_turn


# =============================================================================
# RAW SOURCE LOADERS
# =============================================================================

def source_runs(
    conn,
    model_filter=None,
):

    query = """
        SELECT
            r.run_id,
            r.puzzle_id,
            r.model_name,

            r.solved,
            r.binary_score,
            r.guesses_used,
            r.mistakes_used,

            p.ground_truth_json

        FROM runs r

        JOIN puzzles p
          ON p.puzzle_id = r.puzzle_id

        WHERE r.status = 'completed'
    """

    params = []

    if model_filter:

        query += """
            AND r.model_name = ?
        """

        params.append(
            model_filter
        )

    query += """
        ORDER BY r.run_id
    """

    return conn.execute(
        query,
        params,
    )


def load_raw_turns(
    conn,
    *,
    run_id,
    puzzle_id,
    model,
):

    rows = conn.execute(
        """
        SELECT
            turn_id,
            turn_index,

            parsed_guess_json,
            guess_status,
            feedback_returned,
            parse_success,

            solved_group_indices_before_json,
            solved_group_indices_after_json

        FROM turns

        WHERE run_id = ?

        ORDER BY turn_index
        """,
        (
            run_id,
        ),
    ).fetchall()

    return [
        {
            "run_id":
                run_id,

            "puzzle_id":
                puzzle_id,

            "model":
                model,

            "turn_id":
                turn_id,

            "turn_index":
                turn_index,

            "parsed_guess_json":
                parsed_guess_json,

            "guess_status":
                guess_status,

            "feedback_returned":
                feedback_returned,

            "parse_success":
                parse_success,

            "solved_group_indices_before_json":
                solved_before,

            "solved_group_indices_after_json":
                solved_after,
        }

        for (
            turn_id,
            turn_index,
            parsed_guess_json,
            guess_status,
            feedback_returned,
            parse_success,
            solved_before,
            solved_after,
        )
        in rows
    ]


# =============================================================================
# WRITING
# =============================================================================

def write_step_cache(
    out,
    *,
    run_id,
    puzzle_id,
    model,
    series,
):

    rows = []

    for step in series:

        rows.append(
            (
                run_id,
                puzzle_id,
                model,

                step[
                    "turn_id"
                ],

                step[
                    "turn_index"
                ],

                step[
                    "global_step"
                ],

                step[
                    "candidate_index"
                ],

                step[
                    "step_type"
                ],

                step[
                    "text_source"
                ],

                step[
                    "label"
                ],

                step[
                    "guess_status"
                ],

                step[
                    "kept"
                ],

                step[
                    "well_formed"
                ],

                json.dumps(
                    step[
                        "solved_before"
                    ]
                ),

                json.dumps(
                    step[
                        "raw"
                    ]
                ),

                json.dumps(
                    step[
                        "resolved"
                    ]
                ),

                json.dumps(
                    [
                        sorted(
                            group
                        )

                        for group in step[
                            "partition"
                        ]
                    ]
                ),

                json.dumps(
                    step[
                        "detached"
                    ]
                ),

                step[
                    "hypothesis_pair_f1"
                ],

                step[
                    "hypothesis_best_gold_index"
                ],

                step[
                    "pair_p"
                ],

                step[
                    "pair_r"
                ],

                step[
                    "pair_f1"
                ],

                step[
                    "exact_hits"
                ],

                step[
                    "ari"
                ],

                step[
                    "n_assigned"
                ],

                step[
                    "n_groups"
                ],

                step[
                    "max_group_size"
                ],
            )
        )

    out.executemany(
        """
        INSERT OR REPLACE INTO step_cache(
            run_id,
            puzzle_id,
            model,

            turn_id,
            turn_index,

            global_step,
            candidate_index,

            step_type,
            text_source,

            label,
            guess_status,

            kept,
            well_formed,

            solved_before_json,

            raw_json,
            resolved_json,
            partition_json,
            detached_json,

            hypothesis_pair_f1,
            hypothesis_best_gold_index,

            pair_p,
            pair_r,
            pair_f1,

            exact_hits,
            ari,

            n_assigned,
            n_groups,
            max_group_size
        )
        VALUES (
            ?, ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?,
            ?, ?, ?, ?,
            ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?, ?
        )
        """,
        rows,
    )


def write_turn_metrics(
    out,
    summaries,
):

    rows = []

    for summary in summaries:

        rows.append(
            (
                summary[
                    "run_id"
                ],

                summary[
                    "puzzle_id"
                ],

                summary[
                    "model"
                ],

                summary[
                    "turn_id"
                ],

                summary[
                    "turn_index"
                ],

                summary[
                    "guess_status"
                ],

                summary[
                    "feedback_returned"
                ],

                summary[
                    "parse_success"
                ],

                summary[
                    "solved_before_json"
                ],

                summary[
                    "solved_after_json"
                ],

                summary[
                    "n_candidates"
                ],

                summary[
                    "n_kept"
                ],

                summary[
                    "has_submit_anchor"
                ],

                summary[
                    "q_start"
                ],

                summary[
                    "q_first"
                ],

                summary[
                    "q_pre_submit"
                ],

                summary[
                    "q_submit"
                ],

                summary[
                    "q_end"
                ],

                summary[
                    "q_peak"
                ],

                summary[
                    "q_regret"
                ],

                summary[
                    "within_turn_gain"
                ],

                summary[
                    "action_gain"
                ],

                summary[
                    "context_reset_delta"
                ],

                summary[
                    "next_turn_initial_gain"
                ],

                summary[
                    "response_delta"
                ],

                summary[
                    "m1_turn_churn_total"
                ],

                summary[
                    "m1_turn_churn_per_word"
                ],

                summary[
                    "m1_turn_churn_per_word_step"
                ],
            )
        )

    out.executemany(
        """
        INSERT OR REPLACE INTO turn_metrics(
            run_id,
            puzzle_id,
            model,

            turn_id,
            turn_index,

            guess_status,
            feedback_returned,
            parse_success,

            solved_before_json,
            solved_after_json,

            n_candidates,
            n_kept,
            has_submit_anchor,

            q_start,
            q_first,
            q_pre_submit,
            q_submit,
            q_end,
            q_peak,
            q_regret,

            within_turn_gain,
            action_gain,

            context_reset_delta,
            next_turn_initial_gain,
            response_delta,

            m1_turn_churn_total,
            m1_turn_churn_per_word,
            m1_turn_churn_per_word_step
        )
        VALUES (
            ?, ?, ?,
            ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?,
            ?, ?,
            ?, ?, ?,
            ?, ?, ?
        )
        """,
        rows,
    )


def write_run_metrics(
    out,
    *,
    run_id,
    puzzle_id,
    model,
    n_turns,
    metrics,
):

    out.execute(
        """
        INSERT OR REPLACE INTO run_metrics(
            run_id,
            puzzle_id,
            model,

            n_turns,
            n_steps,
            n_turn_starts,
            n_candidates,
            n_kept,
            n_action_anchors,

            m4_regret,
            m4_peak,
            m4_final,
            m4_peak_frac,
            m4_peak_global_step,
            m4_peak_turn_index,

            m1_churn_total,
            m1_churn_per_word,
            m1_churn_per_word_step,
            m1_churn_transitions,
            m1_correct_abandonment,
            m1_lockin_mean,
            m1_lockin_json,

            m2_blast_duration,
            m2_wrong_word_steps,
            m2_blocked_word_steps,
            m2_blocked_blast_sum,
            m2_mean_blast_radius,
            m2_max_blast_radius,

            m3_dstruct,
            m3_word_violations,
            m3_group_violations,
            m3_size_violations,

            final_exact_groups,
            final_ari,

            trajectory_solved,

            db_solved,
            db_binary_score,

            guesses_used,
            mistakes_used
        )
        VALUES (
            ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?,
            ?,
            ?, ?,
            ?, ?
        )
        """,
        (
            run_id,
            puzzle_id,
            model,

            n_turns,

            metrics[
                "n_steps"
            ],

            metrics[
                "n_turn_starts"
            ],

            metrics[
                "n_candidates"
            ],

            metrics[
                "n_kept"
            ],

            metrics[
                "n_action_anchors"
            ],

            metrics[
                "m4_regret"
            ],

            metrics[
                "m4_peak"
            ],

            metrics[
                "m4_final"
            ],

            metrics[
                "m4_peak_frac"
            ],

            metrics[
                "m4_peak_global_step"
            ],

            metrics[
                "m4_peak_turn_index"
            ],

            metrics[
                "m1_churn_total"
            ],

            metrics[
                "m1_churn_per_word"
            ],

            metrics[
                "m1_churn_per_word_step"
            ],

            metrics[
                "m1_churn_transitions"
            ],

            metrics[
                "m1_correct_abandonment"
            ],

            metrics[
                "m1_lockin_mean"
            ],

            metrics[
                "m1_lockin_json"
            ],

            metrics[
                "m2_blast_duration"
            ],

            metrics[
                "m2_wrong_word_steps"
            ],

            metrics[
                "m2_blocked_word_steps"
            ],

            metrics[
                "m2_blocked_blast_sum"
            ],

            metrics[
                "m2_mean_blast_radius"
            ],

            metrics[
                "m2_max_blast_radius"
            ],

            metrics[
                "m3_dstruct"
            ],

            metrics[
                "m3_word_violations"
            ],

            metrics[
                "m3_group_violations"
            ],

            metrics[
                "m3_size_violations"
            ],

            metrics[
                "final_exact_groups"
            ],

            metrics[
                "final_ari"
            ],

            metrics[
                "trajectory_solved"
            ],

            metrics[
                "db_solved"
            ],

            metrics[
                "db_binary_score"
            ],

            metrics[
                "guesses_used"
            ],

            metrics[
                "mistakes_used"
            ],
        ),
    )


# =============================================================================
# MAIN
# =============================================================================

def run(args):

    tier1 = sqlite3.connect(
        f"file:{args.tier1_db}?mode=ro",
        uri=True,
    )

    tier2 = sqlite3.connect(
        f"file:{args.tier2_db}?mode=ro",
        uri=True,
    )

    out = sqlite3.connect(
        args.out
    )

    out.execute(
        "PRAGMA journal_mode=WAL;"
    )

    ensure_out(
        out
    )

    source_connections = []

    for source_path in (
        args.source_db
    ):

        connection = (
            sqlite3.connect(
                f"file:{source_path}?mode=ro",
                uri=True,
            )
        )

        source_connections.append(
            (
                source_path,
                connection,
            )
        )

    started = (
        time.time()
    )

    scored = 0
    unchanged = 0
    recomputed = 0
    skipped = 0

    stop = False

    for (
        source_path,
        source,
    ) in source_connections:

        if stop:
            break

        for row in source_runs(
            source,
            model_filter=(
                args.model_filter
            ),
        ):

            (
                run_id,
                puzzle_id,
                model,
                solved,
                binary_score,
                guesses_used,
                mistakes_used,
                ground_truth_json,
            ) = row

            if (
                args.run_id
                and run_id
                != args.run_id
            ):
                continue

            if (
                args.limit_runs
                is not None
                and scored
                >= args.limit_runs
            ):

                stop = True
                break

            signature = (
                current_signature(
                    source=source,
                    tier1=tier1,
                    tier2=tier2,
                    run_id=run_id,
                )
            )

            previous_signature = (
                get_processed_signature(
                    out,
                    run_id,
                )
            )

            if (
                previous_signature
                == signature
            ):

                exists = out.execute(
                    """
                    SELECT 1
                    FROM run_metrics
                    WHERE run_id = ?
                    """,
                    (
                        run_id,
                    ),
                ).fetchone()

                if exists:

                    unchanged += 1
                    continue

            if (
                previous_signature
                is not None
            ):

                clear_run_output(
                    out,
                    run_id,
                )

                recomputed += 1

            # Clear stale prior failure message before retrying.
            out.execute(
                """
                DELETE FROM skipped_runs
                WHERE run_id = ?
                """,
                (
                    run_id,
                ),
            )

            out.commit()

            gold = (
                parse_gold_groups(
                    ground_truth_json
                )
            )

            if not gold:

                record_skip(
                    out,

                    run_id=run_id,
                    source_db=source_path,
                    reason=(
                        "invalid ground truth"
                    ),
                )

                skipped += 1
                continue

            raw_turns = (
                load_raw_turns(
                    source,

                    run_id=run_id,
                    puzzle_id=puzzle_id,
                    model=model,
                )
            )

            tier1_candidates = (
                load_tier1_candidates(
                    tier1,
                    run_id,
                )
            )

            tier2_labels = (
                load_tier2_labels(
                    tier2,
                    run_id,
                )
            )

            series, error = (
                build_run_series(
                    raw_turns=raw_turns,

                    candidates_by_turn=(
                        tier1_candidates
                    ),

                    labels_by_turn=(
                        tier2_labels
                    ),

                    gold=gold,
                )
            )

            if error:

                record_skip(
                    out,

                    run_id=run_id,
                    source_db=source_path,
                    reason=error,
                )

                skipped += 1
                continue

            if not series:

                record_skip(
                    out,

                    run_id=run_id,
                    source_db=source_path,
                    reason=(
                        "empty trajectory"
                    ),
                )

                skipped += 1
                continue

            source_run = {
                "solved":
                    solved,

                "binary_score":
                    binary_score,

                "guesses_used":
                    guesses_used,

                "mistakes_used":
                    mistakes_used,
            }

            metrics = (
                compute_run_metrics(
                    series=series,
                    gold=gold,
                    source_run=(
                        source_run
                    ),
                )
            )

            turn_summaries = (
                compute_turn_metrics(
                    series=series,
                    raw_turns=(
                        raw_turns
                    ),
                )
            )

            if not args.no_step_cache:

                write_step_cache(
                    out,

                    run_id=run_id,
                    puzzle_id=puzzle_id,
                    model=model,
                    series=series,
                )

            write_turn_metrics(
                out,
                turn_summaries,
            )

            write_run_metrics(
                out,

                run_id=run_id,
                puzzle_id=puzzle_id,
                model=model,

                n_turns=len(
                    raw_turns
                ),

                metrics=metrics,
            )

            out.execute(
                """
                INSERT OR REPLACE
                INTO processed_runs
                VALUES (
                    ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    run_id,
                    source_path,

                    signature[0],
                    signature[1],
                    signature[2],

                    time.strftime(
                        "%Y-%m-%dT%H:%M:%S"
                    ),
                ),
            )

            out.execute(
                """
                DELETE FROM skipped_runs
                WHERE run_id = ?
                """,
                (
                    run_id,
                ),
            )

            out.commit()

            scored += 1

            if (
                scored % 50
                == 0
            ):

                elapsed = max(
                    time.time()
                    - started,
                    1e-9,
                )

                print(
                    f"{scored} processed this invocation | "
                    f"{unchanged} unchanged skipped | "
                    f"{recomputed} recomputed | "
                    f"{skipped} incomplete/skipped | "
                    f"{scored / elapsed:.2f} runs/s",
                    flush=True,
                )

    elapsed = (
        time.time()
        - started
    )

    print()
    print(
        "DONE\n"
        f"  processed now : {scored}\n"
        f"  unchanged     : {unchanged}\n"
        f"  recomputed    : {recomputed}\n"
        f"  skipped       : {skipped}\n"
        f"  elapsed       : {elapsed:.1f}s"
    )

    tier1.close()
    tier2.close()

    for (
        _,
        connection,
    ) in source_connections:

        connection.close()

    out.close()


# =============================================================================
# CLI
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
        "--tier2-db",
        required=True,
    )

    parser.add_argument(
        "--source-db",
        action="append",
        required=True,
        help=(
            "Raw multiturn DB. "
            "Repeat once per model DB."
        ),
    )

    parser.add_argument(
        "--out",
        required=True,
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
        "--limit-runs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--no-step-cache",
        action="store_true",
    )

    return (
        parser.parse_args()
    )


if __name__ == "__main__":

    run(
        parse_args()
    )
