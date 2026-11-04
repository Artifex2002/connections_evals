#!/usr/bin/env python3
"""
tier3_metrics_multiturn_singleshot.py
=====================================

Multiturn adaptation of the original single-shot metrics_step2.py.

Inside EACH turn we use the same formulation as single-shot:

    Tier1 candidates
        -> Tier2 stance
        -> deterministic refine reconstruction
        -> accumulated partition (last-writer-wins)
        -> pairwise F1 q_t
        -> M1-M4

The only multiturn additions are:

1. Turn start:
       seed partition =
           all groups solved before this turn
           +
           immediately previous submitted guess,
           if that guess is still unresolved.

   Hidden reasoning from the previous turn is NOT carried forward.

2. Within-turn reasoning:
       reconstructed Tier1/Tier2 candidate groups are retained exactly
       as expressed, including groups that revisit earlier/solved words.

   There is NO solved-word stripping, historical special-casing,
   or current-vocabulary projection inside a turn.

3. Turn end:
       the actual submitted guess is applied as a final anchor.

4. step_cache contains explicit:
       turn_start
       reasoning
       submission
   markers.

M1-M4 remain the same operational objects as the original
single-shot implementation.
"""

import argparse
import json
import sqlite3
import time
from collections import Counter
from itertools import combinations
from math import comb


HELD = {
    "propose",
    "refine_add",
    "refine_pivot",
    "commit",
}

POLICY_VERSION = "multiturn_singleshot_v1"


# =====================================================================
# JSON HELPERS
# =====================================================================

def loads_json(s, default=None):
    if s is None:
        return default

    if isinstance(s, (list, dict)):
        return s

    try:
        return json.loads(s)
    except Exception:
        return default


def canon_group(xs):
    if not isinstance(xs, (list, tuple, set, frozenset)):
        return set()

    return {
        str(x).strip().upper()
        for x in xs
        if str(x).strip()
    }


def parse_guess(s):
    obj = loads_json(s)

    if not isinstance(obj, list):
        return None

    g = canon_group(obj)
    return g if len(g) == 4 else None


def parse_word_list(s):
    obj = loads_json(s, [])

    if not isinstance(obj, list):
        return set()

    return canon_group(obj)


def parse_index_list(s):
    obj = loads_json(s, [])

    if not isinstance(obj, list):
        return []

    out = []

    for x in obj:
        try:
            out.append(int(x))
        except Exception:
            pass

    return out


def parse_complete_groups(s):
    """
    Handles current puzzle ground_truth_json:

    [
      {
        "category": "...",
        "group_index": 0,
        "words": [...]
      },
      ...
    ]

    Also tolerates old list-of-lists / dict formats.
    """

    obj = loads_json(s)

    if obj is None:
        return None

    if isinstance(obj, dict):
        raw_groups = list(obj.values())

    elif isinstance(obj, list):
        if (
            obj
            and all(
                isinstance(x, dict)
                and "words" in x
                for x in obj
            )
        ):
            raw_groups = [
                x["words"]
                for x in sorted(
                    obj,
                    key=lambda z: z.get("group_index", 999),
                )
            ]
        else:
            raw_groups = obj

    else:
        return None

    groups = []

    for g in raw_groups:
        if isinstance(g, dict) and "words" in g:
            g = g["words"]

        if not isinstance(g, (list, tuple, set, frozenset)):
            return None

        groups.append(canon_group(g))

    if len(groups) != 4:
        return None

    if any(len(g) != 4 for g in groups):
        return None

    return groups


# =====================================================================
# ORIGINAL SINGLE-SHOT RECONSTRUCTION
# =====================================================================

def reconstruct(cands):
    """
    Same reconstruction rule as the original single-shot script.

    No multiturn-specific stripping or projection happens here.
    Every extracted grouping is treated exactly as it would be in
    the original single-shot trajectory.
    """

    prev = set()
    out = []

    for c in cands:
        raw = set(c["raw"])
        lab = c["label"]

        if lab == "refine_pivot":
            grp = raw

        elif lab == "refine_add":
            if prev and not (raw & prev):
                grp = raw

            elif raw < prev:
                grp = raw

            else:
                grp = prev | raw

        else:
            grp = raw

        out.append({
            **c,
            "resolved": grp,
            "kept": lab in HELD,
            "well_formed": len(grp) <= 4,
        })

        prev = grp

    return out


# =====================================================================
# SCORING
# =====================================================================

def gold_pairs_for(gold):
    return {
        tuple(sorted(p))
        for g in gold
        for p in combinations(sorted(g), 2)
    }


def pairwise_prf(partition, gold_pairs):
    pp = set()

    for s in partition:
        if len(s) >= 2:
            pp |= {
                tuple(sorted(p))
                for p in combinations(sorted(s), 2)
            }

    if not pp:
        return 0.0, 0.0, 0.0

    inter = len(pp & gold_pairs)

    p = inter / len(pp)
    r = inter / len(gold_pairs)

    f = (
        2 * p * r / (p + r)
        if p + r
        else 0.0
    )

    return p, r, f


def exact_hits(partition, gold):
    gs = [
        frozenset(g)
        for g in gold
    ]

    return sum(
        1
        for s in partition
        if frozenset(s) in gs
    )


def adjusted_rand(partition, gold):
    """
    Same ARI implementation as single-shot.

    Only defined for a full 16-word partition.
    """

    assigned = sum(
        len(s)
        for s in partition
    )

    total = sum(
        len(g)
        for g in gold
    )

    if assigned != total or len(partition) != len(gold):
        return None

    gid = {
        w: i
        for i, g in enumerate(gold)
        for w in g
    }

    pid = {
        w: i
        for i, s in enumerate(partition)
        for w in s
    }

    if set(pid) != set(gid):
        return None

    words = list(gid)

    a = [
        gid[w]
        for w in words
    ]

    b = [
        pid[w]
        for w in words
    ]

    n = len(words)

    idx = sum(
        comb(v, 2)
        for v in Counter(zip(a, b)).values()
    )

    sa = sum(
        comb(v, 2)
        for v in Counter(a).values()
    )

    sb = sum(
        comb(v, 2)
        for v in Counter(b).values()
    )

    exp = (
        sa * sb
        / comb(n, 2)
    )

    mx = (
        sa + sb
    ) / 2

    return (
        1.0
        if mx == exp
        else (idx - exp) / (mx - exp)
    )


# =====================================================================
# ORIGINAL LAST-WRITER-WINS PARTITION
# =====================================================================

def initialize_partition(groups):
    word2slot = {}
    slots = []

    for group in groups:
        group = set(group)

        for w in group:
            old = word2slot.get(w)

            if old is not None and w in old:
                old.discard(w)

        slot = set(group)
        slots.append(slot)

        for w in group:
            word2slot[w] = slot

        slots[:] = [
            s
            for s in slots
            if s
        ]

    return word2slot, slots


def apply_group(word2slot, slots, group):
    group = set(group)
    detached = []

    for w in group:
        old = word2slot.get(w)

        if old is not None and w in old:
            old.discard(w)
            detached.append(w)

    slot = set(group)
    slots.append(slot)

    for w in group:
        word2slot[w] = slot

    slots[:] = [
        s
        for s in slots
        if s
    ]

    return detached


def snapshot(
    *,
    run_id,
    puzzle_id,
    model,
    turn_id,
    turn_index,
    global_step,
    turn_step,
    step_type,
    candidate_index,
    label,
    kept,
    well_formed,
    resolved_full,
    resolved,
    projected_out,
    slots,
    detached,
    gold,
    gold_pairs,
    guess_status=None,
):
    partition = [
        set(s)
        for s in slots
        if len(s) >= 2
    ]

    p, r, f = pairwise_prf(
        partition,
        gold_pairs,
    )

    assigned = {
        w
        for s in slots
        for w in s
    }

    return {
        "run_id": run_id,
        "puzzle_id": puzzle_id,
        "model": model,

        "turn_id": turn_id,
        "turn_index": turn_index,

        "global_step": global_step,
        "turn_step": turn_step,
        "step_type": step_type,

        "candidate_index": candidate_index,
        "label": label,
        "kept": int(kept),
        "well_formed": int(well_formed),

        "resolved_full": sorted(resolved_full),
        "resolved": sorted(resolved),
        "projected_out": sorted(projected_out),

        "partition": [
            sorted(s)
            for s in partition
        ],

        "detached": sorted(detached),

        "pair_p": round(p, 4),
        "pair_r": round(r, 4),
        "pair_f1": round(f, 4),

        "exact_hits": exact_hits(
            partition,
            gold,
        ),

        "ari": adjusted_rand(
            partition,
            gold,
        ),

        "n_assigned": len(assigned),

        "n_groups": len(partition),

        "max_group_size": max(
            (
                len(s)
                for s in slots
            ),
            default=0,
        ),

        "guess_status": guess_status,
    }


# =====================================================================
# TURN INITIALIZATION
# =====================================================================

def build_turn_seed(
    turn,
    previous_turn,
    gold,
):
    """
    Start each turn from only the externally visible interaction state:

      - all groups already confirmed solved before this turn
      - the immediately previous submitted guess, if it is not already
        one of those solved groups

    Hidden reasoning from the previous turn is never carried forward.

    remaining_words_before_json is retained as metadata only; it is NOT
    used to strip or project hypotheses.
    """

    solved_indices = parse_index_list(
        turn[
            "solved_group_indices_before_json"
        ]
    )

    solved_groups = [
        set(gold[i])
        for i in solved_indices
        if 0 <= i < len(gold)
    ]

    active_words = parse_word_list(
        turn[
            "remaining_words_before_json"
        ]
    )

    seed = list(
        solved_groups
    )

    previous_guess = None
    previous_status = None

    if previous_turn is not None:
        previous_guess = parse_guess(
            previous_turn[
                "parsed_guess_json"
            ]
        )

        previous_status = previous_turn[
            "guess_status"
        ]

    if previous_guess:
        solved_sets = {
            frozenset(g)
            for g in solved_groups
        }

        # If the previous guess was correct, it is already represented
        # by solved_groups and should not be duplicated.
        if frozenset(previous_guess) not in solved_sets:
            seed.append(
                set(previous_guess)
            )

    return {
        "seed_groups": seed,
        "solved_groups": solved_groups,
        "active_words": active_words,
        "previous_guess": previous_guess,
        "previous_status": previous_status,
    }


# =====================================================================
# ONE TURN = ONE SINGLE-SHOT PASS
# =====================================================================

def process_turn(
    *,
    run_id,
    puzzle_id,
    model,
    turn,
    previous_turn,
    cands,
    gold,
    starting_global_step,
):
    gold_pairs = gold_pairs_for(gold)

    seed = build_turn_seed(
        turn,
        previous_turn,
        gold,
    )

    word2slot, slots = initialize_partition(
        seed["seed_groups"]
    )

    series = []

    global_step = starting_global_step
    turn_step = 0

    # ---------------------------------------------------------
    # TURN START
    # ---------------------------------------------------------

    series.append(
        snapshot(
            run_id=run_id,
            puzzle_id=puzzle_id,
            model=model,

            turn_id=turn["turn_id"],
            turn_index=turn["turn_index"],

            global_step=global_step,
            turn_step=turn_step,

            step_type="turn_start",

            candidate_index=-1,
            label="turn_start",

            kept=False,
            well_formed=True,

            resolved_full=set(),
            resolved=set(),
            projected_out=set(),

            slots=slots,
            detached=[],

            gold=gold,
            gold_pairs=gold_pairs,
        )
    )

    global_step += 1
    turn_step += 1

    # ---------------------------------------------------------
    # REASONING
    # ---------------------------------------------------------

    recon = reconstruct(cands)

    for c in recon:
        detached = []

        if c["kept"] and c["well_formed"]:
            detached = apply_group(
                word2slot,
                slots,
                c["resolved"],
            )

        series.append(
            snapshot(
                run_id=run_id,
                puzzle_id=puzzle_id,
                model=model,

                turn_id=turn["turn_id"],
                turn_index=turn["turn_index"],

                global_step=global_step,
                turn_step=turn_step,

                step_type="reasoning",

                candidate_index=c["candidate_index"],
                label=c["label"],

                kept=c["kept"],
                well_formed=c["well_formed"],

                resolved_full=c["resolved"],
                resolved=c["resolved"],
                projected_out=set(),

                slots=slots,
                detached=detached,

                gold=gold,
                gold_pairs=gold_pairs,
            )
        )

        global_step += 1
        turn_step += 1

    # ---------------------------------------------------------
    # ACTUAL SUBMITTED GUESS
    # ---------------------------------------------------------

    submitted = parse_guess(
        turn[
            "parsed_guess_json"
        ]
    )

    if submitted:
        detached = apply_group(
            word2slot,
            slots,
            submitted,
        )

        max_ci = max(
            [
                c["candidate_index"]
                for c in cands
            ],
            default=-1,
        )

        series.append(
            snapshot(
                run_id=run_id,
                puzzle_id=puzzle_id,
                model=model,

                turn_id=turn["turn_id"],
                turn_index=turn["turn_index"],

                global_step=global_step,
                turn_step=turn_step,

                step_type="submission",

                candidate_index=max_ci + 1,
                label="submit_guess",

                kept=True,
                well_formed=True,

                resolved_full=submitted,
                resolved=submitted,
                projected_out=set(),

                slots=slots,
                detached=detached,

                gold=gold,
                gold_pairs=gold_pairs,

                guess_status=turn["guess_status"],
            )
        )

        global_step += 1
        turn_step += 1

    return {
        **seed,
        "submitted_guess": submitted,
        "series": series,
        "next_global_step": global_step,
    }


# =====================================================================
# ORIGINAL M1-M4
# =====================================================================

def _word_to_group(partition):
    m = {}

    for grp in partition:
        fs = frozenset(grp)

        if len(fs) >= 2:
            for w in fs:
                m[w] = fs

    return m


def _final_partition(series):
    return [
        frozenset(g)
        for g in series[-1]["partition"]
    ]


def m1_churn_count(series):
    sigmaps = [
        _word_to_group(
            s["partition"]
        )
        for s in series
    ]

    words = (
        set().union(
            *[
                set(sm)
                for sm in sigmaps
            ]
        )
        if sigmaps
        else set()
    )

    total = 0

    for w in words:
        prev = None

        for sm in sigmaps:
            cur = sm.get(w)

            if cur != prev:
                if (
                    prev is not None
                    and (
                        cur is None
                        or not (
                            cur >= prev
                            or cur <= prev
                        )
                    )
                ):
                    total += 1

                prev = cur

    return total


def m1_churn(series):
    return round(
        m1_churn_count(series) / 16.0,
        4,
    )


def m1_correct_abandonment(
    series,
    gold,
):
    final = set(
        _final_partition(
            series
        )
    )

    before = [
        set(
            frozenset(g)
            for g in s["partition"]
        )
        for s in series[:-1]
    ]

    ab = 0

    for g in gold:
        fg = frozenset(g)

        if (
            any(
                fg in b
                for b in before
            )
            and fg not in final
        ):
            ab += 1

    return round(
        ab / len(gold),
        4,
    )


def m1_lockin_latencies(series):
    n = len(series)

    if not series:
        return []

    if n <= 1:
        return [
            0.0
            for _ in series[-1]["partition"]
        ]

    per_step = [
        set(
            frozenset(g)
            for g in s["partition"]
        )
        for s in series
    ]

    lats = []

    for fg in _final_partition(series):
        t = n - 1
        lock = n - 1

        while (
            t >= 0
            and fg in per_step[t]
        ):
            lock = t
            t -= 1

        lats.append(
            round(
                lock / (n - 1),
                4,
            )
        )

    return lats


def m2_blast_numerator(
    series,
    gold,
):
    gidx = {
        w: i
        for i, g in enumerate(gold)
        for w in g
    }

    total = 0

    for s in series:
        w2slot = _word_to_group(
            s["partition"]
        )

        for w, slot in w2slot.items():
            gi = gidx.get(w)

            if gi is None:
                continue

            blast = sum(
                1
                for x in slot
                if gidx.get(x) != gi
            )

            if blast == 0:
                continue

            gg = [
                m
                for m in gold[gi]
                if m != w
            ]

            blocked = any(
                (
                    m in w2slot
                    and m not in slot
                )
                for m in gg
            )

            if blocked:
                total += blast

    return total


def m2_blast_duration(
    series,
    gold,
):
    n = len(series)

    return (
        round(
            m2_blast_numerator(
                series,
                gold,
            )
            / (16 * n),
            6,
        )
        if n
        else 0.0
    )


def m3_dstruct(
    series,
    gold,
):
    final = [
        set(g)
        for g in series[-1]["partition"]
    ]

    assigned = (
        set().union(*final)
        if final
        else set()
    )

    all_words = set().union(
        *[
            set(g)
            for g in gold
        ]
    )

    word_viol = len(
        all_words - assigned
    )

    group_viol = abs(
        len(final) - 4
    )

    size_viol = sum(
        abs(len(g) - 4)
        for g in final
    )

    return (
        word_viol
        + group_viol
        + size_viol,

        word_viol,
        group_viol,
        size_viol,
    )


# =====================================================================
# TURN METRICS
# =====================================================================

def compute_turn_metrics(
    series,
    gold,
):
    if not series:
        return None

    q = [
        s["pair_f1"]
        for s in series
    ]

    peak = max(q)
    final = q[-1]

    lats = m1_lockin_latencies(
        series
    )

    d, wv, gv, sv = m3_dstruct(
        series,
        gold,
    )

    return {
        "n_steps": len(series),

        "n_candidates": sum(
            s["step_type"] == "reasoning"
            for s in series
        ),

        "n_kept": sum(
            (
                s["step_type"] == "reasoning"
            )
            and s["kept"]
            and s["well_formed"]
            for s in series
        ),

        "q_start": q[0],
        "q_peak": peak,
        "q_end": final,

        "m4_regret": round(
            peak - final,
            4,
        ),

        "m4_peak": round(
            peak,
            4,
        ),

        "m4_final": round(
            final,
            4,
        ),

        "m4_peak_frac": round(
            q.index(peak)
            / len(q),
            4,
        ),

        "m1_churn": m1_churn(
            series
        ),

        "m1_correct_abandonment":
            m1_correct_abandonment(
                series,
                gold,
            ),

        "m1_lockin_mean": (
            round(
                sum(lats)
                / len(lats),
                4,
            )
            if lats
            else None
        ),

        "m1_lockin_json":
            json.dumps(lats),

        "m2_blast_duration":
            m2_blast_duration(
                series,
                gold,
            ),

        "m3_dstruct": d,
        "m3_word_violations": wv,
        "m3_group_violations": gv,
        "m3_size_violations": sv,

        "final_exact_groups":
            series[-1][
                "exact_hits"
            ],

        "final_ari":
            series[-1][
                "ari"
            ],
    }


# =====================================================================
# RUN METRICS
# =====================================================================

def compute_run_metrics(
    turn_results,
    full_series,
    gold,
    run_row,
):
    if not full_series:
        return None

    q = [
        s["pair_f1"]
        for s in full_series
    ]

    peak = max(q)
    final = q[-1]

    # Do NOT count artificial reassignment caused solely by the
    # turn reset itself as M1 churn.
    #
    # Use original single-shot churn independently inside every
    # turn, then sum raw churn events.

    churn_events = sum(
        m1_churn_count(
            tr["series"]
        )
        for tr in turn_results
    )

    # M2 includes every represented within-turn state, including
    # turn_start, because that is the externally visible state at
    # the beginning of the turn.

    blast_num = sum(
        m2_blast_numerator(
            tr["series"],
            gold,
        )
        for tr in turn_results
    )

    total_steps = sum(
        len(tr["series"])
        for tr in turn_results
    )

    lats = m1_lockin_latencies(
        full_series
    )

    d, wv, gv, sv = m3_dstruct(
        full_series,
        gold,
    )

    return {
        "n_turns": len(turn_results),
        "n_steps": len(full_series),

        "n_candidates": sum(
            s["step_type"] == "reasoning"
            for s in full_series
        ),

        "n_kept": sum(
            (
                s["step_type"] == "reasoning"
            )
            and s["kept"]
            and s["well_formed"]
            for s in full_series
        ),

        "n_submissions": sum(
            s["step_type"] == "submission"
            for s in full_series
        ),

        "m4_regret": round(
            peak - final,
            4,
        ),

        "m4_peak": round(
            peak,
            4,
        ),

        "m4_final": round(
            final,
            4,
        ),

        "m4_peak_frac": round(
            q.index(peak)
            / len(q),
            4,
        ),

        "m1_churn": round(
            churn_events
            / 16.0,
            4,
        ),

        "m1_correct_abandonment":
            m1_correct_abandonment(
                full_series,
                gold,
            ),

        "m1_lockin_mean": (
            round(
                sum(lats)
                / len(lats),
                4,
            )
            if lats
            else None
        ),

        "m1_lockin_json":
            json.dumps(lats),

        "m2_blast_duration": (
            round(
                blast_num
                / (
                    16
                    * total_steps
                ),
                6,
            )
            if total_steps
            else 0.0
        ),

        "m3_dstruct": d,
        "m3_word_violations": wv,
        "m3_group_violations": gv,
        "m3_size_violations": sv,

        "final_exact_groups":
            full_series[-1][
                "exact_hits"
            ],

        "final_ari":
            full_series[-1][
                "ari"
            ],

        "db_solved":
            run_row[
                "solved"
            ],

        "db_binary_score":
            run_row[
                "binary_score"
            ],

        "db_guesses_used":
            run_row[
                "guesses_used"
            ],

        "db_mistakes_used":
            run_row[
                "mistakes_used"
            ],

        "trajectory_solved": int(
            full_series[-1][
                "exact_hits"
            ] == 4
        ),
    }


# =====================================================================
# OUTPUT DB
# =====================================================================

def ensure_out(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS run_metrics(
        run_id TEXT PRIMARY KEY,
        puzzle_id TEXT,
        model TEXT,

        n_turns INTEGER,
        n_steps INTEGER,
        n_candidates INTEGER,
        n_kept INTEGER,
        n_submissions INTEGER,

        m4_regret REAL,
        m4_peak REAL,
        m4_final REAL,
        m4_peak_frac REAL,

        m1_churn REAL,
        m1_correct_abandonment REAL,
        m1_lockin_mean REAL,
        m1_lockin_json TEXT,

        m2_blast_duration REAL,

        m3_dstruct INTEGER,
        m3_word_violations INTEGER,
        m3_group_violations INTEGER,
        m3_size_violations INTEGER,

        final_exact_groups INTEGER,
        final_ari REAL,

        db_solved INTEGER,
        db_binary_score INTEGER,
        db_guesses_used INTEGER,
        db_mistakes_used INTEGER,

        trajectory_solved INTEGER,

        policy_version TEXT
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS turn_metrics(
        run_id TEXT,
        puzzle_id TEXT,
        model TEXT,
        turn_id TEXT,
        turn_index INTEGER,

        previous_guess_json TEXT,
        previous_guess_status TEXT,
        seed_partition_json TEXT,
        solved_before_json TEXT,
        active_words_json TEXT,
        submitted_guess_json TEXT,
        guess_status TEXT,

        n_steps INTEGER,
        n_candidates INTEGER,
        n_kept INTEGER,

        q_start REAL,
        q_peak REAL,
        q_end REAL,

        m4_regret REAL,
        m4_peak REAL,
        m4_final REAL,
        m4_peak_frac REAL,

        m1_churn REAL,
        m1_correct_abandonment REAL,
        m1_lockin_mean REAL,
        m1_lockin_json TEXT,

        m2_blast_duration REAL,

        m3_dstruct INTEGER,
        m3_word_violations INTEGER,
        m3_group_violations INTEGER,
        m3_size_violations INTEGER,

        final_exact_groups INTEGER,
        final_ari REAL,

        PRIMARY KEY(
            run_id,
            turn_id
        )
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS step_cache(
        run_id TEXT,
        puzzle_id TEXT,
        model TEXT,

        turn_id TEXT,
        turn_index INTEGER,

        global_step INTEGER,
        turn_step INTEGER,
        step_type TEXT,

        candidate_index INTEGER,
        label TEXT,
        kept INTEGER,
        well_formed INTEGER,

        resolved_full_json TEXT,
        resolved_json TEXT,
        projected_out_json TEXT,

        partition_json TEXT,
        detached_json TEXT,

        pair_p REAL,
        pair_r REAL,
        pair_f1 REAL,

        exact_hits INTEGER,
        ari REAL,

        n_assigned INTEGER,
        n_groups INTEGER,
        max_group_size INTEGER,

        guess_status TEXT,

        PRIMARY KEY(
            run_id,
            turn_id,
            turn_step
        )
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS processed_runs(
        run_id TEXT PRIMARY KEY,
        raw_turn_count INTEGER,
        tier1_candidate_count INTEGER,
        tier2_label_count INTEGER,
        policy_version TEXT,
        processed_at TEXT
    )
    """)

    conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_step_run_global
    ON step_cache(
        run_id,
        global_step
    )
    """)

    conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_turn_model
    ON turn_metrics(
        model,
        turn_index
    )
    """)

    conn.commit()


# =====================================================================
# INPUT DB HELPERS
# =====================================================================

def open_source_dbs(paths):
    out = []

    for path in paths:
        conn = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
        )

        conn.row_factory = sqlite3.Row

        out.append(
            (
                path,
                conn,
            )
        )

    return out


def find_source_run(
    source_conns,
    run_id,
):
    for path, conn in source_conns:
        row = conn.execute("""
            SELECT
                run_id,
                puzzle_id,
                model_name,
                status,
                solved,
                binary_score,
                guesses_used,
                mistakes_used
            FROM runs
            WHERE run_id = ?
        """, (
            run_id,
        )).fetchone()

        if row:
            return (
                path,
                conn,
                row,
            )

    return None, None, None


def load_gold(
    conn,
    puzzle_id,
):
    row = conn.execute("""
        SELECT
            ground_truth_json
        FROM puzzles
        WHERE puzzle_id = ?
    """, (
        puzzle_id,
    )).fetchone()

    if not row:
        return None

    return parse_complete_groups(
        row[
            "ground_truth_json"
        ]
    )


def load_turns(
    conn,
    run_id,
):
    return conn.execute("""
        SELECT
            run_id,
            turn_id,
            turn_index,
            parsed_guess_json,
            guess_status,
            remaining_words_before_json,
            solved_group_indices_before_json
        FROM turns
        WHERE run_id = ?
        ORDER BY turn_index
    """, (
        run_id,
    )).fetchall()


def load_candidates(
    tier1,
    tier2,
    run_id,
    turn_id,
):
    t1 = tier1.execute("""
        SELECT
            candidate_index,
            wordset_json
        FROM tier1_candidates
        WHERE
            run_id = ?
            AND turn_id = ?
        ORDER BY candidate_index
    """, (
        run_id,
        turn_id,
    )).fetchall()

    labs = {
        ci: lab
        for ci, lab
        in tier2.execute("""
            SELECT
                candidate_index,
                label
            FROM tier2_labels
            WHERE
                run_id = ?
                AND turn_id = ?
        """, (
            run_id,
            turn_id,
        ))
    }

    if len(labs) != len(t1):
        return None

    cands = []

    for ci, gj in t1:
        if ci not in labs:
            return None

        raw = loads_json(
            gj,
            [],
        )

        if not isinstance(
            raw,
            list,
        ):
            return None

        cands.append({
            "candidate_index": ci,
            "raw": canon_group(raw),
            "label": labs[ci],
        })

    return cands


def eligible_run_ids(
    tier1,
    model_filter=None,
):
    query = """
        SELECT DISTINCT
            run_id,
            model
        FROM tier1_candidates
    """

    params = []

    if model_filter:
        query += """
        WHERE model = ?
        """

        params.append(
            model_filter
        )

    query += """
        ORDER BY run_id
    """

    return [
        r[0]
        for r
        in tier1.execute(
            query,
            params,
        ).fetchall()
    ]


def signature(
    tier1,
    tier2,
    source,
    run_id,
):
    raw_turns = source.execute("""
        SELECT COUNT(*)
        FROM turns
        WHERE run_id = ?
    """, (
        run_id,
    )).fetchone()[0]

    n1 = tier1.execute("""
        SELECT COUNT(*)
        FROM tier1_candidates
        WHERE run_id = ?
    """, (
        run_id,
    )).fetchone()[0]

    n2 = tier2.execute("""
        SELECT COUNT(*)
        FROM tier2_labels
        WHERE run_id = ?
    """, (
        run_id,
    )).fetchone()[0]

    return (
        raw_turns,
        n1,
        n2,
    )


# =====================================================================
# DB WRITES
# =====================================================================

def delete_run(
    out,
    run_id,
):
    for table in [
        "step_cache",
        "turn_metrics",
        "run_metrics",
        "processed_runs",
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


def write_steps(
    out,
    series,
):
    sql = """
    INSERT OR REPLACE INTO step_cache(
        run_id,
        puzzle_id,
        model,

        turn_id,
        turn_index,

        global_step,
        turn_step,
        step_type,

        candidate_index,
        label,
        kept,
        well_formed,

        resolved_full_json,
        resolved_json,
        projected_out_json,

        partition_json,
        detached_json,

        pair_p,
        pair_r,
        pair_f1,

        exact_hits,
        ari,

        n_assigned,
        n_groups,
        max_group_size,

        guess_status
    )
    VALUES (
        ?,?,?,?,?,?,?,?,
        ?,?,?,?,
        ?,?,?,
        ?,?,
        ?,?,?,
        ?,?,
        ?,?,?,?
    )
    """

    rows = []

    for s in series:
        rows.append((
            s["run_id"],
            s["puzzle_id"],
            s["model"],

            s["turn_id"],
            s["turn_index"],

            s["global_step"],
            s["turn_step"],
            s["step_type"],

            s["candidate_index"],
            s["label"],
            s["kept"],
            s["well_formed"],

            json.dumps(
                s["resolved_full"]
            ),

            json.dumps(
                s["resolved"]
            ),

            json.dumps(
                s["projected_out"]
            ),

            json.dumps(
                s["partition"]
            ),

            json.dumps(
                s["detached"]
            ),

            s["pair_p"],
            s["pair_r"],
            s["pair_f1"],

            s["exact_hits"],
            s["ari"],

            s["n_assigned"],
            s["n_groups"],
            s["max_group_size"],

            s["guess_status"],
        ))

    out.executemany(
        sql,
        rows,
    )


def write_turn_metric(
    out,
    run_id,
    puzzle_id,
    model,
    turn,
    result,
    m,
):
    out.execute("""
    INSERT OR REPLACE INTO turn_metrics(
        run_id,
        puzzle_id,
        model,
        turn_id,
        turn_index,

        previous_guess_json,
        previous_guess_status,
        seed_partition_json,
        solved_before_json,
        active_words_json,
        submitted_guess_json,
        guess_status,

        n_steps,
        n_candidates,
        n_kept,

        q_start,
        q_peak,
        q_end,

        m4_regret,
        m4_peak,
        m4_final,
        m4_peak_frac,

        m1_churn,
        m1_correct_abandonment,
        m1_lockin_mean,
        m1_lockin_json,

        m2_blast_duration,

        m3_dstruct,
        m3_word_violations,
        m3_group_violations,
        m3_size_violations,

        final_exact_groups,
        final_ari
    )
    VALUES (
        ?,?,?,?,?,?,?,?,
        ?,?,?,?,
        ?,?,?,
        ?,?,?,
        ?,?,?,?,
        ?,?,?,?,
        ?,
        ?,?,?,?,
        ?,?
    )
    """, (
        run_id,
        puzzle_id,
        model,
        turn["turn_id"],
        turn["turn_index"],

        json.dumps(
            sorted(
                result[
                    "previous_guess"
                ]
            )
            if result[
                "previous_guess"
            ]
            else []
        ),

        result[
            "previous_status"
        ],

        json.dumps([
            sorted(g)
            for g
            in result[
                "seed_groups"
            ]
        ]),

        json.dumps(
            parse_index_list(
                turn[
                    "solved_group_indices_before_json"
                ]
            )
        ),

        json.dumps(
            sorted(
                result[
                    "active_words"
                ]
            )
        ),

        json.dumps(
            sorted(
                result[
                    "submitted_guess"
                ]
            )
            if result[
                "submitted_guess"
            ]
            else []
        ),

        turn[
            "guess_status"
        ],

        m["n_steps"],
        m["n_candidates"],
        m["n_kept"],

        m["q_start"],
        m["q_peak"],
        m["q_end"],

        m["m4_regret"],
        m["m4_peak"],
        m["m4_final"],
        m["m4_peak_frac"],

        m["m1_churn"],
        m[
            "m1_correct_abandonment"
        ],
        m[
            "m1_lockin_mean"
        ],
        m[
            "m1_lockin_json"
        ],

        m[
            "m2_blast_duration"
        ],

        m["m3_dstruct"],
        m[
            "m3_word_violations"
        ],
        m[
            "m3_group_violations"
        ],
        m[
            "m3_size_violations"
        ],

        m[
            "final_exact_groups"
        ],

        m["final_ari"],
    ))


def write_run_metric(
    out,
    run_id,
    puzzle_id,
    model,
    m,
):
    out.execute("""
    INSERT OR REPLACE INTO run_metrics(
        run_id,
        puzzle_id,
        model,

        n_turns,
        n_steps,
        n_candidates,
        n_kept,
        n_submissions,

        m4_regret,
        m4_peak,
        m4_final,
        m4_peak_frac,

        m1_churn,
        m1_correct_abandonment,
        m1_lockin_mean,
        m1_lockin_json,

        m2_blast_duration,

        m3_dstruct,
        m3_word_violations,
        m3_group_violations,
        m3_size_violations,

        final_exact_groups,
        final_ari,

        db_solved,
        db_binary_score,
        db_guesses_used,
        db_mistakes_used,

        trajectory_solved,

        policy_version
    )
    VALUES (
        ?,?,?,?,?,?,?,?,
        ?,?,?,?,
        ?,?,?,?,
        ?,
        ?,?,?,?,
        ?,?,
        ?,?,?,?,
        ?,?
    )
    """, (
        run_id,
        puzzle_id,
        model,

        m["n_turns"],
        m["n_steps"],
        m["n_candidates"],
        m["n_kept"],
        m["n_submissions"],

        m["m4_regret"],
        m["m4_peak"],
        m["m4_final"],
        m["m4_peak_frac"],

        m["m1_churn"],
        m[
            "m1_correct_abandonment"
        ],
        m[
            "m1_lockin_mean"
        ],
        m[
            "m1_lockin_json"
        ],

        m[
            "m2_blast_duration"
        ],

        m["m3_dstruct"],
        m[
            "m3_word_violations"
        ],
        m[
            "m3_group_violations"
        ],
        m[
            "m3_size_violations"
        ],

        m[
            "final_exact_groups"
        ],

        m["final_ari"],

        m["db_solved"],
        m[
            "db_binary_score"
        ],
        m[
            "db_guesses_used"
        ],
        m[
            "db_mistakes_used"
        ],

        m[
            "trajectory_solved"
        ],

        POLICY_VERSION,
    ))


# =====================================================================
# MAIN
# =====================================================================

def run(args):
    tier1 = sqlite3.connect(
        f"file:{args.tier1_db}?mode=ro",
        uri=True,
    )

    tier2 = sqlite3.connect(
        f"file:{args.tier2_db}?mode=ro",
        uri=True,
    )

    sources = open_source_dbs(
        args.source_db
    )

    out = sqlite3.connect(
        args.out
    )

    out.execute(
        "PRAGMA journal_mode=WAL;"
    )

    ensure_out(out)

    run_ids = eligible_run_ids(
        tier1,
        args.model_filter,
    )

    if args.skip_runs:
        run_ids = run_ids[
            args.skip_runs:
        ]

    if args.limit_runs is not None:
        run_ids = run_ids[
            :args.limit_runs
        ]

    t0 = time.time()

    processed = 0
    unchanged = 0
    recomputed = 0
    skipped = 0

    for run_id in run_ids:
        _, src, run_row = find_source_run(
            sources,
            run_id,
        )

        if run_row is None:
            skipped += 1
            continue

        sig = signature(
            tier1,
            tier2,
            src,
            run_id,
        )

        # Tier1/Tier2 coverage must agree exactly.
        if sig[1] != sig[2]:
            skipped += 1
            continue

        old = out.execute("""
            SELECT
                raw_turn_count,
                tier1_candidate_count,
                tier2_label_count,
                policy_version
            FROM processed_runs
            WHERE run_id = ?
        """, (
            run_id,
        )).fetchone()

        if (
            old
            and tuple(old[:3]) == sig
            and old[3] == POLICY_VERSION
            and not args.force
        ):
            unchanged += 1
            continue

        if old:
            recomputed += 1

        gold = load_gold(
            src,
            run_row[
                "puzzle_id"
            ],
        )

        if not gold:
            skipped += 1
            continue

        turns = load_turns(
            src,
            run_id,
        )

        if not turns:
            skipped += 1
            continue

        delete_run(
            out,
            run_id,
        )

        puzzle_id = run_row[
            "puzzle_id"
        ]

        model = run_row[
            "model_name"
        ]

        previous_turn = None
        global_step = 0
        full_series = []
        turn_results = []
        incomplete = False

        for turn in turns:
            cands = load_candidates(
                tier1,
                tier2,
                run_id,
                turn[
                    "turn_id"
                ],
            )

            if cands is None:
                incomplete = True
                break

            result = process_turn(
                run_id=run_id,
                puzzle_id=puzzle_id,
                model=model,
                turn=turn,
                previous_turn=previous_turn,
                cands=cands,
                gold=gold,
                starting_global_step=global_step,
            )

            global_step = result[
                "next_global_step"
            ]

            series = result[
                "series"
            ]

            tm = compute_turn_metrics(
                series,
                gold,
            )

            if tm is None:
                incomplete = True
                break

            if not args.no_step_cache:
                write_steps(
                    out,
                    series,
                )

            write_turn_metric(
                out,
                run_id,
                puzzle_id,
                model,
                turn,
                result,
                tm,
            )

            turn_results.append(
                result
            )

            full_series.extend(
                series
            )

            previous_turn = turn

        if incomplete or not full_series:
            out.rollback()
            skipped += 1
            continue

        rm = compute_run_metrics(
            turn_results,
            full_series,
            gold,
            run_row,
        )

        if rm is None:
            out.rollback()
            skipped += 1
            continue

        write_run_metric(
            out,
            run_id,
            puzzle_id,
            model,
            rm,
        )

        out.execute("""
            INSERT OR REPLACE INTO processed_runs
            VALUES (
                ?,?,?,?,?,?
            )
        """, (
            run_id,
            sig[0],
            sig[1],
            sig[2],
            POLICY_VERSION,
            time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
        ))

        out.commit()

        processed += 1

        if processed % 50 == 0:
            elapsed = max(
                time.time() - t0,
                1e-9,
            )

            print(
                f"{processed} processed | "
                f"{unchanged} unchanged | "
                f"{recomputed} recomputed | "
                f"{skipped} skipped | "
                f"{processed / elapsed:.2f} runs/s",
                flush=True,
            )

    elapsed = (
        time.time() - t0
    )

    print()
    print("DONE")

    print(
        f"  policy        : "
        f"{POLICY_VERSION}"
    )

    print(
        f"  processed now : "
        f"{processed}"
    )

    print(
        f"  unchanged     : "
        f"{unchanged}"
    )

    print(
        f"  recomputed    : "
        f"{recomputed}"
    )

    print(
        f"  skipped       : "
        f"{skipped}"
    )

    print(
        f"  elapsed       : "
        f"{elapsed:.1f}s"
    )

    tier1.close()
    tier2.close()

    for _, c in sources:
        c.close()

    out.close()


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--tier1-db",
        default=
            "data/db/tier1_multiturn_all.db",
    )

    ap.add_argument(
        "--tier2-db",
        default=
            "data/db/tier2_v5_all.db",
    )

    ap.add_argument(
        "--source-db",
        action="append",
        required=True,
        help=(
            "Raw DB containing "
            "runs/turns/puzzles. "
            "Repeat once per model DB."
        ),
    )

    ap.add_argument(
        "--out",
        default=
            "data/db/"
            "metrics_multiturn_"
            "singleshot_v1.db",
    )

    ap.add_argument(
        "--model-filter",
        default=None,
    )

    ap.add_argument(
        "--limit-runs",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--skip-runs",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--force",
        action="store_true",
    )

    ap.add_argument(
        "--no-step-cache",
        action="store_true",
    )

    return ap.parse_args()


def main():
    run(
        parse_args()
    )


if __name__ == "__main__":
    main()
