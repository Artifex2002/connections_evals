#!/usr/bin/env python3
"""
plot_multiturn_trajectory.py

Visualize one multiturn NYT Connections reasoning trajectory.

For each run, produces TWO plots:

1. Macro trajectory
   ----------------
   - every reconstructed reasoning state
   - global q_t = pairwise F1
   - exact-group count
   - per-word state heatmap
   - turn boundaries
   - official submission markers
   - actual submitted 4-word guesses
   - feedback entering the next turn

2. Turn-action trajectory
   ----------------------
   Compresses each turn into:
       q_start -> q_peak -> q_action

   where q_action is the state at the official submission if available,
   otherwise the final represented state of that turn.

   Also shows:
       official submitted guess
       guess result / feedback
       transition from one turn's action to the next turn's start

Expected metrics DB:
    step_cache
    run_metrics

Expected raw/source DB:
    puzzles
    runs
    turns
"""

import argparse
import json
import os
import signal
import sqlite3

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


# ---------------------------------------------------------------------
# avoid BrokenPipeError when doing:
#     python ... --list | head -30
# ---------------------------------------------------------------------

try:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
except Exception:
    pass


# ---------------------------------------------------------------------
# visual state definitions
# ---------------------------------------------------------------------

STATE = {
    0: ("unassigned", "#e6e6e6"),
    1: ("misassigned (no gold-mate)", "#d1495b"),
    2: ("contaminated (mixed slot)", "#edae49"),
    3: ("partial-correct (clean, <4)", "#a3c98c"),
    4: ("exact gold group", "#2e8b57"),
}

CMAP = ListedColormap([STATE[i][1] for i in range(5)])
NORM = BoundaryNorm(
    [-.5, .5, 1.5, 2.5, 3.5, 4.5],
    CMAP.N,
)


# ---------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------

def cols(conn, table):
    return {
        row[1]
        for row in conn.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()
    }


def fmt(v, digits=3):
    if v is None:
        return "NA"

    try:
        if np.isnan(v):
            return "NA"
    except Exception:
        pass

    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def clean_word(w):
    return str(w).strip().upper()


def normalize_status(status):
    if status is None:
        return None

    s = str(status).strip()

    if not s:
        return None

    return s.replace("_", " ").upper()


# ---------------------------------------------------------------------
# gold parsing
# ---------------------------------------------------------------------

def parse_groups(s):
    if not s:
        return None

    try:
        obj = json.loads(s)
    except Exception:
        return None

    groups = obj.values() if isinstance(obj, dict) else obj
    out = []

    for g in groups:

        if isinstance(g, dict):
            for key in (
                "words",
                "members",
                "group",
                "items",
            ):
                if key in g:
                    g = g[key]
                    break

        if isinstance(g, (list, tuple)):
            out.append({
                clean_word(w)
                for w in g
            })

    return out if len(out) >= 4 else None


def get_gold(src, puzzle_id):
    pcols = cols(src, "puzzles")

    candidates = [
        "ground_truth_json",
        "answers_json",
        "answers",
        "ground_truth",
        "solution_json",
        "solution",
        "groups_json",
        "groups",
    ]

    for c in candidates:
        if c not in pcols:
            continue

        row = src.execute(
            f"""
            SELECT {c}
            FROM puzzles
            WHERE puzzle_id=?
            """,
            (puzzle_id,),
        ).fetchone()

        if not row:
            continue

        gold = parse_groups(row[0])

        if gold:
            return gold, c

    raise SystemExit(
        "Could not find/parse gold groups in puzzles.\n"
        "Inspect with:\n"
        "sqlite3 RAW.db '.schema puzzles'"
    )


# ---------------------------------------------------------------------
# submitted guess parsing
# ---------------------------------------------------------------------

def parse_guess(s):
    """
    Parse parsed_guess_json robustly.

    Supported examples:

        ["A","B","C","D"]

        {"guess":["A","B","C","D"]}

        {"words":["A","B","C","D"]}
    """

    if not s:
        return None

    if isinstance(s, (list, tuple)):
        obj = s
    else:
        try:
            obj = json.loads(s)
        except Exception:
            return None

    if isinstance(obj, dict):
        for key in (
            "guess",
            "words",
            "group",
            "members",
            "items",
        ):
            if key in obj:
                obj = obj[key]
                break

    if not isinstance(obj, (list, tuple)):
        return None

    out = [
        clean_word(w)
        for w in obj
        if str(w).strip()
    ]

    return out if out else None


def compact_guess(guess):
    """
    Two-line representation:

        WORD1 · WORD2
        WORD3 · WORD4
    """

    if not guess:
        return "NO PARSED GUESS"

    if len(guess) <= 2:
        return " · ".join(guess)

    mid = (len(guess) + 1) // 2

    return (
        " · ".join(guess[:mid])
        + "\n"
        + " · ".join(guess[mid:])
    )


# ---------------------------------------------------------------------
# word state
# ---------------------------------------------------------------------

def word_state(slot, w, gold_grp):
    if not slot:
        return 0

    others = set(slot) - {w}
    mates = others & gold_grp
    wrong = others - gold_grp

    if set(slot) == gold_grp:
        return 4

    if wrong:
        return 2 if mates else 1

    return 3 if mates else 1


# ---------------------------------------------------------------------
# load raw turn actions
# ---------------------------------------------------------------------

def load_turn_actions(src, run_id):
    """
    Returns:

        {
            turn_index: {
                "status": ...,
                "feedback": ...,
                "guess": [...],
                "category": ...,
            }
        }
    """

    tcols = cols(src, "turns")

    required = {
        "run_id",
        "turn_index",
    }

    if not required <= tcols:
        raise SystemExit(
            f"turns table missing required columns: "
            f"{sorted(required - tcols)}"
        )

    optional = []

    for c in (
        "guess_status",
        "feedback_returned",
        "parsed_guess_json",
        "proposed_category",
    ):
        if c in tcols:
            optional.append(c)

    select_cols = [
        "turn_index",
        *optional,
    ]

    rows = src.execute(
        f"""
        SELECT {", ".join(select_cols)}
        FROM turns
        WHERE run_id=?
        ORDER BY turn_index
        """,
        (run_id,),
    ).fetchall()

    actions = {}

    for row in rows:
        d = dict(zip(select_cols, row))

        ti = int(d["turn_index"])

        actions[ti] = {
            "status": d.get("guess_status"),
            "feedback": d.get("feedback_returned"),
            "guess": parse_guess(
                d.get("parsed_guess_json")
            ),
            "category": d.get("proposed_category"),
        }

    return actions


# ---------------------------------------------------------------------
# load metrics + raw source
# ---------------------------------------------------------------------

def load(metrics_db, source_db, run_id):
    # --------------------------------------------------------------
    # metrics
    # --------------------------------------------------------------

    m = sqlite3.connect(
        f"file:{metrics_db}?mode=ro",
        uri=True,
    )

    scols = cols(m, "step_cache")

    if "step_index" in scols:
        order_col = "step_index"

    elif "global_step_index" in scols:
        order_col = "global_step_index"

    else:
        order_col = "rowid"

    required = {
        "run_id",
        "turn_index",
        "step_type",
        "pair_f1",
        "exact_hits",
        "partition_json",
    }

    missing = required - scols

    if missing:
        raise SystemExit(
            "step_cache missing expected columns:\n"
            f"{sorted(missing)}\n\n"
            f"Available:\n{sorted(scols)}"
        )

    steps = m.execute(
        f"""
        SELECT
            {order_col} AS step_index,
            turn_index,
            step_type,
            pair_f1,
            exact_hits,
            partition_json
        FROM step_cache
        WHERE run_id=?
        ORDER BY {order_col}
        """,
        (run_id,),
    ).fetchall()

    if not steps:
        raise SystemExit(
            f"No step_cache rows for run_id={run_id}"
        )

    rcols = cols(m, "run_metrics")

    wanted = [
        "puzzle_id",
        "model",

        "n_turns",
        "n_steps",
        "n_candidates",
        "n_submissions",

        "m1_churn",
        "m1_correct_abandonment",
        "m1_lockin_mean",

        "m2_blast_duration",

        "m3_dstruct",
        "m3_word_violations",
        "m3_group_violations",
        "m3_size_violations",

        "m4_peak",
        "m4_final",
        "m4_regret",
        "m4_peak_frac",

        "final_exact_groups",
        "final_ari",

        "db_solved",
        "db_binary_score",
        "db_guesses_used",
        "db_mistakes_used",

        "trajectory_solved",
        "policy_version",
    ]

    present = [
        c
        for c in wanted
        if c in rcols
    ]

    meta_row = m.execute(
        f"""
        SELECT {",".join(present)}
        FROM run_metrics
        WHERE run_id=?
        """,
        (run_id,),
    ).fetchone()

    if not meta_row:
        raise SystemExit(
            f"No run_metrics row for {run_id}"
        )

    meta = dict(
        zip(present, meta_row)
    )

    m.close()

    # --------------------------------------------------------------
    # source / raw DB
    # --------------------------------------------------------------

    src = sqlite3.connect(
        f"file:{source_db}?mode=ro",
        uri=True,
    )

    puzzle_id = meta.get("puzzle_id")

    if not puzzle_id:
        row = src.execute(
            """
            SELECT puzzle_id
            FROM runs
            WHERE run_id=?
            """,
            (run_id,),
        ).fetchone()

        if not row:
            raise SystemExit(
                "Could not determine puzzle_id."
            )

        puzzle_id = row[0]

        meta["puzzle_id"] = puzzle_id

    gold, gold_col = get_gold(
        src,
        puzzle_id,
    )

    actions = load_turn_actions(
        src,
        run_id,
    )

    src.close()

    return (
        steps,
        meta,
        gold,
        actions,
        gold_col,
    )


# ---------------------------------------------------------------------
# build full trajectory arrays
# ---------------------------------------------------------------------

def build(steps, gold):
    words = [
        w
        for g in gold
        for w in g
    ]

    gmap = {
        w: i
        for i, g in enumerate(gold)
        for w in g
    }

    gset = [
        set(g)
        for g in gold
    ]

    T = len(steps)

    mat = np.zeros(
        (len(words), T),
        dtype=int,
    )

    q = np.array(
        [
            float(s[3])
            if s[3] is not None
            else np.nan
            for s in steps
        ],
        dtype=float,
    )

    exact = np.array(
        [
            int(s[4])
            if s[4] is not None
            else 0
            for s in steps
        ],
        dtype=int,
    )

    turn_idx = np.array(
        [
            int(s[1])
            for s in steps
        ],
        dtype=int,
    )

    step_types = [
        str(s[2])
        for s in steps
    ]

    for ti, s in enumerate(steps):

        try:
            partition = (
                json.loads(s[5])
                if s[5]
                else []
            )

        except Exception:
            partition = []

        w2slot = {}

        for grp in partition:

            if not isinstance(
                grp,
                (list, tuple),
            ):
                continue

            gs = {
                clean_word(w)
                for w in grp
            }

            for w in gs:
                w2slot[w] = gs

        for r, w in enumerate(words):

            mat[r, ti] = word_state(
                w2slot.get(w),
                w,
                gset[gmap[w]],
            )

    return (
        words,
        gmap,
        q,
        exact,
        mat,
        turn_idx,
        step_types,
    )


# ---------------------------------------------------------------------
# turn-level summaries
# ---------------------------------------------------------------------

def summarize_turns(
    q,
    exact,
    turn_idx,
    step_types,
    actions,
):
    """
    For each turn:

        start_q
        peak_q
        action_q
        action position
        guess
        status

    action_q = q at official "submission" step when present,
               otherwise q at final represented state.
    """

    summaries = []

    turns = sorted(
        set(int(x) for x in turn_idx)
    )

    for ti in turns:

        positions = np.where(
            turn_idx == ti
        )[0]

        if not len(positions):
            continue

        start_pos = int(positions[0])
        end_pos = int(positions[-1])

        turn_q = q[positions]

        valid_mask = ~np.isnan(turn_q)

        if valid_mask.any():
            valid_positions = positions[valid_mask]

            peak_local = int(
                np.nanargmax(turn_q)
            )

            peak_pos = int(
                positions[peak_local]
            )

            peak_q = float(
                q[peak_pos]
            )

        else:
            valid_positions = []
            peak_pos = start_pos
            peak_q = np.nan

        submission_positions = [
            int(p)
            for p in positions
            if step_types[int(p)] == "submission"
        ]

        if submission_positions:
            action_pos = submission_positions[-1]
            has_submission_anchor = True

        else:
            action_pos = end_pos
            has_submission_anchor = False

        action = actions.get(
            ti,
            {},
        )

        summaries.append({
            "turn": ti,

            "start_pos": start_pos,
            "peak_pos": peak_pos,
            "action_pos": action_pos,
            "end_pos": end_pos,

            "start_q": float(q[start_pos])
            if not np.isnan(q[start_pos])
            else np.nan,

            "peak_q": peak_q,

            "action_q": float(q[action_pos])
            if not np.isnan(q[action_pos])
            else np.nan,

            "start_exact": int(
                exact[start_pos]
            ),

            "action_exact": int(
                exact[action_pos]
            ),

            "has_submission_anchor":
                has_submission_anchor,

            "guess":
                action.get("guess"),

            "status":
                action.get("status"),

            "feedback":
                action.get("feedback"),

            "category":
                action.get("category"),
        })

    return summaries


# ---------------------------------------------------------------------
# MACRO PLOT
# ---------------------------------------------------------------------

def plot_macro(
    run_id,
    steps,
    meta,
    gold,
    actions,
    out,
):
    (
        words,
        gmap,
        q,
        exact,
        mat,
        turn_idx,
        step_types,
    ) = build(
        steps,
        gold,
    )

    summaries = summarize_turns(
        q,
        exact,
        turn_idx,
        step_types,
        actions,
    )

    T = len(steps)
    x = np.arange(T)

    valid_q = np.where(
        np.isnan(q),
        -np.inf,
        q,
    )

    peak_t = int(
        np.argmax(valid_q)
    )

    turn_starts = [
        s["start_pos"]
        for s in summaries
    ]

    submission_steps = [
        i
        for i, typ in enumerate(step_types)
        if typ == "submission"
    ]

    reasoning_steps = [
        i
        for i, typ in enumerate(step_types)
        if typ == "reasoning"
    ]

    turn_start_steps = [
        i
        for i, typ in enumerate(step_types)
        if typ == "turn_start"
    ]

    figsize_w = max(
        12,
        min(
            34,
            T * 0.035,
        ),
    )

    fig, (ax_a, ax_b) = plt.subplots(
        2,
        1,
        figsize=(figsize_w, 10.2),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.15, 1.4],
            "hspace": 0.08,
        },
    )

    # --------------------------------------------------------------
    # global q_t
    # --------------------------------------------------------------

    ax_a.plot(
        x,
        q,
        "-",
        color="#2e4057",
        lw=1.5,
        zorder=3,
        label="$q_t$ (pairwise F1)",
    )

    ax_a.fill_between(
        x,
        q,
        color="#2e4057",
        alpha=0.06,
        zorder=1,
    )

    # reasoning candidate markers
    if reasoning_steps:
        ax_a.scatter(
            reasoning_steps,
            q[reasoning_steps],
            s=7,
            color="#2e4057",
            alpha=0.27,
            zorder=4,
        )

    # turn start markers
    if turn_start_steps:
        ax_a.scatter(
            turn_start_steps,
            q[turn_start_steps],
            marker="o",
            s=32,
            facecolors="white",
            edgecolors="#6c757d",
            linewidths=0.9,
            zorder=5,
            label="turn start",
        )

    # official submission markers
    if submission_steps:
        ax_a.scatter(
            submission_steps,
            q[submission_steps],
            marker="D",
            s=38,
            color="#7a4a78",
            zorder=7,
            label="official submission",
        )

    # --------------------------------------------------------------
    # annotate actual submitted guesses
    # --------------------------------------------------------------

    for j, s in enumerate(summaries):

        pos = s["action_pos"]

        guess = s["guess"]
        status = normalize_status(
            s["status"]
        )

        if not guess:
            continue

        text = compact_guess(guess)

        if status:
            text += f"\n→ {status}"

        # alternate above/below to reduce collisions
        if j % 2 == 0:
            offset_y = 38
            va = "bottom"

        else:
            offset_y = -42
            va = "top"

        marker = (
            "D"
            if s["has_submission_anchor"]
            else "x"
        )

        # If there was a raw logged guess but no Tier3
        # submission anchor, still show where the turn ended.
        if not s["has_submission_anchor"]:
            ax_a.scatter(
                [pos],
                [q[pos]],
                marker=marker,
                s=38,
                color="#7a4a78",
                zorder=7,
            )

        ax_a.annotate(
            text,
            xy=(
                pos,
                q[pos],
            ),
            xytext=(
                0,
                offset_y,
            ),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=6.4,
            family="monospace",
            linespacing=1.1,
            bbox={
                "boxstyle": "round,pad=0.25",
                "fc": "white",
                "ec": "#7a4a78",
                "alpha": 0.90,
                "lw": 0.6,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "#7a4a78",
                "lw": 0.55,
                "alpha": 0.7,
            },
            zorder=10,
        )

    # --------------------------------------------------------------
    # peak + final
    # --------------------------------------------------------------

    peak_val = meta.get(
        "m4_peak",
        q[peak_t],
    )

    ax_a.scatter(
        [peak_t],
        [q[peak_t]],
        s=140,
        marker="*",
        color="#2e8b57",
        zorder=11,
        label=f"peak {fmt(peak_val)}",
    )

    final_q = meta.get(
        "m4_final",
        q[-1],
    )

    ax_a.scatter(
        [T - 1],
        [q[-1]],
        s=58,
        color="#d1495b",
        zorder=11,
        label=f"final {fmt(final_q)}",
    )

    regret = meta.get(
        "m4_regret"
    )

    if (
        regret is not None
        and regret > 1e-6
    ):
        ax_a.annotate(
            f"M4 regret={fmt(regret)}",
            xy=(
                T - 1,
                q[-1],
            ),
            xytext=(-8, 14),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8.5,
            color="#d1495b",
        )

    # --------------------------------------------------------------
    # turn boundaries + incoming feedback
    # --------------------------------------------------------------

    for s in summaries:

        pos = s["start_pos"]
        ti = s["turn"]

        ax_a.axvline(
            pos - 0.5,
            color="#222222",
            lw=1.0,
            alpha=0.42,
            zorder=0,
        )

        ax_b.axvline(
            pos - 0.5,
            color="white",
            lw=1.4,
            alpha=0.95,
            zorder=5,
        )

        if ti == summaries[0]["turn"]:
            label = f"T{ti}"

        else:
            previous = next(
                (
                    z
                    for z in summaries
                    if z["turn"] == ti - 1
                ),
                None,
            )

            prev_status = (
                normalize_status(
                    previous["status"]
                )
                if previous
                else None
            )

            label = f"T{ti}"

            if prev_status:
                label += (
                    f"\n← {prev_status}"
                )

        ax_a.text(
            pos,
            1.055,
            label,
            ha="left",
            va="bottom",
            fontsize=7,
            clip_on=False,
        )

    # dashed submission lines
    for pos in submission_steps:

        ax_a.axvline(
            pos,
            ls="--",
            color="#7a4a78",
            lw=0.75,
            alpha=0.38,
            zorder=0,
        )

    # --------------------------------------------------------------
    # exact groups on secondary axis
    # --------------------------------------------------------------

    ax_r = ax_a.twinx()

    ax_r.step(
        x,
        exact,
        where="post",
        color="#edae49",
        lw=1.05,
        alpha=0.80,
    )

    ax_r.set_ylim(
        -0.15,
        4.3,
    )

    ax_r.set_yticks(
        range(5)
    )

    ax_r.set_ylabel(
        "exact groups",
        color="#c8902a",
    )

    ax_r.tick_params(
        axis="y",
        colors="#c8902a",
    )

    ax_a.set_ylim(
        -0.03,
        1.05,
    )

    ax_a.set_ylabel(
        "pairwise F1 vs gold"
    )

    ax_a.grid(
        axis="y",
        ls=":",
        alpha=0.35,
    )

    # --------------------------------------------------------------
    # title
    # --------------------------------------------------------------

    solved = meta.get(
        "db_solved"
    )

    state = (
        "SOLVED"
        if solved
        else "UNSOLVED"
    )

    model = meta.get(
        "model",
        "?",
    )

    puzzle_id = meta.get(
        "puzzle_id",
        "?",
    )

    title1 = (
        f"{run_id}  |  "
        f"{model}  |  "
        f"{puzzle_id}  |  "
        f"{state}"
    )

    title2 = (
        f"M1 churn={fmt(meta.get('m1_churn'), 2)}"
        f"   lock-in={fmt(meta.get('m1_lockin_mean'), 2)}"
        f"   |   "
        f"M2 blast×duration={fmt(meta.get('m2_blast_duration'), 2)}"
        f"   |   "
        f"M3 Δstruct={fmt(meta.get('m3_dstruct'), 2)}"
        f"   |   "
        f"M4 peak={fmt(meta.get('m4_peak'))}"
        f" final={fmt(meta.get('m4_final'))}"
        f" regret={fmt(meta.get('m4_regret'))}"
    )

    ax_a.set_title(
        title1
        + "\n"
        + title2,
        fontsize=9.5,
        loc="left",
        pad=18,
    )

    ax_a.legend(
        loc="lower right",
        fontsize=7.4,
        framealpha=0.90,
        ncol=2,
    )

    # --------------------------------------------------------------
    # heatmap
    # --------------------------------------------------------------

    ax_b.imshow(
        mat,
        aspect="auto",
        cmap=CMAP,
        norm=NORM,
        interpolation="nearest",
    )

    ax_b.set_yticks(
        range(len(words))
    )

    ax_b.set_yticklabels(
        words,
        fontsize=7.5,
        family="monospace",
    )

    # gold-group separators
    for k in range(
        1,
        len(gold),
    ):
        ax_b.axhline(
            4 * k - 0.5,
            color="white",
            lw=2.2,
        )

    block_colors = [
        "#3a5a78",
        "#7a4a78",
        "#4a784a",
        "#9a6a3a",
    ]

    for r, w in enumerate(words):

        ax_b.get_yticklabels()[r].set_color(
            block_colors[
                gmap[w] % len(block_colors)
            ]
        )

    ax_b.set_xlabel(
        "global reasoning state  $t$"
    )

    ax_b.set_ylabel(
        "word  (grouped by gold)"
    )

    ax_b.grid(False)

    heat_legend = [
        Patch(
            facecolor=STATE[i][1],
            edgecolor="#999",
            label=STATE[i][0],
        )
        for i in range(5)
    ]

    ax_b.legend(
        handles=heat_legend,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.10,
        ),
        ncol=5,
        fontsize=7.5,
        frameon=False,
    )

    fig.savefig(
        out,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"wrote {out} "
        f"({T} states, "
        f"{len(summaries)} turns)"
    )


# ---------------------------------------------------------------------
# TURN-ACTION PLOT
# ---------------------------------------------------------------------

def plot_turn_action(
    run_id,
    steps,
    meta,
    gold,
    actions,
    out,
):
    (
        words,
        gmap,
        q,
        exact,
        mat,
        turn_idx,
        step_types,
    ) = build(
        steps,
        gold,
    )

    summaries = summarize_turns(
        q,
        exact,
        turn_idx,
        step_types,
        actions,
    )

    n = len(summaries)

    if not n:
        raise SystemExit(
            "No turns to plot."
        )

    # each turn gets:
    #
    #       start   peak   action
    #        t-.2    t      t+.2
    #
    base_x = np.arange(n)

    x_start = base_x - 0.20
    x_peak = base_x
    x_action = base_x + 0.20

    q_start = np.array(
        [s["start_q"] for s in summaries],
        dtype=float,
    )

    q_peak = np.array(
        [s["peak_q"] for s in summaries],
        dtype=float,
    )

    q_action = np.array(
        [s["action_q"] for s in summaries],
        dtype=float,
    )

    # Width scales mildly with number of turns.
    fig_w = max(
        12,
        min(
            26,
            1.55 * n,
        ),
    )

    fig, (ax, ax_actions) = plt.subplots(
        2,
        1,
        figsize=(fig_w, 8.6),
        gridspec_kw={
            "height_ratios": [
                1.65,
                1.15,
            ],
            "hspace": 0.08,
        },
    )

    # --------------------------------------------------------------
    # within-turn start -> peak -> action
    # --------------------------------------------------------------

    for i, s in enumerate(summaries):

        xs = [
            x_start[i],
            x_peak[i],
            x_action[i],
        ]

        ys = [
            q_start[i],
            q_peak[i],
            q_action[i],
        ]

        ax.plot(
            xs,
            ys,
            "-",
            color="#6c757d",
            lw=1.25,
            alpha=0.70,
            zorder=2,
        )

    # start markers
    ax.scatter(
        x_start,
        q_start,
        marker="o",
        s=54,
        facecolors="white",
        edgecolors="#2e4057",
        linewidths=1.2,
        zorder=5,
        label="$q_{start}$",
    )

    # peak markers
    ax.scatter(
        x_peak,
        q_peak,
        marker="*",
        s=125,
        color="#2e8b57",
        zorder=6,
        label="$q_{peak}$",
    )

    # action/submission markers
    ax.scatter(
        x_action,
        q_action,
        marker="D",
        s=50,
        color="#7a4a78",
        zorder=6,
        label="$q_{action}$ / submission",
    )

    # --------------------------------------------------------------
    # feedback transition:
    #
    # submission/action of turn t
    #       ->
    # start state of turn t+1
    # --------------------------------------------------------------

    for i in range(n - 1):

        ax.plot(
            [
                x_action[i],
                x_start[i + 1],
            ],
            [
                q_action[i],
                q_start[i + 1],
            ],
            ":",
            color="#d1495b",
            lw=1.15,
            alpha=0.70,
            zorder=1,
        )

    # feedback labels between turns
    for i in range(n - 1):

        status = normalize_status(
            summaries[i]["status"]
        )

        if not status:
            continue

        xm = (
            x_action[i]
            + x_start[i + 1]
        ) / 2

        ym = (
            q_action[i]
            + q_start[i + 1]
        ) / 2

        ax.annotate(
            status,
            xy=(xm, ym),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#d1495b",
            fontweight="bold",
        )

    # --------------------------------------------------------------
    # q labels for each turn
    # --------------------------------------------------------------

    for i, s in enumerate(summaries):

        ax.annotate(
            fmt(s["peak_q"]),
            xy=(
                x_peak[i],
                q_peak[i],
            ),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#2e8b57",
        )

    ax.set_ylim(
        -0.03,
        1.06,
    )

    ax.set_xlim(
        -0.55,
        n - 0.45,
    )

    ax.set_ylabel(
        "pairwise F1 vs gold"
    )

    ax.set_xticks(
        base_x
    )

    ax.set_xticklabels(
        [
            f"T{s['turn']}"
            for s in summaries
        ],
        fontsize=8,
    )

    ax.grid(
        axis="y",
        ls=":",
        alpha=0.35,
    )

    ax.legend(
        loc="lower right",
        fontsize=8,
        ncol=3,
        framealpha=0.9,
    )

    # --------------------------------------------------------------
    # title
    # --------------------------------------------------------------

    solved = meta.get(
        "db_solved"
    )

    state = (
        "SOLVED"
        if solved
        else "UNSOLVED"
    )

    model = meta.get(
        "model",
        "?",
    )

    puzzle_id = meta.get(
        "puzzle_id",
        "?",
    )

    title1 = (
        f"Turn-action trajectory  |  "
        f"{model}  |  "
        f"{puzzle_id}  |  "
        f"{state}"
    )

    title2 = (
        f"M1 churn={fmt(meta.get('m1_churn'), 2)}"
        f"   |   "
        f"M2 blast×duration={fmt(meta.get('m2_blast_duration'), 2)}"
        f"   |   "
        f"M3 Δstruct={fmt(meta.get('m3_dstruct'), 2)}"
        f"   |   "
        f"M4 peak={fmt(meta.get('m4_peak'))}"
        f" → final={fmt(meta.get('m4_final'))}"
        f"   regret={fmt(meta.get('m4_regret'))}"
    )

    ax.set_title(
        title1
        + "\n"
        + title2,
        fontsize=10,
        loc="left",
        pad=12,
    )

    # --------------------------------------------------------------
    # action strip
    # --------------------------------------------------------------

    ax_actions.set_xlim(
        -0.55,
        n - 0.45,
    )

    ax_actions.set_ylim(
        0,
        1,
    )

    ax_actions.axis(
        "off"
    )

    # explanatory headings
    ax_actions.text(
        -0.48,
        0.92,
        "OFFICIAL ACTIONS",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
    )

    # one action card per turn
    for i, s in enumerate(summaries):

        guess = s["guess"]

        status = normalize_status(
            s["status"]
        )

        category = s["category"]

        if guess:
            guess_text = compact_guess(
                guess
            )

        else:
            guess_text = (
                "NO PARSED\nOFFICIAL GUESS"
            )

        if not status:
            status = "NO STATUS"

        # Main action card
        ax_actions.text(
            i,
            0.58,
            guess_text,
            ha="center",
            va="center",
            fontsize=7.3,
            family="monospace",
            linespacing=1.25,
            bbox={
                "boxstyle": "round,pad=0.38",
                "fc": "#f8f8f8",
                "ec": "#7a4a78",
                "lw": 0.8,
            },
        )

        ax_actions.text(
            i,
            0.20,
            status,
            ha="center",
            va="center",
            fontsize=7.1,
            fontweight="bold",
            color=(
                "#2e8b57"
                if status == "CORRECT"
                else "#d1495b"
                if status in {
                    "INCORRECT",
                    "REPEATED",
                    "INVALID",
                }
                else "#c8902a"
            ),
        )

        ax_actions.text(
            i,
            0.88,
            f"T{s['turn']}",
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
        )

    # arrows between action cards
    for i in range(n - 1):

        ax_actions.annotate(
            "",
            xy=(
                i + 0.63,
                0.58,
            ),
            xytext=(
                i + 0.37,
                0.58,
            ),
            arrowprops={
                "arrowstyle": "->",
                "lw": 0.7,
                "color": "#999999",
            },
        )

    fig.savefig(
        out,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"wrote {out} "
        f"({n} turn-action summaries)"
    )


# ---------------------------------------------------------------------
# output filename helper
# ---------------------------------------------------------------------

def derive_turn_output(
    macro_out,
):
    stem, ext = os.path.splitext(
        macro_out
    )

    if not ext:
        ext = ".png"

    return (
        stem
        + "_turn_action"
        + ext
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--metrics-db",
        required=True,
    )

    ap.add_argument(
        "--source-db",
        required=True,
    )

    ap.add_argument(
        "--run-id",
    )

    ap.add_argument(
        "--out",
        default="multiturn_trajectory.png",
        help=(
            "macro plot output. "
            "Turn-action filename is derived automatically."
        ),
    )

    ap.add_argument(
        "--turn-out",
        default=None,
        help=(
            "optional explicit output path "
            "for turn-action plot"
        ),
    )

    ap.add_argument(
        "--list",
        action="store_true",
    )

    ap.add_argument(
        "--model-filter",
        default=None,
        help=(
            "optional substring filter when using --list"
        ),
    )

    a = ap.parse_args()

    # --------------------------------------------------------------
    # list runs
    # --------------------------------------------------------------

    if a.list:

        c = sqlite3.connect(
            f"file:{a.metrics_db}?mode=ro",
            uri=True,
        )

        q = """
            SELECT
                run_id,
                model,
                puzzle_id,
                n_turns,
                db_solved,
                m1_churn,
                m4_peak,
                m4_final,
                m4_regret
            FROM run_metrics
        """

        params = []

        if a.model_filter:
            q += """
                WHERE model LIKE ?
            """

            params.append(
                f"%{a.model_filter}%"
            )

        q += """
            ORDER BY model, puzzle_id
        """

        try:

            for row in c.execute(
                q,
                params,
            ):

                print(
                    "\t".join(
                        ""
                        if x is None
                        else str(x)
                        for x in row
                    )
                )

        except BrokenPipeError:
            pass

        finally:
            c.close()

        return

    # --------------------------------------------------------------
    # plot one run
    # --------------------------------------------------------------

    if not a.run_id:
        raise SystemExit(
            "Need --run-id or --list"
        )

    (
        steps,
        meta,
        gold,
        actions,
        gold_col,
    ) = load(
        a.metrics_db,
        a.source_db,
        a.run_id,
    )

    print(
        f"gold groups loaded from "
        f"puzzles.{gold_col}"
    )

    macro_out = a.out

    turn_out = (
        a.turn_out
        if a.turn_out
        else derive_turn_output(
            macro_out
        )
    )

    plot_macro(
        a.run_id,
        steps,
        meta,
        gold,
        actions,
        macro_out,
    )

    plot_turn_action(
        a.run_id,
        steps,
        meta,
        gold,
        actions,
        turn_out,
    )


if __name__ == "__main__":
    main()
