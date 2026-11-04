#!/usr/bin/env python3
"""
tier4_compare_multiturn_models_extended.py
==========================================

Extended model-comparison plots for the multiturn Connections setup.

What this script adds beyond the basic Tier-4 comparison:
  1. solved vs unsolved normalized trajectories by model
  2. pairwise model-delta heatmaps on matched puzzles
  3. violin plots for per-run metric distributions
  4. contamination / unassigned burden plots

Expected input DB:
  --metrics-db
      output of tier3_metrics_multiturn_singleshot.py
      with tables:
          run_metrics
          step_cache

Expected raw source DBs:
  --source-db ... (repeatable)
      the multiturn recovery DBs, used only to read:
          puzzles(puzzle_id, ground_truth_json)

Typical usage:
  python -m connections_research.plotting.compare_models \
      --metrics-db data/db/metrics_multiturn_singleshot_full.db \
      --source-db data/db/recovery_v1/dsr1_qwen32b.db \
      --source-db data/db/recovery_v1/dsr1_qwen7b.db \
      --source-db data/db/recovery_v1/qwen3_30b_a3b.db \
      --source-db data/db/recovery_v1/qwq32b.db \
      --scope common \
      --outdir multiturn_model_comparison_extended
"""

import argparse
import json
import math
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Naming helpers
# -----------------------------------------------------------------------------

def short_name(model):
    mapping = {
        "dsr1-qwen32b-structured-recovery-v1": "DS-R1-Qwen32B",
        "dsr1-qwen7b-structured-recovery-v1": "DS-R1-Qwen7B",
        "qwen3-30b-a3b-thinking-structured-recovery-v1": "Qwen3-30B-A3B",
        "qwq32b-structured-recovery-v1": "QwQ-32B",
    }
    return mapping.get(model, model)


# -----------------------------------------------------------------------------
# JSON / gold helpers
# -----------------------------------------------------------------------------

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
        if isinstance(g, (list, tuple)):
            out.append({str(w).strip().upper() for w in g if str(w).strip()})
    return out if len(out) >= 2 else None


def word_state(slot, w, gold_grp):
    """
    Same word-state semantics as the single-shot visualization:

      0 = unassigned
      1 = misassigned (no gold mate)
      2 = contaminated (mixed slot)
      3 = partial-correct (clean, <4)
      4 = exact gold group
    """
    if not slot:
        return 0

    slot = set(slot)
    others = slot - {w}
    mates = others & gold_grp
    wrong = others - gold_grp

    if slot == gold_grp:
        return 4
    if wrong:
        return 2 if mates else 1
    return 3 if mates else 1


def load_gold_map(source_dbs):
    """
    Build puzzle_id -> parsed gold groups from the raw source DBs.
    """
    gold_map = {}
    for db_path in source_dbs:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = conn.execute("""
                SELECT puzzle_id, ground_truth_json
                FROM puzzles
            """).fetchall()
        finally:
            conn.close()

        for puzzle_id, gt_json in rows:
            if puzzle_id in gold_map:
                continue
            groups = parse_groups(gt_json)
            if groups:
                gold_map[puzzle_id] = groups

    return gold_map


# -----------------------------------------------------------------------------
# Load run-level data
# -----------------------------------------------------------------------------

def load_run_metrics(conn):
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT
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
        FROM run_metrics
    """).fetchall()
    return [dict(r) for r in rows]


def rows_by_model(rows):
    out = {}
    for r in rows:
        out.setdefault(r["model"], []).append(r)
    return out


def compute_scope(rows, scope):
    """
    scope = all
        use all runs in run_metrics

    scope = common
        use only puzzles that appear for every model
    """
    by_model = rows_by_model(rows)
    models = sorted(by_model.keys())

    if scope == "all":
        selected = rows
        common_puzzles = None
    else:
        model_puzzle_sets = []
        for m in models:
            model_puzzle_sets.append({r["puzzle_id"] for r in by_model[m]})
        common_puzzles = set.intersection(*model_puzzle_sets) if model_puzzle_sets else set()

        selected = [r for r in rows if r["puzzle_id"] in common_puzzles]

    return selected, models, common_puzzles


def compute_model_order(selected_rows):
    """
    Order models by:
      1) mean m4_final desc
      2) solve rate desc
      3) mean m4_regret asc
      4) model name
    """
    by_model = rows_by_model(selected_rows)

    stats = []
    for model, rows in by_model.items():
        m4_final_vals = [float(r["m4_final"]) for r in rows if r["m4_final"] is not None]
        solved_vals = [int(r["db_solved"]) for r in rows if r["db_solved"] is not None]
        regret_vals = [float(r["m4_regret"]) for r in rows if r["m4_regret"] is not None]

        mean_m4_final = float(np.mean(m4_final_vals)) if m4_final_vals else float("-inf")
        solved_rate = float(np.mean(solved_vals)) if solved_vals else float("-inf")
        mean_m4_regret = float(np.mean(regret_vals)) if regret_vals else float("inf")

        stats.append((model, mean_m4_final, solved_rate, mean_m4_regret))

    stats.sort(key=lambda x: (-x[1], -x[2], x[3], x[0]))
    return [x[0] for x in stats]


# -----------------------------------------------------------------------------
# Load step-cache trajectories
# -----------------------------------------------------------------------------

def load_step_rows(conn, allowed_run_ids):
    """
    Load step_cache rows for the selected run_ids.

    We use the full trajectory order:
        global_step, then a stable within-step ordering.

    We keep all step types that have pair_f1 / partition state.
    """
    conn.row_factory = sqlite3.Row

    query = """
    SELECT
        s.run_id,
        r.model,
        r.puzzle_id,
        r.db_solved,
        s.turn_index,
        s.global_step,
        s.step_type,
        s.candidate_index,
        s.label,
        s.kept,
        s.well_formed,
        s.pair_f1,
        s.exact_hits,
        s.partition_json
    FROM step_cache s
    JOIN run_metrics r
      ON r.run_id = s.run_id
    ORDER BY
        r.model,
        s.run_id,
        s.global_step,
        CASE s.step_type
            WHEN 'turn_start' THEN 0
            WHEN 'reasoning' THEN 1
            WHEN 'submission' THEN 2
            ELSE 3
        END,
        COALESCE(s.candidate_index, -1)
    """

    trajectories = {}
    meta = {}

    for row in conn.execute(query):
        row = dict(row)
        run_id = row["run_id"]

        if run_id not in allowed_run_ids:
            continue

        model = row["model"]
        trajectories.setdefault(model, {}).setdefault(run_id, {
            "q": [],
            "exact": [],
            "partitions": [],
            "step_types": [],
            "turn_indices": [],
            "labels": [],
            "kept": [],
            "well_formed": [],
        })

        if row["pair_f1"] is not None:
            trajectories[model][run_id]["q"].append(float(row["pair_f1"]))
        else:
            trajectories[model][run_id]["q"].append(np.nan)

        if row["exact_hits"] is not None:
            trajectories[model][run_id]["exact"].append(float(row["exact_hits"]))
        else:
            trajectories[model][run_id]["exact"].append(np.nan)

        trajectories[model][run_id]["partitions"].append(row["partition_json"])
        trajectories[model][run_id]["step_types"].append(row["step_type"])
        trajectories[model][run_id]["turn_indices"].append(row["turn_index"])
        trajectories[model][run_id]["labels"].append(row["label"])
        trajectories[model][run_id]["kept"].append(row["kept"])
        trajectories[model][run_id]["well_formed"].append(row["well_formed"])

        meta[run_id] = {
            "model": model,
            "puzzle_id": row["puzzle_id"],
            "db_solved": int(row["db_solved"]) if row["db_solved"] is not None else 0,
        }

    return trajectories, meta


# -----------------------------------------------------------------------------
# Trajectory interpolation
# -----------------------------------------------------------------------------

def interpolate_series(values, grid):
    arr = np.asarray(values, dtype=float)

    mask = np.isfinite(arr)
    arr = arr[mask]

    if arr.size == 0:
        return None
    if arr.size == 1:
        return np.repeat(arr[0], grid.size)

    x_old = np.linspace(0.0, 1.0, arr.size)
    return np.interp(grid, x_old, arr)


def mean_curve(curves):
    return np.vstack(curves).mean(axis=0)


# -----------------------------------------------------------------------------
# Plot 1: solved vs unsolved normalized trajectories
# -----------------------------------------------------------------------------

def plot_solved_unsolved_trajectories(out_dir, scope_tag, model_order, trajectories, trace_meta):
    grid = np.linspace(0.0, 1.0, 101)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(model_order), 1)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    panels = [(1, "Solved Runs"), (0, "Unsolved Runs")]

    for ax, (solved_flag, title) in zip(axes, panels):
        for color, model in zip(colors, model_order):
            traces = trajectories.get(model, {})
            selected = []

            for run_id, payload in traces.items():
                if trace_meta[run_id]["db_solved"] != solved_flag:
                    continue
                curve = interpolate_series(payload["q"], grid)
                if curve is not None:
                    selected.append(curve)

            if not selected:
                continue

            ax.plot(
                grid,
                mean_curve(selected),
                lw=2.0,
                color=color,
                label=f"{short_name(model)} (n={len(selected)})"
            )

        ax.set_title(title)
        ax.set_xlabel("normalized trajectory progress")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", ls=":", alpha=0.4)

    axes[0].set_ylabel("pairwise F1 vs gold")
    axes[1].legend(loc="lower right", fontsize=8, ncol=1, framealpha=0.9)

    fig.suptitle(
        f"Solved vs Unsolved Reasoning Trajectories ({scope_tag} scope)",
        fontsize=14
    )
    fig.tight_layout()

    out_path = out_dir / f"03_solved_vs_unsolved_trajectories_{scope_tag}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


# -----------------------------------------------------------------------------
# Plot 2: pairwise delta heatmaps
# -----------------------------------------------------------------------------

def build_metric_matrix(metric_rows, model_order, metric_name):
    """
    Pair models on puzzle_id and compute mean(row - col).
    """
    vals = {m: {} for m in model_order}

    for row in metric_rows:
        model = row["model"]
        puzzle_id = row["puzzle_id"]
        val = row.get(metric_name, None)
        if model in vals and val is not None:
            vals[model][puzzle_id] = float(val)

    n = len(model_order)
    mat = np.full((n, n), np.nan, dtype=float)

    for i, mi in enumerate(model_order):
        for j, mj in enumerate(model_order):
            common = sorted(set(vals[mi]) & set(vals[mj]))
            if not common:
                continue
            diffs = [vals[mi][p] - vals[mj][p] for p in common]
            mat[i, j] = float(np.mean(diffs))

    return mat


def plot_pairwise_deltas(out_dir, scope_tag, model_order, metric_rows):
    names = [short_name(m) for m in model_order]

    metric_specs = [
        ("db_solved", "Pairwise Δ Solve (row - col)"),
        ("m4_final", "Pairwise Δ M4 Final (row - col)"),
        ("m4_regret", "Pairwise Δ M4 Regret (row - col)"),
        ("m1_churn", "Pairwise Δ M1 Churn (row - col)"),
        ("m2_blast_duration", "Pairwise Δ M2 Blast Duration (row - col)"),
        ("m3_dstruct", "Pairwise Δ M3 Structural Drift (row - col)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(15, 16))
    axes = axes.ravel()

    for ax, (metric, title) in zip(axes, metric_specs):
        mat = build_metric_matrix(metric_rows, model_order, metric)

        finite = mat[np.isfinite(mat)]
        vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
        if vmax == 0:
            vmax = 1.0

        im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax)

        ax.set_title(title)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right")
        ax.set_yticklabels(names)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                txt = "NA" if not np.isfinite(mat[i, j]) else f"{mat[i, j]:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Pairwise Model Delta Heatmaps ({scope_tag} scope, matched by puzzle)",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout()

    out_path = out_dir / f"04_pairwise_delta_heatmaps_{scope_tag}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


# -----------------------------------------------------------------------------
# Plot 3: violin distributions
# -----------------------------------------------------------------------------

def clean_violin_series(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if vals:
        return vals
    return [np.nan]


def add_violin(ax, data, labels, title):
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)

    if "cmeans" in parts:
        parts["cmeans"].set_color("#333333")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", ls=":", alpha=0.3)


def plot_violin_metrics(out_dir, scope_tag, model_order, metric_rows):
    rows_by_model_ = {m: [r for r in metric_rows if r["model"] == m] for m in model_order}
    names = [short_name(m) for m in model_order]

    metric_specs = [
        ("m4_final", "M4 Final"),
        ("m4_regret", "M4 Regret"),
        ("m1_churn", "M1 Churn"),
        ("m1_correct_abandonment", "M1 Correct Abandonment"),
        ("m2_blast_duration", "M2 Blast Duration"),
        ("m3_dstruct", "M3 Structural Drift"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(15, 14))

    for ax, (metric, title) in zip(axes.ravel(), metric_specs):
        data = []
        for m in model_order:
            values = [r[metric] for r in rows_by_model_[m]]
            data.append(clean_violin_series(values))
        add_violin(ax, data, names, title)

    fig.suptitle(
        f"Per-Run Metric Distributions ({scope_tag} scope)",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()

    out_path = out_dir / f"05_violin_metric_distributions_{scope_tag}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


# -----------------------------------------------------------------------------
# Plot 4: contamination / unassigned burdens
# -----------------------------------------------------------------------------

def compute_burdens(trajectories, trace_meta, gold_map):
    rows = []

    for model, runs in trajectories.items():
        for run_id, payload in runs.items():
            puzzle_id = trace_meta[run_id]["puzzle_id"]
            gold = gold_map.get(puzzle_id)

            if not gold:
                continue

            words = [w for g in gold for w in g]
            gmap = {w: i for i, g in enumerate(gold) for w in g}
            gset = [set(g) for g in gold]

            contaminated = 0
            unassigned = 0
            valid_states = 0

            for part_json in payload["partitions"]:
                if not part_json:
                    continue

                try:
                    partition = json.loads(part_json)
                except Exception:
                    continue

                if not isinstance(partition, list):
                    continue

                w2slot = {}
                for grp in partition:
                    if not isinstance(grp, (list, tuple)):
                        continue
                    gs = {str(w).strip().upper() for w in grp if str(w).strip()}
                    for w in gs:
                        w2slot[w] = gs

                for w in words:
                    state = word_state(w2slot.get(w), w, gset[gmap[w]])
                    if state == 2:
                        contaminated += 1
                    elif state == 0:
                        unassigned += 1

                valid_states += 1

            denom = max(16 * valid_states, 1)

            rows.append({
                "run_id": run_id,
                "model": model,
                "contamination_burden": contaminated / denom,
                "unassigned_burden": unassigned / denom,
            })

    return rows


def plot_burdens(out_dir, scope_tag, model_order, burden_rows):
    rows_by_model_ = {m: [r for r in burden_rows if r["model"] == m] for m in model_order}
    names = [short_name(m) for m in model_order]

    contam = [
        clean_violin_series([r["contamination_burden"] for r in rows_by_model_[m]])
        for m in model_order
    ]
    unassigned = [
        clean_violin_series([r["unassigned_burden"] for r in rows_by_model_[m]])
        for m in model_order
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    add_violin(axes[0], contam, names, "Contamination Burden")
    add_violin(axes[1], unassigned, names, "Unassigned Burden")

    fig.suptitle(
        f"Error Burden Across Models ({scope_tag} scope)",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()

    out_path = out_dir / f"06_contamination_unassigned_burden_{scope_tag}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


# -----------------------------------------------------------------------------
# Summary / manifest
# -----------------------------------------------------------------------------

def write_summary(out_dir, scope_tag, selected_rows, model_order):
    by_model = rows_by_model(selected_rows)
    lines = []
    header = [
        "model",
        "runs",
        "solve_rate",
        "mean_m1_churn",
        "mean_m2_blast_duration",
        "mean_m3_dstruct",
        "mean_m4_peak",
        "mean_m4_final",
        "mean_m4_regret",
    ]
    lines.append("\t".join(header))

    for model in model_order:
        rows = by_model[model]
        runs = len(rows)
        solve_rate = np.mean([r["db_solved"] for r in rows]) if rows else np.nan
        mean_m1 = np.mean([r["m1_churn"] for r in rows]) if rows else np.nan
        mean_m2 = np.mean([r["m2_blast_duration"] for r in rows]) if rows else np.nan
        mean_m3 = np.mean([r["m3_dstruct"] for r in rows]) if rows else np.nan
        mean_m4p = np.mean([r["m4_peak"] for r in rows]) if rows else np.nan
        mean_m4f = np.mean([r["m4_final"] for r in rows]) if rows else np.nan
        mean_m4r = np.mean([r["m4_regret"] for r in rows]) if rows else np.nan

        vals = [
            model,
            str(runs),
            f"{solve_rate:.4f}",
            f"{mean_m1:.4f}",
            f"{mean_m2:.4f}",
            f"{mean_m3:.4f}",
            f"{mean_m4p:.4f}",
            f"{mean_m4f:.4f}",
            f"{mean_m4r:.4f}",
        ]
        lines.append("\t".join(vals))

    out_path = out_dir / f"summary_{scope_tag}.tsv"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_manifest(out_dir, outputs, model_order, scope_tag, n_runs):
    manifest = {
        "scope": scope_tag,
        "n_runs": n_runs,
        "models": model_order,
        "plots": [str(p) for p in outputs],
    }
    path = out_dir / "extended_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--metrics-db", required=True)
    ap.add_argument(
        "--source-db",
        action="append",
        required=True,
        help="Repeatable. Raw recovery DBs used to read puzzles.ground_truth_json",
    )
    ap.add_argument(
        "--scope",
        choices=["all", "common"],
        default="common",
        help="all = use all runs; common = use only puzzles available for every model",
    )
    ap.add_argument(
        "--outdir", "--out-dir",
        dest="outdir",
        default="multiturn_model_comparison_extended",
    )

    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(f"file:{args.metrics_db}?mode=ro", uri=True)

    all_run_rows = load_run_metrics(conn)
    selected_rows, all_models, common_puzzles = compute_scope(all_run_rows, args.scope)
    model_order = compute_model_order(selected_rows)

    allowed_run_ids = {r["run_id"] for r in selected_rows}
    trajectories, trace_meta = load_step_rows(conn, allowed_run_ids)
    conn.close()

    gold_map = load_gold_map(args.source_db)
    burden_rows = compute_burdens(trajectories, trace_meta, gold_map)

    outputs = []
    outputs.append(
        plot_solved_unsolved_trajectories(
            out_dir, args.scope, model_order, trajectories, trace_meta
        )
    )
    outputs.append(
        plot_pairwise_deltas(
            out_dir, args.scope, model_order, selected_rows
        )
    )
    outputs.append(
        plot_violin_metrics(
            out_dir, args.scope, model_order, selected_rows
        )
    )
    outputs.append(
        plot_burdens(
            out_dir, args.scope, model_order, burden_rows
        )
    )

    summary_path = write_summary(out_dir, args.scope, selected_rows, model_order)
    manifest_path = write_manifest(out_dir, outputs, model_order, args.scope, len(selected_rows))

    print("=" * 80)
    print("MULTITURN EXTENDED MODEL COMPARISON")
    print("=" * 80)
    print(f"metrics db : {args.metrics_db}")
    print(f"scope      : {args.scope}")
    print(f"outdir     : {out_dir}")
    print(f"runs used  : {len(selected_rows)}")
    print(f"models     : {len(model_order)}")
    if args.scope == "common":
        print(f"common puzzles: {len(common_puzzles) if common_puzzles is not None else 0}")
    print()

    print("Model order:")
    for m in model_order:
        print(f"  - {m}  [{short_name(m)}]")

    print()
    print(f"wrote {summary_path}")
    print(f"wrote {manifest_path}")
    for path in outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
