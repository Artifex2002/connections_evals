"""Exercise extraction, policy dispatch, and SQLite outputs without a model server."""

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from test_connections_game import GROUPS, WORDS


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("solved_only", "previous_guess", "a_plus_h_v1", "a_plus_h_v2")


def run_module(module, *arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(ROOT / "src"), environment.get("PYTHONPATH")])
    )
    return subprocess.run(
        [sys.executable, "-m", module, *map(str, arguments)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )


@pytest.fixture
def analysis_inputs(tmp_path):
    """Two official turns, with explicit candidates and known stance annotations."""
    raw = tmp_path / "raw.db"
    tier1 = tmp_path / "tier1.db"
    tier2 = tmp_path / "tier2.db"
    gold = [sorted(group.words) for group in GROUPS]
    with sqlite3.connect(raw) as connection:
        connection.executescript("""
            CREATE TABLE puzzles (
                puzzle_id TEXT PRIMARY KEY, words_json TEXT, ground_truth_json TEXT
            );
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY, puzzle_id TEXT, model_name TEXT,
                status TEXT, solved INTEGER, binary_score INTEGER,
                guesses_used INTEGER, mistakes_used INTEGER
            );
            CREATE TABLE turns (
                run_id TEXT, turn_id TEXT, turn_index INTEGER,
                reasoning_trace TEXT, visible_response TEXT,
                reasoning_extraction_method TEXT, guess_status TEXT,
                parse_success INTEGER, parsed_guess_json TEXT,
                feedback_returned TEXT, remaining_words_before_json TEXT,
                solved_group_indices_before_json TEXT,
                solved_group_indices_after_json TEXT
            );
        """)
        connection.execute("INSERT INTO puzzles VALUES (?, ?, ?)", (
            "fixture", json.dumps(WORDS), json.dumps(gold),
        ))
        connection.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
            "run", "fixture", "fixture-model", "completed", 0, 0, 2, 1,
        ))
        for turn_index in range(2):
            guess = gold[0] if turn_index == 0 else gold[1][:3] + [gold[2][0]]
            reasoning = "\n".join(
                ", ".join(group) + " form a group."
                for group in (gold if turn_index == 0 else [guess])
            )
            connection.execute("INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                "run", f"turn-{turn_index}", turn_index, reasoning, "",
                "think_tags", "correct" if turn_index == 0 else "one_away", 1,
                json.dumps(guess), "Correct" if turn_index == 0 else "One word away",
                json.dumps(WORDS if turn_index == 0 else [w for w in WORDS if w not in gold[0]]),
                json.dumps([] if turn_index == 0 else [0]), json.dumps([0]),
            ))
    run_module("connections_research.extraction.candidates", "--src", raw, "--out", tier1)
    # Human-specified labels isolate relocation/SQLite integration from model inference.
    with sqlite3.connect(tier1) as source, sqlite3.connect(tier2) as target:
        target.execute("""CREATE TABLE tier2_labels (
            run_id TEXT, puzzle_id TEXT, turn_id TEXT, turn_index INTEGER,
            model TEXT, candidate_index INTEGER, text_source TEXT,
            label TEXT, raw_response TEXT
        )""")
        rows = source.execute("""SELECT run_id, puzzle_id, turn_id, turn_index,
            model, candidate_index, text_source FROM tier1_candidates""").fetchall()
        assert len(rows) == 5
        target.executemany("INSERT INTO tier2_labels VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           [(*row, "propose", "fixture annotation") for row in rows])
    return raw, tier1, tier2


@pytest.mark.parametrize("policy", POLICIES)
def test_pipeline_scores_and_resumes(policy, analysis_inputs, tmp_path):
    raw, tier1, tier2 = analysis_inputs
    output = tmp_path / f"{policy}.db"
    args = ("--policy", policy, "--tier1-db", tier1, "--tier2-db", tier2,
            "--source-db", raw, "--out", output)
    run_module("connections_research.metrics.compute", *args)
    with sqlite3.connect(output) as connection:
        scores = connection.execute(
            "SELECT m4_peak, m4_final FROM run_metrics WHERE run_id = 'run'"
        ).fetchone()
        assert scores is not None
        assert scores[0] == pytest.approx(1.0)
        assert 0 < scores[1] < scores[0]
        assert connection.execute("SELECT count(*) FROM turn_metrics").fetchone()[0] == 2
        step_count = connection.execute("SELECT count(*) FROM step_cache").fetchone()[0]
        assert step_count >= 9
    run_module("connections_research.metrics.compute", *args)
    with sqlite3.connect(output) as connection:
        assert connection.execute("SELECT count(*) FROM run_metrics").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM step_cache").fetchone()[0] == step_count


@pytest.mark.parametrize("policy", POLICIES)
def test_policy_help_dispatch(policy):
    result = run_module("connections_research.metrics.compute", "--policy", policy, "--help")
    assert "--source-db" in result.stdout
    assert "--tier2-db" in result.stdout
