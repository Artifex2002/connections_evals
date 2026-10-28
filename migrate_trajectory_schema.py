import sqlite3
from pathlib import Path


DATABASE_PATH = Path("data/db/connections.db")


CREATE_RUNS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,

    puzzle_id TEXT NOT NULL,
    model_provider TEXT NOT NULL,
    model_name TEXT NOT NULL,

    shuffle_seed INTEGER NOT NULL,
    max_mistakes INTEGER NOT NULL DEFAULT 4,

    decoding_settings_json TEXT NOT NULL DEFAULT '{}',
    system_prompt TEXT,
    run_metadata_json TEXT NOT NULL DEFAULT '{}',

    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,

    status TEXT NOT NULL DEFAULT 'running'
        CHECK (
            status IN (
                'running',
                'completed',
                'failed',
                'aborted'
            )
        ),

    solved INTEGER
        CHECK (solved IN (0, 1)),

    binary_score INTEGER
        CHECK (binary_score IN (0, 1)),

    guesses_used INTEGER,
    mistakes_used INTEGER,

    final_predicted_groups_json TEXT,
    final_remaining_words_json TEXT,
    failure_message TEXT,

    total_latency_ms REAL,
    total_prompt_tokens INTEGER,
    total_completion_tokens INTEGER,
    total_tokens INTEGER,

    FOREIGN KEY (puzzle_id)
        REFERENCES puzzles(puzzle_id)
        ON DELETE RESTRICT
);
"""


CREATE_TURNS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS turns (
    turn_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,

    turn_index INTEGER NOT NULL,

    prompt_shown TEXT NOT NULL,
    feedback_returned TEXT,

    reasoning_trace TEXT,
    raw_output TEXT NOT NULL,

    parsed_guess_json TEXT,
    proposed_category TEXT,

    parse_success INTEGER NOT NULL
        CHECK (parse_success IN (0, 1)),

    parse_error TEXT,

    guess_status TEXT
        CHECK (
            guess_status IS NULL
            OR guess_status IN (
                'correct',
                'one_away',
                'incorrect',
                'invalid',
                'repeated',
                'already_finished'
            )
        ),

    matched_group_index INTEGER,
    matched_category TEXT,

    remaining_words_before_json TEXT NOT NULL,
    remaining_words_after_json TEXT,

    solved_group_indices_before_json TEXT NOT NULL,
    solved_group_indices_after_json TEXT,

    mistakes_used_before INTEGER NOT NULL,
    mistakes_used_after INTEGER,

    mistakes_remaining_after INTEGER,

    game_over_after INTEGER
        CHECK (
            game_over_after IS NULL
            OR game_over_after IN (0, 1)
        ),

    solved_after INTEGER
        CHECK (
            solved_after IS NULL
            OR solved_after IN (0, 1)
        ),

    latency_ms REAL,

    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,

    provider_request_id TEXT,
    model_metadata_json TEXT NOT NULL DEFAULT '{}',

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (run_id)
        REFERENCES runs(run_id)
        ON DELETE CASCADE,

    UNIQUE (run_id, turn_index)
);
"""


CREATE_RUN_PUZZLE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_runs_puzzle_id
ON runs(puzzle_id);
"""


CREATE_RUN_MODEL_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_runs_model
ON runs(model_provider, model_name);
"""


CREATE_RUN_STATUS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_runs_status
ON runs(status);
"""


CREATE_TURNS_RUN_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_turns_run_id
ON turns(run_id);
"""


def main() -> None:
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Database does not exist: {DATABASE_PATH}"
        )

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON;")

        connection.execute(CREATE_RUNS_TABLE_SQL)
        connection.execute(CREATE_TURNS_TABLE_SQL)

        connection.execute(CREATE_RUN_PUZZLE_INDEX_SQL)
        connection.execute(CREATE_RUN_MODEL_INDEX_SQL)
        connection.execute(CREATE_RUN_STATUS_INDEX_SQL)
        connection.execute(CREATE_TURNS_RUN_INDEX_SQL)

        connection.commit()

        tables = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        ).fetchall()

    print("Trajectory schema created successfully.")
    print("Database:", DATABASE_PATH.resolve())
    print("Tables:")

    for table in tables:
        print("-", table[0])


if __name__ == "__main__":
    main()
