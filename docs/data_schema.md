# Data and schema notes

Large data remains local under `data/`, which is ignored by Git. This snapshot includes schema-producing code but no experiment databases.

| Database stage | Principal tables | Source |
| --- | --- | --- |
| Raw puzzle and interaction data | `puzzles`, `runs`, `turns` | Data import, collection, trajectory logging |
| Tier 1 | `tier1_candidates`, `tier1_turns`, `done_turns`, `tier1_run_meta` | `extraction.candidates` |
| Tier 2 | `tier2_labels`, `tier2_done`, `tier2_run_meta` | `extraction.stance` |
| Tier 3 | `run_metrics`, `turn_metrics`, `step_cache`, `processed_runs` and policy-dependent skip metadata | Selected reconstruction policy |

Puzzle identifiers connect runs to puzzle vocabulary and gold groups. Run and turn identifiers connect the extraction, stance, and metric databases. JSON columns store word lists, groups, settings, feedback-related state, and metadata. Keep original source databases immutable during analysis and record their checksums in result manifests.

## Existing schema limitations

`scripts/migrations/create_trajectory_schema.py` is the original historical initializer. It uses `CREATE TABLE IF NOT EXISTS`; it is not a complete migration system and does not upgrade existing tables. It predates logger fields such as `runs.auto_complete_final_group`, exposed-response metadata, and answer-recovery columns. Running it alone does not create a database compatible with every current logger operation.

Use a compatible existing research database for collection. Before supporting fresh collection installations, recover the actual schema/migrations from the experiment environment and add a verified migration path. This organization pass preserves the initializer rather than inventing an unverified experiment schema.

`game.puzzles` still loads from its historical `data/db/connections.db` path. The runner's `--database` option selects its logging database, not a new puzzle-repository path. When using a separate model-run database, preserve the expected puzzle source and ensure the logging database contains compatible puzzle records. This behavior predates the reorganization.

## Suggested local directories

```text
data/raw/           Original downloaded dataset
data/processed/     Normalized JSONL and validation reports
data/db/            Raw interactions and Tier 1–3 SQLite databases
checkpoints/        Resumable collection checkpoints
logs/               Cluster and collection logs
```

Exact column definitions remain in the owning modules; the notes above are an inventory, not a claim that all historical schemas are identical.
