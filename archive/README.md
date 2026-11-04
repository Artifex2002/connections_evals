# Historical snapshots

`snapshots/` preserves original old runners, prompt/client backups, Slurm backups, and scratch files byte-for-byte from commit `aabbc94`. Their original relative paths are retained within this directory.

These files are not installed, tested, or used by active workflows. Their imports and commands intentionally retain their historical form. Some use older APIs; `test_trajectory_logging.py` also contains a pre-existing syntax error. The original `run_model.py` is an unfinished multi-turn scaffold, not a single-shot runner.

Keep these snapshots until their experimental provenance has been checked. Older runner names may represent different memory protocols, rather than redundant copies of the current runner. See [the full migration inventory](../docs/file_migration.md).
