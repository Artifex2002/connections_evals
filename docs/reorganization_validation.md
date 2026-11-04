# Organization-pass validation

Compared with the original `aabbc94` snapshot:

- All 15 archived files retain their original bytes.
- All relocated Python implementations retain the original syntax-tree structure after normalizing import paths and extracting existing CLI guard bodies into `main()` functions. The model-comparison usage example was also updated to its new module path.
- All active Python files parse successfully. Historical snapshots are excluded because they include a pre-existing broken logger diagnostic.
- `python -m pytest -q`: **14 passed**, covering the six existing game checks, all four policy help routes, and extraction-to-metrics SQLite integration/resume behavior for all four policies.
- On the two-turn synthetic fixture, the original and relocated implementations produced exactly equal rows in `step_cache`, `turn_metrics`, and `run_metrics` for every policy.
- Editable installation and six lightweight console-command help routes succeeded: one-game collection, batch collection, extraction, metric selection, model comparison, and trajectory plotting.
- The aggregate plotter generated its four PNG outputs from the synthetic metrics database.
- All five active Slurm launchers passed `bash -n`.

Validation used Python 3.12 in a temporary environment. It did not run GPU inference, the Hugging Face classifier, cluster jobs, or the original full research datasets. Dependency ranges in `pyproject.toml` are installation metadata, not a recovered historical experiment environment.

Existing research behavior and schema limitations are documented in [the pipeline](pipeline.md), [policy notes](reconstruction_policies.md), and [schema notes](data_schema.md). No paper results were regenerated or reassigned to a policy during this pass.
