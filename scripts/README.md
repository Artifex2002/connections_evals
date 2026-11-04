# Operational scripts

Run Python scripts from the repository root after installing the package, or set `PYTHONPATH=src` for source-tree execution.

- `data/download_and_inspect.py`: downloads the Hugging Face dataset; requires the `data` extra.
- `data/normalize_dataset.py`: writes normalized JSONL and a validation report.
- `data/import_puzzles.py`: imports the normalized puzzle records into SQLite.
- `migrations/create_trajectory_schema.py`: retained historical initializer; read [schema limitations](../docs/data_schema.md) before use.
- `diagnostics/play_manual.py`: manually play a puzzle from the local database.
- `diagnostics/nemotron_transformers.py`: model-specific GPU diagnostic; requires the `hf` extra and model weights.
- `slurm/`: existing cluster launch scripts with updated module paths. They export the repository's `src/` on `PYTHONPATH`, so the existing cluster environment need not reinstall the package. `PROJECT_DIR` can be overridden; cluster/Conda/model paths and Slurm resource settings otherwise retain their original defaults.

Example Slurm submission from the repository root:

```bash
mkdir -p logs
sbatch scripts/slurm/run_structured_model_all.sh
```

That launcher requires `MODEL_ID`, `SERVED_MODEL`, `DATABASE_PATH`, and `CHECKPOINT_PATH` to be exported, as before. Review the script and the cluster environment before submitting a job.
