# Multi-turn workflow

Run these commands from the repository root after installing the appropriate extras. Input databases and model weights are not included. Replace `model_runs.db` and the classifier model path with actual inputs. Use separate outputs for each reconstruction policy.

## Collection

With a compatible existing puzzle/logging database and a running inference server:

```bash
connections-run-one \
  --puzzle-id nyt_2024-06-03 \
  --database data/db/connections.db \
  --model YOUR_SERVED_MODEL \
  --base-url http://127.0.0.1:8000/v1
```

For batches, inspect `connections-run-batch --help` or use the launchers under `scripts/slurm/`. See [schema notes](../../docs/data_schema.md) before trying fresh collection.

## Tier 1–3 analysis

```bash
mkdir -p data/db

connections-extract \
  --src data/db/model_runs.db \
  --out data/db/tier1_multiturn_all.db

connections-label \
  --tier1-db data/db/tier1_multiturn_all.db \
  --out data/db/tier2_v5_all.db \
  --classifier hf \
  --model /path/to/Qwen2.5-7B-Instruct

connections-metrics --policy previous_guess \
  --tier1-db data/db/tier1_multiturn_all.db \
  --tier2-db data/db/tier2_v5_all.db \
  --source-db data/db/model_runs.db \
  --out data/db/metrics_previous_guess.db
```

The policy is an explicit example, not a declaration of the paper's canonical implementation. Consult [policy differences](../../docs/reconstruction_policies.md). Tier 1 accepts one source database per invocation and can accumulate completed runs into the same output. Tier 3 accepts repeated `--source-db` arguments.

## Tier 4 figures

The aggregate plotting command expects the `previous_guess` output schema:

```bash
connections-compare \
  --metrics-db data/db/metrics_previous_guess.db \
  --source-db data/db/model_runs.db \
  --scope common \
  --outdir results/generated/multiturn_common

connections-plot \
  --metrics-db data/db/metrics_previous_guess.db \
  --source-db data/db/model_runs.db \
  --list
```

To render a listed run, replace `--list` with `--run-id RUN_ID --out results/generated/trajectory.png`. Use `--scope all` for descriptive within-model aggregate plots; preserve the scope and puzzle coverage in the result manifest. Bulk generated outputs are local; copy deliberately selected figures to `results/figures/` for versioning.
