# Connections research

Research code for measuring how language models discover, revise, preserve, and submit candidate solutions to NYT Connections puzzles.

This snapshot contains the **multi-turn collection and analysis pipeline**, including adaptations of the original single-shot metrics. The original single-shot collection and analysis project has not yet been added. In particular, the policy named `previous_guess` is a multi-turn adaptation, not a single-shot experiment runner.

## Interactive project showcase

Explore [Finding Is Not Acting](https://artifex2002.github.io/connections_evals/), an interactive research companion with trajectory replays, a demonstration puzzle, and paper-reported comparisons. The replay examples are explicitly illustrative pending verified experiment exports.

The showcase is maintained on [`codex/project-showcase`](https://github.com/Artifex2002/connections_evals/tree/codex/project-showcase) and published from `docs/` through GitHub Pages. See [the showcase README](docs/README.md) for local preview and data integration.

## Repository layout

```text
configs/                     Reference experiment settings and paper configuration notes
src/connections_research/
  game/                      Game environment and puzzle database access
  models/                    Inference client and generation/answer parsing
  prompts/                   Current structured-memory multi-turn prompts
  runners/                   One-game and batch multi-turn collection
  storage/                   SQLite trajectory logging
  extraction/                Tier 1 candidates and Tier 2 stance classification
  reconstruction/policies/   Four preserved reconstruction/metric implementations
  metrics/                   Explicit policy-selection CLI
  plotting/                  Tier 4 trajectory and model-comparison plots
scripts/
  data/                      Download, normalize, and import puzzles
  migrations/                Historical schema initializer
  diagnostics/               Manual play and model-backend diagnostics
  slurm/                     Cluster launch scripts
analysis/                    Multi-turn workflows and paper figure inventory
tests/                       Automated offline tests
docs/                        Pipeline, schema, policy, and migration documentation
data/                        Local datasets/databases; ignored by Git
results/                     Curated manifests, tables, and figures
archive/                     Original historical scripts and backup snapshots
```

## Setup

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

Install extras for the stages you need:

```bash
python -m pip install -e '.[inference]'   # OpenAI-compatible / vLLM client
python -m pip install -e '.[hf]'          # Hugging Face stance classifier and GPU diagnostic
python -m pip install -e '.[plots]'       # NumPy and Matplotlib plotting
python -m pip install -e '.[data]'        # Dataset download
```

The inference server (for example, vLLM) is managed separately in the cluster environment. Dependency ranges describe installation requirements; they are not a recovered lockfile for the original experiments. Record the actual environment for new runs.

## Entry points

| Command | Purpose |
| --- | --- |
| `connections-run-one --help` | Collect one multi-turn game |
| `connections-run-batch --help` | Collect a batch of games |
| `connections-extract --help` | Tier 1 lexical candidate extraction |
| `connections-label --help` | Tier 2 stance classification; requires the `hf` extra |
| `connections-metrics --help` | Select one of four preserved reconstruction policies |
| `connections-metrics --policy previous_guess --help` | Inspect a policy's original metric arguments |
| `connections-compare --help` | Aggregate model comparisons; requires `plots` |
| `connections-plot --help` | Individual trajectory visualization; requires `plots` |

Each implementation also supports `python -m connections_research.<module>`. Paths supplied on the command line, and historical default data paths, remain relative to the working directory. Run workflows from the repository root.

Start with [the multi-turn workflow](analysis/multiturn/README.md). Read [reconstruction policies](docs/reconstruction_policies.md) before choosing a metric implementation. There is deliberately no default policy in the new selector.

## Data and reproducibility

The original datasets, experiment databases, checkpoints, and full paper outputs are not included. The older schema initializer does not supply all columns required by the current logger; see [schema notes](docs/data_schema.md). Collection requires a compatible existing database and inference server.

`configs/` currently contains **reference settings**, not automatically loaded runtime configuration. The preserved CLIs and Slurm scripts take arguments as before. Store a completed copy of the [result manifest template](results/manifests/example.json) alongside each curated result to record its code revision, inputs, policy, environment, and commands.

For this organization pass, research algorithms, prompts, metric definitions, CLI defaults, and policy versions were preserved. Imports and launch paths were updated, and entry points were added. Policy-specific metrics remain together with their reconstruction implementation to avoid changing the research logic during relocation.

Old commands and files are mapped in [the migration inventory](docs/file_migration.md). Files in `archive/` are historical references, not supported entry points.

See [organization-pass validation](docs/reorganization_validation.md) for the offline checks and their limits.
