# Pipeline

The existing Tier 1–4 terminology remains useful for describing the research workflow.

1. **Prepare puzzles.** `scripts/data/download_and_inspect.py` downloads the source dataset, `normalize_dataset.py` validates and normalizes it, and `import_puzzles.py` imports puzzle records into SQLite.
2. **Collect interactions.** `runners.multiturn` plays a single game; `runners.batch` schedules puzzle runs. `game.environment` evaluates official submissions, `prompts.multiturn` builds the structured history, and `storage.trajectory_logger` records generations, actions, feedback, and recovery metadata.
3. **Tier 1: extract candidates.** `extraction.candidates` reads completed runs and extracts literal puzzle-word sets from each turn's reasoning, falling back to the visible response when needed. This stage does not classify stance.
4. **Tier 2: label stance.** `extraction.stance` classifies already-extracted spans. Its existing stub mode is a development tool, not a research-quality annotation source.
5. **Tier 3: reconstruct and score.** Choose an explicit implementation through `metrics.compute --policy NAME`. Each policy retains its original reconstruction rules and M1–M4 formulas. Never mix outputs from different policies into an unlabelled result database.
6. **Tier 4: visualize.** `plotting.trajectory` renders per-run dynamics; `plotting.compare_models` produces aggregate comparisons. Record the scope (`all` or `common`) and puzzle coverage.

## Collection details retained from the snapshot

- Subsequent model turns receive structured action/category/feedback history, not the previous full reasoning trace.
- Truncated or unparseable generations may receive an additional answer-recovery call.
- Repeated and invalid guesses consume no mistake; the model-turn cap still limits the run.
- Models explicitly submit the final remaining group; the active runner disables automatic completion.
- Data locations and decoding defaults were not changed as part of organization.

The single-shot collection pipeline is not present yet. Shared extraction/reconstruction logic is not evidence that the original single-shot experiments can already be reproduced from this repository.

## Module boundaries

The package now separates the game, model interface, prompts, storage, extraction, policy implementations, and plotting. The four large Tier 3 files still contain both reconstruction and metrics. Extracting common metric functions is a later refactor that should be checked against recorded outputs; this pass does not assume the implementations are interchangeable.
