# File migration map

Paths in the original `aabbc94` snapshot map to the following locations.

| Original path | Current path |
| --- | --- |
| `0.0` | `archive/snapshots/0.0` |
| `import_puzzles.py` | `scripts/data/import_puzzles.py` |
| `inspect_dataset.py` | `scripts/data/download_and_inspect.py` |
| `migrate_trajectory_schema.py` | `scripts/migrations/create_trajectory_schema.py` |
| `normalize_dataset.py` | `scripts/data/normalize_dataset.py` |
| `play_manual.py` | `scripts/diagnostics/play_manual.py` |
| `run_all_puzzles.py` | `src/connections_research/runners/batch.py` |
| `run_all_puzzles_old.py` | `archive/snapshots/run_all_puzzles_old.py` |
| `run_model.py` | `archive/snapshots/run_model.py` |
| `run_one_puzzle.py` | `src/connections_research/runners/multiturn.py` |
| `run_one_puzzle.py.save` | `archive/snapshots/run_one_puzzle.py.save` |
| `run_one_puzzle.py.save.1` | `archive/snapshots/run_one_puzzle.py.save.1` |
| `run_one_puzzle_old.py` | `archive/snapshots/run_one_puzzle_old.py` |
| `run_one_puzzle_old_2.py` | `archive/snapshots/run_one_puzzle_old_2.py` |
| `slurm/run_nemotron_all.sh` | `scripts/slurm/run_nemotron_all.sh` |
| `slurm/run_nemotron_structured_memory_all.sh` | `scripts/slurm/run_nemotron_structured_memory_all.sh` |
| `slurm/run_structured_model_all.sh` | `scripts/slurm/run_structured_model_all.sh` |
| `slurm/run_structured_model_all.sh.bak_20260822` | `archive/snapshots/slurm/run_structured_model_all.sh.bak_20260822` |
| `slurm/run_structured_model_all.sh.bak_mp` | `archive/snapshots/slurm/run_structured_model_all.sh.bak_mp` |
| `slurm/run_structured_model_all.sh.save` | `archive/snapshots/slurm/run_structured_model_all.sh.save` |
| `slurm/run_tier2_v5.sh` | `scripts/slurm/run_tier2_v5.sh` |
| `slurm/run_tier2_v5_shard.sh` | `scripts/slurm/run_tier2_v5_shard.sh` |
| `src/connections_game.py` | `src/connections_research/game/environment.py` |
| `src/model_client.py` | `src/connections_research/models/client.py` |
| `src/model_client.py.bak` | `archive/snapshots/src/model_client.py.bak` |
| `src/model_output_parser.py` | `src/connections_research/models/output_parser.py` |
| `src/prompt_builder.py` | `src/connections_research/prompts/multiturn.py` |
| `src/prompt_builder_old.py` | `archive/snapshots/src/prompt_builder_old.py` |
| `src/puzzle_repository.py` | `src/connections_research/game/puzzles.py` |
| `src/trajectory_logger.py` | `src/connections_research/storage/trajectory_logger.py` |
| `test_nemotron_transformers.py` | `scripts/diagnostics/nemotron_transformers.py` |
| `test_trajectory_logging.py` | `archive/snapshots/test_trajectory_logging.py` |
| `tier1_scale_extract.py` | `src/connections_research/extraction/candidates.py` |
| `tier2_scale_hf_multiturn.py` | `src/connections_research/extraction/stance.py` |
| `tier3_metrics_multiturn.py` | `src/connections_research/reconstruction/policies/solved_only.py` |
| `tier3_metrics_multiturn_a_plus_h.py` | `src/connections_research/reconstruction/policies/a_plus_h_v1.py` |
| `tier3_metrics_multiturn_a_plus_h_v2.py` | `src/connections_research/reconstruction/policies/a_plus_h_v2.py` |
| `tier3_metrics_multiturn_singleshot.py` | `src/connections_research/reconstruction/policies/previous_guess.py` |
| `tier4_compare_multiturn_models.py` | `src/connections_research/plotting/compare_models.py` |
| `tier4_compare_multiturn_models.py.bak` | `archive/snapshots/tier4_compare_multiturn_models.py.bak` |
| `tier4_plot_multiturn_trajectory.py` | `src/connections_research/plotting/trajectory.py` |
| `turn2_prompt.txt` | `archive/snapshots/turn2_prompt.txt` |

Historical snapshots are retained byte-for-byte and are not supported entry points.
The empty original `src/__init__.py` was replaced by package initializers.
The existing game tests remain in `tests/` with their imports updated.
