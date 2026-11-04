# Paper reproduction inventory

This inventory describes available code, not verified regeneration of the draft. The original experiment databases and exact figure commands are absent.

| Draft figure | Available implementation | Remaining inputs/work |
| --- | --- | --- |
| 1: example single-shot/multi-turn trajectories | `plotting.trajectory` covers the multi-turn component | Original single-shot visualization, chosen run IDs, and combined layout |
| 2: feedback-conditioned dynamics | No dedicated reproduction script identified in this snapshot | Recover plotting/aggregation code and exact run selection |
| 3–5: single-shot summaries and comparisons | Original single-shot scripts absent | Recover original pipeline and data |
| 6: solved/unsolved multi-turn curves | `plotting.compare_models`, trajectory plot output | Metrics snapshot, all-source DBs, and `--scope all` |
| 7: matched multi-turn differences | `plotting.compare_models`, pairwise delta output | Exact scope, puzzle overlaps, metric snapshot, and source DBs |
| 8: per-run metric distributions | `plotting.compare_models`, violin output | Metrics snapshot, source DBs, and `--scope all` |

For each recovered figure, save the exact command, policy/version, code commit, input checksums, coverage, and output path in `results/manifests/`. Record verified selections under `configs/paper/`. Do not infer paper provenance from a filename containing `singleshot` or `full`.
