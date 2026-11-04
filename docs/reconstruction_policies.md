# Reconstruction policies

All four implementations were retained, with existing formulas, defaults, and policy-version strings. Their names describe the differences without implying that the newest implementation is the paper's canonical method.

| Selector | Original file | Turn-start state | Within-turn differences | Stored policy version |
| --- | --- | --- | --- | --- |
| `solved_only` | `tier3_metrics_multiturn.py` | Confirmed groups only | Original held-candidate accumulation | No explicit policy-version constant |
| `previous_guess` | `tier3_metrics_multiturn_singleshot.py` | Confirmed groups plus the immediately preceding unresolved guess | Single-shot-style accumulation; no solved-word stripping or historical special-casing | `multiturn_singleshot_v1` |
| `a_plus_h_v1` | `tier3_metrics_multiturn_a_plus_h.py` | Confirmed groups only, locked | Exact historical guesses do not mutate the global state; other candidates have solved words stripped | `a_plus_h_v1` |
| `a_plus_h_v2` | `tier3_metrics_multiturn_a_plus_h_v2.py` | Confirmed groups only, locked | Historical equality is metadata; a currently held historical guess may mutate state when it does not overlap solved words | `a_plus_h_v2_history_stance` |

Each implementation lives in `src/connections_research/reconstruction/policies/`. The metric functions remain with the policy they originally accompanied. See each module's documentation for detailed state and scoring semantics.

## Running a policy

```bash
connections-metrics --policy previous_guess \
  --tier1-db data/db/tier1_multiturn_all.db \
  --tier2-db data/db/tier2_v5_all.db \
  --source-db data/db/model_runs.db \
  --out data/db/metrics_previous_guess.db
```

`--source-db` can be repeated for additional model databases. For exact supported flags, use `connections-metrics --policy NAME --help`. Direct module execution also remains available, for example:

```bash
python -m connections_research.reconstruction.policies.a_plus_h_v2 --help
```

## Reproducibility boundaries

- Use a separate output database per policy and input snapshot. Resume bookkeeping is not a general content-hash cache; overwriting labels without changing row counts may require a fresh output database.
- The original `solved_only` implementation has different cache metadata from later variants.
- `previous_guess` is **multi-turn**, despite its old filename containing `singleshot`.
- The original paper-generation database is not included, so no policy is labelled as a verified source of every draft result.
- `plotting.compare_models` expects the schema produced by `previous_guess`. Other policies have different field names; do not assume their output is directly interchangeable with that plotting command.
