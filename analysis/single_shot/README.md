# Single-shot workflow pending

The original single-shot runner, original `metrics_step2.py`, and scripts reproducing the 14-model study have not yet been recovered into this repository.

Add the recovered workflow here, put reusable implementations under `src/connections_research/`, and record experiment settings under `configs/single_shot/`. Preserve the original prompts, budgets, input schema, final-partition parsing, and model/puzzle coverage before attempting to share implementations with multi-turn experiments.

The `previous_guess` reconstruction policy is a multi-turn adaptation of single-shot logic. It does not replace the missing original single-shot pipeline.
