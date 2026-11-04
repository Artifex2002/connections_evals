# Experiment configuration references

These files document experiment settings; the current CLIs do **not** automatically load JSON configs. Continue supplying their existing CLI arguments or Slurm environment variables.

`multiturn/structured_memory.example.json` records defaults visible in the current structured-model launcher and runner. It is not a verified manifest of the paper's actual runs. The package install metadata likewise does not recover the original environment lockfile.

When adding executable configuration support later, keep explicit policy names and record the fully resolved settings with each result. Do not silently change historical defaults during that refactor.
