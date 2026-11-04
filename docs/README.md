# Finding Is Not Acting — interactive showcase

[Live page](https://artifex2002.github.io/connections_evals/) · [Portfolio](https://artifex2002.github.io/)

A buildless research companion exploring discovery, preservation, and action in Connections reasoning. It includes a playable demonstration puzzle, three annotated trajectory replays, a peak-versus-final comparison, and a methodology walkthrough.

## Source and preview

- `index.html`, `styles.css`, `app.js`: page content, responsive design, and interactions.
- `data/showcase.json`: separate replay dataset and reported comparison values.
- `.nojekyll`: serve these files directly through GitHub Pages.

From the repository root:

```sh
python3 -m http.server 4178 --bind 127.0.0.1 --directory docs
```

Open http://127.0.0.1:4178/. No dependencies, backend, or API keys are required. Use HTTP because the replay fetches JSON. Google Fonts has system fallbacks.

Check changes with:

```sh
python3 scripts/validate_showcase.py
node --check docs/app.js
git diff --check
```

## Data and publication

The puzzle, hero, and selectable replays are constructed examples, clearly labeled as illustrative. Their pairwise-F1 values are computed from their displayed partitions. Study counts and the three-model comparison come from the supplied paper; comparison peak means are derived from its rounded final-quality and regret means. The unpublished PDF and internal experiment databases are not included.

To replace the examples, follow [DATA_INTEGRATION.md](DATA_INTEGRATION.md). Curated runs with exact reconstructed states, official actions, feedback, and policy provenance are sufficient. Add author credits and an approved public paper link when available.

GitHub Pages serves `docs/` on `codex/project-showcase`. Pushes to that branch update the site. Keep relative asset URLs so it works under `/connections_evals/`; keep the research implementation on `main` separate from showcase-only changes.
