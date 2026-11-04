# Replacing the illustrative data

The browser loads `data/showcase.json`. Prefer small, curated JSON exports over shipping SQLite databases to visitors. Full raw databases do not need to be pushed to GitHub for this site.

## First export

Choose three runs illustrating preservation, regression, and repeated actions. Ideally choose runs of the same puzzle so the page can compare behavior without changing task difficulty. This version uses one puzzle shared by all replays. If selected real runs use different puzzles, add a puzzle collection keyed by puzzle ID and update the replay selector to resolve the current run's puzzle.

Keep the existing `regression`, `success`, and `loop` IDs for a direct replacement, or update the three tab IDs, button `data-run` values, initial selection, and hero's selected run in `app.js` together. The browser currently computes scores instead of trusting stored quality values; preserve partitions exactly as reconstructed.

### Puzzle

- `id`: original puzzle identifier
- `source`: dataset/source attribution
- `words`: the sixteen unique puzzle words in presentation order
- `groups`: four objects with a `category` and four unique `words`

### Run

- `id`: stable UI selector (`regression`, `success`, `loop`)
- `title`: editorial title for the selected example
- `regime`: visible single-shot or multi-turn label; remove "constructed example" only after verification
- `model`: exact model identifier
- `steps`: ordered reconstructed partition states

### State

- `partition`: array of represented groups, each an array of exact puzzle words. Groups may be incomplete or oversized. A word must not occupy two groups at once. Unassigned words are rendered automatically.
- `kind`: `reasoning` or `submission`
- `title`, `description`: concise human-reviewed annotations; never fabricate model quotations
- `action`: official four-word submission when applicable; never infer an action from a candidate mentioned in reasoning
- `feedback`: `CORRECT`, `ONE-AWAY`, `INCORRECT`, or `REPEATED`
- `action_key`: short stable letter per unordered guess so A → B → A becomes visible
- Recommended additional metadata: original run ID, turn ID/index, candidate index, stance, text source, and a short verified excerpt. The UI does not yet render these extra fields.

### Provenance

Record the code commit, input snapshot/checksums, reconstruction selector and stored policy version, classifier version, run selection, and model revision/settings. Mark `status` as `verified` only after checking the exports. Update the visible demonstration labels and the "What's real" disclosure at the same time; changing the JSON label alone deliberately does not remove those safeguards.

The repository preserves four reconstruction policies. Do not assume the newest policy produced the paper. Identify the exact policy/snapshot before replacing published results. Preserve official submission anchors and turn boundaries from the source pipeline.

## Aggregate results

The existing comparison requires each model's mean `final` quality and mean `regret` on the **same 568-puzzle single-shot subset**. Use unrounded exports when available. A future multi-turn comparison should be a separate dataset with puzzle-matched scope, intersection sizes, uncertainty definition, and run counts; do not substitute unequal all-run coverage into the single-shot chart.

## If sharing databases instead

We can work from a local database path or a private source. Export the puzzle/run/turn records and corresponding Tier 3 `step_cache`/metrics for selected runs, plus Tier 1/Tier 2 records if trace excerpts and stances are needed. Provide a table inventory first because schemas differ by policy. Omit credentials, provider identifiers, unrelated prompts, and other fields the showcase does not use.

## Publication content still needed

- Final author/contributor attribution and affiliations
- A paper or preprint URL approved for sharing
- Selected run IDs and short annotations explaining why each is representative or exceptional
- Source attribution for actual puzzle content
- Optional validation summary for the showcased stance labels
