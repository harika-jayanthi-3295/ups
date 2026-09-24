# Agent Guide — UPS RFID Truck-Location Project

You are working in `/home/ups`, an RFID tag-localization research project.
**This is not a git repository** — there is no version control here, so
there's no `git diff`/`git log` to lean on and no easy undo for destructive
file operations. Be extra careful with anything that deletes or overwrites
files; there is no safety net. `wiki/log.md` is the closest thing this
project has to a commit history — keep it honest.

## This repo uses the "LLM Wiki" pattern

`wiki/` is not a one-time documentation dump. It's a **persistent,
compounding knowledge base** that you (the agent) read from and write to
every session. The three layers:

- **Raw sources** — `data_files/` (raw archives, ground-truth CSVs,
  `TestLog.docx`), the codebase itself (`backend/`, `dashboard/`), and
  generated artifacts (`backend/ml/artifacts/metrics.json`,
  `dashboard/data/*.json`). Read from these, don't treat them as things
  you maintain by hand — `data_files/raw_data/` and
  `data_files/ground_truth_data/` in particular are immutable source
  data; never edit or regenerate them yourself.
- **The wiki** (`wiki/*.md`) — you own this layer entirely. Read
  [`wiki/Home.md`](wiki/Home.md) first for orientation, then
  [`wiki/index.md`](wiki/index.md) to find the page relevant to your task.
- **This file** — the schema. It's the conventions doc: how the wiki is
  structured, and what to do when you ingest a new source, answer a
  query, or lint the wiki. Update *this file* when you discover a
  workflow that isn't captured here yet.

### Wiki conventions

- Every content page starts with YAML frontmatter: `type` (`overview`,
  `reference`, `workflow`, or `index`), `updated` (date of last edit),
  and optionally `sources` (repo paths it's derived from) and `tags`.
  Update `updated` whenever you edit a page's content.
- [`wiki/index.md`](wiki/index.md) is the content catalog — one line per
  page, grouped by category, plus an "Open threads" section for
  unresolved questions. Update it whenever you add, rename, or
  meaningfully re-scope a page.
- [`wiki/log.md`](wiki/log.md) is the append-only chronological record.
  Every entry starts with `## [YYYY-MM-DD] type | Title` where `type` is
  `ingest`, `query`, or `lint` — this makes it `grep`-able
  (`grep "^## \[" wiki/log.md | tail -5`). Append an entry for anything
  that changes the wiki's content, however small.
- Don't create new top-level `.md` files outside `wiki/` for
  documentation — extend an existing wiki page or add a new one under
  `wiki/`, linked from `index.md`.

### Operation: Ingest

Something changed in the raw sources and the wiki needs to catch up.
Concretely, in this repo, a "new source" is one of:

| New source | What to do |
|---|---|
| New `.zip` dropped in `data_files/raw_data/` | Run the Workflow 1 steps in [`wiki/Running-The-Project.md`](wiki/Running-The-Project.md) (`build_dataset.py`). If it introduces a new firmware version, antenna placement, or box layout, update [`wiki/Data-Collection.md`](wiki/Data-Collection.md)'s experiment narrative. |
| `TestLog.docx` gets new rows | Update the phased narrative in [`wiki/Data-Collection.md`](wiki/Data-Collection.md) — don't just append, fold the new phase into the existing numbered list so the story stays coherent. |
| A `run_full_pipeline.py` retrain | Diff the new `backend/ml/artifacts/metrics.json` against the table in [`wiki/ML-Pipeline.md`](wiki/ML-Pipeline.md). If accuracy moved meaningfully (either direction), update the table and add a one-line explanation if you can identify the cause (feature change, new runs, label change). |
| Code change in `backend/` or `dashboard/` | Update the relevant reference page (`Preprocessing-Pipeline.md`, `ML-Pipeline.md`, `EDA-Scripts.md`, or `Dashboard.md`). If it fixes one of the gotchas below, remove that bullet instead of leaving it stale. |
| A decision made in conversation (e.g. "we're doing Phase B now") | File it into the relevant page's narrative (don't leave it only in chat history) and resolve the matching item in `wiki/index.md`'s "Open threads" if there is one. |

After any ingest: update `wiki/index.md` if the page set or summaries
changed, and always append a `wiki/log.md` entry.

### Operation: Query

When asked a question about the project: read `wiki/index.md` first,
open the page(s) it points to, and answer from there — only fall back to
re-reading raw source files (code, CSVs, JSON) when the wiki doesn't
cover it or you suspect it's stale (check the page's `updated` date
against reality). If your answer is a synthesis worth keeping — a
comparison, a root-cause explanation, a new connection across pages —
file it back into the wiki (a new page, or a new section on an existing
one) rather than letting it live only in the conversation. Log it as a
`query` entry in `wiki/log.md`.

### Operation: Lint

Periodically (or when asked to "check the wiki"), look for:

- **Metrics drift**: does `wiki/ML-Pipeline.md`'s numbers table still
  match `backend/ml/artifacts/metrics.json`?
- **Uningested data**: any `data_files/raw_data/*.zip` not reflected in
  `dashboard/data/manifest.json`, or `TestLog.docx` rows not reflected in
  `wiki/Data-Collection.md`?
- **Orphans/staleness**: pages not linked from `index.md`; `updated`
  dates far older than a related source's last change; resolved items
  still sitting in `index.md`'s "Open threads".
- **Contradictions**: e.g. a gotcha below that the code no longer
  exhibits, or two pages describing the same mechanism differently.

Log findings (and fixes, if you make them) as a `lint` entry in
`wiki/log.md`.

## Repo shape (see `wiki/Home.md` for the diagram)

- `data_files/` — raw zips, extracted archives, ground-truth CSVs/docs.
  Treat everything under `data_files/raw_data/` and
  `data_files/ground_truth_data/` as **source data — do not edit or
  regenerate it**. `data_files/extracted/` is a derived cache (safe to
  delete and re-extract via `backend/preprocessing/unzip_raw_data.py`).
- `backend/preprocessing/` — turns raw zips into `dashboard/data/{manifest,
  antennas,runs/*}.json`. See `wiki/Preprocessing-Pipeline.md`.
- `backend/eda/` — standalone research scripts, not part of the automated
  pipeline. See `wiki/EDA-Scripts.md`.
- `backend/ml/` — feature engineering + 7 model variants (`model_0`..
  `model_6`) + train/predict/evaluate scripts + `artifacts/*.pt`. See
  `wiki/ML-Pipeline.md`.
- `dashboard/` — static site (no build step, no framework, vendored D3).
  `dashboard/data/*.json` is **entirely generated** by the backend
  scripts — never hand-edit it, regenerate it instead.
- `HOWTO_new_data.md`, `wiki/Running-The-Project.md` — the canonical
  operational workflows (add new data, retrain, serve dashboard).

## Things that will bite you if you don't know them

1. **Stale docstrings in `backend/ml/model_*`**: several model files
   (`gnn_model.py`, `gnn_pc_model.py`, `cnn_model.py`) describe "four
   classification heads" from an old design. The real code (`HEADS` in
   `common.py`/`labels.py`) has collapsed this to a single 35-way joint
   class. Trust the code, not those comments — and don't propagate the
   stale description into new code or docs.
2. **`predict.py` vs `ensemble_predict.py` write to the same files**
   (`dashboard/data/predictions/<testId>.json`) with different sample
   sets and different schemas (2 vs. 3 model keys per tag). Running
   `ensemble_predict.py` after `predict.py` silently changes what the
   dashboard sees. Check `wiki/ML-Pipeline.md#inference` before touching
   either script.
3. **`data_files/ground_truth_data/README.md` is inaccurate** — it
   describes `xPseudo`/`yPseudo` CSV columns that don't exist. The real
   sketch-projection mechanism is `SURFACE_ANCHORS` in
   `build_dataset.py`. Don't trust that README for pixel-mapping logic.
4. **Layout detection is filename-based and only knows `1`/`2`**
   (`detect_layout()` in `build_dataset.py`). A new archive without
   `Layout01`/`Layout02` in its name will be skipped with a `[skip]`
   message, not an error you'll notice by default.
5. **No test suite, no linter config, no `requirements.txt` at the repo
   root** — only `backend/ml/requirements.txt` (torch, torch-geometric,
   scikit-learn, numpy). There is nothing to run for automated
   verification beyond re-running the relevant script and inspecting its
   output/printed metrics.
6. **Scripts are not packaged** — every `backend/ml/model_N/train_*.py`
   manually inserts its parent dirs into `sys.path`. Run them from their
   own directory (or via `run_full_pipeline.py`, which sets `cwd`
   correctly) rather than assuming they're importable from anywhere.
7. **`instruction_to_follow.txt`** at the repo root is a leftover ad-hoc
   task note ("run the models... update the dashboard with model
   predictions") — it describes Workflow 2 in
   `wiki/Running-The-Project.md`, not a standing instruction to act on
   automatically.

(If you fix the underlying issue for any of these, remove the bullet as
part of that ingest — per the Lint operation above, a stale gotcha is
itself a wiki bug.)

## How to verify a change

There's no CI. After touching:
- **preprocessing** (`backend/preprocessing/`): re-run
  `python3 backend/preprocessing/build_dataset.py` and check its printed
  warnings (`reads reference unknown antennas`, `[skip]` lines) plus spot
  check one `dashboard/data/runs/*.json` file.
- **ML code** (`backend/ml/`): re-run the specific `train_*.py` you
  touched (or the full `run_full_pipeline.py` if the change is shared
  code like `features.py`/`common.py`) and compare the printed accuracy
  against the table in `wiki/ML-Pipeline.md` — a big regression usually
  means a feature/label bug.
- **dashboard** (`dashboard/`): serve it (`python3 -m http.server 8765`
  from `dashboard/`) and manually click through a run + tag selection in
  a browser; there's no automated frontend test.

## Where to add new documentation

Update the relevant page under `wiki/` (and `wiki/index.md`/`wiki/log.md`
per the conventions above) rather than adding new top-level `.md` files.
