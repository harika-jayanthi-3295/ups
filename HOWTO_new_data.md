# How to process a new raw data file

This describes the end-to-end steps to add a newly-collected RFID test run
(a new `.zip` from the reader software) to the dashboard.

## 1. Drop the zip in place

Copy the new archive into:

```
data_files/raw_data/
```

No renaming is required, but the filename **must contain `Layout01` or
`Layout02`** (case-insensitive, `Layout2`/`Layout2` also accepted) so the
pipeline knows whether to position boxes 001-004 using
`data_files/ground_truth_data/TagLoc1.csv` (first-run ordering) or
`TagLoc2.csv` (ascending ordering used from run 7 onward). If a future data
collection introduces a **third** box layout, see "Adding a new layout"
below before proceeding.

## 2. Run the build script

From the repo root:

```bash
python3 backend/preprocessing/build_dataset.py
```

This will:
1. Extract any `.zip` in `data_files/raw_data/` that hasn't been extracted
   yet into `data_files/extracted/<archive name>/` (already-extracted
   archives are skipped; pass `--force` to `unzip_raw_data.py` directly if
   you need to re-extract one).
2. Walk every extracted archive for `test_<N>_<timestamp>/` folders
   containing a `config/` and `raw_data/` subfolder.
3. Join each tag's raw reads (`raw_data/<EPC>.json`) with antenna metadata
   (`AntLoc1.csv`) and tag/box metadata (the layout-appropriate `TagLoc*.csv`).
4. Write one file per run to `dashboard/data/runs/<test_id>.json`, plus
   `dashboard/data/manifest.json` (the list of runs) and
   `dashboard/data/antennas.json` (static antenna positions).

Run folders that already exist across two archives (e.g. the same test also
bundled inside an "AllRev12x" combined zip) are automatically deduplicated by
test folder name — the first archive processed wins.

Re-running the script is always safe: it's idempotent and only needs to be
re-run when new zips are added.

## 3. Refresh the dashboard

The dashboard is a static site — no build step. From the repo root:

```bash
cd dashboard
python3 -m http.server 8765
```

Open `http://localhost:8765/index.html` in a browser and reload. The new
run will appear in the "Test run" dropdown (sorted by timestamp).

## Adding a new layout (Layout03+)

1. Add the new ground-truth file, e.g. `data_files/ground_truth_data/TagLoc3.csv`,
   with the same columns as `TagLoc1.csv`/`TagLoc2.csv`.
2. In `backend/preprocessing/build_dataset.py`:
   - Extend the `detect_layout()` regex/logic to recognize the new layout
     name and return `"3"`.
   - Add `"3": load_tags("TagLoc3.csv")` to the `tags_by_layout` dict in `main()`.
3. Re-run `python3 backend/preprocessing/build_dataset.py`.

## Troubleshooting

- **`cannot detect layout (1 or 2) from archive name`**: rename the zip to
  include `Layout01`/`Layout02`, or extend `detect_layout()`.
- **`reads reference unknown antennas`** warning: a raw read's
  `reader_ip` (mapped to reader number as `<last octet> - 100`) or
  `antenna` number doesn't match any row in `AntLoc1.csv` — check whether a
  new reader/antenna was added to the physical setup and update
  `AntLoc1.csv` accordingly.
- **A tag shows "unknown ground-truth position"**: its EPC isn't present in
  the layout's `TagLoc*.csv`. Check for typos in the EPC or a box that
  hasn't been added to the ground-truth file yet.
- **Antenna/tag markers look misplaced on the sketch**: the pixel positions
  are computed by linear interpolation between two calibration points per
  wall surface (`SURFACE_ANCHORS` in `build_dataset.py`), read by eye off
  `CargoArea01.png`. Tweak those coordinates if the sketch image changes.
