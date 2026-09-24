"""Shared paths, label vocabularies and the run-level train/test split.

All seven models consume the same samples produced from
dashboard/data/runs/*.json, so everything that defines the prediction problem
lives here to keep the pipelines consistent.

Two things here are worth reading before touching a model:

* **The corpus is a TX-power sweep.** Each run was collected at a single power
  from 30 dBm down to 3 dBm. Below 15 dBm the runs are effectively empty --
  61% of tags produce zero reads at 12 dBm, 99.6% at 3 dBm -- so they are
  excluded from the learning problem by MIN_POWER_DBM and reported separately.
  Power itself is carried through as a model input, because it shifts every
  RSSI value by up to 27 dB.

* **The label is a single joint class, not four independent ones.** Only 35 of
  the 3*3*4*8 = 288 (region, row, reader, antenna) combinations are physically
  realisable, because reader/antenna is the antenna nearest the tag and that
  antenna's own side/row is almost always the tag's. Predicting four
  independent heads let the models emit an impossible descriptor 13.9% of the
  time. JOINT_LABELS is the realisable set, derived from ground truth.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DATA = PROJECT_ROOT / "dashboard" / "data"
RUNS_DIR = DASHBOARD_DATA / "runs"
ANTENNAS_JSON = DASHBOARD_DATA / "antennas.json"
MANIFEST_JSON = DASHBOARD_DATA / "manifest.json"
PREDICTIONS_DIR = DASHBOARD_DATA / "predictions"
READ_PATTERNS_JSON = DASHBOARD_DATA / "read_patterns.json"
GT_DIR = PROJECT_ROOT / "data_files" / "ground_truth_data"

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

# --- corpus filter -----------------------------------------------------------
# Runs below this TX power carry almost no reads and cannot be predicted by any
# model; they stay in the manifest (and in the power-sweep report) but are kept
# out of train and test.
MIN_POWER_DBM = 15.0

# Normalisation constants for TX power as a model input: roughly centres the
# 15-30 dBm band on zero with unit-ish scale.
POWER_REF = 22.5
POWER_SCALE = 6.0


def normalize_power(power_dbm: float | None) -> float:
    if power_dbm is None:
        return 0.0
    return (float(power_dbm) - POWER_REF) / POWER_SCALE


# --- prediction target vocabularies -----------------------------------------
# A prediction is the human-readable descriptor "<row> <region> near reader R
# antenna A". region/row come from the tag ground truth; reader/antenna come
# from the antenna physically closest to the tag.
REGIONS = ["Aisle", "Left", "Right"]
ROWS = ["Floor", "Middle", "Top"]
READERS = [1, 2, 3, 4]
ANTENNAS = [1, 2, 3, 4, 5, 6, 7, 8]

# The four components of a descriptor. These are reporting axes only -- the
# model has a single head over JOINT_LABELS and the components are recovered by
# decomposing its prediction.
SUBHEADS = ["region", "row", "reader", "antenna"]


def _load_antenna_xyz() -> list[tuple[float, float, float, int, int]]:
    """(x, y, z, reader, antenna) for every antenna, in (reader, antenna) order."""
    raw = json.loads(ANTENNAS_JSON.read_text())
    ants = [
        (float(a["x"]), float(a["y"]), float(a["z"]), int(a["reader"]), int(a["antenna"]))
        for a in raw
    ]
    ants.sort(key=lambda a: (a[3], a[4]))
    return ants


def _nearest_antenna_idx(x: float, y: float, z: float, ants) -> int:
    return min(
        range(len(ants)),
        key=lambda i: (ants[i][0] - x) ** 2 + (ants[i][1] - y) ** 2 + (ants[i][2] - z) ** 2,
    )


def build_joint_vocab() -> list[tuple[int, int, int, int]]:
    """Every (region, row, reader, antenna) index tuple that ground truth realises.

    Derived from the antenna catalog plus both TagLoc layouts, so the vocabulary
    is a property of the physical installation and does not shift with whichever
    runs happen to be loaded.
    """
    ants = _load_antenna_xyz()
    reg_idx = {v: i for i, v in enumerate(REGIONS)}
    row_idx = {v: i for i, v in enumerate(ROWS)}
    rdr_idx = {v: i for i, v in enumerate(READERS)}
    ant_idx = {v: i for i, v in enumerate(ANTENNAS)}

    seen: set[tuple[int, int, int, int]] = set()
    for csv_path in sorted(GT_DIR.glob("TagLoc*.csv")):
        with csv_path.open() as fh:
            for row in csv.DictReader(fh):
                side, shelf = row.get("Side"), row.get("Row")
                if side not in reg_idx or shelf not in row_idx:
                    continue
                near = ants[_nearest_antenna_idx(float(row["X"]), float(row["Y"]), float(row["Z"]), ants)]
                seen.add((reg_idx[side], row_idx[shelf], rdr_idx[near[3]], ant_idx[near[4]]))
    return sorted(seen)


JOINT_LABELS: list[tuple[int, int, int, int]] = build_joint_vocab()
JOINT_TO_IDX: dict[tuple[int, int, int, int], int] = {t: i for i, t in enumerate(JOINT_LABELS)}
N_JOINT = len(JOINT_LABELS)

# The single classification head every model emits. Kept in the same
# (name, n_classes) shape the models already iterate over.
HEADS = [("joint", N_JOINT)]


def joint_to_parts(joint_idx: int) -> dict[str, int]:
    """Joint class index -> the four component class indices."""
    region, row, reader, antenna = JOINT_LABELS[int(joint_idx)]
    return {"region": region, "row": row, "reader": reader, "antenna": antenna}


def parts_to_joint(parts: dict[str, int]) -> int | None:
    """The four component class indices -> joint class index, or None if the
    combination is not physically realisable."""
    return JOINT_TO_IDX.get(
        (int(parts["region"]), int(parts["row"]), int(parts["reader"]), int(parts["antenna"]))
    )


def descriptor(row: str, region: str, reader: int, antenna: int) -> str:
    """Format a prediction the way the task phrases it."""
    return f"{row} {region} near reader {reader} antenna {antenna}"


# --- run selection and split -------------------------------------------------

# Fraction of usable runs used for training.
TRAIN_FRACTION = 0.40
SPLIT_SEED = 13


def run_power(run: dict) -> float | None:
    p = run.get("transmitPower")
    return None if p is None else float(p)


EXCLUDE_ARCHIVE_SUBSTRINGS = ("ceilingantenna",)


def usable_runs(manifest: list[dict]) -> list[dict]:
    """Runs that carry enough signal to learn from: TX power >= MIN_POWER_DBM.

    A run with no recorded power is kept rather than silently dropped, so a
    missing config shows up as a data problem instead of a shrinking corpus.
    """
    out = []
    for run in manifest:
        # Ceiling-antenna runs use shifted box positions that TagLoc2.csv doesn't reflect.
        if any(s in run.get("sourceArchive", "") for s in EXCLUDE_ARCHIVE_SUBSTRINGS):
            continue
        p = run_power(run)
        if p is None or p >= MIN_POWER_DBM:
            out.append(run)
    return out


def excluded_runs(manifest: list[dict]) -> list[dict]:
    usable = {r["testId"] for r in usable_runs(manifest)}
    return [r for r in manifest if r["testId"] not in usable]


def stratified_run_split(manifest: list[dict]) -> tuple[list[str], list[str]]:
    """Pick ~TRAIN_FRACTION of the usable runs for training, stratified by
    layout and TX power so both layouts and the whole surviving power range are
    represented in train. Deterministic for a fixed manifest.
    """
    runs = usable_runs(manifest)

    by_stratum: dict[tuple[str, float], list[dict]] = {}
    for run in runs:
        key = (str(run["layout"]), run_power(run) if run_power(run) is not None else -1.0)
        by_stratum.setdefault(key, []).append(run)

    train_ids: list[str] = []
    for key in sorted(by_stratum):
        group = sorted(by_stratum[key], key=lambda r: (r["readCount"], r["testId"]))
        n_train = max(1, round(len(group) * TRAIN_FRACTION))
        # Evenly spaced pick across the read-density-sorted list.
        step = len(group) / n_train
        picked = {int(i * step) for i in range(n_train)}
        train_ids.extend(group[idx]["testId"] for idx in range(len(group)) if idx in picked)

    train_set = set(train_ids)
    test_ids = [r["testId"] for r in runs if r["testId"] not in train_set]
    return train_ids, test_ids
