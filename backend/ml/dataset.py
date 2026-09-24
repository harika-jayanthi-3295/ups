"""Load every run into per-tag samples and expose the run-level split.

A Sample is one tag observed in one run. It carries the standardized 32-slot
feature matrix, the present-flags and the four class labels, plus metadata used
for evaluation and dashboard export.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from common import MANIFEST_JSON, RUNS_DIR, run_power, stratified_run_split, usable_runs
from features import DYN_DIM, FeatureScaler, tag_dynamic_matrix
from labels import Antenna, load_antennas, tag_labels


@dataclass
class Sample:
    epc: str
    test_id: str
    layout: str
    box: str
    tag_num: int
    dyn: np.ndarray          # [32, DYN_DIM] raw (unscaled)
    present: np.ndarray      # [32]
    labels: dict             # joint class index + its four components
    total_reads: int
    power_dbm: float | None = None


@dataclass
class Dataset:
    antennas: list[Antenna]
    antenna_order: list[str]
    train: list[Sample]
    test: list[Sample]
    scaler: FeatureScaler
    train_ids: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)

    @property
    def antenna_xyz(self) -> np.ndarray:
        return np.array([[a.x, a.y, a.z] for a in self.antennas], dtype=np.float32)


def _load_manifest() -> list[dict]:
    return json.loads(MANIFEST_JSON.read_text())


def _samples_for_run(run_path, antennas, antenna_order, power_dbm) -> list[Sample]:
    run = json.loads(run_path.read_text())
    out: list[Sample] = []
    for epc, tag in run["tags"].items():
        labels = tag_labels(tag, antennas)
        if labels is None:
            continue
        dyn, present = tag_dynamic_matrix(tag, antenna_order, power_dbm)
        out.append(
            Sample(
                epc=epc,
                test_id=run["testId"],
                layout=str(run["layout"]),
                box=str(tag.get("box", "?")),
                tag_num=int(tag.get("tagNum", 0)),
                dyn=dyn,
                present=present,
                labels=labels,
                total_reads=int(present.sum()),
                power_dbm=power_dbm,
            )
        )
    return out


def build_dataset() -> Dataset:
    antennas = load_antennas()
    antenna_order = [a.key for a in antennas]
    manifest = _load_manifest()
    train_ids, test_ids = stratified_run_split(manifest)
    train_set = set(train_ids)

    train: list[Sample] = []
    test: list[Sample] = []
    # Only runs above MIN_POWER_DBM; the rest are reported, not learned from.
    for run in usable_runs(manifest):
        path = RUNS_DIR / f"{run['testId']}.json"
        if not path.exists():
            continue
        samples = _samples_for_run(path, antennas, antenna_order, run_power(run))
        (train if run["testId"] in train_set else test).extend(samples)

    dyn_stack = np.stack([s.dyn for s in train]) if train else np.zeros((1, len(antenna_order), DYN_DIM))
    present_stack = np.stack([s.present for s in train]) if train else np.zeros((1, len(antenna_order)))
    scaler = FeatureScaler.fit(dyn_stack, present_stack)

    return Dataset(
        antennas=antennas,
        antenna_order=antenna_order,
        train=train,
        test=test,
        scaler=scaler,
        train_ids=train_ids,
        test_ids=test_ids,
    )
