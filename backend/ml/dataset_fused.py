"""Load runs into fused RSSI + phase/channel samples for model_5 / model_6.

Mirrors dataset.py and dataset_pc.py but each sample carries the concatenated
feature matrix plus both indicator flags. Uses the same run-level split as every
other model so the numbers are directly comparable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from common import MANIFEST_JSON, RUNS_DIR, run_power, stratified_run_split, usable_runs
from features_fused import FUSED_DIM, FusedScaler, tag_fused_matrix
from labels import Antenna, load_antennas, tag_labels


@dataclass
class FusedSample:
    epc: str
    test_id: str
    layout: str
    box: str
    tag_num: int
    feat: np.ndarray         # [32, FUSED_DIM] raw (unscaled)
    present: np.ndarray        # [32] antenna read this tag
    multi_channel: np.ndarray  # [32] slot had >= 3 distinct channels
    labels: dict
    total_reads: int
    power_dbm: float | None = None


@dataclass
class FusedDataset:
    antennas: list[Antenna]
    antenna_order: list[str]
    train: list[FusedSample]
    test: list[FusedSample]
    scaler: FusedScaler
    train_ids: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)

    @property
    def antenna_xyz(self) -> np.ndarray:
        return np.array([[a.x, a.y, a.z] for a in self.antennas], dtype=np.float32)


def _load_manifest() -> list[dict]:
    return json.loads(MANIFEST_JSON.read_text())


def _samples_for_run(run_path, antennas, antenna_order, power_dbm) -> list[FusedSample]:
    run = json.loads(run_path.read_text())
    out: list[FusedSample] = []
    for epc, tag in run["tags"].items():
        labels = tag_labels(tag, antennas)
        if labels is None:
            continue
        feat, present, multi_channel = tag_fused_matrix(tag, antenna_order, power_dbm)
        out.append(
            FusedSample(
                epc=epc,
                test_id=run["testId"],
                layout=str(run["layout"]),
                box=str(tag.get("box", "?")),
                tag_num=int(tag.get("tagNum", 0)),
                feat=feat,
                present=present,
                multi_channel=multi_channel,
                labels=labels,
                total_reads=int(present.sum()),
                power_dbm=power_dbm,
            )
        )
    return out


def build_fused_dataset() -> FusedDataset:
    antennas = load_antennas()
    antenna_order = [a.key for a in antennas]
    manifest = _load_manifest()
    train_ids, test_ids = stratified_run_split(manifest)
    train_set = set(train_ids)

    train: list[FusedSample] = []
    test: list[FusedSample] = []
    for run in usable_runs(manifest):
        path = RUNS_DIR / f"{run['testId']}.json"
        if not path.exists():
            continue
        samples = _samples_for_run(path, antennas, antenna_order, run_power(run))
        (train if run["testId"] in train_set else test).extend(samples)

    n_ant = len(antenna_order)
    feat_stack = np.stack([s.feat for s in train]) if train else np.zeros((1, n_ant, FUSED_DIM))
    present_stack = np.stack([s.present for s in train]) if train else np.zeros((1, n_ant))
    valid_stack = np.stack([s.multi_channel for s in train]) if train else np.zeros((1, n_ant))
    scaler = FusedScaler.fit(feat_stack, present_stack, valid_stack)

    return FusedDataset(
        antennas=antennas,
        antenna_order=antenna_order,
        train=train,
        test=test,
        scaler=scaler,
        train_ids=train_ids,
        test_ids=test_ids,
    )
