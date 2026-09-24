"""Load runs into phase/channel samples for model_2 / model_3 / model_4.

Mirrors dataset.py but each sample carries the phase/channel feature matrix,
the present flags, and the antenna x channel phase image. Uses the same
run-level train/test split so results are comparable to the RSSI models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from common import MANIFEST_JSON, RUNS_DIR, run_power, stratified_run_split, usable_runs
from features_pc import PC_DIM, PCScaler, tag_pc_matrix, tag_phase_image
from labels import Antenna, load_antennas, tag_labels


@dataclass
class PCSample:
    epc: str
    test_id: str
    layout: str
    box: str
    tag_num: int
    pc: np.ndarray           # [32, PC_DIM] raw (unscaled)
    present: np.ndarray      # [32]
    img: np.ndarray          # [IMG_PLANES, 32, N_CHAN]
    labels: dict
    total_reads: int
    power_dbm: float | None = None


@dataclass
class PCDataset:
    antennas: list[Antenna]
    antenna_order: list[str]
    train: list[PCSample]
    test: list[PCSample]
    scaler: PCScaler
    train_ids: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)

    @property
    def antenna_xyz(self) -> np.ndarray:
        return np.array([[a.x, a.y, a.z] for a in self.antennas], dtype=np.float32)


def _load_manifest() -> list[dict]:
    return json.loads(MANIFEST_JSON.read_text())


def _samples_for_run(run_path, antennas, antenna_order, power_dbm) -> list[PCSample]:
    run = json.loads(run_path.read_text())
    out: list[PCSample] = []
    for epc, tag in run["tags"].items():
        labels = tag_labels(tag, antennas)
        if labels is None:
            continue
        pc, present = tag_pc_matrix(tag, antenna_order, power_dbm)
        img = tag_phase_image(tag, antenna_order)
        out.append(
            PCSample(
                epc=epc,
                test_id=run["testId"],
                layout=str(run["layout"]),
                box=str(tag.get("box", "?")),
                tag_num=int(tag.get("tagNum", 0)),
                pc=pc,
                present=present,
                img=img,
                labels=labels,
                total_reads=int(present.sum()),
                power_dbm=power_dbm,
            )
        )
    return out


def build_pc_dataset() -> PCDataset:
    antennas = load_antennas()
    antenna_order = [a.key for a in antennas]
    manifest = _load_manifest()
    train_ids, test_ids = stratified_run_split(manifest)
    train_set = set(train_ids)

    train: list[PCSample] = []
    test: list[PCSample] = []
    for run in usable_runs(manifest):
        path = RUNS_DIR / f"{run['testId']}.json"
        if not path.exists():
            continue
        samples = _samples_for_run(path, antennas, antenna_order, run_power(run))
        (train if run["testId"] in train_set else test).extend(samples)

    pc_stack = np.stack([s.pc for s in train]) if train else np.zeros((1, len(antenna_order), PC_DIM))
    present_stack = np.stack([s.present for s in train]) if train else np.zeros((1, len(antenna_order)))
    scaler = PCScaler.fit(pc_stack, present_stack)

    return PCDataset(
        antennas=antennas,
        antenna_order=antenna_order,
        train=train,
        test=test,
        scaler=scaler,
        train_ids=train_ids,
        test_ids=test_ids,
    )
