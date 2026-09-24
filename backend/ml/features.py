"""Per-antenna feature extraction + standardization.

Each tag in a run is described by a fixed 32-slot matrix (one slot per antenna,
in load_antennas() order). A slot holds the read statistics for that antenna,
or zeros if the antenna never saw the tag. A present-flag column distinguishes
"read with value 0" from "not read".

The last column carries the run's TX power, broadcast to every slot. It is a
per-sample constant rather than a per-antenna measurement, but broadcasting it
is what lets every model pick it up without an architecture change: each model
either pools over slots or attends across them, so a constant column survives
the readout. It matters because the corpus spans 15-30 dBm and power offsets
every RSSI value in the row above it -- without it the model has to infer the
power level from the RSSI distribution it is trying to interpret.
"""

from __future__ import annotations

import math

import numpy as np

from common import normalize_power

# Dynamic per-antenna columns (order matters - reused by the scaler).
DYN_COLS = ["log_count", "avgRssi", "minRssi", "maxRssi", "readShare", "tx_power_norm"]
DYN_DIM = len(DYN_COLS)


def xyz_norm_stats(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis (mean, std) for standardizing antenna positions."""
    mean = xyz.mean(axis=0)
    std = xyz.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def tag_dynamic_matrix(
    tag: dict, antenna_order: list[str], power_dbm: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dyn[32, DYN_DIM], present[32]) for one tag."""
    reads = tag.get("reads", {})
    total = sum(r.get("count", 0) for r in reads.values()) or 1
    dyn = np.zeros((len(antenna_order), DYN_DIM), dtype=np.float32)
    present = np.zeros(len(antenna_order), dtype=np.float32)
    pwr = normalize_power(power_dbm)
    for i, key in enumerate(antenna_order):
        r = reads.get(key)
        if not r:
            continue
        present[i] = 1.0
        count = float(r.get("count", 0))
        dyn[i] = (
            math.log1p(count),
            float(r.get("avgRssi", 0.0)),
            float(r.get("minRssi", 0.0)),
            float(r.get("maxRssi", 0.0)),
            count / total,
            pwr,
        )
    return dyn, present


class FeatureScaler:
    """Standardize the dynamic columns using stats from *present* slots only."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    @classmethod
    def fit(cls, dyn_stack: np.ndarray, present_stack: np.ndarray) -> "FeatureScaler":
        mask = present_stack.reshape(-1) > 0
        rows = dyn_stack.reshape(-1, dyn_stack.shape[-1])[mask]
        if rows.size == 0:
            return cls(np.zeros(dyn_stack.shape[-1]), np.ones(dyn_stack.shape[-1]))
        return cls(rows.mean(axis=0), rows.std(axis=0))

    def transform(self, dyn: np.ndarray, present: np.ndarray) -> np.ndarray:
        scaled = (dyn - self.mean) / self.std
        scaled = scaled * present[..., None]  # keep absent slots at zero
        return scaled.astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureScaler":
        return cls(np.array(d["mean"], dtype=np.float32), np.array(d["std"], dtype=np.float32))
