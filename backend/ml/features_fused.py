"""Fused RSSI + phase/channel feature extraction for model_5 / model_6.

model_0/model_1 describe an antenna slot by RSSI statistics; model_3/model_4
use the phase/channel statistics instead. Fusing them lets a slot carry both.

This used to fuse in the phase-slope range estimate, which is why fusion made
the RSSI models *worse* rather than better: the estimate was noise (see
features_pc for the measurement that shows why) and it came with a hard 0.0
sentinel indistinguishable from a genuine zero range. Both range columns are
gone. What remains from the phase side is the circular phase spread and the
fraction of the band an antenna covered.

Column order is DYN_COLS then PC_KEEP and is reused by FusedScaler.
"""

from __future__ import annotations

import numpy as np

from features import DYN_COLS, DYN_DIM, tag_dynamic_matrix
from features_pc import PC_COLS, tag_pc_matrix

# PC's log_count duplicates RSSI's (count == len(pc) for every read in the
# corpus) and PC's tx_power_norm duplicates RSSI's exactly. Keeping either
# would add a perfectly collinear column.
PC_DROP = {"log_count", "tx_power_norm"}
PC_KEEP = [c for c in PC_COLS if c not in PC_DROP]
_PC_KEEP_IDX = [PC_COLS.index(c) for c in PC_KEEP]

# Source-prefixed so the two blocks stay distinguishable in the column order.
FUSED_COLS = [f"rssi_{c}" for c in DYN_COLS] + [f"pc_{c}" for c in PC_KEEP]
FUSED_DIM = DYN_DIM + len(PC_KEEP)

# Two binary indicators appended at graph/token build time (not standardized):
# present, and multi_channel.
FLAG_DIM = 2

# A slot seen on at least this many distinct channels has usable frequency
# diversity in its phase statistics. Currently only ~36% of slots clear it,
# which is the main thing more reader dwell would change.
MIN_CHANNELS_FOR_DIVERSITY = 3


def tag_multi_channel(tag: dict, antenna_order: list[str]) -> np.ndarray:
    """Per-antenna flag: was this slot seen on >= MIN_CHANNELS_FOR_DIVERSITY
    distinct channels? Marks which phase statistics are worth trusting."""
    reads = tag.get("reads", {})
    valid = np.zeros(len(antenna_order), dtype=np.float32)
    for i, key in enumerate(antenna_order):
        r = reads.get(key)
        if not r or not r.get("pc"):
            continue
        channels = {c for c, _phase, _rssi in r["pc"] if c is not None}
        if len(channels) >= MIN_CHANNELS_FOR_DIVERSITY:
            valid[i] = 1.0
    return valid


def tag_fused_matrix(
    tag: dict, antenna_order: list[str], power_dbm: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (feat[32, FUSED_DIM], present[32], multi_channel[32]) for one tag.

    present comes from the RSSI side, which is the superset: every read in the
    corpus carries a pc block, so a slot with phase data always has RSSI too.
    """
    dyn, present = tag_dynamic_matrix(tag, antenna_order, power_dbm)
    pc, _pc_present = tag_pc_matrix(tag, antenna_order, power_dbm)
    feat = np.concatenate([dyn, pc[:, _PC_KEEP_IDX]], axis=1).astype(np.float32)
    return feat, present, tag_multi_channel(tag, antenna_order)


class FusedScaler:
    """Standardize fused columns over the slots where they are defined.

    Every remaining column is defined wherever the antenna read the tag at all,
    so a single `present` mask covers the whole matrix.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    @classmethod
    def fit(cls, feat_stack: np.ndarray, present_stack: np.ndarray, valid_stack: np.ndarray) -> "FusedScaler":
        dim = feat_stack.shape[-1]
        rows = feat_stack.reshape(-1, dim)[present_stack.reshape(-1) > 0]
        if rows.size == 0:
            return cls(np.zeros(dim), np.ones(dim))
        return cls(rows.mean(axis=0), rows.std(axis=0))

    def transform(self, feat: np.ndarray, present: np.ndarray, valid: np.ndarray) -> np.ndarray:
        scaled = (feat - self.mean) / self.std
        # Unread slots collapse to exactly zero, matching the flag columns.
        return (scaled * present[..., None]).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FusedScaler":
        return cls(np.array(d["mean"], dtype=np.float32), np.array(d["std"], dtype=np.float32))
