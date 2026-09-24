"""Phase/channel feature extraction.

RFID backscatter phase relates to antenna-tag distance d by
    phi(f) = (4*pi*f/c) * d + phi0   (mod 2*pi)
so in principle the slope of unwrapped phase vs frequency gives an unambiguous
range, d = (c / (4*pi)) * dphi/df, with the cabling offset phi0 cancelling.

That estimator has been removed, because this corpus cannot support it. The
reader does hop all 50 channels (902.75-927.25 MHz), but a given tag is seen on
a mean of 2.4 distinct channels per antenna, and 21,931 of 21,941
(tag, antenna, channel) cells hold exactly one read. Fitting a line through
~3 scattered points -- over a band where a whole metre of range produces only
about one radian of total sweep -- returns noise, and abs(slope) was hiding the
sign flips that gave it away. The models built on it scored 15-30% exact match
against 63% for plain RSSI, and fusing it made the RSSI models worse.

What survives here is the phase information that does not need a sweep: read
volume, the circular spread of phase, and how much of the band the antenna
covered. Absolute per-(antenna, channel) phase is far more reproducible than
this module previously assumed -- R = 0.96-0.99 across sessions and across TX
power levels -- but exploiting it needs a masked (antenna x channel) complex
representation rather than a per-slot scalar, which is a separate change.
"""

from __future__ import annotations

import math

import numpy as np

from common import normalize_power

# Fixed frequency-channel grid (MHz) used for the antenna x channel image.
CHAN_MIN = 902.75
CHAN_STEP = 0.5
N_CHAN = 50
CHAN_SPAN = CHAN_STEP * (N_CHAN - 1)  # total hop span in MHz

# Per-antenna phase/channel feature columns (order reused by the scaler).
PC_COLS = [
    "log_count",       # read volume
    "phase_std",       # circular spread of phase
    "chan_span",       # fraction of the band covered by this antenna's reads
    "tx_power_norm",   # run TX power, broadcast (see features.py)
]
PC_DIM = len(PC_COLS)

# Image planes for the model_2 "spectrogram": phase-gradient cos, sin, present.
# The gradient (dphase/dchannel) is offset-invariant and proportional to range.
IMG_PLANES = 3


def chan_to_idx(channel: float) -> int:
    return int(round((channel - CHAN_MIN) / CHAN_STEP))


def _antenna_pc_features(pc: list, pwr: float) -> np.ndarray:
    """One antenna's PC_DIM feature row from its [channel, phase, rssi] reads."""
    row = np.zeros(PC_DIM, dtype=np.float32)
    if not pc:
        return row
    arr = np.asarray(pc, dtype=np.float64)
    channels = arr[:, 0]
    phases = arr[:, 1]
    count = float(len(pc))

    ph_rad = np.deg2rad(phases)
    cos_m = float(np.cos(ph_rad).mean())
    sin_m = float(np.sin(ph_rad).mean())
    resultant = math.hypot(cos_m, sin_m)
    # Circular std = sqrt(-2 ln R); 0 (tight) .. large (spread).
    phase_std = math.sqrt(max(0.0, -2.0 * math.log(resultant))) if resultant > 1e-6 else math.pi
    chan_span = (channels.max() - channels.min()) / CHAN_SPAN if channels.size else 0.0

    row[0] = math.log1p(count)
    row[1] = phase_std
    row[2] = float(chan_span)
    row[3] = pwr
    return row


def tag_pc_matrix(
    tag: dict, antenna_order: list[str], power_dbm: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pc[32, PC_DIM], present[32]) for one tag."""
    reads = tag.get("reads", {})
    pc = np.zeros((len(antenna_order), PC_DIM), dtype=np.float32)
    present = np.zeros(len(antenna_order), dtype=np.float32)
    pwr = normalize_power(power_dbm)
    for i, key in enumerate(antenna_order):
        r = reads.get(key)
        if not r or not r.get("pc"):
            continue
        present[i] = 1.0
        pc[i] = _antenna_pc_features(r["pc"], pwr)
    return pc, present


def tag_phase_image(tag: dict, antenna_order: list[str]) -> np.ndarray:
    """Antenna x channel phase-gradient "spectrogram": [IMG_PLANES, 32, N_CHAN].

    For each antenna the per-channel mean phase is differenced against the next
    occupied channel to form dphase/dchannel, encoded as (cos, sin) at the lower
    channel index; plane 2 marks where a gradient exists. The gradient cancels
    the run-specific absolute-phase offset and is proportional to range, so the
    image generalises across runs where the raw phase image does not.
    """
    n_ant = len(antenna_order)
    img = np.zeros((IMG_PLANES, n_ant, N_CHAN), dtype=np.float32)
    reads = tag.get("reads", {})
    for i, key in enumerate(antenna_order):
        r = reads.get(key)
        if not r or not r.get("pc"):
            continue
        # Mean phase per occupied channel index.
        cos_acc: dict[int, float] = {}
        sin_acc: dict[int, float] = {}
        cnt: dict[int, int] = {}
        for channel, phase, _rssi in r["pc"]:
            if channel is None or phase is None:
                continue
            j = chan_to_idx(channel)
            if not (0 <= j < N_CHAN):
                continue
            rad = math.radians(phase)
            cos_acc[j] = cos_acc.get(j, 0.0) + math.cos(rad)
            sin_acc[j] = sin_acc.get(j, 0.0) + math.sin(rad)
            cnt[j] = cnt.get(j, 0) + 1
        occ = sorted(cnt)
        mean_ph = {j: math.atan2(sin_acc[j] / cnt[j], cos_acc[j] / cnt[j]) for j in occ}
        for a, b in zip(occ, occ[1:]):
            step = (b - a)
            # Per-channel-step phase increment, wrapped to (-pi, pi].
            dphi = (mean_ph[b] - mean_ph[a]) / step
            dphi = math.atan2(math.sin(dphi), math.cos(dphi))
            img[0, i, a] = math.cos(dphi)
            img[1, i, a] = math.sin(dphi)
            img[2, i, a] = 1.0
    return img


class PCScaler:
    """Standardize PC columns using stats from *present* slots only."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    @classmethod
    def fit(cls, pc_stack: np.ndarray, present_stack: np.ndarray) -> "PCScaler":
        mask = present_stack.reshape(-1) > 0
        rows = pc_stack.reshape(-1, pc_stack.shape[-1])[mask]
        if rows.size == 0:
            return cls(np.zeros(pc_stack.shape[-1]), np.ones(pc_stack.shape[-1]))
        return cls(rows.mean(axis=0), rows.std(axis=0))

    def transform(self, pc: np.ndarray, present: np.ndarray) -> np.ndarray:
        scaled = (pc - self.mean) / self.std
        scaled = scaled * present[..., None]
        return scaled.astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "PCScaler":
        return cls(np.array(d["mean"], dtype=np.float32), np.array(d["std"], dtype=np.float32))
