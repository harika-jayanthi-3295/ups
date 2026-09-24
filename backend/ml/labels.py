"""Antenna catalog + label derivation.

Labels for a tag: region and row come straight from the tag's ground truth;
the nearest reader/antenna is the antenna whose 3D position minimises Euclidean
distance to the tag.

Those four components are not independent -- only 35 of their 288 combinations
are physically realisable -- so training targets a single joint class over
common.JOINT_LABELS. The component indices are still carried alongside it for
reporting, and are recovered from a prediction with joint_to_parts().
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from common import (
    ANTENNAS,
    ANTENNAS_JSON,
    JOINT_TO_IDX,
    READERS,
    REGIONS,
    ROWS,
    joint_to_parts,
)


@dataclass
class Antenna:
    key: str
    reader: int
    antenna: int
    side: str
    row: str
    x: float
    y: float
    z: float
    o: float
    px: float | None
    py: float | None


def load_antennas() -> list[Antenna]:
    raw = json.loads(ANTENNAS_JSON.read_text())
    ants = [
        Antenna(
            key=a["rdrAnt"],
            reader=int(a["reader"]),
            antenna=int(a["antenna"]),
            side=a["side"],
            row=a["row"],
            x=float(a["x"]),
            y=float(a["y"]),
            z=float(a["z"]),
            o=float(a.get("o", 0.0)),
            px=a.get("px"),
            py=a.get("py"),
        )
        for a in raw
    ]
    # Stable ordering by (reader, antenna) — used as the fixed node/token order.
    ants.sort(key=lambda a: (a.reader, a.antenna))
    return ants


def nearest_antenna(tag: dict, antennas: list[Antenna]) -> Antenna:
    tx, ty, tz = float(tag["x"]), float(tag["y"]), float(tag["z"])
    return min(
        antennas,
        key=lambda a: (a.x - tx) ** 2 + (a.y - ty) ** 2 + (a.z - tz) ** 2,
    )


# Class-index maps.
REGION_TO_IDX = {v: i for i, v in enumerate(REGIONS)}
ROW_TO_IDX = {v: i for i, v in enumerate(ROWS)}
READER_TO_IDX = {v: i for i, v in enumerate(READERS)}
ANTENNA_TO_IDX = {v: i for i, v in enumerate(ANTENNAS)}


def tag_labels(tag: dict, antennas: list[Antenna]) -> dict | None:
    """Return {joint, region, row, reader, antenna} class indices, or None if the
    tag has no usable ground truth.

    "joint" is the training target; the four components are kept for reporting.
    A tag whose component tuple is outside JOINT_LABELS is dropped rather than
    forced into a class, since the vocabulary is built from the same ground
    truth and a miss means the tag's position is inconsistent with the catalog.
    """
    if tag.get("side") not in REGION_TO_IDX or tag.get("row") not in ROW_TO_IDX:
        return None
    near = nearest_antenna(tag, antennas)
    parts = {
        "region": REGION_TO_IDX[tag["side"]],
        "row": ROW_TO_IDX[tag["row"]],
        "reader": READER_TO_IDX[near.reader],
        "antenna": ANTENNA_TO_IDX[near.antenna],
    }
    joint = JOINT_TO_IDX.get(
        (parts["region"], parts["row"], parts["reader"], parts["antenna"])
    )
    if joint is None:
        return None
    return {"joint": joint, **parts}


def decode_labels(idx: dict) -> dict:
    """Class indices -> readable values + descriptor string.

    Accepts either a full {region,row,reader,antenna} mapping or a {"joint": i}
    prediction, which is expanded first.
    """
    if "region" not in idx and "joint" in idx:
        idx = joint_to_parts(idx["joint"])
    region = REGIONS[idx["region"]]
    row = ROWS[idx["row"]]
    reader = READERS[idx["reader"]]
    antenna = ANTENNAS[idx["antenna"]]
    from common import descriptor

    return {
        "region": region,
        "row": row,
        "reader": reader,
        "antenna": antenna,
        "descriptor": descriptor(row, region, reader, antenna),
    }
