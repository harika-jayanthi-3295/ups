"""Build the dashboard dataset from ground-truth CSVs + extracted raw RFID reads.

For each RFID test run found under data_files/extracted/, this script joins the
per-read events (reader/antenna/RSSI/timestamp) with the antenna and tag ground
truth (position, side, row, distance) and writes one JSON file per run plus a
manifest, ready to be served to the D3 dashboard as static files.

It also computes an approximate pixel position (px, py) for every antenna and
every tag/box, projected onto the 1200x600 CargoArea01.png sketch. The sketch
has no real xPseudo/yPseudo columns to drive this, so positions are derived by
linear interpolation between two calibration points per wall surface (Side +
Row), read by eye off the antenna labels drawn in the sketch. Both antennas and
boxes share the same Side/Row/Dist columns, so the same calibration table and
formula work for both.

Usage:
    python build_dataset.py                # extract any new zips, build everything
    python build_dataset.py --skip-extract  # assume data_files/extracted/ is current
    python build_dataset.py --out DIR       # override output dir (default dashboard/data)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unzip_raw_data  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GT_DIR = PROJECT_ROOT / "data_files" / "ground_truth_data"
RAW_DIR = PROJECT_ROOT / "data_files" / "raw_data"
EXTRACTED_DIR = PROJECT_ROOT / "data_files" / "extracted"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
DEFAULT_OUT = DASHBOARD_DIR / "data"

TEST_DIR_RE = re.compile(r"^test_(\d+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")

# Pixel calibration points (Y=18.5 -> "low", Y=191 -> "high") read off the
# antenna labels in CargoArea01.png (1200x600), keyed by the real Y coordinate
# (distance along the truck) rather than the CSV "Dist" column: Dist is a
# nominal box-ordering label that is NOT recomputed when boxes 001-004 are
# reshuffled between layouts (TagLoc1 vs TagLoc2), so it can't be trusted for
# placement — Y always reflects the box's actual physical position. Aisle/Floor
# has no antennas to anchor on, so its two points are a reasoned estimate along
# the aisle centerline.
SURFACE_ANCHORS: dict[tuple[str, str], dict[str, tuple[float, float]]] = {
    ("Left", "Floor"): {"low": (48, 413), "high": (361, 381)},
    ("Left", "Middle"): {"low": (48, 266), "high": (361, 290)},
    ("Left", "Top"): {"low": (48, 113), "high": (361, 155)},
    ("Right", "Floor"): {"low": (1141, 413), "high": (835, 381)},
    ("Right", "Middle"): {"low": (1141, 266), "high": (835, 290)},
    ("Right", "Top"): {"low": (1141, 113), "high": (835, 155)},
    ("Aisle", "Floor"): {"low": (601, 440), "high": (601, 600)},
}
Y_LOW, Y_HIGH = 18.5, 191.0
# Ceiling antennas aren't on a Y-interpolated line; fixed sketch positions.
CEILING_PX: dict[str, tuple[float, float]] = {
    "Rdr-2-Ant-8": (601, 35),
    "Rdr-4-Ant-8": (601, 71),
}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def project_to_sketch(rdr_ant: str | None, side: str, row: str, y_pos: float) -> tuple[float, float] | None:
    """Return an (px, py) pixel position on CargoArea01.png for a Side/Row/Y."""
    if rdr_ant in CEILING_PX:
        return CEILING_PX[rdr_ant]
    anchors = SURFACE_ANCHORS.get((side, row))
    if anchors is None:
        return None
    frac = clamp((y_pos - Y_LOW) / (Y_HIGH - Y_LOW), 0.0, 1.0)
    lx, ly = anchors["low"]
    hx, hy = anchors["high"]
    return (lx + (hx - lx) * frac, ly + (hy - ly) * frac)


def load_antennas() -> dict[str, dict[str, Any]]:
    """Load AntLoc1.csv keyed by Rdr-Ant (e.g. 'Rdr-1-Ant-1')."""
    antennas: dict[str, dict[str, Any]] = {}
    with (GT_DIR / "AntLoc1.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            rdr, ant = int(row["Rdr"]), int(row["Ant"])
            side, shelf, dist = row["Side"], row["Row"], float(row["Dist"])
            rdr_ant = row["Rdr-Ant"]
            px = project_to_sketch(rdr_ant, side, shelf, float(row["Y"]))
            antennas[rdr_ant] = {
                "rdrAnt": rdr_ant,
                "reader": rdr,
                "antenna": ant,
                "orientation": row["Orientation"],
                "side": side,
                "row": shelf,
                "dist": dist,
                "x": float(row["X"]),
                "y": float(row["Y"]),
                "z": float(row["Z"]),
                "o": float(row["O"]),
                "px": px[0] if px else None,
                "py": px[1] if px else None,
            }
    return antennas


def load_tags(csv_name: str) -> dict[str, dict[str, Any]]:
    """Load a TagLoc*.csv keyed by lowercase EPC."""
    tags: dict[str, dict[str, Any]] = {}
    with (GT_DIR / csv_name).open(newline="") as f:
        for row in csv.DictReader(f):
            epc = row["EPC"].strip().lower()
            side, shelf, dist = row["Side"], row["Row"], float(row["Dist"])
            px = project_to_sketch(None, side, shelf, float(row["Y"]))
            tags[epc] = {
                "epc": epc,
                "box": row["Box#"],
                "tagNum": int(row["Tag#"]),
                "orientation": row["Orientation"],
                "side": side,
                "row": shelf,
                "dist": dist,
                "x": float(row["X"]),
                "y": float(row["Y"]),
                "z": float(row["Z"]),
                "px": px[0] if px else None,
                "py": px[1] if px else None,
            }
    return tags


def detect_layout(archive_stem: str) -> str:
    """Return '1' or '2' from an archive name like 'Rev121-Layout01_2026-08-28'.

    Special case: ceiling antenna tests (e.g. '9-16-testdata-ceilingantenna...')
    are Layout 2 (post-09-01 reordering).
    """
    # Check for explicit layout in filename
    m = re.search(r"layout0*([12])(?!\d)", archive_stem, re.IGNORECASE)
    if m:
        return m.group(1)

    # Ceiling antenna tests are Layout 2
    if "ceilingantenna" in archive_stem.lower():
        return "2"

    raise ValueError(f"cannot detect layout (1 or 2) from archive name: {archive_stem!r}")


def reader_from_ip(reader_ip: str | None) -> int | None:
    """'192.168.100.101' -> 1, '192.168.100.104' -> 4."""
    if not reader_ip:
        return None
    try:
        last_octet = int(reader_ip.rsplit(".", 1)[-1])
    except ValueError:
        return None
    rdr = last_octet - 100
    return rdr if rdr > 0 else None


def find_test_dirs(extracted_root: Path) -> list[tuple[str, str, Path]]:
    """Return (archive_stem, test_id, path) for every test_N_<timestamp> dir found."""
    found: list[tuple[str, str, Path]] = []
    for archive_dir in sorted(p for p in extracted_root.iterdir() if p.is_dir()):
        for candidate in sorted(archive_dir.rglob("test_*")):
            if not candidate.is_dir():
                continue
            if not TEST_DIR_RE.match(candidate.name):
                continue
            if (candidate / "config").is_dir() and (candidate / "raw_data").is_dir():
                found.append((archive_dir.name, candidate.name, candidate))
    return found


def parse_run_config(test_dir: Path) -> dict[str, Any]:
    config_path = test_dir / "config" / "location_burst.json"
    try:
        cfg = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "antennas": cfg.get("antennas"),
        "transmitPower": cfg.get("transmitPower"),
        "radioStopConditions": cfg.get("radioStopConditions"),
    }


def build_run(
    archive_stem: str,
    test_id: str,
    test_dir: Path,
    antennas: dict[str, dict[str, Any]],
    tags: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    m = TEST_DIR_RE.match(test_id)
    test_num, test_ts = m.group(1), m.group(2)
    layout = detect_layout(archive_stem)

    tag_entries: dict[str, dict[str, Any]] = {}
    total_reads = 0
    unmatched_antennas: set[str] = set()

    for epc_file in sorted((test_dir / "raw_data").glob("*.json")):
        epc = epc_file.stem.lower()
        try:
            events = json.loads(epc_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        tag_meta = tags.get(epc)
        per_antenna: dict[str, list[dict[str, Any]]] = defaultdict(list)
        series: list[dict[str, Any]] = []

        for event in events:
            data = event.get("data", {})
            rdr = reader_from_ip(event.get("reader_ip"))
            ant = data.get("antenna")
            if rdr is None or ant is None:
                continue
            rdr_ant = f"Rdr-{rdr}-Ant-{ant}"
            rssi = data.get("peakRssi")
            ts = event.get("timestamp")
            per_antenna[rdr_ant].append(
                {"rssi": rssi, "phase": data.get("phase"), "channel": data.get("channel"), "ts": ts}
            )
            series.append({"t": ts, "antenna": rdr_ant, "rssi": rssi})
            if rdr_ant not in antennas:
                unmatched_antennas.add(rdr_ant)

        reads_summary: dict[str, dict[str, Any]] = {}
        for rdr_ant, events_for_ant in per_antenna.items():
            rssi_vals = [e["rssi"] for e in events_for_ant if e["rssi"] is not None]
            ts_vals = sorted(e["ts"] for e in events_for_ant if e["ts"])
            reads_summary[rdr_ant] = {
                "count": len(events_for_ant),
                "avgRssi": sum(rssi_vals) / len(rssi_vals) if rssi_vals else None,
                "minRssi": min(rssi_vals) if rssi_vals else None,
                "maxRssi": max(rssi_vals) if rssi_vals else None,
                "firstSeen": ts_vals[0] if ts_vals else None,
                "lastSeen": ts_vals[-1] if ts_vals else None,
                # Per-read [channel_MHz, phase_deg, rssi] for phase-frequency ranging.
                "pc": [
                    [e["channel"], e["phase"], e["rssi"]]
                    for e in events_for_ant
                    if e["channel"] is not None and e["phase"] is not None
                ],
            }

        series.sort(key=lambda e: e["t"] or "")
        total_reads += len(series)

        entry: dict[str, Any] = {"epc": epc, "reads": reads_summary, "series": series}
        entry.update(tag_meta if tag_meta else {"box": None, "tagNum": None, "side": None,
                                                 "row": None, "dist": None, "x": None,
                                                 "y": None, "z": None, "px": None, "py": None})
        tag_entries[epc] = entry

    if unmatched_antennas:
        print(f"  [warn] {test_id}: reads reference unknown antennas: {sorted(unmatched_antennas)}")

    return {
        "testId": test_id,
        "testNum": int(test_num),
        "timestamp": test_ts,
        "layout": layout,
        "sourceArchive": archive_stem,
        "config": parse_run_config(test_dir),
        "tagCount": len(tag_entries),
        "readCount": total_reads,
        "tags": tag_entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-extract", action="store_true",
                        help="don't run the zip-extraction step first")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output directory for generated JSON (default: {DEFAULT_OUT})")
    args = parser.parse_args(argv)

    if not args.skip_extract:
        print("== extracting any new raw_data zips ==")
        unzip_raw_data.main(["--src", str(RAW_DIR), "--out", str(EXTRACTED_DIR)])

    out_dir: Path = args.out
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print("== loading ground truth ==")
    antennas = load_antennas()
    tags_by_layout = {"1": load_tags("TagLoc1.csv"), "2": load_tags("TagLoc2.csv")}

    (out_dir / "antennas.json").write_text(json.dumps(list(antennas.values()), indent=2))

    print("== finding test runs ==")
    test_dirs = find_test_dirs(EXTRACTED_DIR)
    seen_ids: set[str] = set()
    manifest: list[dict[str, Any]] = []

    for archive_stem, test_id, test_dir in test_dirs:
        if test_id in seen_ids:
            print(f"  [dedupe] {test_id} already processed (from another archive), skipping {archive_stem}")
            continue
        seen_ids.add(test_id)

        try:
            layout = detect_layout(archive_stem)
        except ValueError as exc:
            print(f"  [skip] {test_id} in {archive_stem}: {exc}")
            continue

        print(f"  [run] {test_id} (layout {layout}, from {archive_stem})")
        run_data = build_run(archive_stem, test_id, test_dir, antennas, tags_by_layout[layout])
        (runs_dir / f"{test_id}.json").write_text(json.dumps(run_data))

        manifest.append({
            "testId": run_data["testId"],
            "testNum": run_data["testNum"],
            "timestamp": run_data["timestamp"],
            "layout": run_data["layout"],
            "sourceArchive": run_data["sourceArchive"],
            "tagCount": run_data["tagCount"],
            "readCount": run_data["readCount"],
            # Each run is a single TX power level from the collection sweep
            # (30 dBm down to 3 dBm). It shifts every RSSI value by up to 27 dB,
            # so downstream it is both a corpus filter and a model input.
            "transmitPower": (run_data.get("config") or {}).get("transmitPower"),
        })

    manifest.sort(key=lambda r: r["timestamp"])
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {len(manifest)} run(s) + antennas.json + manifest.json to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
