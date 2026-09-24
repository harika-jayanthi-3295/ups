"""Step-1: antenna-vs-tag read-pattern summary per test run.

Writes dashboard/data/read_patterns.json (consumed by the dashboard) and prints
a compact per-run table: how many antennas were active, how much each antenna
contributed, tag coverage, and a weak-run flag for low-read runs.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))

from common import MANIFEST_JSON, READ_PATTERNS_JSON, RUNS_DIR

WEAK_READ_THRESHOLD = 800  # runs below this are flagged as weak/sparse


def summarize_run(run: dict) -> dict:
    reads_per_ant: dict[str, int] = defaultdict(int)
    rssi_sum: dict[str, float] = defaultdict(float)
    tags_per_ant: dict[str, int] = defaultdict(int)
    antennas_per_tag: list[int] = []

    for tag in run["tags"].values():
        hit = tag.get("reads", {})
        if hit:
            antennas_per_tag.append(len(hit))
        for key, r in hit.items():
            c = r.get("count", 0)
            reads_per_ant[key] += c
            rssi_sum[key] += r.get("avgRssi", 0.0) * c
            tags_per_ant[key] += 1

    avg_rssi = {k: round(rssi_sum[k] / reads_per_ant[k], 1) for k in reads_per_ant if reads_per_ant[k]}
    top = sorted(reads_per_ant.items(), key=lambda kv: kv[1], reverse=True)[:5]
    n_tags_detected = sum(1 for t in run["tags"].values() if t.get("reads"))

    return {
        "testId": run["testId"],
        "layout": str(run["layout"]),
        "tagCount": run["tagCount"],
        "readCount": run["readCount"],
        "activeAntennas": len(reads_per_ant),
        "tagsDetected": n_tags_detected,
        "avgAntennasPerTag": round(sum(antennas_per_tag) / len(antennas_per_tag), 2) if antennas_per_tag else 0.0,
        "readsPerAntenna": dict(sorted(reads_per_ant.items())),
        "avgRssiPerAntenna": avg_rssi,
        "tagsPerAntenna": dict(sorted(tags_per_ant.items())),
        "topAntennas": [{"antenna": k, "reads": v} for k, v in top],
        "weak": run["readCount"] < WEAK_READ_THRESHOLD,
    }


def main() -> None:
    manifest = json.loads(MANIFEST_JSON.read_text())
    summaries = []
    for entry in manifest:
        path = RUNS_DIR / f"{entry['testId']}.json"
        if not path.exists():
            continue
        summaries.append(summarize_run(json.loads(path.read_text())))

    READ_PATTERNS_JSON.write_text(json.dumps(summaries, indent=2))

    print(f"{'run':<32} {'lay':<3} {'reads':>7} {'act':>4} {'det':>4} {'a/tag':>6} {'top antenna':>14} weak")
    print("-" * 86)
    for s in summaries:
        top = s["topAntennas"][0]["antenna"] if s["topAntennas"] else "-"
        print(
            f"{s['testId']:<32} {s['layout']:<3} {s['readCount']:>7} {s['activeAntennas']:>4} "
            f"{s['tagsDetected']:>4} {s['avgAntennasPerTag']:>6} {top:>14} {'YES' if s['weak'] else ''}"
        )
    print(f"\nWrote {READ_PATTERNS_JSON}")


if __name__ == "__main__":
    main()
