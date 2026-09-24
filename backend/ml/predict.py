"""Export per-run predictions for the dashboard.

For every run, writes dashboard/data/predictions/<testId>.json mapping each EPC
to the GNN and Transformer predicted descriptors plus the actual one, so the
dashboard can show predicted-vs-actual position for any tag.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "model_0"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "model_1"))

import numpy as np

from common import ARTIFACTS_DIR, HEADS, PREDICTIONS_DIR
from dataset import build_dataset
from gnn_model import load_gnn, predict_gnn
from labels import decode_labels
from transformer_model import load_transformer, predict_transformer


def _decode(preds, i) -> dict:
    return decode_labels({name: int(preds[name][i]) for name, _ in HEADS})


def main() -> None:
    ds = build_dataset()
    gnn = load_gnn(ARTIFACTS_DIR / "gnn.pt")
    tr = load_transformer(ARTIFACTS_DIR / "transformer.pt")

    all_samples = ds.train + ds.test
    split_of = {tid: "train" for tid in ds.train_ids}
    split_of.update({tid: "test" for tid in ds.test_ids})

    gnn_preds = predict_gnn(gnn, all_samples)
    tr_preds = predict_transformer(tr, all_samples)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    by_run: dict[str, dict] = {}
    for i, s in enumerate(all_samples):
        entry = by_run.setdefault(s.test_id, {})
        entry[s.epc] = {
            "actual": decode_labels(s.labels),
            "gnn": _decode(gnn_preds, i),
            "transformer": _decode(tr_preds, i),
        }

    # Runs can leave the corpus (e.g. dropped by MIN_POWER_DBM). Their old
    # prediction files would otherwise linger and be served as current.
    current = {f"{tid}.json" for tid in by_run} | {"index.json"}
    for stale in PREDICTIONS_DIR.glob("*.json"):
        if stale.name not in current:
            print(f"  removing stale prediction file {stale.name}")
            stale.unlink()

    index = []
    for test_id, tags in by_run.items():
        (PREDICTIONS_DIR / f"{test_id}.json").write_text(
            json.dumps({"testId": test_id, "split": split_of.get(test_id, "test"), "tags": tags})
        )
        index.append({"testId": test_id, "split": split_of.get(test_id, "test"), "predictedTags": len(tags)})
    (PREDICTIONS_DIR / "index.json").write_text(json.dumps(index, indent=2))
    print(f"wrote {len(index)} prediction files to {PREDICTIONS_DIR}")


if __name__ == "__main__":
    main()
