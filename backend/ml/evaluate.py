"""Evaluate just the GNN and the Transformer on the held-out runs.

Superseded by evaluate_all.py, which covers all seven models and adds the
per-TX-power breakdown. Kept as a quick two-model check, and writes to its own
file so it cannot overwrite the full evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "model_0"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "model_1"))

import numpy as np

from common import ARTIFACTS_DIR, HEADS
from dataset import build_dataset
from gnn_model import load_gnn, predict_gnn
from nn_common import metrics_from_preds, save_json
from transformer_model import load_transformer, predict_transformer


def _trues(samples):
    return {name: np.array([s.labels[name] for s in samples]) for name, _ in HEADS}


def _subset(preds, mask):
    return {name: np.asarray(v)[mask] for name, v in preds.items()}


def evaluate_model(name, preds, samples) -> dict:
    trues = _trues(samples)
    layouts = np.array([s.layout for s in samples])
    runs = np.array([s.test_id for s in samples])

    result = {"overall": metrics_from_preds(preds, trues), "perLayout": {}, "perRun": {}}
    for layout in sorted(set(layouts)):
        mask = layouts == layout
        result["perLayout"][layout] = metrics_from_preds(_subset(preds, mask), _subset(trues, mask))
    for run in sorted(set(runs)):
        mask = runs == run
        result["perRun"][run] = metrics_from_preds(_subset(preds, mask), _subset(trues, mask))
    return result


def _head_acc(block) -> str:
    return "  ".join(f"{name}={block[name]['accuracy']:.2f}" for name, _ in HEADS)


def main() -> None:
    ds = build_dataset()
    gnn = load_gnn(ARTIFACTS_DIR / "gnn.pt")
    tr = load_transformer(ARTIFACTS_DIR / "transformer.pt")

    gnn_preds = predict_gnn(gnn, ds.test)
    tr_preds = predict_transformer(tr, ds.test)

    results = {
        "split": {"trainRuns": ds.train_ids, "testRuns": ds.test_ids,
                  "trainSamples": len(ds.train), "testSamples": len(ds.test)},
        "gnn": evaluate_model("gnn", gnn_preds, ds.test),
        "transformer": evaluate_model("transformer", tr_preds, ds.test),
    }
    save_json(ARTIFACTS_DIR / "metrics_two_model.json", results)

    print(f"\nHeld-out: {len(ds.test_ids)} runs, {len(ds.test)} tags\n")
    for model in ("gnn", "transformer"):
        o = results[model]["overall"]
        print(f"[{model}] overall  {_head_acc(o)}  exact={o['descriptorExactMatch']:.2f}")
        for layout, block in results[model]["perLayout"].items():
            print(f"    layout {layout}: {_head_acc(block)}  exact={block['descriptorExactMatch']:.2f}  (n={block['n']})")
    print(f"\nwrote {ARTIFACTS_DIR / 'metrics_two_model.json'}")


if __name__ == "__main__":
    main()
