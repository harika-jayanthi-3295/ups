"""Ensemble the GNN and the Transformer over the joint label.

The two models are averaged in probability space. The previous version averaged
predicted *class indices*, which was meaningless even for the old four-head
setup and is worse for a 35-way joint label, where the midpoint of classes 3
and 31 is an unrelated class 17.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ml_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(ml_dir))
sys.path.insert(0, str(ml_dir / "model_0"))
sys.path.insert(0, str(ml_dir / "model_1"))

import numpy as np
import torch

from common import ARTIFACTS_DIR, HEADS, PREDICTIONS_DIR
from dataset import build_dataset
from labels import decode_labels
from nn_common import metrics_from_preds

from gnn_model import build_graphs, load_gnn
from transformer_model import load_transformer, samples_to_tensors


@torch.no_grad()
def gnn_proba(bundle, samples, batch_size: int = 256) -> np.ndarray:
    from torch_geometric.loader import DataLoader

    graphs = build_graphs(
        samples, bundle["scaler"], bundle["xyz"], bundle["xyz_norm"], bundle["edge_index"]
    )
    out = []
    for batch in DataLoader(graphs, batch_size=batch_size):
        logits = bundle["model"](batch.x, batch.edge_index, batch.ant_idx, batch.batch)
        out.append(torch.softmax(logits["joint"], dim=1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def transformer_proba(bundle, samples, batch_size: int = 256) -> np.ndarray:
    feats, _ = samples_to_tensors(samples, bundle["scaler"])
    out = []
    for i in range(0, feats.shape[0], batch_size):
        logits = bundle["model"](feats[i : i + batch_size])
        out.append(torch.softmax(logits["joint"], dim=1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ds = build_dataset()
    gnn_bundle = load_gnn(ARTIFACTS_DIR / "gnn.pt")
    trans_bundle = load_transformer(ARTIFACTS_DIR / "transformer.pt")

    p_gnn = gnn_proba(gnn_bundle, ds.test)
    p_trans = transformer_proba(trans_bundle, ds.test)
    p_ens = (p_gnn + p_trans) / 2.0

    preds = {
        "gnn": {"joint": p_gnn.argmax(1)},
        "transformer": {"joint": p_trans.argmax(1)},
        "ensemble": {"joint": p_ens.argmax(1)},
    }
    trues = {name: np.array([s.labels[name] for s in ds.test]) for name, _ in HEADS}

    # Same on-disk schema as predict.py so the dashboard can read either.
    by_run: dict[str, dict] = defaultdict(dict)
    for i, sample in enumerate(ds.test):
        by_run[sample.test_id][sample.epc] = {
            "actual": decode_labels(sample.labels),
            **{k: decode_labels({"joint": int(v["joint"][i])}) for k, v in preds.items()},
        }

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for run_id, tags in by_run.items():
        (PREDICTIONS_DIR / f"{run_id}.json").write_text(
            json.dumps({"testId": run_id, "split": "test", "tags": tags})
        )
        index.append({"testId": run_id, "split": "test", "predictedTags": len(tags)})
    (PREDICTIONS_DIR / "index.json").write_text(json.dumps(index, indent=2))
    print(f"saved ensemble predictions for {len(by_run)} runs")

    print("\n=== exact match ===")
    for key in ("gnn", "transformer", "ensemble"):
        m = metrics_from_preds(preds[key], trues)
        print(f"  {key:12s} {m['descriptorExactMatch']:.4f}")


if __name__ == "__main__":
    main()
