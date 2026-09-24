"""Hard example mining: identify and retrain on misclassified samples."""

from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "model_0"))
sys.path.insert(0, str(BASE / "model_1"))

import numpy as np
from collections import defaultdict

from common import HEADS
from dataset import build_dataset
from gnn_model import load_gnn, predict_gnn
from transformer_model import load_transformer, predict_transformer
from nn_common import metrics_from_preds


def identify_hard_examples(preds: dict, trues: dict, threshold: float = 0.5) -> list[int]:
    """Identify samples where model got predictions wrong.
    
    Args:
        preds: dict[head] = np.array of predictions
        trues: dict[head] = np.array of true labels
        threshold: confidence threshold (not used for hard negatives, but for future impl)
    
    Returns:
        List of indices where at least one head was wrong
    """
    hard_indices = set()
    
    # Find samples with any incorrect prediction
    for name, _ in HEADS:
        pred = np.asarray(preds[name])
        true = np.asarray(trues[name])
        wrong = pred != true
        hard_indices.update(np.where(wrong)[0].tolist())
    
    return sorted(list(hard_indices))


def main() -> None:
    print("=== Hard Example Mining ===\n")
    
    from common import ARTIFACTS_DIR
    
    print("Building dataset...")
    ds = build_dataset()
    
    print("Loading models...")
    gnn_bundle = load_gnn(ARTIFACTS_DIR / "gnn.pt")
    trans_bundle = load_transformer(ARTIFACTS_DIR / "transformer.pt")
    
    print("\nAnalyzing current performance...")
    gnn_preds = predict_gnn(gnn_bundle, ds.test)
    trans_preds = predict_transformer(trans_bundle, ds.test)
    trues = {name: np.array([s.labels[name] for s in ds.test]) for name, _ in HEADS}
    
    m_gnn = metrics_from_preds(gnn_preds, trues)
    m_trans = metrics_from_preds(trans_preds, trues)
    
    print(f"GNN exact match: {m_gnn['descriptorExactMatch']:.4f}")
    print(f"Transformer exact match: {m_trans['descriptorExactMatch']:.4f}")
    
    # Find hard examples
    print("\nIdentifying hard examples in training set...")
    hard_gnn = identify_hard_examples(
        predict_gnn(gnn_bundle, ds.train),
        {name: np.array([s.labels[name] for s in ds.train]) for name, _ in HEADS}
    )
    hard_trans = identify_hard_examples(
        predict_transformer(trans_bundle, ds.train),
        {name: np.array([s.labels[name] for s in ds.train]) for name, _ in HEADS}
    )
    
    hard_indices = set(hard_gnn) | set(hard_trans)
    print(f"Hard examples: {len(hard_indices)} out of {len(ds.train)} ({100*len(hard_indices)/len(ds.train):.1f}%)")
    
    # Show breakdown by head
    gnn_train_preds = predict_gnn(gnn_bundle, ds.train)
    trans_train_preds = predict_transformer(trans_bundle, ds.train)
    train_trues = {name: np.array([s.labels[name] for s in ds.train]) for name, _ in HEADS}
    
    print("\nHard examples by prediction head (GNN):")
    for name, _ in HEADS:
        pred = np.asarray(gnn_train_preds[name])
        true = np.asarray(train_trues[name])
        wrong = (pred != true).sum()
        print(f"  {name:8s}: {wrong:4d} errors ({100*wrong/len(true):.1f}%)")
    
    print("\nHard examples by prediction head (Transformer):")
    for name, _ in HEADS:
        pred = np.asarray(trans_train_preds[name])
        true = np.asarray(train_trues[name])
        wrong = (pred != true).sum()
        print(f"  {name:8s}: {wrong:4d} errors ({100*wrong/len(true):.1f}%)")


if __name__ == "__main__":
    main()
