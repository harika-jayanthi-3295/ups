"""Train model_6 (fused RSSI + phase/channel Transformer) -> artifacts/transformer_fused.pt.

Hyperparameters match model_4's train_transformer_pc.py so the comparison
isolates the fused feature set.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from common import ARTIFACTS_DIR, HEADS
from dataset_fused import build_fused_dataset
from nn_common import class_weights, metrics_from_preds, multi_head_loss, set_seed
from transformer_fused_model import TagTransformerFused, samples_to_tensors, save_transformer_fused

D_MODEL = 64
NHEAD = 4
LAYERS = 2
EPOCHS = 200
LR = 1e-3
BATCH = 64


def _xyz_norm(xyz):
    mean = xyz.mean(axis=0)
    std = xyz.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (xyz - mean) / std


def main() -> None:
    set_seed()
    ds = build_fused_dataset()
    print(f"train runs={len(ds.train_ids)} samples={len(ds.train)} | "
          f"test runs={len(ds.test_ids)} samples={len(ds.test)}")

    xyz_norm = _xyz_norm(ds.antenna_xyz)

    feats, labels = samples_to_tensors(ds.train, ds.scaler)
    model = TagTransformerFused(xyz_norm, d_model=D_MODEL, nhead=NHEAD, layers=LAYERS)
    weights = class_weights(ds.train)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    n = feats.shape[0]
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            logits = model(feats[idx])
            targets = {name: labels[name][idx] for name, _ in HEADS}
            loss = multi_head_loss(logits, targets, weights)
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        if epoch % 15 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss={total / n:.4f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / "transformer_fused.pt"
    save_transformer_fused(model, ds.scaler, ds.antenna_order, xyz_norm, D_MODEL, NHEAD, LAYERS, path)

    from transformer_fused_model import load_transformer_fused, predict_transformer_fused

    bundle = load_transformer_fused(path)
    preds = predict_transformer_fused(bundle, ds.train)
    trues = {name: np.array([s.labels[name] for s in ds.train]) for name, _ in HEADS}
    m = metrics_from_preds(preds, trues)
    print("train metrics:", {k: (round(v["accuracy"], 3) if isinstance(v, dict) else v) for k, v in m.items()})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
