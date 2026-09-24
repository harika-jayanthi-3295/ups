"""Train the Step-3 Transformer on the 40% train split -> artifacts/transformer.pt."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from common import ARTIFACTS_DIR, HEADS
from dataset import build_dataset
from features import xyz_norm_stats
from nn_common import class_weights, metrics_from_preds, multi_head_loss, set_seed
from transformer_model import TagTransformer, samples_to_tensors, save_transformer

D_MODEL = 64
NHEAD = 4
LAYERS = 2
EPOCHS = 200
LR = 1e-3
BATCH = 64


def main() -> None:
    set_seed()
    ds = build_dataset()
    print(f"train runs={len(ds.train_ids)} samples={len(ds.train)} | "
          f"test runs={len(ds.test_ids)} samples={len(ds.test)}")

    mean, std = xyz_norm_stats(ds.antenna_xyz)
    xyz_norm = (ds.antenna_xyz - mean) / std

    feats, labels = samples_to_tensors(ds.train, ds.scaler)
    model = TagTransformer(xyz_norm, d_model=D_MODEL, nhead=NHEAD, layers=LAYERS)
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
    path = ARTIFACTS_DIR / "transformer.pt"
    save_transformer(model, ds.scaler, ds.antenna_order, xyz_norm, D_MODEL, NHEAD, LAYERS, path)

    from transformer_model import load_transformer, predict_transformer

    bundle = load_transformer(path)
    preds = predict_transformer(bundle, ds.train)
    trues = {name: np.array([s.labels[name] for s in ds.train]) for name, _ in HEADS}
    m = metrics_from_preds(preds, trues)
    print("train metrics:", {k: (round(v["accuracy"], 3) if isinstance(v, dict) else v) for k, v in m.items()})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
