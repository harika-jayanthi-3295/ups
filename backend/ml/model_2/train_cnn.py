"""Train model_2 (phase/channel CNN) on the 40% split -> artifacts/cnn_pc.pt."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from common import ARTIFACTS_DIR, HEADS
from dataset_pc import build_pc_dataset
from nn_common import class_weights, metrics_from_preds, multi_head_loss, set_seed
from cnn_model import PhaseCNN, samples_to_images, save_cnn

WIDTH = 32
EPOCHS = 120
LR = 1e-3
BATCH = 64


def main() -> None:
    set_seed()
    ds = build_pc_dataset()
    print(f"train runs={len(ds.train_ids)} samples={len(ds.train)} | "
          f"test runs={len(ds.test_ids)} samples={len(ds.test)}")

    imgs, labels = samples_to_images(ds.train)
    model = PhaseCNN(n_antennas=len(ds.antenna_order), width=WIDTH)
    weights = class_weights(ds.train)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    n = imgs.shape[0]
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            logits = model(imgs[idx])
            targets = {name: labels[name][idx] for name, _ in HEADS}
            loss = multi_head_loss(logits, targets, weights)
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss={total / n:.4f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / "cnn_pc.pt"
    save_cnn(model, ds.antenna_order, WIDTH, path)

    from cnn_model import load_cnn, predict_cnn

    bundle = load_cnn(path)
    preds = predict_cnn(bundle, ds.train)
    trues = {name: np.array([s.labels[name] for s in ds.train]) for name, _ in HEADS}
    m = metrics_from_preds(preds, trues)
    print("train metrics:", {k: (round(v["accuracy"], 3) if isinstance(v, dict) else v) for k, v in m.items()})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
