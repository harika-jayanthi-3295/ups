"""Train the Step-2 GNN on the 40% train split and save artifacts/gnn.pt."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from common import ARTIFACTS_DIR, HEADS
from dataset import build_dataset
from features import xyz_norm_stats
from gnn_model import GNN, build_edge_index, build_graphs, save_gnn
from nn_common import class_weights, metrics_from_preds, multi_head_loss, set_seed

HIDDEN = 96
EMB_DIM = 16
EPOCHS = 150
LR = 1e-3
BATCH = 64


def main() -> None:
    set_seed()
    ds = build_dataset()
    print(f"train runs={len(ds.train_ids)} samples={len(ds.train)} | "
          f"test runs={len(ds.test_ids)} samples={len(ds.test)}")

    xyz = ds.antenna_xyz
    xyz_norm = xyz_norm_stats(xyz)
    edge_index = build_edge_index(ds.antennas)

    train_graphs = build_graphs(ds.train, ds.scaler, xyz, xyz_norm, edge_index)
    loader = DataLoader(train_graphs, batch_size=BATCH, shuffle=True)

    model = GNN(len(ds.antenna_order), hidden=HIDDEN, emb_dim=EMB_DIM)
    weights = class_weights(ds.train)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for batch in loader:
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.ant_idx, batch.batch)
            targets = {name: batch[name] for name, _ in HEADS}
            loss = multi_head_loss(logits, targets, weights)
            loss.backward()
            opt.step()
            total += float(loss) * batch.num_graphs
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss={total / len(train_graphs):.4f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / "gnn.pt"
    save_gnn(model, ds.scaler, ds.antenna_order, xyz, xyz_norm, edge_index, HIDDEN, EMB_DIM, path)

    # quick train-set sanity metrics
    from gnn_model import load_gnn, predict_gnn

    bundle = load_gnn(path)
    preds = predict_gnn(bundle, ds.train)
    trues = {name: np.array([s.labels[name] for s in ds.train]) for name, _ in HEADS}
    m = metrics_from_preds(preds, trues)
    print("train metrics:", {k: (round(v["accuracy"], 3) if isinstance(v, dict) else v) for k, v in m.items()})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
