"""Train model_3 (phase/channel GNN) on the 40% split -> artifacts/gnn_pc.pt."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from common import ARTIFACTS_DIR, HEADS
from dataset_pc import build_pc_dataset
from features_pc import PCScaler  # noqa: F401  (kept for parity/imports)
from gnn_pc_model import GNNPC, build_edge_index, build_graphs, save_gnn_pc
from nn_common import metrics_from_preds, multi_head_loss, set_seed
from nn_common import class_weights

HIDDEN = 96
EMB_DIM = 16
EPOCHS = 150
LR = 1e-3
BATCH = 64


def _xyz_norm(xyz):
    mean = xyz.mean(axis=0)
    std = xyz.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _class_weights(samples):
    # class_weights() lives in nn_common and expects .labels — PCSample has it.
    return class_weights(samples)


def main() -> None:
    set_seed()
    ds = build_pc_dataset()
    print(f"train runs={len(ds.train_ids)} samples={len(ds.train)} | "
          f"test runs={len(ds.test_ids)} samples={len(ds.test)}")

    xyz = ds.antenna_xyz
    xyz_norm = _xyz_norm(xyz)
    edge_index = build_edge_index(ds.antennas)

    train_graphs = build_graphs(ds.train, ds.scaler, xyz, xyz_norm, edge_index)
    loader = DataLoader(train_graphs, batch_size=BATCH, shuffle=True)

    model = GNNPC(len(ds.antenna_order), hidden=HIDDEN, emb_dim=EMB_DIM)
    weights = _class_weights(ds.train)
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
    path = ARTIFACTS_DIR / "gnn_pc.pt"
    save_gnn_pc(model, ds.scaler, ds.antenna_order, xyz, xyz_norm, edge_index, HIDDEN, EMB_DIM, path)

    from gnn_pc_model import load_gnn_pc, predict_gnn_pc

    bundle = load_gnn_pc(path)
    preds = predict_gnn_pc(bundle, ds.train)
    trues = {name: np.array([s.labels[name] for s in ds.train]) for name, _ in HEADS}
    m = metrics_from_preds(preds, trues)
    print("train metrics:", {k: (round(v["accuracy"], 3) if isinstance(v, dict) else v) for k, v in m.items()})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
