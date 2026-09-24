"""model_3: GNN over the antenna graph using phase/channel range features.

Same message-passing architecture as model_0's GNN, but each node carries the
phase/channel-derived features (estimated range, phase cos/sin, fit confidence)
instead of RSSI statistics, plus the antenna's static position.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_max_pool, global_mean_pool

from common import HEADS
from features_pc import PC_DIM, PCScaler
from dataset_pc import PCSample

NODE_STATIC_DIM = 3  # normalized x, y, z
NODE_FEAT_DIM = PC_DIM + 1 + NODE_STATIC_DIM  # pc + present + xyz


def build_edge_index(antennas, k: int = 6) -> torch.Tensor:
    """kNN-by-position edges plus same-reader edges, undirected."""
    xyz = np.array([[a.x, a.y, a.z] for a in antennas], dtype=np.float32)
    n = len(antennas)
    edges: set[tuple[int, int]] = set()
    for i in range(n):
        d = ((xyz - xyz[i]) ** 2).sum(axis=1)
        for j in np.argsort(d)[1 : k + 1]:
            edges.add((i, int(j)))
            edges.add((int(j), i))
    for i in range(n):
        for j in range(n):
            if i != j and antennas[i].reader == antennas[j].reader:
                edges.add((i, j))
    src, dst = zip(*sorted(edges))
    return torch.tensor([src, dst], dtype=torch.long)


def sample_to_graph(sample, scaler, xyz_norm, xyz, edge_index) -> Data:
    mean, std = xyz_norm
    scaled = scaler.transform(sample.pc, sample.present)           # [32, PC_DIM]
    xyz_n = (xyz - mean) / std                                     # [32, 3]
    x = np.concatenate([scaled, sample.present[:, None], xyz_n], axis=1).astype(np.float32)
    data = Data(x=torch.from_numpy(x), edge_index=edge_index)
    for name, _ in HEADS:
        data[name] = torch.tensor([sample.labels[name]], dtype=torch.long)
    data.ant_idx = torch.arange(x.shape[0], dtype=torch.long)
    return data


class GNNPC(nn.Module):
    def __init__(self, n_antennas: int, hidden: int = 96, emb_dim: int = 16):
        super().__init__()
        self.node_emb = nn.Embedding(n_antennas, emb_dim)
        self.conv1 = SAGEConv(NODE_FEAT_DIM + emb_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        trunk = 2 * hidden
        self.heads = nn.ModuleDict({name: nn.Linear(trunk, n_cls) for name, n_cls in HEADS})

    def forward(self, x, edge_index, ant_idx, batch):
        h = torch.cat([x, self.node_emb(ant_idx)], dim=1)
        h = self.act(self.conv1(h, edge_index))
        h = self.dropout(h)
        h = self.act(self.conv2(h, edge_index))
        h = self.act(self.conv3(h, edge_index))
        pooled = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        return {name: head(pooled) for name, head in self.heads.items()}


def build_graphs(samples, scaler, xyz, xyz_norm, edge_index):
    return [sample_to_graph(s, scaler, xyz_norm, xyz, edge_index) for s in samples]


def save_gnn_pc(model, scaler, antenna_order, xyz, xyz_norm, edge_index, hidden, emb_dim, path):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "scaler": scaler.to_dict(),
            "antenna_order": antenna_order,
            "xyz": xyz.tolist(),
            "xyz_norm": [xyz_norm[0].tolist(), xyz_norm[1].tolist()],
            "edge_index": edge_index.tolist(),
            "hidden": hidden,
            "emb_dim": emb_dim,
        },
        path,
    )


def load_gnn_pc(path):
    ckpt = torch.load(path, weights_only=False)
    xyz = np.array(ckpt["xyz"], dtype=np.float32)
    xyz_norm = (np.array(ckpt["xyz_norm"][0], dtype=np.float32), np.array(ckpt["xyz_norm"][1], dtype=np.float32))
    edge_index = torch.tensor(ckpt["edge_index"], dtype=torch.long)
    model = GNNPC(len(ckpt["antenna_order"]), hidden=ckpt["hidden"], emb_dim=ckpt["emb_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {
        "model": model,
        "scaler": PCScaler.from_dict(ckpt["scaler"]),
        "antenna_order": ckpt["antenna_order"],
        "xyz": xyz,
        "xyz_norm": xyz_norm,
        "edge_index": edge_index,
    }


@torch.no_grad()
def predict_gnn_pc(bundle, samples, batch_size: int = 256):
    from torch_geometric.loader import DataLoader

    graphs = build_graphs(samples, bundle["scaler"], bundle["xyz"], bundle["xyz_norm"], bundle["edge_index"])
    model = bundle["model"]
    preds = {name: [] for name, _ in HEADS}
    for batch in DataLoader(graphs, batch_size=batch_size):
        logits = model(batch.x, batch.edge_index, batch.ant_idx, batch.batch)
        for name, _ in HEADS:
            preds[name].append(logits[name].argmax(dim=1).cpu().numpy())
    return {name: np.concatenate(v) if v else np.array([]) for name, v in preds.items()}
