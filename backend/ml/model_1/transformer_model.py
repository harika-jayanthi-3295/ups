"""Step-3: Transformer with position embeddings for tag zone prediction.

Each of the 32 antennas is a token. A token's value is that tag's read
statistics at the antenna; its identity/position come from a learned antenna
embedding plus a position embedding of the antenna's real (x, y, z). A CLS
token aggregates the self-attention output and feeds four classification heads.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from common import HEADS
from features import DYN_DIM, FeatureScaler
from dataset import Sample

TOKEN_FEAT_DIM = DYN_DIM + 1  # dyn stats + present flag


def samples_to_tensors(samples: list[Sample], scaler: FeatureScaler):
    """Return (feats[N,32,TOKEN_FEAT_DIM], labels{name:[N]})."""
    feats = np.stack(
        [
            np.concatenate(
                [scaler.transform(s.dyn, s.present), s.present[:, None]], axis=1
            )
            for s in samples
        ]
    ).astype(np.float32)
    labels = {
        name: torch.tensor([s.labels[name] for s in samples], dtype=torch.long)
        for name, _ in HEADS
    }
    return torch.from_numpy(feats), labels


class TagTransformer(nn.Module):
    def __init__(self, xyz_norm: np.ndarray, d_model: int = 64, nhead: int = 4, layers: int = 2):
        super().__init__()
        n_ant = xyz_norm.shape[0]
        self.register_buffer("xyz", torch.from_numpy(xyz_norm.astype(np.float32)))
        self.input_proj = nn.Linear(TOKEN_FEAT_DIM, d_model)
        self.ant_emb = nn.Embedding(n_ant, d_model)
        self.pos_proj = nn.Linear(3, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(d_model, n_cls) for name, n_cls in HEADS}
        )

    def forward(self, feats: torch.Tensor) -> dict:
        b, n, _ = feats.shape
        idx = torch.arange(n, device=feats.device)
        tok = self.input_proj(feats) + self.ant_emb(idx) + self.pos_proj(self.xyz)
        cls = self.cls.expand(b, -1, -1)
        seq = torch.cat([cls, tok], dim=1)
        out = self.encoder(seq)
        pooled = self.norm(out[:, 0])
        return {name: head(pooled) for name, head in self.heads.items()}


def save_transformer(model, scaler, antenna_order, xyz_norm, d_model, nhead, layers, path):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "scaler": scaler.to_dict(),
            "antenna_order": antenna_order,
            "xyz_norm": xyz_norm.tolist(),
            "d_model": d_model,
            "nhead": nhead,
            "layers": layers,
        },
        path,
    )


def load_transformer(path):
    ckpt = torch.load(path, weights_only=False)
    xyz_norm = np.array(ckpt["xyz_norm"], dtype=np.float32)
    model = TagTransformer(xyz_norm, d_model=ckpt["d_model"], nhead=ckpt["nhead"], layers=ckpt["layers"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {
        "model": model,
        "scaler": FeatureScaler.from_dict(ckpt["scaler"]),
        "antenna_order": ckpt["antenna_order"],
    }


@torch.no_grad()
def predict_transformer(bundle, samples, batch_size: int = 256):
    feats, _ = samples_to_tensors(samples, bundle["scaler"])
    model = bundle["model"]
    preds = {name: [] for name, _ in HEADS}
    for i in range(0, feats.shape[0], batch_size):
        logits = model(feats[i : i + batch_size])
        for name, _ in HEADS:
            preds[name].append(logits[name].argmax(dim=1).cpu().numpy())
    return {name: np.concatenate(v) if v else np.array([]) for name, v in preds.items()}
