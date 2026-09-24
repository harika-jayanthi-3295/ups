"""model_2: 2D CNN over the antenna x channel phase "spectrogram".

Each tag is rendered as a [3, 32, 50] image: plane 0/1 hold the mean cos/sin of
phase in every (antenna, channel) cell, plane 2 is the present mask. Convolving
along the channel (frequency) axis lets the network read the phase-vs-frequency
structure that encodes antenna-tag range, much like a mel-spectrogram exposes
frequency structure for ASR. A shared conv trunk feeds the four heads.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from common import HEADS
from features_pc import IMG_PLANES, N_CHAN
from dataset_pc import PCSample


def samples_to_images(samples):
    """Return (imgs[N, IMG_PLANES, 32, N_CHAN], labels{name:[N]})."""
    imgs = np.stack([s.img for s in samples]).astype(np.float32)
    labels = {
        name: torch.tensor([s.labels[name] for s in samples], dtype=torch.long)
        for name, _ in HEADS
    }
    return torch.from_numpy(imgs), labels


class PhaseCNN(nn.Module):
    def __init__(self, n_antennas: int = 32, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(IMG_PLANES, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 2 * width, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # pool only along channel/freq axis
            nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * width),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 5)),
        )
        trunk = 4 * width * 4 * 5
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(trunk, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.heads = nn.ModuleDict({name: nn.Linear(256, n_cls) for name, n_cls in HEADS})

    def forward(self, x: torch.Tensor) -> dict:
        h = self.features(x)
        h = self.proj(h)
        return {name: head(h) for name, head in self.heads.items()}


def save_cnn(model, antenna_order, width, path):
    torch.save(
        {"state_dict": model.state_dict(), "antenna_order": antenna_order, "width": width},
        path,
    )


def load_cnn(path):
    ckpt = torch.load(path, weights_only=False)
    model = PhaseCNN(n_antennas=len(ckpt["antenna_order"]), width=ckpt["width"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {"model": model, "antenna_order": ckpt["antenna_order"]}


@torch.no_grad()
def predict_cnn(bundle, samples, batch_size: int = 256):
    imgs, _ = samples_to_images(samples)
    model = bundle["model"]
    preds = {name: [] for name, _ in HEADS}
    for i in range(0, imgs.shape[0], batch_size):
        logits = model(imgs[i : i + batch_size])
        for name, _ in HEADS:
            preds[name].append(logits[name].argmax(dim=1).cpu().numpy())
    return {name: np.concatenate(v) if v else np.array([]) for name, v in preds.items()}
