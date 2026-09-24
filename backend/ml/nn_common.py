"""Shared training/eval helpers for every model."""

from __future__ import annotations

import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from common import ARTIFACTS_DIR, HEADS, SUBHEADS, joint_to_parts
from dataset import Sample


def set_seed(seed: int = 13) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(samples: list[Sample]) -> dict[str, torch.Tensor]:
    """Inverse-frequency weights per head to counter class imbalance."""
    weights = {}
    for name, n_cls in HEADS:
        counts = np.zeros(n_cls, dtype=np.float64)
        for s in samples:
            counts[s.labels[name]] += 1
        counts = np.where(counts == 0, 1.0, counts)
        w = counts.sum() / (n_cls * counts)
        weights[name] = torch.tensor(w, dtype=torch.float32)
    return weights


def multi_head_loss(logits: dict, targets: dict, weights: dict | None = None) -> torch.Tensor:
    total = 0.0
    for name, _ in HEADS:
        w = weights[name].to(logits[name].device) if weights else None
        total = total + F.cross_entropy(logits[name], targets[name], weight=w)
    return total


def expand_joint(joint: np.ndarray) -> dict[str, np.ndarray]:
    """Joint class indices -> {region,row,reader,antenna} index arrays."""
    parts = [joint_to_parts(int(j)) for j in np.asarray(joint).ravel()]
    return {name: np.array([p[name] for p in parts]) for name in SUBHEADS}


def metrics_from_preds(preds: dict, trues: dict) -> dict:
    """Per-component accuracy / macro-F1 plus full-descriptor exact match.

    The model has one head over the joint label, so exact match *is* the joint
    accuracy -- the components are decomposed only so the numbers stay
    comparable to the previous four-head runs. That equality is the point of
    the joint head: a four-head model could and did emit component
    combinations that no tag position can produce.
    """
    p_joint = np.asarray(preds["joint"])
    t_joint = np.asarray(trues["joint"])
    p_parts = expand_joint(p_joint)
    t_parts = expand_joint(t_joint)

    out: dict = {}
    for name in SUBHEADS:
        out[name] = {
            "accuracy": float((p_parts[name] == t_parts[name]).mean()),
            "macroF1": float(f1_score(t_parts[name], p_parts[name], average="macro", zero_division=0)),
        }
    out["joint"] = {
        "accuracy": float((p_joint == t_joint).mean()),
        "macroF1": float(f1_score(t_joint, p_joint, average="macro", zero_division=0)),
    }
    out["descriptorExactMatch"] = float((p_joint == t_joint).mean())
    out["n"] = int(len(t_joint))
    return out


def metrics_with_coverage(preds: dict, trues: dict, samples: list) -> dict:
    """metrics_from_preds plus how much of the result rests on tags that were
    never read.

    A tag no antenna saw has an all-zero feature row and is unpredictable by
    construction. Those samples are kept in the headline number so it stays
    honest, and split out here so a coverage problem cannot be mistaken for a
    model problem.
    """
    out = metrics_from_preds(preds, trues)
    read = np.array([s.total_reads > 0 for s in samples])
    out["noReadRate"] = float((~read).mean())
    if read.any():
        sub = metrics_from_preds(
            {"joint": np.asarray(preds["joint"])[read]},
            {"joint": np.asarray(trues["joint"])[read]},
        )
        out["exactMatchOnRead"] = sub["descriptorExactMatch"]
        out["nRead"] = sub["n"]
    else:
        out["exactMatchOnRead"] = 0.0
        out["nRead"] = 0
    return out


def save_json(path, payload) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
