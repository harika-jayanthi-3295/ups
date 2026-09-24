"""Evaluate all seven models on the held-out runs and write artifacts/metrics.json.

model_0 GNN (RSSI) and model_1 Transformer (RSSI) reuse the original dataset;
model_2 CNN, model_3 GNN and model_4 Transformer use the phase/channel dataset;
model_5 GNN and model_6 Transformer use the fused RSSI + phase/channel dataset.

Results are broken down by TX power as well as by layout and run. That is not
cosmetic: the corpus is a power sweep and a pooled average silently mixes runs
where every tag is readable with runs where most are not, so a single headline
number says more about which powers landed in the test split than about the
model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
for sub in ("model_0", "model_1", "model_2", "model_3", "model_4", "model_5", "model_6"):
    sys.path.insert(0, str(BASE / sub))

import numpy as np

from common import (
    ARTIFACTS_DIR,
    HEADS,
    MANIFEST_JSON,
    MIN_POWER_DBM,
    N_JOINT,
    SUBHEADS,
    excluded_runs,
)
from dataset import build_dataset
from dataset_pc import build_pc_dataset
from dataset_fused import build_fused_dataset
from nn_common import metrics_from_preds, metrics_with_coverage, save_json

from gnn_model import load_gnn, predict_gnn
from transformer_model import load_transformer, predict_transformer
from cnn_model import load_cnn, predict_cnn
from gnn_pc_model import load_gnn_pc, predict_gnn_pc
from transformer_pc_model import load_transformer_pc, predict_transformer_pc
from gnn_fused_model import load_gnn_fused, predict_gnn_fused
from transformer_fused_model import load_transformer_fused, predict_transformer_fused


def _trues(samples):
    return {name: np.array([s.labels[name] for s in samples]) for name, _ in HEADS}


def _subset(preds, mask):
    return {name: np.asarray(v)[mask] for name, v in preds.items()}


def evaluate_model(preds, samples) -> dict:
    trues = _trues(samples)
    layouts = np.array([s.layout for s in samples])
    runs = np.array([s.test_id for s in samples])
    powers = np.array([s.power_dbm if s.power_dbm is not None else -1.0 for s in samples])

    result = {
        "overall": metrics_with_coverage(preds, trues, samples),
        "perPower": {},
        "perLayout": {},
        "perRun": {},
    }
    for power in sorted(set(powers.tolist()), reverse=True):
        mask = powers == power
        result["perPower"][f"{power:g}"] = metrics_with_coverage(
            _subset(preds, mask), _subset(trues, mask), [s for s, m in zip(samples, mask) if m]
        )
    for layout in sorted(set(layouts)):
        mask = layouts == layout
        result["perLayout"][layout] = metrics_from_preds(_subset(preds, mask), _subset(trues, mask))
    for run in sorted(set(runs)):
        mask = runs == run
        result["perRun"][run] = metrics_from_preds(_subset(preds, mask), _subset(trues, mask))
    return result


def _head_acc(block) -> str:
    return "  ".join(f"{name}={block[name]['accuracy']:.2f}" for name in SUBHEADS)


LABELS = {
    "gnn": "model_0 GNN (RSSI)",
    "transformer": "model_1 Transformer (RSSI)",
    "cnn_pc": "model_2 CNN (phase image)",
    "gnn_pc": "model_3 GNN (phase/channel)",
    "transformer_pc": "model_4 Transformer (phase/channel)",
    "gnn_fused": "model_5 GNN (fused RSSI+PC)",
    "transformer_fused": "model_6 Transformer (fused RSSI+PC)",
}


def main() -> None:
    ds = build_dataset()
    dpc = build_pc_dataset()
    dfu = build_fused_dataset()

    gnn = load_gnn(ARTIFACTS_DIR / "gnn.pt")
    tr = load_transformer(ARTIFACTS_DIR / "transformer.pt")
    cnn = load_cnn(ARTIFACTS_DIR / "cnn_pc.pt")
    gnn_pc = load_gnn_pc(ARTIFACTS_DIR / "gnn_pc.pt")
    tr_pc = load_transformer_pc(ARTIFACTS_DIR / "transformer_pc.pt")
    gnn_fu = load_gnn_fused(ARTIFACTS_DIR / "gnn_fused.pt")
    tr_fu = load_transformer_fused(ARTIFACTS_DIR / "transformer_fused.pt")

    manifest = json.loads(MANIFEST_JSON.read_text())
    dropped = excluded_runs(manifest)

    results = {
        "split": {
            "trainRuns": ds.train_ids,
            "testRuns": ds.test_ids,
            "trainSamples": len(ds.train),
            "testSamples": len(ds.test),
            "minPowerDbm": MIN_POWER_DBM,
            "nJointClasses": N_JOINT,
            "excludedRuns": [
                {"testId": r["testId"], "transmitPower": r.get("transmitPower")} for r in dropped
            ],
        },
        "gnn": evaluate_model(predict_gnn(gnn, ds.test), ds.test),
        "transformer": evaluate_model(predict_transformer(tr, ds.test), ds.test),
        "cnn_pc": evaluate_model(predict_cnn(cnn, dpc.test), dpc.test),
        "gnn_pc": evaluate_model(predict_gnn_pc(gnn_pc, dpc.test), dpc.test),
        "transformer_pc": evaluate_model(predict_transformer_pc(tr_pc, dpc.test), dpc.test),
        "gnn_fused": evaluate_model(predict_gnn_fused(gnn_fu, dfu.test), dfu.test),
        "transformer_fused": evaluate_model(predict_transformer_fused(tr_fu, dfu.test), dfu.test),
    }
    save_json(ARTIFACTS_DIR / "metrics.json", results)

    print(f"\nCorpus: TX power >= {MIN_POWER_DBM:g} dBm, {N_JOINT} joint classes")
    dropped_desc = ", ".join(f"{r['testId']} ({r.get('transmitPower')} dBm)" for r in dropped)
    print(f"Excluded as unlearnable: {dropped_desc or 'none'}")
    print(f"Held-out: {len(ds.test_ids)} runs, {len(ds.test)} tags\n")

    for key, name in LABELS.items():
        o = results[key]["overall"]
        print(f"[{name:38s}] {_head_acc(o)}  exact={o['descriptorExactMatch']:.3f}")

    print("\nexact match by TX power:")
    powers = sorted(results["transformer"]["perPower"], key=float, reverse=True)
    print(f"  {'dBm':>5} {'n':>6} {'noRead':>7}  " + "  ".join(f"{k:>9}" for k in LABELS))
    for p in powers:
        row = f"  {p:>5} {results['transformer']['perPower'][p]['n']:6d} " \
              f"{results['transformer']['perPower'][p]['noReadRate']*100:6.1f}%  "
        row += "  ".join(f"{results[k]['perPower'][p]['descriptorExactMatch']*100:8.1f}%" for k in LABELS)
        print(row)

    print(f"\nwrote {ARTIFACTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
