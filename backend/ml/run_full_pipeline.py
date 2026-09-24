#!/usr/bin/env python3
"""Train all seven models, export predictions, and evaluate.

Every model now targets the single joint label over the 35 realisable
(region, row, reader, antenna) combinations, and every dataset is filtered to
runs at or above common.MIN_POWER_DBM.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from common import ARTIFACTS_DIR, DASHBOARD_DATA, MIN_POWER_DBM, N_JOINT, SUBHEADS

TRAIN_STEPS = [
    ("model_0", "train_gnn.py", "GNN (RSSI)"),
    ("model_1", "train_transformer.py", "Transformer (RSSI)"),
    ("model_2", "train_cnn.py", "CNN (phase image)"),
    ("model_3", "train_gnn_pc.py", "GNN (phase/channel)"),
    ("model_4", "train_transformer_pc.py", "Transformer (phase/channel)"),
    ("model_5", "train_gnn_fused.py", "GNN (fused)"),
    ("model_6", "train_transformer_fused.py", "Transformer (fused)"),
]


def run(cmd: list[str], cwd: Path, description: str) -> bool:
    print(f"\n{'=' * 64}\n>>> {description}\n{'=' * 64}")
    if subprocess.run(cmd, cwd=cwd).returncode != 0:
        print(f"FAILED: {description}")
        return False
    return True


def main() -> int:
    print(f"Pipeline: {N_JOINT}-class joint label, TX power >= {MIN_POWER_DBM:g} dBm")

    for i, (sub, script, name) in enumerate(TRAIN_STEPS, 1):
        if not run([sys.executable, script], BASE / sub, f"[{i}/{len(TRAIN_STEPS)}] training {name}"):
            return 1

    if not run([sys.executable, "evaluate_all.py"], BASE, "evaluating all models"):
        return 1
    if not run([sys.executable, "predict.py"], BASE, "exporting dashboard predictions"):
        return 1

    metrics_file = ARTIFACTS_DIR / "metrics.json"
    if metrics_file.exists():
        # The dashboard serves its own static copy; keep it in step with the run.
        shutil.copyfile(metrics_file, DASHBOARD_DATA / "metrics.json")
        metrics = json.loads(metrics_file.read_text())
        print(f"\n{'=' * 64}\nFINAL RESULTS\n{'=' * 64}")
        for model in [k for k in metrics if k != "split"]:
            m = metrics[model]["overall"]
            heads = "  ".join(f"{h}={m[h]['accuracy']:.3f}" for h in SUBHEADS)
            print(f"{model:20s} {heads}  EXACT={m['descriptorExactMatch']:.4f}")

    print("\nDashboard: cd /home/ups/dashboard && python3 -m http.server 8765")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
