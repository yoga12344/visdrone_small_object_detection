"""
run_exp5_thresholds.py
======================
Experiment 5: Validation-Only Confidence Threshold Analysis
Analyzes precision-recall-F1 operating trade-offs across confidence thresholds [0.20, 0.25, 0.30, 0.40, 0.50]
strictly on the 106-image validation set.

Scientific Objective:
  In aerial drone surveillance, the threshold controls the false alarm vs detection
  completeness trade-off. We determine the optimal F1 operating point on validation data
  and FREEZE this threshold before final test evaluation.

Outputs:
  - results/experiments/thresholds/threshold_sweep.csv
  - results/experiments/thresholds/threshold_f1_curve.png
  - Optimal frozen threshold definition
"""

import gc
import json
import logging
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "scripts"))
from research_evaluator import evaluate_yolo_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp5_Thresholds")

EXP_ROOT = PROJECT_ROOT / "results" / "experiments" / "thresholds"
EXP_ROOT.mkdir(parents=True, exist_ok=True)


def main():
    logger.info("==================================================================")
    logger.info("  EXPERIMENT 5: VALIDATION-ONLY CONFIDENCE THRESHOLD ANALYSIS     ")
    logger.info("==================================================================")

    weights_path = PROJECT_ROOT / "results" / "trained" / "yolov8s" / "weights" / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}")

    thresholds = [0.20, 0.25, 0.30, 0.40, 0.50]
    results = []

    for conf in thresholds:
        logger.info(f"--- Evaluating Confidence Threshold {conf:.2f} on Validation Set ---")
        eval_res = evaluate_yolo_model(
            weights_path,
            split="val",
            imgsz=640,
            conf=conf,
            device=0,
            model_name=f"YOLOv8s_conf{int(conf*100)}"
        )
        results.append({
            "Confidence_Threshold": conf,
            "mAP@0.5": eval_res["mAP50"],
            "Precision": eval_res["precision"],
            "Recall": eval_res["recall"],
            "F1": eval_res["f1"],
            "Small_AP": eval_res["small_ap"],
            "Medium_AP": eval_res["medium_ap"],
            "Large_AP": eval_res["large_ap"],
            "Latency_ms": eval_res["latency_ms"],
            "FPS": eval_res["fps"],
            "Num_Predictions": eval_res["num_predictions"]
        })

    df = pd.DataFrame(results)
    csv_path = EXP_ROOT / "threshold_sweep.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved threshold sweep data to %s", csv_path)

    # Plot Precision, Recall, F1 curves
    plt.figure(figsize=(8, 5))
    plt.plot(df["Confidence_Threshold"], df["Precision"], marker='o', label="Precision", color="#1f77b4", linewidth=2)
    plt.plot(df["Confidence_Threshold"], df["Recall"], marker='s', label="Recall", color="#ff7f0e", linewidth=2)
    plt.plot(df["Confidence_Threshold"], df["F1"], marker='^', label="F1 Score", color="#2ca02c", linewidth=2.5)

    best_idx = df["F1"].idxmax()
    best_conf = df.loc[best_idx, "Confidence_Threshold"]
    best_f1 = df.loc[best_idx, "F1"]

    plt.axvline(best_conf, color="red", linestyle="--", alpha=0.7, label=f"Optimal F1 (conf={best_conf}, F1={best_f1:.4f})")
    plt.title("Confidence Threshold vs Detection Performance (Validation Set)", fontsize=13, fontweight="bold")
    plt.xlabel("Confidence Threshold", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    plot_path = EXP_ROOT / "threshold_f1_curve.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Saved threshold curve plot to %s", plot_path)

    print("\n" + "="*80)
    print("EXPERIMENT 5 SUMMARY (Confidence Threshold Analysis on Validation Set):")
    print(df.to_string(index=False))
    print(f"\nOptimal Operating Point (Maximum F1): conf = {best_conf:.2f} (F1 = {best_f1:.4f})")
    print("This threshold will be frozen for final evaluation.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
