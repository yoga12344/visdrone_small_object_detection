"""
run_exp2_resolution.py
======================
Experiment 2: Input Resolution Ablation
Compares input resolutions: 640x640 vs 768x768 vs 1024x1024.

Strict Protocol:
  - Uses the benchmark YOLOv8s architecture
  - Same dataset: data/processed/ (Validation split: 106 images)
  - Evaluates effect on small (<32px), medium (32-96px), and large (>96px) objects
  - Measures latency (ms/frame), FPS, and Peak VRAM (MB) with CUDA synchronization
  - Saves artifacts in:
      results/experiments/resolution/640/
      results/experiments/resolution/768/
      results/experiments/resolution/1024/
"""

import gc
import json
import logging
import os
import sys
from pathlib import Path
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "scripts"))
from research_evaluator import evaluate_yolo_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp2_Resolution")

EXP_ROOT = PROJECT_ROOT / "results" / "experiments" / "resolution"
EXP_ROOT.mkdir(parents=True, exist_ok=True)


def main():
    logger.info("==================================================================")
    logger.info("  EXPERIMENT 2: INPUT RESOLUTION ABLATION (VALIDATION SET)        ")
    logger.info("==================================================================")

    # Use YOLOv8s weights
    weights_path = PROJECT_ROOT / "results" / "trained" / "yolov8s" / "weights" / "best.pt"
    if not weights_path.exists():
        weights_path = PROJECT_ROOT / "results" / "experiments" / "yolo_model_size" / "yolov8s" / "weights" / "best.pt"
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}")

    resolutions = [640, 768, 1024]
    results = []

    for res in resolutions:
        save_dir = EXP_ROOT / str(res)
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"--- Evaluating Input Resolution {res}x{res} ---")

        eval_res = evaluate_yolo_model(
            weights_path,
            split="val",
            imgsz=res,
            conf=0.25,
            device=0,
            model_name=f"YOLOv8s_{res}px",
            save_dir=str(save_dir)
        )
        results.append(eval_res)

    # Summary table
    df = pd.DataFrame([
        {
            "Resolution": f"{r['imgsz']}x{r['imgsz']}",
            "mAP@0.5": r["mAP50"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1": r["f1"],
            "Small_AP": r["small_ap"],
            "Medium_AP": r["medium_ap"],
            "Large_AP": r["large_ap"],
            "Latency_ms": r["latency_ms"],
            "FPS": r["fps"],
            "Peak_VRAM_MB": r["peak_vram_mb"]
        }
        for r in results
    ])
    summary_csv = EXP_ROOT / "resolution_comparison.csv"
    df.to_csv(summary_csv, index=False)
    logger.info("Saved resolution comparison to %s", summary_csv)
    print("\n" + "="*80)
    print("EXPERIMENT 2 SUMMARY (Input Resolution Ablation on Validation Set):")
    print(df.to_string(index=False))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
