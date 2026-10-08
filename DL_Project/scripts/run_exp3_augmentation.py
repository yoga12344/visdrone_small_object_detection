"""
run_exp3_augmentation.py
========================
Experiment 3: Aerial-Specific Augmentation Ablation
Compares controlled augmentation strategies on VisDrone:
  - Baseline (Default YOLO augmentations)
  - Set A: Geometric (horizontal flip, small rotation, scale jitter)
  - Set B: Photometric (HSV color variation, brightness/contrast)
  - Set C: Blur & Noise / No-Mosaic (preserves tiny spatial features without aggressive mosaic cuts)
  - Set D: Combined (Balanced aerial augmentation)

Protocol:
  - Trains for 15 epochs per set using identical seed=42 and base model (YOLOv8n)
  - Evaluates on the 106-image validation split
  - Records mAP@0.5, Small AP, Medium AP, Large AP, Precision, Recall, F1
  - Saves artifacts in results/experiments/augmentation/{strategy}/
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
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp3_Augmentation")

DATA_YAML = str(PROJECT_ROOT / "data" / "processed" / "data.yaml")
EXP_ROOT = PROJECT_ROOT / "results" / "experiments" / "augmentation"
EXP_ROOT.mkdir(parents=True, exist_ok=True)

AUG_CONFIGS = {
    "baseline": {
        "mosaic": 1.0,
        "degrees": 0.0,
        "fliplr": 0.5,
        "scale": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4
    },
    "geometric": {
        "mosaic": 0.5,
        "degrees": 15.0,
        "fliplr": 0.5,
        "flipud": 0.1,
        "scale": 0.3,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0
    },
    "photometric": {
        "mosaic": 0.5,
        "degrees": 0.0,
        "fliplr": 0.0,
        "scale": 0.0,
        "hsv_h": 0.03,
        "hsv_s": 0.8,
        "hsv_v": 0.6
    },
    "blur_noise": {
        "mosaic": 0.0,  # Disable mosaic to test pure spatial preservation without boundary artifacts
        "degrees": 5.0,
        "fliplr": 0.5,
        "scale": 0.1,
        "hsv_h": 0.015,
        "hsv_s": 0.4,
        "hsv_v": 0.4
    },
    "combined": {
        "mosaic": 0.5,
        "degrees": 10.0,
        "fliplr": 0.5,
        "flipud": 0.1,
        "scale": 0.2,
        "hsv_h": 0.02,
        "hsv_s": 0.6,
        "hsv_v": 0.5
    }
}


def train_and_eval_aug(strategy_name: str, aug_params: dict, epochs: int = 15):
    save_dir = EXP_ROOT / strategy_name
    weights_path = save_dir / "weights" / "best.pt"

    if not weights_path.exists():
        logger.info(f"=== Training Augmentation Strategy: {strategy_name} ({epochs} epochs) ===")
        torch.cuda.empty_cache()
        gc.collect()

        model = YOLO("yolov8n.pt")
        train_args = {
            "data": DATA_YAML,
            "epochs": epochs,
            "imgsz": 640,
            "batch": 16,
            "workers": 0,
            "device": "0",
            "seed": 42,
            "deterministic": True,
            "project": str(EXP_ROOT),
            "name": strategy_name,
            "exist_ok": True,
            "amp": True,
            "plots": True,
            "verbose": False
        }
        train_args.update(aug_params)
        model.train(**train_args)
        del model
        torch.cuda.empty_cache()
        gc.collect()
    else:
        logger.info(f"Existing weights found for {strategy_name} at {weights_path}")

    # Evaluate on validation set
    logger.info(f"=== Evaluating {strategy_name} on 106 Validation Images ===")
    eval_res = evaluate_yolo_model(
        weights_path,
        split="val",
        imgsz=640,
        conf=0.25,
        device=0,
        model_name=f"Aug_{strategy_name}",
        save_dir=str(save_dir)
    )
    return eval_res


def main():
    logger.info("==================================================================")
    logger.info("  EXPERIMENT 3: AERIAL-SPECIFIC AUGMENTATION ABLATION (VAL SET)   ")
    logger.info("==================================================================")

    results = []
    for strat_name, aug_params in AUG_CONFIGS.items():
        res = train_and_eval_aug(strat_name, aug_params, epochs=15)
        results.append(res)

    df = pd.DataFrame([
        {
            "Augmentation": r["model"].replace("Aug_", ""),
            "mAP@0.5": r["mAP50"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1": r["f1"],
            "Small_AP": r["small_ap"],
            "Medium_AP": r["medium_ap"],
            "Large_AP": r["large_ap"],
            "Latency_ms": r["latency_ms"],
            "FPS": r["fps"]
        }
        for r in results
    ])
    summary_path = EXP_ROOT / "augmentation_comparison.csv"
    df.to_csv(summary_path, index=False)
    logger.info("Saved augmentation comparison to %s", summary_path)
    print("\n" + "="*80)
    print("EXPERIMENT 3 SUMMARY (Augmentation Ablation on Validation Set):")
    print(df.to_string(index=False))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
