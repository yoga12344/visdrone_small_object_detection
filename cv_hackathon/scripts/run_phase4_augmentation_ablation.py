"""
run_phase4_augmentation_ablation.py
Execute Phase 4 Augmentation Ablation on VisDrone validation set and save to results/final_optimization/augmentation_ablation.csv.
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
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from research_evaluator import evaluate_yolo_model
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase4_Augmentation")

DATA_YAML = str(PROJECT_ROOT / "data" / "processed" / "data.yaml")
FINAL_ROOT = PROJECT_ROOT / "results" / "final_optimization"
TRAIN_ROOT = FINAL_ROOT / "training" / "augmentation"
TRAIN_ROOT.mkdir(parents=True, exist_ok=True)

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
    "spatial_preserve": {
        "mosaic": 0.0,  # No mosaic cuts: preserves tiny object integrity
        "degrees": 5.0,
        "fliplr": 0.5,
        "scale": 0.1,
        "hsv_h": 0.015,
        "hsv_s": 0.4,
        "hsv_v": 0.4
    },
    "combined_aerial": {
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
    save_dir = TRAIN_ROOT / strategy_name
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
            "device": 0,
            "seed": 42,
            "deterministic": True,
            "project": str(TRAIN_ROOT),
            "name": strategy_name,
            "exist_ok": True,
            "amp": True,
            "plots": False,
            "verbose": False
        }
        train_args.update(aug_params)
        model.train(**train_args)
        del model
        torch.cuda.empty_cache()
        gc.collect()
    else:
        logger.info(f"Existing weights found for {strategy_name} at {weights_path}")

    # Evaluate on 106 validation images
    logger.info(f"=== Evaluating {strategy_name} on Validation Set ===")
    eval_res = evaluate_yolo_model(
        str(weights_path),
        split="val",
        imgsz=640,
        conf=0.25,
        device=0,
        model_name=f"Aug_{strategy_name}"
    )
    return eval_res

def main():
    logger.info("=" * 60)
    logger.info("PHASE 4: AERIAL AUGMENTATION ABLATION ON VALIDATION SET")
    logger.info("=" * 60)

    results = []
    for strat_name, aug_params in AUG_CONFIGS.items():
        res = train_and_eval_aug(strat_name, aug_params, epochs=15)
        results.append(res)

    df = pd.DataFrame([
        {
            "strategy": r["model"].replace("Aug_", ""),
            "val_mAP50": r["mAP50"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "small_ap": r["small_ap"],
            "medium_ap": r["medium_ap"],
            "large_ap": r["large_ap"],
            "latency_ms": r["latency_ms"],
            "fps": r["fps"]
        }
        for r in results
    ])
    
    out_csv = FINAL_ROOT / "augmentation_ablation.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Phase 4 Augmentation Ablation saved to {out_csv}")
    print("\n" + "=" * 80)
    print("PHASE 4 SUMMARY (Augmentation Ablation on Validation Set):")
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
