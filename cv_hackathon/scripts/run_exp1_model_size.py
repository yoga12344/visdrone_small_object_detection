"""
run_exp1_model_size.py
======================
Experiment 1: Controlled YOLO Model Size Comparison
Compares YOLOv8n, YOLOv8s, YOLOv8m, and YOLO11n on VisDrone2019.

Strict Protocol:
  - Identical dataset: data/processed/data.yaml (422 train, 106 val, 20 test locked)
  - Identical image size: 640x640
  - Identical random seed: 42
  - Evaluated strictly on the 106-image validation set (test set remains untouched)
  - Records: mAP@0.5, Precision, Recall, F1, Small AP, Med AP, Large AP, Mean IoU,
    Latency (ms), FPS, Parameters, Model Size (MB), Peak VRAM (MB)
"""

import gc
import json
import logging
import os
import shutil
import sys
from pathlib import Path
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "scripts"))
from research_evaluator import evaluate_yolo_model
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp1_ModelSize")

DATA_YAML = str(PROJECT_ROOT / "data" / "processed" / "data.yaml")
EXP_ROOT = PROJECT_ROOT / "results" / "experiments" / "yolo_model_size"
EXP_ROOT.mkdir(parents=True, exist_ok=True)


def train_and_eval_model(model_name: str, base_weights: str, batch_size: int, epochs: int = 25):
    target_dir = EXP_ROOT / model_name
    weights_path = target_dir / "weights" / "best.pt"
    
    # 1. Train if not already trained
    if not weights_path.exists():
        logger.info(f"=== Starting Training for {model_name} (epochs={epochs}, batch={batch_size}) ===")
        torch.cuda.empty_cache()
        gc.collect()
        
        model = YOLO(base_weights)
        train_results = model.train(
            data=DATA_YAML,
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            workers=0,
            device="0",
            seed=42,
            deterministic=True,
            project=str(EXP_ROOT),
            name=model_name,
            exist_ok=True,
            amp=True,
            plots=True,
            verbose=True,
        )
        del model
        torch.cuda.empty_cache()
        gc.collect()
    else:
        logger.info(f"Existing weights found for {model_name} at {weights_path}")

    # 2. Evaluate on validation set
    logger.info(f"=== Evaluating {model_name} on 106 Validation Images ===")
    eval_res = evaluate_yolo_model(
        weights_path,
        split="val",
        imgsz=640,
        conf=0.25,
        device=0,
        model_name=model_name,
        save_dir=str(target_dir)
    )
    return eval_res


def main():
    logger.info("==================================================================")
    logger.info("  EXPERIMENT 1: YOLO MODEL SIZE COMPARISON (VALIDATION SET)       ")
    logger.info("==================================================================")

    results = []

    # 1. YOLOv8s (re-use existing fine-tuned baseline weights)
    yolov8s_existing = PROJECT_ROOT / "results" / "trained" / "yolov8s" / "weights" / "best.pt"
    yolov8s_target_weights = EXP_ROOT / "yolov8s" / "weights" / "best.pt"
    if yolov8s_existing.exists() and not yolov8s_target_weights.exists():
        (EXP_ROOT / "yolov8s" / "weights").mkdir(parents=True, exist_ok=True)
        shutil.copy2(yolov8s_existing, yolov8s_target_weights)
        # Also copy training plots if available
        for plot_file in (PROJECT_ROOT / "results" / "trained" / "yolov8s").glob("*.png"):
            shutil.copy2(plot_file, EXP_ROOT / "yolov8s")

    res_8s = evaluate_yolo_model(
        yolov8s_target_weights,
        split="val",
        imgsz=640,
        conf=0.25,
        device=0,
        model_name="YOLOv8s",
        save_dir=str(EXP_ROOT / "yolov8s")
    )
    results.append(res_8s)

    # 2. YOLOv8n
    res_8n = train_and_eval_model("yolov8n", "yolov8n.pt", batch_size=16, epochs=25)
    results.append(res_8n)

    # 3. YOLO11n
    res_11n = train_and_eval_model("yolo11n", "yolo11n.pt", batch_size=16, epochs=25)
    results.append(res_11n)

    # 4. YOLOv8m
    res_8m = train_and_eval_model("yolov8m", "yolov8m.pt", batch_size=8, epochs=25)
    results.append(res_8m)

    # Compile Summary Table
    df = pd.DataFrame([
        {
            "Model": r["model"],
            "Params": r["parameters"],
            "Size_MB": r["model_size_mb"],
            "mAP@0.5": r["mAP50"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1": r["f1"],
            "Small_AP": r["small_ap"],
            "Medium_AP": r["medium_ap"],
            "Large_AP": r["large_ap"],
            "Mean_IoU": r["mean_iou"],
            "Latency_ms": r["latency_ms"],
            "FPS": r["fps"],
            "Peak_VRAM_MB": r["peak_vram_mb"]
        }
        for r in results
    ])
    summary_path = EXP_ROOT / "model_size_comparison.csv"
    df.to_csv(summary_path, index=False)
    logger.info("Saved model size comparison to %s", summary_path)
    print("\n" + "="*80)
    print("EXPERIMENT 1 SUMMARY (Validation Set):")
    print(df.to_string(index=False))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
