"""
run_phase9_two_stage_hybrid.py
Phase 9: Selective Two-Stage System (YOLOv8s 100ep + selective Faster R-CNN patch refinement).
Evaluates on the 106-image Validation split.
Saves metrics and comparison to results/final_optimization/hybrid_two_stage_comparison.csv.
"""
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "data"))

from research_evaluator import load_split_ground_truth, evaluate_yolo_model
from run_exp9_two_stage_hybrid import load_faster_rcnn, predict_two_stage
import metrics
import small_objects
import visdrone_parser
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase9_Hybrid")

FINAL_ROOT = PROJECT_ROOT / "results" / "final_optimization"
FINAL_ROOT.mkdir(parents=True, exist_ok=True)

YOLO_CKPT = FINAL_ROOT / "training" / "yolov8s_100ep_640" / "weights" / "best.pt"

def main():
    logger.info("=" * 60)
    logger.info("PHASE 9: SELECTIVE TWO-STAGE REFINEMENT (VAL SET)")
    logger.info("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gts, img_paths, _ = load_split_ground_truth("val")

    assert YOLO_CKPT.exists(), f"YOLO checkpoint {YOLO_CKPT} not found"
    yolo_model = YOLO(str(YOLO_CKPT))
    frcnn_model = load_faster_rcnn(device)

    # Warmup
    if img_paths:
        _ = predict_two_stage(img_paths[0], yolo_model, frcnn_model, device, conf=0.25)

    timings = []
    all_predictions = []

    logger.info(f"Evaluating selective two-stage hybrid on {len(img_paths)} validation images...")
    for img_path in img_paths:
        img_id = Path(img_path).stem
        with Image.open(img_path) as im:
            W, H = im.size

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        preds = predict_two_stage(img_path, yolo_model, frcnn_model, device, conf=0.25)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        timings.append(elapsed_ms / 1000.0)

        for p in preds:
            b = p["box_xyxy"]
            x1 = max(0.0, float(b[0]))
            y1 = max(0.0, float(b[1]))
            x2 = min(float(W), float(b[2]))
            y2 = min(float(H), float(b[3]))
            c = p["class_id"]
            s = p["score"]
            if x2 > x1 and y2 > y1 and 0 <= c <= 9:
                all_predictions.append({
                    "image_id": img_id,
                    "class_id": c,
                    "score": s,
                    "box_xyxy": [x1, y1, x2, y2]
                })

    mean_lat_s = float(np.mean(timings)) if timings else 0.0
    latency_ms = mean_lat_s * 1000.0
    fps = 1.0 / mean_lat_s if mean_lat_s > 0 else 0.0

    det_metrics = metrics.compute_metrics_for_model(
        gts, all_predictions, iou_thresh=0.50, class_names=visdrone_parser.CLASS_NAMES
    )
    size_metrics = metrics.compute_size_metrics(gts, all_predictions, iou_thresh=0.50)

    hybrid_res = {
        "model": "Selective_Two_Stage_Hybrid",
        "val_mAP50": round(float(det_metrics.get("mAP", 0.0)), 4),
        "precision": round(float(det_metrics.get("precision", 0.0)), 4),
        "recall": round(float(det_metrics.get("recall", 0.0)), 4),
        "f1": round(float(det_metrics.get("f1", 0.0)), 4),
        "small_ap": round(float(size_metrics.get("small", {}).get("ap", 0.0) or 0.0), 4),
        "medium_ap": round(float(size_metrics.get("medium", {}).get("ap", 0.0) or 0.0), 4),
        "large_ap": round(float(size_metrics.get("large", {}).get("ap", 0.0) or 0.0), 4),
        "mean_iou": round(float(det_metrics.get("mean_iou", 0.0)), 4),
        "latency_ms": round(float(latency_ms), 2),
        "fps": round(float(fps), 2)
    }

    df = pd.DataFrame([hybrid_res])
    out_csv = FINAL_ROOT / "hybrid_two_stage_comparison.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved hybrid comparison to {out_csv}")

    print("\n" + "=" * 80)
    print("PHASE 9 SUMMARY (Selective Two-Stage System on Val Set):")
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
