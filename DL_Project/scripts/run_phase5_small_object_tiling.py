"""
run_phase5_small_object_tiling.py
Phase 5 & 6: Small-Object Spatial Preservation via Tiling and High-Resolution Inference.
Evaluates the 100-epoch trained YOLOv8s on the validation set under:
  1. Standard Native 640px
  2. Standard Native 1024px
  3. Tiled 2x2 Grid (Overlap 0.20)
  4. Tiled 3x2 Grid (Overlap 0.25)
Saves results to results/final_optimization/small_object_tiling_comparison.csv
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "data"))

from research_evaluator import load_split_ground_truth, evaluate_yolo_model
from run_exp4_small_objects import predict_tiled
import metrics
import small_objects
import visdrone_parser
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase5_Tiling")

FINAL_ROOT = PROJECT_ROOT / "results" / "final_optimization"
FINAL_ROOT.mkdir(parents=True, exist_ok=True)

CKPT_PATH = FINAL_ROOT / "training" / "yolov8s_100ep_640" / "weights" / "best.pt"

def evaluate_custom_tiled(model, gts, img_paths, n_cols=2, n_rows=2, overlap=0.20, conf=0.25, name="Tiled"):
    logger.info(f"=== Evaluating {name} on {len(img_paths)} validation images ===")
    
    # Warmup
    if img_paths:
        _ = predict_tiled(model, img_paths[0], n_cols=n_cols, n_rows=n_rows, overlap=overlap, conf=conf, device=0)

    timings = []
    all_predictions = []

    for img_path in img_paths:
        img_id = Path(img_path).stem
        with Image.open(img_path) as im:
            W, H = im.size

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        preds = predict_tiled(
            model=model,
            img_path=img_path,
            n_cols=n_cols,
            n_rows=n_rows,
            overlap=overlap,
            conf=conf,
            iou_nms=0.45,
            device=0
        )
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

    res = {
        "strategy": name,
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
    return res

def main():
    logger.info("=" * 60)
    logger.info("PHASE 5 & 6: SPATIAL TILING & HIGH-RES ABLATION (VAL SET)")
    logger.info("=" * 60)

    assert CKPT_PATH.exists(), f"Checkpoint {CKPT_PATH} not found"
    model = YOLO(str(CKPT_PATH))
    gts, img_paths, _ = load_split_ground_truth("val")

    results = []

    # 1. Native 640px
    logger.info("Evaluating Standard 640px...")
    res_640 = evaluate_yolo_model(
        str(CKPT_PATH), split="val", imgsz=640, conf=0.25, device=0, model_name="Native_640px"
    )
    results.append({
        "strategy": "Native 640px",
        "val_mAP50": res_640["mAP50"],
        "precision": res_640["precision"],
        "recall": res_640["recall"],
        "f1": res_640["f1"],
        "small_ap": res_640["small_ap"],
        "medium_ap": res_640["medium_ap"],
        "large_ap": res_640["large_ap"],
        "mean_iou": res_640["mean_iou"],
        "latency_ms": res_640["latency_ms"],
        "fps": res_640["fps"]
    })

    # 2. Native 1024px
    logger.info("Evaluating Native 1024px...")
    res_1024 = evaluate_yolo_model(
        str(CKPT_PATH), split="val", imgsz=1024, conf=0.25, device=0, model_name="Native_1024px"
    )
    results.append({
        "strategy": "Native 1024px",
        "val_mAP50": res_1024["mAP50"],
        "precision": res_1024["precision"],
        "recall": res_1024["recall"],
        "f1": res_1024["f1"],
        "small_ap": res_1024["small_ap"],
        "medium_ap": res_1024["medium_ap"],
        "large_ap": res_1024["large_ap"],
        "mean_iou": res_1024["mean_iou"],
        "latency_ms": res_1024["latency_ms"],
        "fps": res_1024["fps"]
    })

    # 3. Tiled 2x2 Grid (20% overlap)
    res_tiled_2x2 = evaluate_custom_tiled(
        model, gts, img_paths, n_cols=2, n_rows=2, overlap=0.20, conf=0.25, name="Tiled 2x2 (20% overlap)"
    )
    results.append(res_tiled_2x2)

    # 4. Tiled 3x2 Grid (25% overlap)
    res_tiled_3x2 = evaluate_custom_tiled(
        model, gts, img_paths, n_cols=3, n_rows=2, overlap=0.25, conf=0.25, name="Tiled 3x2 (25% overlap)"
    )
    results.append(res_tiled_3x2)

    df = pd.DataFrame(results)
    out_csv = FINAL_ROOT / "small_object_tiling_comparison.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved tiling comparison to {out_csv}")

    print("\n" + "=" * 80)
    print("PHASE 5 & 6 SUMMARY (Small-Object Spatial Preservation on Val Set):")
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
