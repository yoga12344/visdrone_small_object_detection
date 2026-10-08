"""
run_phase_ij_advanced_adaptive_tiling.py
Phase I & J: Advanced & Adaptive Tiling Optimization on the 106-image Validation set.
Compares:
  1. Standard Native 640px (Baseline)
  2. Standard Native 1024px (High-Res)
  3. Tiled 2x2 Grid + 10% overlap
  4. Tiled 2x2 Grid + 20% overlap
  5. Tiled 2x2 Grid + 25% overlap
  6. Tiled 3x2 Grid + 20% overlap
  7. Adaptive Tiling: Fast 640 scan -> trigger 2x2 tiling only on images with detected small objects
Saves all validation metrics to results/performance_maximization/tiling_ablation.csv
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
logger = logging.getLogger("PhaseIJ_Tiling")

OUT_DIR = PROJECT_ROOT / "results" / "performance_maximization"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = PROJECT_ROOT / "results" / "final_optimization" / "training" / "yolov8s_100ep_640" / "weights" / "best.pt"

def evaluate_custom_tiled(model, gts, img_paths, n_cols=2, n_rows=2, overlap=0.20, conf=0.20, nms_iou=0.50, name="Tiled"):
    logger.info(f"=== Evaluating {name} (conf={conf}, nms={nms_iou}) on {len(img_paths)} val images ===")
    
    # Warmup
    if img_paths:
        _ = predict_tiled(model, img_paths[0], n_cols=n_cols, n_rows=n_rows, overlap=overlap, conf=conf, iou_nms=nms_iou, device=0)

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
            iou_nms=nms_iou,
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

    return {
        "configuration": name,
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

def evaluate_adaptive_tiling(model, gts, img_paths, conf=0.20, nms_iou=0.50, name="Adaptive_Tiling_2x2"):
    logger.info(f"=== Evaluating {name} on {len(img_paths)} val images ===")
    timings = []
    all_predictions = []
    tiles_triggered = 0

    for img_path in img_paths:
        img_id = Path(img_path).stem
        with Image.open(img_path) as im:
            W, H = im.size

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        # Fast 640 full-frame scan
        full_res = model(img_path, imgsz=640, conf=conf, iou=nms_iou, device=0, verbose=False)[0]
        preds = []
        has_small_targets = False

        if full_res.boxes is not None and len(full_res.boxes) > 0:
            bxs = full_res.boxes.xyxy.cpu().numpy()
            clss = full_res.boxes.cls.cpu().numpy().astype(int)
            scs = full_res.boxes.conf.cpu().numpy()
            for b, c, s in zip(bxs, clss, scs):
                preds.append({"box_xyxy": b, "class_id": int(c), "score": float(s)})
                w = b[2] - b[0]
                h = b[3] - b[1]
                if w < 40 or h < 40:
                    has_small_targets = True

        # If small targets detected or high density, run high-resolution 2x2 tile refinement
        if has_small_targets or len(preds) > 15:
            tiles_triggered += 1
            preds = predict_tiled(
                model=model,
                img_path=img_path,
                n_cols=2,
                n_rows=2,
                overlap=0.20,
                conf=conf,
                iou_nms=nms_iou,
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
    logger.info("Adaptive tiling triggered for %d / %d images (%.1f%%).", tiles_triggered, len(img_paths), 100.0 * tiles_triggered / len(img_paths))

    det_metrics = metrics.compute_metrics_for_model(
        gts, all_predictions, iou_thresh=0.50, class_names=visdrone_parser.CLASS_NAMES
    )
    size_metrics = metrics.compute_size_metrics(gts, all_predictions, iou_thresh=0.50)

    return {
        "configuration": name,
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

def main():
    print("=" * 70)
    print("PHASE I & J: ADVANCED & ADAPTIVE TILING (VAL SET)")
    print("=" * 70)

    assert CKPT_PATH.exists(), f"Checkpoint {CKPT_PATH} not found"
    model = YOLO(str(CKPT_PATH))
    gts, img_paths, _ = load_split_ground_truth("val")

    results = []

    # 1. Native 640px (Tuned conf=0.20, nms=0.50)
    r_640 = evaluate_yolo_model(str(CKPT_PATH), split="val", imgsz=640, conf=0.20, iou_nms=0.50, device=0, model_name="Native_640px")
    results.append({
        "configuration": "Native 640px (Tuned)",
        "val_mAP50": r_640["mAP50"], "precision": r_640["precision"], "recall": r_640["recall"], "f1": r_640["f1"],
        "small_ap": r_640["small_ap"], "medium_ap": r_640["medium_ap"], "large_ap": r_640["large_ap"],
        "mean_iou": r_640["mean_iou"], "latency_ms": r_640["latency_ms"], "fps": r_640["fps"]
    })

    # 2. Native 1024px (Tuned conf=0.20, nms=0.50)
    r_1024 = evaluate_yolo_model(str(CKPT_PATH), split="val", imgsz=1024, conf=0.20, iou_nms=0.50, device=0, model_name="Native_1024px")
    results.append({
        "configuration": "Native 1024px (Tuned)",
        "val_mAP50": r_1024["mAP50"], "precision": r_1024["precision"], "recall": r_1024["recall"], "f1": r_1024["f1"],
        "small_ap": r_1024["small_ap"], "medium_ap": r_1024["medium_ap"], "large_ap": r_1024["large_ap"],
        "mean_iou": r_1024["mean_iou"], "latency_ms": r_1024["latency_ms"], "fps": r_1024["fps"]
    })

    # 3. Tiled 2x2 + 10% overlap
    results.append(evaluate_custom_tiled(model, gts, img_paths, n_cols=2, n_rows=2, overlap=0.10, conf=0.20, nms_iou=0.50, name="Tiled 2x2 (10% overlap)"))

    # 4. Tiled 2x2 + 20% overlap
    results.append(evaluate_custom_tiled(model, gts, img_paths, n_cols=2, n_rows=2, overlap=0.20, conf=0.20, nms_iou=0.50, name="Tiled 2x2 (20% overlap)"))

    # 5. Tiled 2x2 + 25% overlap
    results.append(evaluate_custom_tiled(model, gts, img_paths, n_cols=2, n_rows=2, overlap=0.25, conf=0.20, nms_iou=0.50, name="Tiled 2x2 (25% overlap)"))

    # 6. Tiled 3x2 + 20% overlap
    results.append(evaluate_custom_tiled(model, gts, img_paths, n_cols=3, n_rows=2, overlap=0.20, conf=0.20, nms_iou=0.50, name="Tiled 3x2 (20% overlap)"))

    # 7. Adaptive Tiling
    results.append(evaluate_adaptive_tiling(model, gts, img_paths, conf=0.20, nms_iou=0.50, name="Adaptive Tiling (Selective 2x2)"))

    df = pd.DataFrame(results)
    out_csv = OUT_DIR / "tiling_ablation.csv"
    df.to_csv(out_csv, index=False)
    print("\n" + "=" * 80)
    print("PHASE I & J ADVANCED TILING ABLATION RESULTS:")
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
