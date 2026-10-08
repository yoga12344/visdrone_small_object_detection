"""
run_phase12_locked_test_evaluation.py
Phase 12: Final Evaluation on the Untouched, Locked 20-Image Test Set.
Evaluates all final frozen models ONCE against the authentic VisDrone test annotations:
  1. YOLOv8s Baseline (25ep, 640px)
  2. YOLOv8s Final Trained (100ep, 640px)
  3. YOLOv8s Final High-Res (100ep, 1024px)
  4. YOLOv8s Final Tiled (100ep, 2x2 Tiled with 20% overlap)
  5. Faster R-CNN (ResNet-50-FPN)
  6. Mask R-CNN (ResNet-50-FPN)
  7. RF-DETR Nano
  8. Selective Two-Stage Hybrid (YOLOv8s 100ep + Faster R-CNN)
Saves results to results/final_optimization/final_test_benchmark_comparison.csv and json.
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
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "data"))

from research_evaluator import load_split_ground_truth, evaluate_yolo_model
from run_exp4_small_objects import predict_tiled
from run_exp9_two_stage_hybrid import load_faster_rcnn, predict_two_stage
import metrics
import small_objects
import visdrone_parser
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase12_FinalTest")

FINAL_ROOT = PROJECT_ROOT / "results" / "final_optimization"
FINAL_ROOT.mkdir(parents=True, exist_ok=True)

def eval_tiled_test(model, gts, img_paths, n_cols=2, n_rows=2, overlap=0.20, conf=0.25, name="YOLOv8s_100ep_Tiled2x2"):
    logger.info(f"=== Evaluating {name} on Locked Test Set ===")
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

    return {
        "model": name,
        "test_mAP50": round(float(det_metrics.get("mAP", 0.0)), 4),
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

def eval_hybrid_test(yolo_model, frcnn_model, gts, img_paths, device, name="Selective_Two_Stage_Hybrid"):
    logger.info(f"=== Evaluating {name} on Locked Test Set ===")
    timings = []
    all_predictions = []

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

    return {
        "model": name,
        "test_mAP50": round(float(det_metrics.get("mAP", 0.0)), 4),
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
    logger.info("=" * 60)
    logger.info("PHASE 12: FINAL BENCHMARK ON LOCKED 20-IMAGE TEST SET")
    logger.info("=" * 60)

    gts, img_paths, _ = load_split_ground_truth("test")
    assert len(img_paths) == 20, f"Expected exactly 20 test images, got {len(img_paths)}"
    logger.info("Confirmed 20 test images with %d ground truth targets.", len(gts))

    results = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. YOLOv8s Baseline (25ep, 640px)
    base_ckpt = PROJECT_ROOT / "results" / "trained" / "yolov8s" / "weights" / "best.pt"
    if base_ckpt.exists():
        r = evaluate_yolo_model(str(base_ckpt), split="test", imgsz=640, conf=0.25, device=0, model_name="YOLOv8s_Baseline_25ep")
        results.append({
            "model": "YOLOv8s (25ep Baseline)",
            "test_mAP50": r["mAP50"], "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
            "small_ap": r["small_ap"], "medium_ap": r["medium_ap"], "large_ap": r["large_ap"],
            "mean_iou": r["mean_iou"], "latency_ms": r["latency_ms"], "fps": r["fps"]
        })

    # 2. YOLOv8s Final (100ep, 640px)
    opt_ckpt = FINAL_ROOT / "training" / "yolov8s_100ep_640" / "weights" / "best.pt"
    assert opt_ckpt.exists(), f"Optimized YOLOv8s checkpoint not found at {opt_ckpt}"
    r_100_640 = evaluate_yolo_model(str(opt_ckpt), split="test", imgsz=640, conf=0.25, device=0, model_name="YOLOv8s_Final_100ep_640")
    results.append({
        "model": "YOLOv8s (100ep, 640px)",
        "test_mAP50": r_100_640["mAP50"], "precision": r_100_640["precision"], "recall": r_100_640["recall"], "f1": r_100_640["f1"],
        "small_ap": r_100_640["small_ap"], "medium_ap": r_100_640["medium_ap"], "large_ap": r_100_640["large_ap"],
        "mean_iou": r_100_640["mean_iou"], "latency_ms": r_100_640["latency_ms"], "fps": r_100_640["fps"]
    })

    # 3. YOLOv8s Final High-Res (100ep, 1024px)
    r_100_1024 = evaluate_yolo_model(str(opt_ckpt), split="test", imgsz=1024, conf=0.25, device=0, model_name="YOLOv8s_Final_100ep_1024")
    results.append({
        "model": "YOLOv8s (100ep, 1024px)",
        "test_mAP50": r_100_1024["mAP50"], "precision": r_100_1024["precision"], "recall": r_100_1024["recall"], "f1": r_100_1024["f1"],
        "small_ap": r_100_1024["small_ap"], "medium_ap": r_100_1024["medium_ap"], "large_ap": r_100_1024["large_ap"],
        "mean_iou": r_100_1024["mean_iou"], "latency_ms": r_100_1024["latency_ms"], "fps": r_100_1024["fps"]
    })

    # 4. YOLOv8s Final Tiled (100ep, 2x2 Tiled)
    opt_yolo = YOLO(str(opt_ckpt))
    r_tiled = eval_tiled_test(opt_yolo, gts, img_paths, n_cols=2, n_rows=2, overlap=0.20, conf=0.25, name="YOLOv8s (100ep, 2x2 Tiled)")
    results.append(r_tiled)

    # 5. Faster R-CNN (ResNet-50-FPN)
    with open(PROJECT_ROOT / "results" / "trained" / "faster_rcnn" / "test_metrics.json") as f:
        frcnn_meta = json.load(f)
    results.append({
        "model": "Faster R-CNN (ResNet-50-FPN)",
        "test_mAP50": frcnn_meta.get("mAP50"), "precision": frcnn_meta.get("precision"), "recall": frcnn_meta.get("recall"), "f1": frcnn_meta.get("f1"),
        "small_ap": frcnn_meta.get("small_ap"), "medium_ap": frcnn_meta.get("medium_ap"), "large_ap": frcnn_meta.get("large_ap"),
        "mean_iou": frcnn_meta.get("mean_iou"), "latency_ms": frcnn_meta.get("latency_ms"), "fps": frcnn_meta.get("fps")
    })

    # 6. Mask R-CNN (ResNet-50-FPN)
    with open(PROJECT_ROOT / "results" / "trained" / "mask_rcnn" / "test_metrics.json") as f:
        mrcnn_meta = json.load(f)
    results.append({
        "model": "Mask R-CNN (ResNet-50-FPN)",
        "test_mAP50": mrcnn_meta.get("mAP50"), "precision": mrcnn_meta.get("precision"), "recall": mrcnn_meta.get("recall"), "f1": mrcnn_meta.get("f1"),
        "small_ap": mrcnn_meta.get("small_ap"), "medium_ap": mrcnn_meta.get("medium_ap"), "large_ap": mrcnn_meta.get("large_ap"),
        "mean_iou": mrcnn_meta.get("mean_iou"), "latency_ms": mrcnn_meta.get("latency_ms"), "fps": mrcnn_meta.get("fps")
    })

    # 7. RF-DETR Nano
    with open(PROJECT_ROOT / "results" / "trained" / "rf_detr_nano" / "test_metrics.json") as f:
        rfdetr_meta = json.load(f)
    results.append({
        "model": "RF-DETR Nano",
        "test_mAP50": rfdetr_meta.get("mAP50"), "precision": rfdetr_meta.get("precision"), "recall": rfdetr_meta.get("recall"), "f1": rfdetr_meta.get("f1"),
        "small_ap": rfdetr_meta.get("small_ap"), "medium_ap": rfdetr_meta.get("medium_ap"), "large_ap": rfdetr_meta.get("large_ap"),
        "mean_iou": rfdetr_meta.get("mean_iou"), "latency_ms": rfdetr_meta.get("latency_ms"), "fps": rfdetr_meta.get("fps")
    })

    # 8. Selective Two-Stage Hybrid
    frcnn_model = load_faster_rcnn(device)
    r_hybrid = eval_hybrid_test(opt_yolo, frcnn_model, gts, img_paths, device, name="Selective Two-Stage Hybrid")
    results.append(r_hybrid)

    df = pd.DataFrame(results)
    out_csv = FINAL_ROOT / "final_test_benchmark_comparison.csv"
    df.to_csv(out_csv, index=False)
    
    out_json = FINAL_ROOT / "final_test_benchmark_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Final benchmark saved to {out_csv} and {out_json}")
    print("\n" + "=" * 90)
    print("FINAL TEST SET BENCHMARK COMPARISON (Locked 20-Image Independent Test Set):")
    print(df.to_string(index=False))
    print("=" * 90 + "\n")

if __name__ == "__main__":
    main()
