"""
run_phase_q_final_test.py
Phase Q: Single Leakage-Free Final Evaluation on the Locked 20-Image Independent Test Set.
Evaluates the Frozen Configurations:
  1. YOLOv8s Baseline (25ep, 640px, default conf=0.25)
  2. YOLOv8s 100ep Native 640px (Tuned conf=0.20, nms=0.50)
  3. YOLOv8s 100ep Native 1024px (Best Real-Time: conf=0.20, nms=0.50)
  4. YOLOv8s 100ep Tiled 2x2 (Best Practical Tiling: conf=0.20, nms=0.50, overlap=0.20)
  5. YOLOv8s 100ep Tiled 3x2 (Best Accuracy & Small Object: conf=0.20, nms=0.50, overlap=0.20)
  6. Faster R-CNN (ResNet-50-FPN)
  7. Mask R-CNN (ResNet-50-FPN)
  8. RF-DETR Nano
  9. Selective Two-Stage Hybrid (YOLOv8s + Faster R-CNN)
Saves results to results/performance_maximization/FINAL_RESULTS.csv and json.
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
from run_exp4_small_objects import predict_tiled
from run_exp9_two_stage_hybrid import load_faster_rcnn, predict_two_stage
import metrics
import small_objects
import visdrone_parser
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PhaseQ_FinalTest")

OUT_DIR = PROJECT_ROOT / "results" / "performance_maximization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OPT_CKPT = PROJECT_ROOT / "results" / "final_optimization" / "training" / "yolov8s_100ep_640" / "weights" / "best.pt"
BASE_CKPT = PROJECT_ROOT / "results" / "trained" / "yolov8s" / "weights" / "best.pt"

def eval_tiled_test(model, gts, img_paths, n_cols=2, n_rows=2, overlap=0.20, conf=0.20, nms_iou=0.50, name="Tiled"):
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
    print("=" * 80)
    print("PHASE Q: SINGLE LEAKAGE-FREE FINAL TEST ON LOCKED TEST SET")
    print("=" * 80)

    gts, img_paths, _ = load_split_ground_truth("test")
    assert len(img_paths) == 20, f"Expected 20 test images, got {len(img_paths)}"
    logger.info("Confirmed 20 test images with %d ground truth targets.", len(gts))

    results = []
    opt_yolo = YOLO(str(OPT_CKPT))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. YOLOv8s Baseline (25ep, 640px)
    r_base = evaluate_yolo_model(str(BASE_CKPT), split="test", imgsz=640, conf=0.25, device=0, model_name="YOLOv8s_Baseline_25ep")
    results.append({
        "configuration": "YOLOv8s (25ep Baseline)",
        "test_mAP50": r_base["mAP50"], "precision": r_base["precision"], "recall": r_base["recall"], "f1": r_base["f1"],
        "small_ap": r_base["small_ap"], "medium_ap": r_base["medium_ap"], "large_ap": r_base["large_ap"],
        "mean_iou": r_base["mean_iou"], "latency_ms": r_base["latency_ms"], "fps": r_base["fps"]
    })

    # 2. YOLOv8s 100ep Native 640px (Tuned: conf=0.20, nms=0.50)
    r_640 = evaluate_yolo_model(str(OPT_CKPT), split="test", imgsz=640, conf=0.20, iou_nms=0.50, device=0, model_name="YOLOv8s_100ep_640_Tuned")
    results.append({
        "configuration": "YOLOv8s (100ep, 640px, Tuned)",
        "test_mAP50": r_640["mAP50"], "precision": r_640["precision"], "recall": r_640["recall"], "f1": r_640["f1"],
        "small_ap": r_640["small_ap"], "medium_ap": r_640["medium_ap"], "large_ap": r_640["large_ap"],
        "mean_iou": r_640["mean_iou"], "latency_ms": r_640["latency_ms"], "fps": r_640["fps"]
    })

    # 3. YOLOv8s 100ep Native 1024px (Best Real-Time: conf=0.20, nms=0.50)
    r_1024 = evaluate_yolo_model(str(OPT_CKPT), split="test", imgsz=1024, conf=0.20, iou_nms=0.50, device=0, model_name="YOLOv8s_100ep_1024_Tuned")
    results.append({
        "configuration": "YOLOv8s (100ep, 1024px, Best Real-Time)",
        "test_mAP50": r_1024["mAP50"], "precision": r_1024["precision"], "recall": r_1024["recall"], "f1": r_1024["f1"],
        "small_ap": r_1024["small_ap"], "medium_ap": r_1024["medium_ap"], "large_ap": r_1024["large_ap"],
        "mean_iou": r_1024["mean_iou"], "latency_ms": r_1024["latency_ms"], "fps": r_1024["fps"]
    })

    # 4. YOLOv8s 100ep Tiled 2x2 (Best Practical Tiling)
    results.append(eval_tiled_test(opt_yolo, gts, img_paths, n_cols=2, n_rows=2, overlap=0.20, conf=0.20, nms_iou=0.50, name="YOLOv8s (100ep, Tiled 2x2, Tuned)"))

    # 5. YOLOv8s 100ep Tiled 3x2 (Best Accuracy & Small Object)
    results.append(eval_tiled_test(opt_yolo, gts, img_paths, n_cols=3, n_rows=2, overlap=0.20, conf=0.20, nms_iou=0.50, name="YOLOv8s (100ep, Tiled 3x2, Best Accuracy)"))

    # 6. Faster R-CNN (ResNet-50-FPN)
    with open(PROJECT_ROOT / "results" / "trained" / "faster_rcnn" / "test_metrics.json") as f:
        frcnn_meta = json.load(f)
    results.append({
        "configuration": "Faster R-CNN (ResNet-50-FPN)",
        "test_mAP50": frcnn_meta.get("mAP50"), "precision": frcnn_meta.get("precision"), "recall": frcnn_meta.get("recall"), "f1": frcnn_meta.get("f1"),
        "small_ap": frcnn_meta.get("small_ap"), "medium_ap": frcnn_meta.get("medium_ap"), "large_ap": frcnn_meta.get("large_ap"),
        "mean_iou": frcnn_meta.get("mean_iou"), "latency_ms": frcnn_meta.get("latency_ms"), "fps": frcnn_meta.get("fps")
    })

    # 7. Mask R-CNN (ResNet-50-FPN)
    with open(PROJECT_ROOT / "results" / "trained" / "mask_rcnn" / "test_metrics.json") as f:
        mrcnn_meta = json.load(f)
    results.append({
        "configuration": "Mask R-CNN (ResNet-50-FPN)",
        "test_mAP50": mrcnn_meta.get("mAP50"), "precision": mrcnn_meta.get("precision"), "recall": mrcnn_meta.get("recall"), "f1": mrcnn_meta.get("f1"),
        "small_ap": mrcnn_meta.get("small_ap"), "medium_ap": mrcnn_meta.get("medium_ap"), "large_ap": mrcnn_meta.get("large_ap"),
        "mean_iou": mrcnn_meta.get("mean_iou"), "latency_ms": mrcnn_meta.get("latency_ms"), "fps": mrcnn_meta.get("fps")
    })

    # 8. RF-DETR Nano
    with open(PROJECT_ROOT / "results" / "trained" / "rf_detr_nano" / "test_metrics.json") as f:
        rfdetr_meta = json.load(f)
    results.append({
        "configuration": "RF-DETR Nano",
        "test_mAP50": rfdetr_meta.get("mAP50"), "precision": rfdetr_meta.get("precision"), "recall": rfdetr_meta.get("recall"), "f1": rfdetr_meta.get("f1"),
        "small_ap": rfdetr_meta.get("small_ap"), "medium_ap": rfdetr_meta.get("medium_ap"), "large_ap": rfdetr_meta.get("large_ap"),
        "mean_iou": rfdetr_meta.get("mean_iou"), "latency_ms": rfdetr_meta.get("latency_ms"), "fps": rfdetr_meta.get("fps")
    })

    # 9. Selective Two-Stage Hybrid
    frcnn_model = load_faster_rcnn(device)
    timings = []
    all_predictions = []
    for img_path in img_paths:
        img_id = Path(img_path).stem
        with Image.open(img_path) as im:
            W, H = im.size
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        preds = predict_two_stage(img_path, opt_yolo, frcnn_model, device, conf=0.20)
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event) / 1000.0)
        for p in preds:
            b = p["box_xyxy"]
            x1 = max(0.0, float(b[0]))
            y1 = max(0.0, float(b[1]))
            x2 = min(float(W), float(b[2]))
            y2 = min(float(H), float(b[3]))
            c = p["class_id"]
            s = p["score"]
            if x2 > x1 and y2 > y1 and 0 <= c <= 9:
                all_predictions.append({"image_id": img_id, "class_id": c, "score": s, "box_xyxy": [x1, y1, x2, y2]})
    mean_lat_s = float(np.mean(timings)) if timings else 0.0
    det_metrics = metrics.compute_metrics_for_model(gts, all_predictions, iou_thresh=0.50, class_names=visdrone_parser.CLASS_NAMES)
    size_metrics = metrics.compute_size_metrics(gts, all_predictions, iou_thresh=0.50)
    results.append({
        "configuration": "Selective Two-Stage Hybrid (Tuned)",
        "test_mAP50": round(float(det_metrics.get("mAP", 0.0)), 4),
        "precision": round(float(det_metrics.get("precision", 0.0)), 4),
        "recall": round(float(det_metrics.get("recall", 0.0)), 4),
        "f1": round(float(det_metrics.get("f1", 0.0)), 4),
        "small_ap": round(float(size_metrics.get("small", {}).get("ap", 0.0) or 0.0), 4),
        "medium_ap": round(float(size_metrics.get("medium", {}).get("ap", 0.0) or 0.0), 4),
        "large_ap": round(float(size_metrics.get("large", {}).get("ap", 0.0) or 0.0), 4),
        "mean_iou": round(float(det_metrics.get("mean_iou", 0.0)), 4),
        "latency_ms": round(float(mean_lat_s * 1000.0), 2),
        "fps": round(float(1.0 / mean_lat_s if mean_lat_s > 0 else 0.0), 2)
    })

    df = pd.DataFrame(results)
    out_csv = OUT_DIR / "FINAL_RESULTS.csv"
    df.to_csv(out_csv, index=False)

    out_json = OUT_DIR / "FINAL_RESULTS.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 95)
    print("FINAL RESULTS (LOCKED 20-IMAGE INDEPENDENT TEST SET):")
    print(df.to_string(index=False))
    print("=" * 95 + "\n")

if __name__ == "__main__":
    main()
