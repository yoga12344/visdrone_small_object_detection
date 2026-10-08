"""
run_exp4_small_objects.py
=========================
Experiment 4: Small-Object Recovery via Tiled / Sliced Inference
Evaluates spatial-preservation tiled inference vs standard 640px inference on VisDrone.

Scientific Principle:
  Aerial images often have native resolutions of 1360x765 or higher. Downsampling
  to 640x640 reduces a 16px pedestrian or 12px bicycle to under 8px, below the
  effective receptive field and stride limit of single-stage detectors.
  Tiled inference slices the native image into overlapping tiles (e.g. 640x640 with
  20% overlap), runs inference at native spatial resolution, projects coordinates
  back, and performs global Non-Maximum Suppression (NMS).

Protocol:
  - Base Model: YOLOv8s (fine-tuned checkpoint)
  - Split: 106-image Validation split
  - Compares:
      1. Standard Full-Image 640px
      2. Tiled 2x2 Grid (overlap 0.20)
      3. Tiled 3x2 Grid (overlap 0.25)
  - Evaluates: Small AP (<32px), Medium AP, Large AP, overall mAP@0.5, Latency, FPS
  - Saves artifacts in results/experiments/small_object_strategy/
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
sys.path.append(str(PROJECT_ROOT / "scripts"))
sys.path.append(str(PROJECT_ROOT / "evaluation"))
sys.path.append(str(PROJECT_ROOT / "data"))

from research_evaluator import load_split_ground_truth
import metrics
import small_objects
import visdrone_parser
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp4_SmallObjects")

EXP_ROOT = PROJECT_ROOT / "results" / "experiments" / "small_object_strategy"
EXP_ROOT.mkdir(parents=True, exist_ok=True)


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thresh: float = 0.50):
    """Greedy Non-Maximum Suppression per class."""
    if len(boxes) == 0:
        return []

    keep_indices = []
    unique_classes = np.unique(classes)

    for c in unique_classes:
        idxs = np.where(classes == c)[0]
        c_boxes = boxes[idxs]
        c_scores = scores[idxs]

        order = c_scores.argsort()[::-1]
        c_keep = []

        while order.size > 0:
            i = order[0]
            c_keep.append(idxs[i])

            if order.size == 1:
                break

            xx1 = np.maximum(c_boxes[i, 0], c_boxes[order[1:], 0])
            yy1 = np.maximum(c_boxes[i, 1], c_boxes[order[1:], 1])
            xx2 = np.minimum(c_boxes[i, 2], c_boxes[order[1:], 2])
            yy2 = np.minimum(c_boxes[i, 3], c_boxes[order[1:], 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            area_i = (c_boxes[i, 2] - c_boxes[i, 0]) * (c_boxes[i, 3] - c_boxes[i, 1])
            area_rem = (c_boxes[order[1:], 2] - c_boxes[order[1:], 0]) * (c_boxes[order[1:], 3] - c_boxes[order[1:], 1])
            union = area_i + area_rem - inter

            iou = np.zeros_like(inter)
            valid = union > 0
            iou[valid] = inter[valid] / union[valid]

            remaining = np.where(iou <= iou_thresh)[0]
            order = order[remaining + 1]

        keep_indices.extend(c_keep)

    return sorted(keep_indices)


def predict_tiled(
    model: YOLO,
    img_path: str,
    n_cols: int = 2,
    n_rows: int = 2,
    overlap: float = 0.20,
    conf: float = 0.25,
    iou_nms: float = 0.45,
    device: int = 0
):
    """
    Slices the input image into an n_cols x n_rows grid with overlap,
    runs YOLO inference on each tile and the full resized image,
    projects bounding boxes back to global coordinates, and merges with global NMS.
    """
    im = Image.open(img_path).convert("RGB")
    W, H = im.size

    tile_w = int(W / (n_cols - (n_cols - 1) * overlap))
    tile_h = int(H / (n_rows - (n_rows - 1) * overlap))

    step_x = int(tile_w * (1.0 - overlap))
    step_y = int(tile_h * (1.0 - overlap))

    all_boxes = []
    all_scores = []
    all_classes = []

    # 1. Full-image inference to anchor medium/large global context
    full_res = model(img_path, imgsz=640, conf=conf, iou=iou_nms, device=device, verbose=False)[0]
    if full_res.boxes is not None and len(full_res.boxes) > 0:
        b = full_res.boxes.xyxy.cpu().numpy()
        s = full_res.boxes.conf.cpu().numpy()
        c = full_res.boxes.cls.cpu().numpy().astype(int)
        all_boxes.extend(b)
        all_scores.extend(s)
        all_classes.extend(c)

    # 2. Tile-level high-resolution inference
    for r in range(n_rows):
        y1 = min(r * step_y, max(0, H - tile_h))
        y2 = min(y1 + tile_h, H)
        for col in range(n_cols):
            x1 = min(col * step_x, max(0, W - tile_w))
            x2 = min(x1 + tile_w, W)

            tile_crop = im.crop((x1, y1, x2, y2))
            tile_res = model(tile_crop, imgsz=640, conf=conf, iou=iou_nms, device=device, verbose=False)[0]

            if tile_res.boxes is not None and len(tile_res.boxes) > 0:
                tb = tile_res.boxes.xyxy.cpu().numpy()
                ts = tile_res.boxes.conf.cpu().numpy()
                tc = tile_res.boxes.cls.cpu().numpy().astype(int)

                # Project back to full image coordinates
                tb[:, [0, 2]] += x1
                tb[:, [1, 3]] += y1

                all_boxes.extend(tb)
                all_scores.extend(ts)
                all_classes.extend(tc)

    if not all_boxes:
        return []

    boxes_arr = np.array(all_boxes)
    scores_arr = np.array(all_scores)
    classes_arr = np.array(all_classes)

    keep_idx = nms_xyxy(boxes_arr, scores_arr, classes_arr, iou_thresh=iou_nms)

    results = []
    for idx in keep_idx:
        b = boxes_arr[idx]
        x1 = max(0.0, float(b[0]))
        y1 = max(0.0, float(b[1]))
        x2 = min(float(W), float(b[2]))
        y2 = min(float(H), float(b[3]))
        c = int(classes_arr[idx])
        s = float(scores_arr[idx])
        if x2 > x1 and y2 > y1 and 0 <= c <= 9:
            results.append({"box_xyxy": [x1, y1, x2, y2], "class_id": c, "score": s})

    return results


def evaluate_strategy(strategy_name: str, predictor_fn, split_gts, split_img_paths, device=0):
    logger.info("Evaluating %s on %d validation images...", strategy_name, len(split_img_paths))
    timings = []
    all_predictions = []

    # Warmup
    for _ in range(3):
        _ = predictor_fn(split_img_paths[0])

    for img_path in split_img_paths:
        img_id = Path(img_path).stem

        t0 = time.perf_counter()
        preds = predictor_fn(img_path)
        dt = time.perf_counter() - t0
        timings.append(dt)

        for p in preds:
            all_predictions.append({
                "image_id": img_id,
                "class_id": p["class_id"],
                "score": p["score"],
                "box_xyxy": p["box_xyxy"]
            })

    latency_ms = float(np.mean(timings)) * 1000.0
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    det_metrics = metrics.compute_metrics_for_model(
        split_gts, all_predictions, iou_thresh=0.50, class_names=visdrone_parser.CLASS_NAMES
    )
    size_metrics = metrics.compute_size_metrics(split_gts, all_predictions, iou_thresh=0.50)

    res = {
        "Strategy": strategy_name,
        "mAP@0.5": round(float(det_metrics.get("mAP", 0.0)), 4),
        "Precision": round(float(det_metrics.get("precision", 0.0)), 4),
        "Recall": round(float(det_metrics.get("recall", 0.0)), 4),
        "F1": round(float(det_metrics.get("f1", 0.0)), 4),
        "Small_AP": round(float(size_metrics.get("small", {}).get("ap", 0.0) or 0.0), 4),
        "Medium_AP": round(float(size_metrics.get("medium", {}).get("ap", 0.0) or 0.0), 4),
        "Large_AP": round(float(size_metrics.get("large", {}).get("ap", 0.0) or 0.0), 4),
        "Mean_IoU": round(float(det_metrics.get("mean_iou", 0.0)), 4),
        "Latency_ms": round(float(latency_ms), 2),
        "FPS": round(float(fps), 2),
        "Num_Detections": len(all_predictions)
    }
    return res


def main():
    logger.info("==================================================================")
    logger.info("  EXPERIMENT 4: SMALL OBJECT STRATEGY (TILED / SLICED INFERENCE)  ")
    logger.info("==================================================================")

    weights_path = PROJECT_ROOT / "results" / "trained" / "yolov8s" / "weights" / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}")

    model = YOLO(str(weights_path))
    val_gts, val_img_paths, _ = load_split_ground_truth("val")

    results = []

    # 1. Standard 640px Inference
    def pred_standard(p):
        res = model(p, imgsz=640, conf=0.25, device=0, verbose=False)[0]
        out = []
        if res.boxes is not None and len(res.boxes) > 0:
            bxs = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            scs = res.boxes.conf.cpu().numpy()
            for b, c, s in zip(bxs, clss, scs):
                out.append({"box_xyxy": b, "class_id": c, "score": s})
        return out

    r_std = evaluate_strategy("Standard_640px", pred_standard, val_gts, val_img_paths)
    results.append(r_std)

    # 2. Tiled 2x2 Grid (overlap 0.20)
    def pred_tile_2x2(p):
        return predict_tiled(model, p, n_cols=2, n_rows=2, overlap=0.20, conf=0.25, device=0)

    r_tile2 = evaluate_strategy("Tiled_2x2_Grid", pred_tile_2x2, val_gts, val_img_paths)
    results.append(r_tile2)

    # 3. Tiled 3x2 Grid (overlap 0.25)
    def pred_tile_3x2(p):
        return predict_tiled(model, p, n_cols=3, n_rows=2, overlap=0.25, conf=0.25, device=0)

    r_tile3 = evaluate_strategy("Tiled_3x2_Grid", pred_tile_3x2, val_gts, val_img_paths)
    results.append(r_tile3)

    df = pd.DataFrame(results)
    out_csv = EXP_ROOT / "small_object_strategy_comparison.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Saved small-object strategy comparison to %s", out_csv)

    print("\n" + "="*80)
    print("EXPERIMENT 4 SUMMARY (Small-Object Recovery on Validation Set):")
    print(df.to_string(index=False))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
