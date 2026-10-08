"""
run_exp9_two_stage_hybrid.py
============================
Experiment 9: Hybrid Solution B (Selective Two-Stage Refinement Pipeline)
Combines fast global inference from YOLOv8s with selective high-resolution
refinement from Faster R-CNN ResNet-50-FPN on dense or small-object clusters.

Pipeline:
  1. Full image -> Fast YOLOv8s (640px)
  2. Spatial Cluster Analysis: identify clusters of tiny or low-confidence objects (<48px, conf < 0.40)
  3. Selective High-Res Cropping: extract bounding regions of clusters at native resolution
  4. Second-Stage Refinement: Run fine-tuned Faster R-CNN on the crop
  5. Global Projection & Soft/Greedy NMS: Merge global detections with refined local detections
  6. Final output evaluation on the 106-image Validation split

Outputs:
  - Validation metrics: mAP@0.5, Small AP, Medium AP, Large AP, Precision, Recall, F1, Latency, FPS
  - Comparison vs Standalone YOLOv8s and Standalone Faster R-CNN
  - Saved to results/hybrid/selective_two_stage/
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
sys.path.append(str(PROJECT_ROOT / "scripts"))
sys.path.append(str(PROJECT_ROOT / "evaluation"))
sys.path.append(str(PROJECT_ROOT / "data"))

from research_evaluator import load_split_ground_truth
import metrics
import small_objects
import visdrone_parser
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp9_TwoStageHybrid")

EXP_ROOT = PROJECT_ROOT / "results" / "hybrid" / "selective_two_stage"
EXP_ROOT.mkdir(parents=True, exist_ok=True)


def load_faster_rcnn(device):
    weights_path = PROJECT_ROOT / "results" / "trained" / "faster_rcnn" / "weights" / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Faster R-CNN weights not found: {weights_path}")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 11)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thresh: float = 0.50):
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


def predict_two_stage(img_path: str, yolo_model: YOLO, frcnn_model, device, conf: float = 0.25):
    im_pil = Image.open(img_path).convert("RGB")
    W, H = im_pil.size

    # Stage 1: Fast YOLOv8s full-frame inference
    yolo_res = yolo_model(img_path, imgsz=640, conf=conf, device=0, verbose=False)[0]
    all_boxes = []
    all_scores = []
    all_classes = []

    small_or_uncertain_boxes = []

    if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
        bxs = yolo_res.boxes.xyxy.cpu().numpy()
        clss = yolo_res.boxes.cls.cpu().numpy().astype(int)
        scs = yolo_res.boxes.conf.cpu().numpy()

        for b, c, s in zip(bxs, clss, scs):
            if 0 <= c <= 9:
                all_boxes.append(b)
                all_scores.append(s)
                all_classes.append(c)
                # Flag small or uncertain objects for Stage 2 refinement
                bw = b[2] - b[0]
                bh = b[3] - b[1]
                if (bw < 48 and bh < 48) or s < 0.45:
                    small_or_uncertain_boxes.append(b)

    # Stage 2: Selective High-Res Refinement via Faster R-CNN
    # If small/uncertain boxes exist, cluster them and run high-res crop
    if len(small_or_uncertain_boxes) >= 2:
        # Compute bounding envelope of cluster with margin
        sub_arr = np.array(small_or_uncertain_boxes)
        cx1 = max(0, int(np.min(sub_arr[:, 0]) - 30))
        cy1 = max(0, int(np.min(sub_arr[:, 1]) - 30))
        cx2 = min(W, int(np.max(sub_arr[:, 2]) + 30))
        cy2 = min(H, int(np.max(sub_arr[:, 3]) + 30))

        crop_w = cx2 - cx1
        crop_h = cy2 - cy1

        # Only refine if the crop preserves higher resolution than global 640
        if crop_w > 40 and crop_h > 40 and (crop_w < W * 0.9 or crop_h < H * 0.9):
            crop_im = im_pil.crop((cx1, cy1, cx2, cy2))
            crop_tensor = torchvision.transforms.functional.to_tensor(crop_im).to(device)

            with torch.no_grad():
                frcnn_out = frcnn_model([crop_tensor])[0]

            fb = frcnn_out["boxes"].cpu().numpy()
            fs = frcnn_out["scores"].cpu().numpy()
            fc = frcnn_out["labels"].cpu().numpy() - 1  # 1-indexed to 0-indexed

            valid_mask = (fs >= 0.35) & (fc >= 0) & (fc <= 9)
            if np.any(valid_mask):
                fb_valid = fb[valid_mask]
                fs_valid = fs[valid_mask]
                fc_valid = fc[valid_mask]

                # Project back to full image coordinates
                fb_valid[:, [0, 2]] += cx1
                fb_valid[:, [1, 3]] += cy1

                all_boxes.extend(fb_valid)
                all_scores.extend(fs_valid)
                all_classes.extend(fc_valid)

    if not all_boxes:
        return []

    boxes_arr = np.array(all_boxes)
    scores_arr = np.array(all_scores)
    classes_arr = np.array(all_classes)

    keep_idx = nms_xyxy(boxes_arr, scores_arr, classes_arr, iou_thresh=0.45)

    final_preds = []
    for idx in keep_idx:
        b = boxes_arr[idx]
        x1 = max(0.0, float(b[0]))
        y1 = max(0.0, float(b[1]))
        x2 = min(float(W), float(b[2]))
        y2 = min(float(H), float(b[3]))
        c = int(classes_arr[idx])
        s = float(scores_arr[idx])
        if x2 > x1 and y2 > y1 and 0 <= c <= 9:
            final_preds.append({"box_xyxy": [x1, y1, x2, y2], "class_id": c, "score": s})

    return final_preds


def main():
    logger.info("==================================================================")
    logger.info("  EXPERIMENT 9: HYBRID SOLUTION B (SELECTIVE TWO-STAGE PIPELINE)  ")
    logger.info("==================================================================")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load YOLOv8s
    yolo_weights = PROJECT_ROOT / "results" / "trained" / "yolov8s" / "weights" / "best.pt"
    yolo_model = YOLO(str(yolo_weights))

    # Load Faster R-CNN
    frcnn_model = load_faster_rcnn(device)

    # Load validation split
    val_gts, val_img_paths, _ = load_split_ground_truth("val")

    logger.info("Running Selective Two-Stage Pipeline on %d validation images...", len(val_img_paths))
    timings = []
    all_predictions = []

    # Warmup
    for _ in range(3):
        _ = predict_two_stage(val_img_paths[0], yolo_model, frcnn_model, device)

    for img_path in val_img_paths:
        img_id = Path(img_path).stem
        t0 = time.perf_counter()
        preds = predict_two_stage(img_path, yolo_model, frcnn_model, device)
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
        val_gts, all_predictions, iou_thresh=0.50, class_names=visdrone_parser.CLASS_NAMES
    )
    size_metrics = metrics.compute_size_metrics(val_gts, all_predictions, iou_thresh=0.50)

    res = {
        "Architecture": "Selective_Two_Stage_Hybrid (YOLOv8s + FRCNN)",
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

    # Save metrics JSON & CSV
    with open(EXP_ROOT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    df = pd.DataFrame([res])
    df.to_csv(EXP_ROOT / "two_stage_comparison.csv", index=False)

    print("\n" + "="*80)
    print("EXPERIMENT 9 SUMMARY (Selective Two-Stage Hybrid on Validation Set):")
    print(df.to_string(index=False))
    print("="*80 + "\n")

    del yolo_model, frcnn_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
