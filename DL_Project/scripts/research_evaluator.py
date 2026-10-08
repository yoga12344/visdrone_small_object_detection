"""
research_evaluator.py
=====================
Unified evaluation engine for the Final Research Optimization and Ablation Phase.

Provides standardized, reproducible evaluation of object detection models
on either the validation split (for ablations/tuning) or the locked test split.

Metrics computed:
  - mAP@0.5 (true VOC/COCO style area under PR curve)
  - Precision, Recall, F1
  - Size-stratified AP: Small (<32px), Medium (32-96px), Large (>96px)
  - Mean IoU of matched true positives
  - Latency (ms/frame) via synchronized GPU timer
  - Throughput (FPS)
  - Parameter count and disk model size (MB)
  - Peak GPU VRAM allocated (MB)
"""

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "data"))
sys.path.append(str(PROJECT_ROOT / "evaluation"))
sys.path.append(str(PROJECT_ROOT / "inference"))

import confusion_matrix
import metrics
import small_objects
import visdrone_parser
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ResearchEvaluator")


def load_split_ground_truth(split: str = "val") -> Tuple[List[Dict], List[str], Dict]:
    """
    Load ground truth annotations and image paths for a given split ('val' or 'test').
    Returns (filtered_gts, image_paths, stats).
    """
    split_file = PROJECT_ROOT / "data" / "processed" / "splits" / f"{split}_ids.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, "r", encoding="utf-8") as f:
        image_ids = [line.strip() for line in f if line.strip()]

    raw_val_dir = PROJECT_ROOT / "data" / "raw" / "VisDrone2019-DET-val"
    imgs_dir = raw_val_dir / "images"
    anns_dir = raw_val_dir / "annotations"

    all_gts, _, stats = visdrone_parser.parse_visdrone_dataset(str(imgs_dir), str(anns_dir))

    # Keep only targets for images in split and valid classes 0..9
    split_gts = [g for g in all_gts if g["image_id"] in image_ids and 0 <= g["class_id"] <= 9]
    split_img_paths = [str(imgs_dir / f"{iid}.jpg") for iid in image_ids]

    logger.info("Loaded %s split: %d images, %d valid ground-truth targets.",
                split, len(split_img_paths), len(split_gts))
    return split_gts, split_img_paths, stats


def evaluate_yolo_model(
    model_or_weights,
    split: str = "val",
    imgsz: int = 640,
    conf: float = 0.25,
    iou_nms: float = 0.45,
    device: int = 0,
    model_name: str = "YOLO",
    save_dir: Optional[str] = None,
    custom_predictor=None
) -> Dict:
    """
    Evaluates a YOLO model or custom predictor on the given split with synchronized GPU timing.
    """
    gts, img_paths, _ = load_split_ground_truth(split)

    if torch.cuda.is_available() and device >= 0:
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        dev_str = f"cuda:{device}"
    else:
        dev_str = "cpu"

    if isinstance(model_or_weights, (str, Path)):
        weights_path = Path(model_or_weights)
        model_size_mb = weights_path.stat().st_size / (1024 * 1024)
        model = YOLO(str(weights_path))
    else:
        model = model_or_weights
        model_size_mb = 0.0

    param_count = sum(p.numel() for p in model.model.parameters()) if hasattr(model, "model") else 0

    # Model warm-up on GPU
    logger.info("Warming up %s on %s...", model_name, dev_str)
    if len(img_paths) > 0:
        warmup_img = img_paths[0]
        for _ in range(5):
            if custom_predictor:
                _ = custom_predictor(warmup_img)
            else:
                _ = model(warmup_img, imgsz=imgsz, conf=conf, iou=iou_nms, device=device, verbose=False)

    logger.info("Running evaluation for %s on %d %s images (imgsz=%d, conf=%.2f)...",
                model_name, len(img_paths), split, imgsz, conf)

    timings = []
    all_predictions = []

    for img_path in img_paths:
        img_id = Path(img_path).stem
        with Image.open(img_path) as im:
            W, H = im.size

        if torch.cuda.is_available() and dev_str.startswith("cuda"):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            if custom_predictor:
                preds = custom_predictor(img_path)
            else:
                results = model(img_path, imgsz=imgsz, conf=conf, iou=iou_nms, device=device, verbose=False)[0]
                preds = []
                if results.boxes is not None and len(results.boxes) > 0:
                    bxs = results.boxes.xyxy.cpu().numpy()
                    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
                    scores = results.boxes.conf.cpu().numpy()
                    for b, c, s in zip(bxs, cls_ids, scores):
                        preds.append({"box_xyxy": b, "class_id": int(c), "score": float(s)})
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            timings.append(elapsed_ms / 1000.0)
        else:
            t0 = time.perf_counter()
            if custom_predictor:
                preds = custom_predictor(img_path)
            else:
                results = model(img_path, imgsz=imgsz, conf=conf, iou=iou_nms, device=device, verbose=False)[0]
                preds = []
                if results.boxes is not None and len(results.boxes) > 0:
                    bxs = results.boxes.xyxy.cpu().numpy()
                    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
                    scores = results.boxes.conf.cpu().numpy()
                    for b, c, s in zip(bxs, cls_ids, scores):
                        preds.append({"box_xyxy": b, "class_id": int(c), "score": float(s)})
            dt = time.perf_counter() - t0
            timings.append(dt)

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

    mean_latency_s = float(np.mean(timings)) if timings else 0.0
    latency_ms = mean_latency_s * 1000.0
    fps = 1.0 / mean_latency_s if mean_latency_s > 0 else 0.0

    peak_vram_mb = 0.0
    if torch.cuda.is_available() and dev_str.startswith("cuda"):
        try:
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        except Exception:
            peak_vram_mb = 0.0

    # Compute official benchmark metrics
    det_metrics = metrics.compute_metrics_for_model(
        gts, all_predictions, iou_thresh=0.50, class_names=visdrone_parser.CLASS_NAMES
    )
    size_metrics = metrics.compute_size_metrics(gts, all_predictions, iou_thresh=0.50)

    mAP50 = det_metrics.get("mAP", 0.0)
    prec = det_metrics.get("precision", 0.0)
    rec = det_metrics.get("recall", 0.0)
    f1 = det_metrics.get("f1", 0.0)
    small_ap = size_metrics.get("small", {}).get("ap", 0.0) or 0.0
    medium_ap = size_metrics.get("medium", {}).get("ap", 0.0) or 0.0
    large_ap = size_metrics.get("large", {}).get("ap", 0.0) or 0.0
    mean_iou = det_metrics.get("mean_iou", 0.0)

    res = {
        "model": model_name,
        "split": split,
        "imgsz": imgsz,
        "conf": conf,
        "iou_nms": iou_nms,
        "num_images": len(img_paths),
        "num_gt_targets": len(gts),
        "num_predictions": len(all_predictions),
        "mAP50": round(float(mAP50), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "small_ap": round(float(small_ap), 4),
        "medium_ap": round(float(medium_ap), 4),
        "large_ap": round(float(large_ap), 4),
        "mean_iou": round(float(mean_iou), 4),
        "latency_ms": round(float(latency_ms), 2),
        "fps": round(float(fps), 2),
        "parameters": int(param_count),
        "model_size_mb": round(float(model_size_mb), 2),
        "peak_vram_mb": round(float(peak_vram_mb), 2),
        "per_class_ap": det_metrics.get("per_class_AP", {})
    }

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

        # Build and plot confusion matrix
        cm_mat, cm_names = confusion_matrix.build_confusion_matrix(
            gts, all_predictions, visdrone_parser.CLASS_NAMES, iou_thresh=0.50
        )
        cm_plot_file = save_path / "confusion_matrix.png"
        confusion_matrix.plot_confusion_matrix(
            cm_mat, cm_names, f"Confusion Matrix: {model_name} ({split.title()} Split)", str(cm_plot_file)
        )
        logger.info("Saved evaluation artifacts to %s", save_path)

    # Clean up GPU
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return res
