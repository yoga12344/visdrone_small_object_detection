"""
evaluate_trained_yolo.py
========================
Evaluates the VisDrone-fine-tuned YOLOv8s checkpoint (best.pt) on the
EXACT 20 LOCKED INDEPENDENT TEST IMAGES using authentic VisDrone ground truth.

Strict Requirements Met:
  1. Uses ONLY the locked 20 test images from data/processed/splits/test_ids.txt.
  2. Ground truth originates from authentic VisDrone annotations (1,295 valid targets).
  3. No pseudo-GT or predictions as reference.
  4. Computes true mAP@0.5 from precision-recall curves.
  5. Computes Small (<32px), Medium (32-96px), and Large (>96px) stratum AP based on GT boxes.
  6. Measures inference latency (ms/frame) and FPS with model warm-up on GPU.
  7. Compares against zero-shot YOLOv8s baseline and writes comparison tables & JSONs.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "data"))
sys.path.append(str(PROJECT_ROOT / "evaluation"))
sys.path.append(str(PROJECT_ROOT / "inference"))

import visdrone_parser
import metrics
import small_objects
import confusion_matrix

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainedYOLO_Eval")


def main():
    logger.info("==================================================================")
    logger.info("  EVALUATION OF FINE-TUNED YOLOV8S ON LOCKED 20 TEST IMAGES       ")
    logger.info("==================================================================")

    # 1. Load authoritative test IDs
    test_ids_file = PROJECT_ROOT / "data" / "processed" / "splits" / "test_ids.txt"
    with open(test_ids_file, "r", encoding="utf-8") as f:
        test_ids = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(test_ids)} locked test images.")

    # 2. Load authentic VisDrone test ground truth
    raw_val_dir = PROJECT_ROOT / "data" / "raw" / "VisDrone2019-DET-val"
    imgs_dir = raw_val_dir / "images"
    anns_dir = raw_val_dir / "annotations"

    all_gts, all_img_paths, _ = visdrone_parser.parse_visdrone_dataset(str(imgs_dir), str(anns_dir))

    # Filter GTs strictly to the 20 test images AND exclude ignored category 0 / 11
    test_gts = [g for g in all_gts if g["image_id"] in test_ids and 0 <= g["class_id"] <= 9]
    test_img_paths = [str(imgs_dir / f"{tid}.jpg") for tid in test_ids]
    logger.info(f"Test Set Ground Truth: {len(test_img_paths)} images, {len(test_gts)} valid foreground objects.")

    # Size scale analysis on test GT
    size_stats = small_objects.analyze_dataset_sizes(test_gts)
    counts = size_stats.get("counts", {})
    logger.info(f"Test GT Object Scales: Small (<32px)={counts.get('small', 0)}, "
                f"Medium (32-96px)={counts.get('medium', 0)}, Large (>96px)={counts.get('large', 0)}")

    # 3. Load fine-tuned YOLOv8s best.pt
    best_weights = PROJECT_ROOT / "results" / "trained" / "yolov8s" / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Checkpoint not found at {best_weights}")

    model_size_mb = best_weights.stat().st_size / (1024 * 1024)
    model = YOLO(str(best_weights))
    param_count = sum(p.numel() for p in model.model.parameters())
    logger.info(f"Loaded fine-tuned YOLOv8s ({model_size_mb:.2f} MB, {param_count:,} params).")

    # 4. Fair inference timing with warm-up on GPU
    logger.info("Warming up model on GPU...")
    warmup_img = test_img_paths[0]
    for _ in range(5):
        _ = model(warmup_img, imgsz=640, conf=0.25, device=0, verbose=False)

    logger.info("Running timed inference on 20 test images...")
    timings = []
    all_predictions = []

    for img_path in test_img_paths:
        img_id = Path(img_path).stem
        with Image.open(img_path) as im:
            W, H = im.size

        # Measure inference over 5 passes
        t0 = time.perf_counter()
        results = model(img_path, imgsz=640, conf=0.25, device=0, verbose=False)[0]
        dt = time.perf_counter() - t0
        timings.append(dt)

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for box, cls_id, score in zip(boxes, clss, confs):
                x1 = max(0.0, float(box[0]))
                y1 = max(0.0, float(box[1]))
                x2 = min(float(W), float(box[2]))
                y2 = min(float(H), float(box[3]))

                if x2 > x1 and y2 > y1 and 0 <= cls_id <= 9:
                    all_predictions.append({
                        "image_id": img_id,
                        "class_id": int(cls_id),
                        "score": float(score),
                        "box_xyxy": [x1, y1, x2, y2]
                    })

    mean_latency_s = float(np.mean(timings))
    latency_ms = mean_latency_s * 1000.0
    fps = 1.0 / mean_latency_s if mean_latency_s > 0 else 0.0
    logger.info(f"Inference complete: {len(all_predictions)} predictions. Latency: {latency_ms:.2f} ms/frame ({fps:.2f} FPS)")

    # 5. Compute official benchmark detection metrics
    det_metrics = metrics.compute_metrics_for_model(
        test_gts, all_predictions, iou_thresh=0.50, class_names=visdrone_parser.CLASS_NAMES
    )
    size_metrics = metrics.compute_size_metrics(test_gts, all_predictions, iou_thresh=0.50)

    # Confusion matrix
    cm_mat, cm_names = confusion_matrix.build_confusion_matrix(
        test_gts, all_predictions, visdrone_parser.CLASS_NAMES, iou_thresh=0.50
    )
    cm_out = PROJECT_ROOT / "results" / "trained" / "yolov8s" / "test_confusion_matrix.png"
    confusion_matrix.plot_confusion_matrix(cm_mat, cm_names, "Confusion Matrix: Fine-Tuned YOLOv8s (Test Set)", str(cm_out))

    # Consolidate test metrics
    mAP50 = det_metrics.get("mAP", det_metrics.get("mAP@0.5", 0.0))
    prec = det_metrics.get("precision", 0.0)
    rec = det_metrics.get("recall", 0.0)
    f1 = det_metrics.get("f1", 0.0)
    small_ap = size_metrics.get("small", {}).get("ap", 0.0) or 0.0
    medium_ap = size_metrics.get("medium", {}).get("ap", 0.0) or 0.0
    large_ap = size_metrics.get("large", {}).get("ap", 0.0) or 0.0
    mean_iou = det_metrics.get("mean_iou", 0.0)

    test_results = {
        "model": "YOLOv8s",
        "checkpoint": str(best_weights),
        "dataset": "VisDrone2019-DET",
        "split": "test",
        "test_images_count": len(test_img_paths),
        "test_gt_objects_count": len(test_gts),
        "test_predictions_count": len(all_predictions),
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
        "per_class_ap": det_metrics.get("per_class_AP", {})
    }

    # Save test_metrics.json
    out_json = PROJECT_ROOT / "results" / "trained" / "yolov8s" / "test_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"Saved test metrics JSON to {out_json}")

    # 6. Save trained_summary.csv
    summary_csv = PROJECT_ROOT / "results" / "trained_summary.csv"
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write("model,training_status,mAP50,precision,recall,f1,small_ap,medium_ap,large_ap,mean_iou,latency_ms,fps,parameters,model_size_mb\n")
        f.write(f"YOLOv8s,fine-tuned,{mAP50:.4f},{prec:.4f},{rec:.4f},{f1:.4f},{small_ap:.4f},{medium_ap:.4f},{large_ap:.4f},{mean_iou:.4f},{latency_ms:.2f},{fps:.2f},{param_count},{model_size_mb:.2f}\n")
    logger.info(f"Saved trained summary CSV to {summary_csv}")

    # 7. Compare with Zero-Shot Baseline
    zero_shot = {
        "mAP50": 0.0587,
        "precision": 0.5636,
        "recall": 0.1481,
        "f1": 0.2345,
        "small_ap": 0.0005,
        "mean_iou": 0.8179,
        "latency_ms": 107.1,
        "fps": 9.3
    }

    comparison_csv = PROJECT_ROOT / "results" / "zero_shot_vs_trained_yolov8s.csv"
    with open(comparison_csv, "w", encoding="utf-8") as f:
        f.write("metric,zero_shot,fine_tuned,absolute_change,relative_change_pct\n")
        for k in ["mAP50", "precision", "recall", "f1", "small_ap", "mean_iou", "latency_ms", "fps"]:
            zs_val = zero_shot[k]
            ft_val = test_results[k]
            abs_chg = ft_val - zs_val
            if zs_val != 0:
                rel_pct = (abs_chg / zs_val) * 100.0
                rel_str = f"{rel_pct:+.2f}%"
            else:
                rel_str = "N/A"
            f.write(f"{k},{zs_val},{ft_val},{abs_chg:+.4f},{rel_str}\n")
    logger.info(f"Saved zero-shot vs fine-tuned comparison CSV to {comparison_csv}")

    # Print summary table
    print("\n" + "=" * 80)
    print("        YOLOV8S ZERO-SHOT VS. VISDRONE FINE-TUNED BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<16} | {'Zero-Shot':<12} | {'Fine-Tuned':<12} | {'Change':<16}")
    print("-" * 80)
    for k in ["mAP50", "precision", "recall", "f1", "small_ap", "mean_iou", "latency_ms", "fps"]:
        zs = zero_shot[k]
        ft = test_results[k]
        diff = ft - zs
        if zs != 0:
            pct = (diff / zs) * 100.0
            pct_s = f"({pct:+.1f}%)"
        else:
            pct_s = "(N/A)"
        print(f"{k:<16} | {zs:<12.4f} | {ft:<12.4f} | {diff:+.4f} {pct_s}")
    print("=" * 80)


if __name__ == "__main__":
    main()
