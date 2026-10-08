"""
run_benchmark.py
================
Master Benchmark Execution Pipeline for CV Hackathon.

Strict Requirements Met:
  1. Uses REAL VisDrone-DET ground-truth annotations (NO pseudo-GT!).
  2. Benchmarks all 5 architectures on the EXACT same test image set.
  3. Pretrained COCO classes are mapped rigorously to VisDrone classes.
  4. Proper IoU matching (0.50 threshold), true precision-recall curve mAP.
  5. Detailed Small-Object (<32x32px) sensitivity breakdown.
  6. Generates full suite of metric JSONs, comparison plots, and confusion matrices.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

# Adjust paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "data"))
sys.path.append(str(PROJECT_ROOT / "evaluation"))
sys.path.append(str(PROJECT_ROOT / "inference"))
sys.path.append(str(PROJECT_ROOT / "visualization"))

import visdrone_parser
import metrics
import small_objects
import confusion_matrix
import benchmark_plots

from yolo_inferencer import YOLOv8sInferencer
from rcnn_inferencer import FasterRCNNInferencer, MaskRCNNInferencer
from rfdetr_inferencer import RFDETRNanoInferencer
from sam_inferencer import YOLOSAMInferencer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Benchmark")


def main():
    logger.info("==================================================================")
    logger.info("   VISDRONE COMPREHENSIVE MULTI-MODEL RESEARCH BENCHMARK          ")
    logger.info("==================================================================")

    # 1. Dataset Discovery
    val_raw_dir = PROJECT_ROOT / "data" / "raw" / "VisDrone2019-DET-val"
    imgs_dir = val_raw_dir / "images"
    anns_dir = val_raw_dir / "annotations"

    if not imgs_dir.exists() or not anns_dir.exists():
        raise FileNotFoundError(f"VisDrone val directory not found at {val_raw_dir}")

    # Parse full VisDrone ground truth
    logger.info("Loading Real VisDrone Ground Truth annotations...")
    all_gts, all_img_paths, stats = visdrone_parser.parse_visdrone_dataset(str(imgs_dir), str(anns_dir))
    logger.info(f"Loaded {len(all_img_paths)} total images and {len(all_gts)} valid GT annotations.")

    # Select representative evaluation subset (e.g. 20 representative aerial images for CPU feasibility)
    eval_image_ids = sorted(list(all_img_paths.keys()))[:20]
    eval_img_paths = [all_img_paths[img_id] for img_id in eval_image_ids]
    eval_gts = [g for g in all_gts if g['image_id'] in eval_image_ids]

    # Analyze sizes in the test subset
    size_stats = small_objects.analyze_dataset_sizes(eval_gts)
    counts = size_stats.get('counts', {})
    logger.info(f"Evaluation Subset: {len(eval_img_paths)} images, {len(eval_gts)} ground truth objects.")
    logger.info(f"Object scale breakdown: Small={counts.get('small', 0)}, "
                f"Medium={counts.get('medium', 0)}, Large={counts.get('large', 0)}")

    # 2. Instantiate Models
    yolo_weights_path = str(PROJECT_ROOT.parent / "yolov8s.pt")
    
    models = {
        "YOLOv8s": YOLOv8sInferencer(weights_path=yolo_weights_path, conf_threshold=0.25),
        "Faster R-CNN": FasterRCNNInferencer(conf_threshold=0.30),
        "Mask R-CNN": MaskRCNNInferencer(conf_threshold=0.30),
        "RF-DETR Nano": RFDETRNanoInferencer(conf_threshold=0.25),
        "YOLOv8 + SAM": YOLOSAMInferencer(yolo_weights=yolo_weights_path, conf_threshold=0.25)
    }

    results = {}
    cm_dir = PROJECT_ROOT / "results" / "confusion_matrices"
    os.makedirs(cm_dir, exist_ok=True)

    # 3. Benchmark Execution Loop
    for name, model in models.items():
        logger.info(f"\n---> Benchmarking {name}...")
        try:
            model.load_model()
            preds, timing = model.predict_dataset(eval_img_paths)
            
            # Format predictions for evaluation engine
            pred_dicts = [p.to_dict() for p in preds]

            # Calculate Detection & Small Object Metrics
            det_metrics = metrics.compute_metrics_for_model(
                eval_gts, pred_dicts, iou_thresh=0.50, class_names=visdrone_parser.CLASS_NAMES
            )
            size_metrics = metrics.compute_size_metrics(eval_gts, pred_dicts, iou_thresh=0.50)

            # Confusion Matrix
            cm_mat, cm_names = confusion_matrix.build_confusion_matrix(
                eval_gts, pred_dicts, visdrone_parser.CLASS_NAMES, iou_thresh=0.50
            )
            cm_plot_path = str(cm_dir / f"confusion_{name.lower().replace(' ', '_').replace('+', '_')}.png")
            confusion_matrix.plot_confusion_matrix(cm_mat, cm_names, f"Confusion Matrix: {name} on VisDrone", cm_plot_path)

            # Check model parameters & footprint
            info = model.get_model_info()
            params_m = (info.get('param_count') or 0) / 1e6
            size_mb = info.get('size_mb') or 0.0

            # Consolidate
            mAP50 = det_metrics.get('mAP', det_metrics.get('mAP@0.5', 0.0))
            prec = det_metrics.get('precision', 0.0)
            rec = det_metrics.get('recall', 0.0)
            f1 = det_metrics.get('f1', 0.0)
            small_map = size_metrics.get('small', {}).get('ap', 0.0) or 0.0
            mean_iou = det_metrics.get('mean_iou', 0.0)

            results[name] = {
                "mAP50": round(float(mAP50), 4),
                "Precision": round(float(prec), 4),
                "Recall": round(float(rec), 4),
                "F1": round(float(f1), 4),
                "Small_mAP": round(float(small_map), 4),
                "MeanIoU": round(float(mean_iou), 4),
                "Time_ms": round(float(timing['mean_s'] * 1000), 2),
                "FPS": round(float(timing['fps']), 2),
                "Params_M": round(float(params_m), 2),
                "Size_MB": round(float(size_mb), 2)
            }
            logger.info(f"Completed {name}: mAP@0.5={results[name]['mAP50']}, Latency={results[name]['Time_ms']}ms, FPS={results[name]['FPS']}")
        except Exception as e:
            logger.error(f"Error evaluating {name}: {e}", exc_info=True)
            results[name] = {
                "mAP50": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0,
                "Small_mAP": 0.0, "MeanIoU": 0.0, "Time_ms": 0.0, "FPS": 0.0,
                "Params_M": 0.0, "Size_MB": 0.0, "error": str(e)
            }

    # 4. Save Metrics & Comparison Tables
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # 5. Generate Visual Charts
    plots_dir = PROJECT_ROOT / "results" / "plots"
    benchmark_plots.plot_all_benchmark_charts(results, str(plots_dir))

    # 6. Print Markdown Summary Table
    print("\n" + "=" * 95)
    print("                    FINAL VISDRONE BENCHMARK RESULTS")
    print("=" * 95)
    hdr = f"| {'Model':<15} | {'mAP@0.5':<8} | {'Precision':<9} | {'Recall':<7} | {'F1':<6} | {'Small mAP':<9} | {'Mean IoU':<8} | {'Time (ms)':<9} | {'FPS':<6} |"
    sep = f"|{'-'*17}|{'-'*10}|{'-'*11}|{'-'*9}|{'-'*8}|{'-'*11}|{'-'*10}|{'-'*11}|{'-'*8}|"
    print(hdr)
    print(sep)
    for m, d in results.items():
        row = f"| {m:<15} | {d['mAP50']:<8.4f} | {d['Precision']:<9.4f} | {d['Recall']:<7.4f} | {d['F1']:<6.4f} | {d['Small_mAP']:<9.4f} | {d['MeanIoU']:<8.4f} | {d['Time_ms']:<9.1f} | {d['FPS']:<6.1f} |"
        print(row)
    print("=" * 95)


if __name__ == "__main__":
    main()
