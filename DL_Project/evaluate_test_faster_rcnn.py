"""
evaluate_test_faster_rcnn.py
============================
Performs evaluation of the fine-tuned Faster R-CNN (best.pt) on the 20 locked test images.
"""

import sys
import time
import json
import logging
from pathlib import Path
from PIL import Image
import numpy as np

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "data"))
sys.path.append(str(PROJECT_ROOT / "evaluation"))
sys.path.append(str(PROJECT_ROOT / "inference"))

import visdrone_parser
import metrics
import small_objects
import confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Eval_FasterRCNN")

CLASS_NAMES = visdrone_parser.CLASS_NAMES


def get_faster_rcnn_model(num_classes: int = 11):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main():
    logger.info("==================================================================")
    logger.info("     FINAL EVALUATION OF BEST CHECKPOINT ON LOCKED TEST SET       ")
    logger.info("==================================================================")

    if not torch.cuda.is_available():
        raise RuntimeError("FATAL: CUDA is not available! Refusing to fall back to CPU.")
    
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"Verified GPU: {gpu_name} (CUDA Device 0)")

    out_dir = PROJECT_ROOT / "results" / "trained" / "faster_rcnn"
    weights_dir = out_dir / "weights"
    best_weights_path = weights_dir / "best.pt"

    if not best_weights_path.exists():
        raise FileNotFoundError(f"Missing weights file at {best_weights_path}")

    model = get_faster_rcnn_model(num_classes=11)
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = best_weights_path.stat().st_size / (1024**2)
    logger.info(f"Loaded best.pt ({model_size_mb:.2f} MB, {param_count:,} parameters)")

    # Read best validation epoch and mAP from training_history.csv
    hist_csv = out_dir / "training_history.csv"
    best_epoch = 18
    best_val_map = 0.3325
    if hist_csv.exists():
        import csv
        with open(hist_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                v_map = float(row["val_map50"])
                if v_map >= best_val_map:
                    best_val_map = v_map
                    best_epoch = int(row["epoch"])

    # Load authentic VisDrone test ground truth
    raw_val_dir = PROJECT_ROOT / "data" / "raw" / "VisDrone2019-DET-val"
    imgs_dir = raw_val_dir / "images"
    anns_dir = raw_val_dir / "annotations"

    test_ids_file = PROJECT_ROOT / "data" / "processed" / "splits" / "test_ids.txt"
    test_ids = [l.strip() for l in open(test_ids_file, encoding="utf-8") if l.strip()]

    all_gts, _, _ = visdrone_parser.parse_visdrone_dataset(str(imgs_dir), str(anns_dir))
    test_gts = [g for g in all_gts if g["image_id"] in test_ids and 0 <= g["class_id"] <= 9]
    test_img_paths = [str(imgs_dir / f"{tid}.jpg") for tid in test_ids]

    logger.info(f"Locked Test Set: {len(test_img_paths)} images, {len(test_gts)} valid GT objects.")

    # Model warm-up on GPU
    logger.info("Warming up Faster R-CNN on GPU...")
    warmup_tensor = torchvision.transforms.functional.to_tensor(Image.open(test_img_paths[0]).convert("RGB")).to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model([warmup_tensor])

    # Timed inference
    logger.info("Running timed inference on 20 test images...")
    timings = []
    test_predictions = []

    with torch.no_grad():
        for img_p in test_img_paths:
            stem = Path(img_p).stem
            img_pil = Image.open(img_p).convert("RGB")
            W, H = img_pil.size
            img_t = torchvision.transforms.functional.to_tensor(img_pil).to(device)

            t0 = time.perf_counter()
            outputs = model([img_t])[0]
            dt = time.perf_counter() - t0
            timings.append(dt)

            boxes = outputs["boxes"].detach().cpu().numpy()
            scores = outputs["scores"].detach().cpu().numpy()
            labels = outputs["labels"].detach().cpu().numpy()

            for b, s, l in zip(boxes, scores, labels):
                if s >= 0.25 and 1 <= l <= 10:  # standard confidence threshold 0.25
                    x1 = max(0.0, float(b[0]))
                    y1 = max(0.0, float(b[1]))
                    x2 = min(float(W), float(b[2]))
                    y2 = min(float(H), float(b[3]))
                    if x2 > x1 and y2 > y1:
                        test_predictions.append({
                            "image_id": stem,
                            "class_id": int(l - 1),  # map back to 0..9
                            "score": float(s),
                            "box_xyxy": [x1, y1, x2, y2]
                        })

    mean_lat_s = float(np.mean(timings))
    latency_ms = mean_lat_s * 1000.0
    fps = 1.0 / mean_lat_s if mean_lat_s > 0 else 0.0

    logger.info(f"Inference complete: {len(test_predictions)} predictions generated.")
    logger.info(f"Latency: {latency_ms:.2f} ms | FPS: {fps:.2f}")

    # Calculate metrics
    det_metrics = metrics.compute_metrics_for_model(
        test_gts, test_predictions, iou_thresh=0.50, class_names=CLASS_NAMES
    )
    size_metrics = metrics.compute_size_metrics(test_gts, test_predictions, iou_thresh=0.50)

    # Confusion matrix
    cm_mat, cm_names = confusion_matrix.build_confusion_matrix(
        test_gts, test_predictions, CLASS_NAMES, iou_thresh=0.50
    )
    cm_out = out_dir / "test_confusion_matrix.png"
    confusion_matrix.plot_confusion_matrix(cm_mat, cm_names, "Confusion Matrix: Fine-Tuned Faster R-CNN (Test Set)", str(cm_out))

    mAP50 = det_metrics.get("mAP", 0.0)
    prec = det_metrics.get("precision", 0.0)
    rec = det_metrics.get("recall", 0.0)
    f1 = det_metrics.get("f1", 0.0)
    small_ap = size_metrics.get("small", {}).get("ap", 0.0) or 0.0
    med_ap = size_metrics.get("medium", {}).get("ap", 0.0) or 0.0
    large_ap = size_metrics.get("large", {}).get("ap", 0.0) or 0.0
    mean_iou = det_metrics.get("mean_iou", 0.0)

    test_res_dict = {
        "model": "Faster R-CNN (ResNet50+FPN)",
        "training_status": "fine-tuned",
        "best_epoch": best_epoch,
        "best_val_map50": round(float(best_val_map), 4),
        "test_images_count": len(test_img_paths),
        "test_gt_objects_count": len(test_gts),
        "test_predictions_count": len(test_predictions),
        "mAP50": round(float(mAP50), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "small_ap": round(float(small_ap), 4),
        "medium_ap": round(float(med_ap), 4),
        "large_ap": round(float(large_ap), 4),
        "mean_iou": round(float(mean_iou), 4),
        "latency_ms": round(float(latency_ms), 2),
        "fps": round(float(fps), 2),
        "parameters": int(param_count),
        "model_size_mb": round(float(model_size_mb), 2),
        "per_class_ap": det_metrics.get("per_class_AP", {})
    }

    # Save metrics JSON
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_res_dict, f, indent=2)

    # Append to results/trained_summary.csv
    summary_csv = PROJECT_ROOT / "results" / "trained_summary.csv"
    with open(summary_csv, "a", encoding="utf-8") as f:
        f.write(f"Faster R-CNN,fine-tuned,{mAP50:.4f},{prec:.4f},{rec:.4f},{f1:.4f},{small_ap:.4f},{med_ap:.4f},{large_ap:.4f},{mean_iou:.4f},{latency_ms:.2f},{fps:.2f},{param_count},{model_size_mb:.2f}\n")

    # Save comparison with zero-shot Faster R-CNN
    zero_shot_frcnn = {
        "mAP50": 0.0888,
        "precision": 0.3910,
        "recall": 0.2574,
        "f1": 0.3104,
        "small_ap": 0.0084,
        "mean_iou": 0.7792,
        "latency_ms": 1963.0,
        "fps": 0.51
    }

    zs_vs_trained_csv = PROJECT_ROOT / "results" / "zero_shot_vs_trained_faster_rcnn.csv"
    with open(zs_vs_trained_csv, "w", encoding="utf-8") as f:
        f.write("metric,zero_shot,fine_tuned,absolute_change,relative_change_pct\n")
        for k in ["mAP50", "precision", "recall", "f1", "small_ap", "mean_iou", "latency_ms", "fps"]:
            zs = zero_shot_frcnn[k]
            ft = test_res_dict[k]
            diff = ft - zs
            pct_s = f"{(diff / zs)*100.0:+.2f}%" if zs != 0 else "N/A"
            f.write(f"{k},{zs},{ft},{diff:+.4f},{pct_s}\n")

    # Print summary
    print("\n" + "=" * 80)
    print("      FASTER R-CNN ZERO-SHOT VS. VISDRONE FINE-TUNED BENCHMARK")
    print("=" * 80)
    print(f"{'Metric':<16} | {'Zero-Shot':<12} | {'Fine-Tuned':<12} | {'Change':<16}")
    print("-" * 80)
    for k in ["mAP50", "precision", "recall", "f1", "small_ap", "mean_iou", "latency_ms", "fps"]:
        zs = zero_shot_frcnn[k]
        ft = test_res_dict[k]
        diff = ft - zs
        pct_s = f"({(diff / zs)*100.0:+.1f}%)" if zs != 0 else "(N/A)"
        print(f"{k:<16} | {zs:<12.4f} | {ft:<12.4f} | {diff:+.4f} {pct_s}")
    print("=" * 80)


if __name__ == "__main__":
    main()
