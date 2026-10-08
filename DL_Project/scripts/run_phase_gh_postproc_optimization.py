"""
run_phase_gh_postproc_optimization.py
Phase G & H: Systematic Confidence and NMS Optimization on the 106-image Validation set.
Sweeps:
  Confidence: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
  NMS IoU:    [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
Saves validation sweeps and identifies optimal operating points for:
  1. Maximum F1 score
  2. Maximum mAP@0.5
  3. Maximum Small-Object AP
"""
import sys
from pathlib import Path
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "data"))

from research_evaluator import evaluate_yolo_model
from ultralytics import YOLO

OUT_DIR = PROJECT_ROOT / "results" / "performance_maximization"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = PROJECT_ROOT / "results" / "final_optimization" / "training" / "yolov8s_100ep_640" / "weights" / "best.pt"

def sweep_confidence():
    print("=" * 60)
    print("PHASE G: CONFIDENCE THRESHOLD OPTIMIZATION (VAL SET)")
    print("=" * 60)
    confs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    results = []

    for conf in confs:
        print(f"Evaluating conf = {conf:.2f} (iou=0.45)...")
        r = evaluate_yolo_model(
            model_or_weights=str(CKPT_PATH),
            split="val",
            imgsz=640,
            conf=conf,
            iou_nms=0.45,
            device=0,
            model_name=f"yolov8s_conf{conf:.2f}"
        )
        results.append({
            "conf_threshold": conf,
            "val_mAP50": r["mAP50"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "small_ap": r["small_ap"],
            "medium_ap": r["medium_ap"],
            "large_ap": r["large_ap"],
            "mean_iou": r["mean_iou"],
            "latency_ms": r["latency_ms"],
            "fps": r["fps"]
        })
        print(f"Conf {conf:.2f}: mAP={r['mAP50']:.4f}, Prec={r['precision']:.4f}, Rec={r['recall']:.4f}, F1={r['f1']:.4f}, Small_AP={r['small_ap']:.4f}")

    df = pd.DataFrame(results)
    out_csv = OUT_DIR / "confidence_threshold_sweep.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nConfidence sweep saved to {out_csv}")
    best_f1_row = df.loc[df['f1'].idxmax()]
    print(f"Optimal Conf for F1: {best_f1_row['conf_threshold']:.2f} (F1 = {best_f1_row['f1']:.4f})")
    return best_f1_row['conf_threshold']

def sweep_nms(best_conf: float):
    print("\n" + "=" * 60)
    print(f"PHASE H: NMS IoU THRESHOLD OPTIMIZATION (VAL SET, conf={best_conf:.2f})")
    print("=" * 60)
    ious = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    results = []

    for iou in ious:
        print(f"Evaluating nms_iou = {iou:.2f}...")
        r = evaluate_yolo_model(
            model_or_weights=str(CKPT_PATH),
            split="val",
            imgsz=640,
            conf=best_conf,
            iou_nms=iou,
            device=0,
            model_name=f"yolov8s_nms{iou:.2f}"
        )
        results.append({
            "nms_iou": iou,
            "conf_threshold": best_conf,
            "val_mAP50": r["mAP50"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "small_ap": r["small_ap"],
            "medium_ap": r["medium_ap"],
            "large_ap": r["large_ap"],
            "mean_iou": r["mean_iou"],
            "latency_ms": r["latency_ms"],
            "fps": r["fps"]
        })
        print(f"IoU {iou:.2f}: mAP={r['mAP50']:.4f}, Prec={r['precision']:.4f}, Rec={r['recall']:.4f}, F1={r['f1']:.4f}, Small_AP={r['small_ap']:.4f}")

    df = pd.DataFrame(results)
    out_csv = OUT_DIR / "nms_threshold_sweep.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nNMS sweep saved to {out_csv}")
    best_mAP_row = df.loc[df['val_mAP50'].idxmax()]
    print(f"Optimal NMS IoU for mAP50: {best_mAP_row['nms_iou']:.2f} (mAP50 = {best_mAP_row['val_mAP50']:.4f})")

def main():
    best_conf = sweep_confidence()
    sweep_nms(best_conf)

if __name__ == "__main__":
    main()
