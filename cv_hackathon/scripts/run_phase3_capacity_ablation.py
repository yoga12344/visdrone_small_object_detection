"""
run_phase3_capacity_ablation.py
Evaluate model capacity across YOLO scales: YOLOv8n, YOLOv8s (100ep), YOLOv8m, and YOLO11n on the validation set.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd
import torch
from research_evaluator import evaluate_yolo_model

def main():
    print("=" * 60)
    print("PHASE 3: MODEL CAPACITY ABLATION ON VALIDATION SET")
    print("=" * 60)
    
    models_to_eval = [
        {"name": "yolov8n", "path": "results/experiments/yolo_model_size/yolov8n/weights/best.pt", "family": "YOLOv8"},
        {"name": "yolov8s_100ep", "path": "results/final_optimization/training/yolov8s_100ep_640/weights/best.pt", "family": "YOLOv8"},
        {"name": "yolov8m", "path": "results/experiments/yolo_model_size/yolov8m/weights/best.pt", "family": "YOLOv8"},
        {"name": "yolo11n", "path": "results/experiments/yolo_model_size/yolo11n/weights/best.pt", "family": "YOLO11"},
    ]
    
    results = []
    
    for m in models_to_eval:
        p = Path(m["path"])
        assert p.exists(), f"Checkpoint {p} not found"
        print(f"\nEvaluating {m['name']} ({p})...")
        
        metrics = evaluate_yolo_model(
            model_or_weights=str(p),
            split="val",
            imgsz=640,
            conf=0.25,
            iou_nms=0.45,
            device=0,
            model_name=m["name"]
        )
        
        row = {
            "model": m["name"],
            "family": m["family"],
            "parameters": metrics["parameters"],
            "size_mb": metrics["model_size_mb"],
            "val_mAP50": metrics["mAP50"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "small_ap": metrics["small_ap"],
            "medium_ap": metrics["medium_ap"],
            "large_ap": metrics["large_ap"],
            "latency_ms": metrics["latency_ms"],
            "fps": metrics["fps"]
        }
        results.append(row)
        print(f"{m['name']}: mAP50 = {metrics['mAP50']:.4f}, Small AP = {metrics['small_ap']:.4f}, FPS = {metrics['fps']:.2f}")

    df = pd.DataFrame(results)
    out_csv = Path("results/final_optimization/capacity_ablation.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nCapacity ablation saved to {out_csv}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
