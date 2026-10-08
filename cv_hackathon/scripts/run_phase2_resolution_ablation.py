"""
run_phase2_resolution_ablation.py
Evaluate the trained YOLOv8s 100-epoch model on the 106-image validation set across 640, 768, and 1024 resolutions.
"""
import sys
from pathlib import Path

# Add project root and scripts dir to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd
import torch
from research_evaluator import evaluate_yolo_model

def main():
    print("=" * 60)
    print("PHASE 2: RESOLUTION ABLATION ON VALIDATION SET")
    print("=" * 60)
    
    ckpt_path = Path("results/final_optimization/training/yolov8s_100ep_640/weights/best.pt")
    assert ckpt_path.exists(), f"Checkpoint {ckpt_path} not found"
    
    resolutions = [640, 768, 1024]
    results = []
    
    for res in resolutions:
        print(f"\nEvaluating resolution: {res}x{res}...")
        
        metrics = evaluate_yolo_model(
            model_or_weights=str(ckpt_path),
            split="val",
            imgsz=res,
            conf=0.25,
            iou_nms=0.45,
            device=0,
            model_name=f"yolov8s_res{res}"
        )
        
        row = {
            "resolution": res,
            "imgsz": f"{res}x{res}",
            "val_mAP50": metrics["mAP50"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "small_ap": metrics["small_ap"],
            "medium_ap": metrics["medium_ap"],
            "large_ap": metrics["large_ap"],
            "mean_iou": metrics["mean_iou"],
            "latency_ms": metrics["latency_ms"],
            "fps": metrics["fps"]
        }
        results.append(row)
        print(f"Res {res}: mAP50 = {metrics['mAP50']:.4f}, Small AP = {metrics['small_ap']:.4f}, FPS = {metrics['fps']:.2f}")

    df = pd.DataFrame(results)
    out_csv = Path("results/final_optimization/resolution_ablation.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResolution ablation saved to {out_csv}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
