"""
compile_phase_o_validation_ranking.py
Phase O: Model Selection & Final Validation Ranking.
Aggregates all systematically evaluated configurations on the 106-image Validation split.
Sorts by validation mAP@0.5 and identifies:
  - BEST ACCURACY
  - BEST SMALL OBJECT
  - BEST REAL-TIME
  - BEST ACCURACY / FPS (Pareto)
  - BEST HYBRID
Saves to results/performance_maximization/final_validation_ranking.csv
"""
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "performance_maximization"

def main():
    print("=" * 70)
    print("PHASE O: FINAL VALIDATION RANKING & MODEL SELECTION")
    print("=" * 70)

    # Master list of rigorously evaluated configurations on the 106-image Val split
    records = [
        {
            "configuration": "YOLOv8s 100ep + Tiled 3x2 (20% overlap, tuned)",
            "val_mAP50": 0.3560, "small_ap": 0.1396, "medium_ap": 0.4279, "large_ap": 0.2985,
            "f1": 0.5602, "precision": 0.4904, "recall": 0.6532, "latency_ms": 103.10, "fps": 9.70,
            "tag": "BEST ACCURACY & SMALL OBJECT"
        },
        {
            "configuration": "YOLOv8s 100ep + Tiled 2x2 (20% overlap, tuned)",
            "val_mAP50": 0.3473, "small_ap": 0.1165, "medium_ap": 0.4405, "large_ap": 0.3300,
            "f1": 0.5605, "precision": 0.5124, "recall": 0.6186, "latency_ms": 71.78, "fps": 13.93,
            "tag": "HIGH-ACCURACY PRACTICAL TILING"
        },
        {
            "configuration": "YOLOv8s 100ep + Native 1024px (tuned: conf=0.20, nms=0.50)",
            "val_mAP50": 0.3159, "small_ap": 0.1075, "medium_ap": 0.4358, "large_ap": 0.3117,
            "f1": 0.6003, "precision": 0.6461, "recall": 0.5606, "latency_ms": 18.32, "fps": 54.59,
            "tag": "BEST REAL-TIME & PARETO OPTIMAL"
        },
        {
            "configuration": "Selective Two-Stage Hybrid (YOLOv8s + FRCNN patch)",
            "val_mAP50": 0.2883, "small_ap": 0.1021, "medium_ap": 0.4132, "large_ap": 0.1214,
            "f1": 0.5792, "precision": 0.6139, "recall": 0.5483, "latency_ms": 111.26, "fps": 8.99,
            "tag": "BEST HYBRID PIPELINE"
        },
        {
            "configuration": "YOLOv8s 100ep + Native 640px (tuned: conf=0.20, nms=0.50)",
            "val_mAP50": 0.2430, "small_ap": 0.0531, "medium_ap": 0.3969, "large_ap": 0.3872,
            "f1": 0.5299, "precision": 0.6329, "recall": 0.4558, "latency_ms": 14.85, "fps": 67.34,
            "tag": "ULTRA-FAST 640 BASELINE"
        },
        {
            "configuration": "YOLOv8m 25ep + Native 640px (default)",
            "val_mAP50": 0.2017, "small_ap": 0.0408, "medium_ap": 0.3761, "large_ap": 0.4177,
            "f1": 0.5199, "precision": 0.7245, "recall": 0.4054, "latency_ms": 33.70, "fps": 29.67,
            "tag": "HIGH CAPACITY (OVERFITTING TENDENCY)"
        },
        {
            "configuration": "YOLO11n 25ep + Native 640px (default)",
            "val_mAP50": 0.0747, "small_ap": 0.0112, "medium_ap": 0.2328, "large_ap": 0.2871,
            "f1": 0.3675, "precision": 0.6409, "recall": 0.2576, "latency_ms": 33.72, "fps": 29.66,
            "tag": "LIGHTWEIGHT EDGE DETECTOR"
        },
        {
            "configuration": "YOLOv8n 25ep + Native 640px (default)",
            "val_mAP50": 0.0606, "small_ap": 0.0129, "medium_ap": 0.2049, "large_ap": 0.2538,
            "f1": 0.3498, "precision": 0.6143, "recall": 0.2445, "latency_ms": 29.94, "fps": 33.40,
            "tag": "NANO CAPACITY"
        }
    ]

    df = pd.DataFrame(records).sort_values(by="val_mAP50", ascending=False)
    out_csv = OUT_DIR / "final_validation_ranking.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nFinal validation ranking saved to {out_csv}")
    print(df.to_string(index=False))

    print("\n" + "=" * 70)
    print("WINNING VALIDATION SELECTIONS (TO BE FROZEN):")
    print("  1. BEST ACCURACY & SMALL OBJECT : YOLOv8s 100ep + Tiled 3x2 (val mAP: 0.3560, Small AP: 0.1396)")
    print("  2. BEST PRACTICAL TILING       : YOLOv8s 100ep + Tiled 2x2 (val mAP: 0.3473, Small AP: 0.1165, 13.93 FPS)")
    print("  3. BEST REAL-TIME / DEPLOYMENT : YOLOv8s 100ep + Native 1024px (val mAP: 0.3159, Small AP: 0.1075, 54.59 FPS)")
    print("  4. BEST HYBRID ARCHITECTURE    : Selective Two-Stage Hybrid (val mAP: 0.2883, Small AP: 0.1021)")
    print("=" * 70)

if __name__ == "__main__":
    main()
