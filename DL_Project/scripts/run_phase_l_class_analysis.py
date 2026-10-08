"""
run_phase_l_class_analysis.py
Phase L: Class-Specific AP and Error Taxonomy on the 106-image Validation set.
Analyzes:
  - Per-class AP for all 10 VisDrone categories
  - Easiest vs Hardest classes
  - Category count distribution and recall failure points
Saves to results/performance_maximization/class_specific_analysis.csv
"""
import sys
from pathlib import Path
import pandas as pd
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "data"))

from research_evaluator import evaluate_yolo_model
import visdrone_parser

OUT_DIR = PROJECT_ROOT / "results" / "performance_maximization"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = PROJECT_ROOT / "results" / "final_optimization" / "training" / "yolov8s_100ep_640" / "weights" / "best.pt"

def main():
    print("=" * 60)
    print("PHASE L: CLASS-SPECIFIC PERFORMANCE ANALYSIS (VAL SET)")
    print("=" * 60)
    
    r = evaluate_yolo_model(
        str(CKPT_PATH),
        split="val",
        imgsz=1024,
        conf=0.20,
        iou_nms=0.50,
        device=0,
        model_name="YOLOv8s_1024_Tuned"
    )

    per_class_ap = r["per_class_ap"]
    rows = []
    for cid, cname in enumerate(visdrone_parser.CLASS_NAMES):
        ap = per_class_ap.get(cname, 0.0)
        rows.append({
            "class_id": cid,
            "class_name": cname,
            "AP50": round(float(ap), 4)
        })

    df = pd.DataFrame(rows).sort_values(by="AP50", ascending=False)
    out_csv = OUT_DIR / "class_specific_analysis.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nClass analysis saved to {out_csv}")
    print(df.to_string(index=False))

    print("\nEasiest Class:", df.iloc[0]["class_name"], f"(AP50 = {df.iloc[0]['AP50']:.4f})")
    print("Hardest Class:", df.iloc[-1]["class_name"], f"(AP50 = {df.iloc[-1]['AP50']:.4f})")

if __name__ == "__main__":
    main()
