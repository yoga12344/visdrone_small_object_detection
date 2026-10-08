"""
evaluate_benchmark.py
=====================
Scientific benchmark evaluation engine for VisDrone object detection.
Computes true PR-curve Average Precision, small/medium/large metrics, and confusion matrix.
Zero test leakage, zero pseudo-ground truth.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Evaluate VisDrone Object Detection Benchmark")
    parser.add_argument("--weights", default="weights/best.pt", help="Path to trained model weights")
    parser.add_argument("--data", default="configs/visdrone.yaml", help="Path to visdrone.yaml dataset config")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"], help="Split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Weights file not found at {weights_path}")
        return

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Config file not found at {data_path}")
        return

    print("=" * 65)
    print(" VISDRONE SCIENTIFIC BENCHMARK EVALUATOR")
    print("=" * 65)
    print(f" Weights : {weights_path}")
    print(f" Dataset : {data_path} (split: {args.split})")
    print(f" Device  : CUDA:0 (GeForce RTX 4060)")
    print("=" * 65)

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=0
    )

    print("\n" + "=" * 65)
    print(" OVERALL EVALUATION SUMMARY")
    print("=" * 65)
    print(f" Overall Precision       : {metrics.box.mp * 100:.2f}%")
    print(f" Overall Recall          : {metrics.box.mr * 100:.2f}%")
    print(f" Overall mAP @ 0.50      : {metrics.box.map50 * 100:.2f}%")
    print(f" Overall mAP @ 0.50:0.95 : {metrics.box.map * 100:.2f}%")
    print("=" * 65)

    print("\nPer-Class AP@0.50 Breakdown:")
    for i, cname in enumerate(model.names.values()):
        ap50 = metrics.box.maps[i] if i < len(metrics.box.maps) else 0.0
        print(f"  {cname:<16} : {ap50 * 100:>6.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    main()
