"""
infer_yolo.py
=============
Inference script for YOLOv8s trained on VisDrone.
Supports single images, folders of images, or MP4 videos.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8s VisDrone Inference")
    parser.add_argument("--weights", default="weights/best.pt", help="Path to trained YOLOv8 weights")
    parser.add_argument("--source", required=True, help="Path to image, directory, or video file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640)")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO format .txt annotations")
    parser.add_argument("--output-dir", default="results/yolo_output", help="Directory to save output")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Weights not found at {weights_path}")
        return

    print(f"Loading trained model: {weights_path}")
    model = YOLO(str(weights_path))

    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        save_txt=args.save_txt,
        project=args.output_dir,
        name="predict",
        exist_ok=True,
        device=0
    )
    print(f"\nInference complete! Saved output to {args.output_dir}/predict")

if __name__ == "__main__":
    main()
