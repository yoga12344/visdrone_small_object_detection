"""
auto_annotate.py
================
Automated annotation tool using the newly trained VisDrone YOLOv8s model.
Generates YOLO format .txt annotations, visual overlay images, and Roboflow-ready packages.

Usage:
  python scripts/auto_annotate.py --source "path/to/images_or_video" --conf 0.25 --save-visual
"""

import os
import sys
import argparse
import shutil
import zipfile
from pathlib import Path
from ultralytics import YOLO

VISDRONE_CLASSES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

def parse_args():
    parser = argparse.ArgumentParser(description="Auto-annotate images or videos using trained VisDrone YOLOv8s")
    parser.add_argument(
        "--weights",
        type=str,
        default="results/full_visdrone_training/yolov8s_full_6471imgs/weights/best.pt",
        help="Path to trained YOLOv8 weights"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to directory of images, or path to a video file"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions (default: 0.25)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU NMS threshold (default: 0.45)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution (default: 640)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/auto_annotated",
        help="Directory where annotations and images will be saved"
    )
    parser.add_argument(
        "--save-visual",
        action="store_true",
        help="Save visual preview images with bounding box overlays"
    )
    parser.add_argument(
        "--create-roboflow-zip",
        action="store_true",
        help="Bundle images, labels, and data.yaml into a Roboflow-ready zip archive"
    )
    return parser.parse_args()

def run_auto_annotation():
    args = parse_args()

    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        print(f"Error: Weights file not found at {weights_path}")
        sys.exit(1)

    source_path = Path(args.source).resolve()
    if not source_path.exists():
        print(f"Error: Source path not found at {source_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir).resolve()
    labels_out_dir = out_dir / "labels"
    images_out_dir = out_dir / "images"
    visual_out_dir = out_dir / "visual_previews"

    labels_out_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_visual:
        visual_out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" VISDRONE AI-ASSISTED AUTO-ANNOTATION ENGINE")
    print("=" * 65)
    print(f" Model Weights : {weights_path}")
    print(f" Source Data   : {source_path}")
    print(f" Confidence    : {args.conf}")
    print(f" Output Folder : {out_dir}")
    print("=" * 65)

    model = YOLO(str(weights_path))

    # Run inference with txt saving
    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save_txt=True,
        save_conf=True,
        save=args.save_visual,
        project=str(out_dir / "ultralytics_run"),
        name="predict",
        exist_ok=True,
        device=0
    )

    # Summarize findings
    class_counts = {name: 0 for name in VISDRONE_CLASSES}
    total_boxes = 0
    total_images = len(results)

    ultra_labels_dir = out_dir / "ultralytics_run" / "predict" / "labels"
    ultra_visual_dir = out_dir / "ultralytics_run" / "predict"

    for r in results:
        img_name = Path(r.path).name
        stem = Path(r.path).stem

        # Copy original image to images_out_dir
        dest_img = images_out_dir / img_name
        if not dest_img.exists() and Path(r.path).is_file():
            shutil.copy2(r.path, dest_img)

        # Count detected classes
        if r.boxes is not None and len(r.boxes) > 0:
            for c in r.boxes.cls.cpu().numpy():
                cls_idx = int(c)
                if 0 <= cls_idx < len(VISDRONE_CLASSES):
                    class_counts[VISDRONE_CLASSES[cls_idx]] += 1
                    total_boxes += 1

        # Copy generated txt label
        src_txt = ultra_labels_dir / f"{stem}.txt"
        dest_txt = labels_out_dir / f"{stem}.txt"
        if src_txt.exists():
            shutil.copy2(src_txt, dest_txt)
        else:
            # Create empty text file for background / no-detection images
            dest_txt.touch()

        # Copy visual preview if requested
        if args.save_visual:
            src_vis = ultra_visual_dir / img_name
            if src_vis.exists():
                shutil.copy2(src_vis, visual_out_dir / img_name)

    # Write data.yaml and classes.txt for standard export
    classes_txt_path = out_dir / "classes.txt"
    with open(classes_txt_path, "w") as f:
        for c in VISDRONE_CLASSES:
            f.write(f"{c}\n")

    yaml_content = f"""names:
{chr(10).join([f'  {i}: {c}' for i, c in enumerate(VISDRONE_CLASSES)])}
nc: 10
train: ./images
val: ./images
"""
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print("\nAnnotation Summary:")
    print(f"Total Images Processed : {total_images}")
    print(f"Total Bounding Boxes    : {total_boxes}")
    print("-" * 40)
    for cname, count in class_counts.items():
        print(f"  {cname:<16} : {count:>5}")
    print("-" * 40)
    print(f"Annotations saved to   : {labels_out_dir}")
    print(f"Classes config saved to: {classes_txt_path}")

    if args.create_roboflow_zip:
        zip_path = out_dir / "auto_annotated_roboflow.zip"
        print(f"\nPackaging into Roboflow ZIP: {zip_path}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(yaml_path, arcname="data.yaml")
            zf.write(classes_txt_path, arcname="classes.txt")
            for img_file in images_out_dir.glob("*.*"):
                zf.write(img_file, arcname=f"images/{img_file.name}")
            for txt_file in labels_out_dir.glob("*.txt"):
                zf.write(txt_file, arcname=f"labels/{txt_file.name}")
        print(f"Roboflow ZIP created successfully ({zip_path.stat().st_size / 1e6:.2f} MB)")

    print("\nDone! All annotations are ready.")

if __name__ == "__main__":
    run_auto_annotation()
