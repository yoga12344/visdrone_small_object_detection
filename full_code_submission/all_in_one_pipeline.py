"""
all_in_one_pipeline.py
======================
Comprehensive All-in-One Vision Pipeline for VisDrone Aerial Object Detection.
Contains complete implementations for:
  1. Dataset Format Conversion (VisDrone CSV -> YOLO normalized format)
  2. Model Training (YOLOv8s on full VisDrone with Windows-safe workers=0)
  3. Inference (Images & Video with bounding boxes & class confidence)
  4. AI-Assisted Auto-Annotation (Automatic YOLO .txt label generation)
  5. Scientific Evaluation (Correct P-R curve mAP@0.50, no pseudo-GT, optimal matching)

Author: CV Hackathon Team
Hardware: NVIDIA GeForce RTX 4060 GPU / CUDA 12.1 / PyTorch
"""

import os
import sys
import argparse
import time
import shutil
import zipfile
from pathlib import Path
import numpy as np
import cv2

# Ensure ultralytics is available
try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics is not installed. Run: pip install ultralytics")

# VisDrone 10 Target Classes
VISDRONE_CLASSES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

# VisDrone raw CSV category mapping (category 0 is ignored region, 1-10 are targets)
VISDRONE_CAT_TO_CLASS = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9   # motor
}

# COCO to VisDrone class mapping (for pretrained COCO backbones)
COCO_TO_VISDRONE = {
    0: 0,   # person -> pedestrian
    1: 2,   # bicycle -> bicycle
    2: 3,   # car -> car
    3: 9,   # motorcycle -> motor
    5: 8,   # bus -> bus
    7: 5    # truck -> truck
}


# ==============================================================================
# MODULE 1: DATASET CONVERTER
# ==============================================================================
def convert_visdrone_split(images_dir, annotations_dir, output_labels_dir):
    """
    Converts VisDrone CSV annotation text files into normalized YOLO format:
    <class_id> <x_center> <y_center> <width> <height>
    Skips ignored regions (category 0).
    """
    images_path = Path(images_dir)
    annotations_path = Path(annotations_dir)
    labels_out = Path(output_labels_dir)
    labels_out.mkdir(parents=True, exist_ok=True)

    ann_files = list(annotations_path.glob("*.txt"))
    print(f"Converting {len(ann_files)} annotation files from {annotations_dir}...")

    total_boxes = 0
    ignored_boxes = 0

    for ann_file in ann_files:
        stem = ann_file.stem
        img_candidates = list(images_path.glob(f"{stem}.*"))
        if not img_candidates:
            continue

        img_path = img_candidates[0]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        yolo_lines = []
        with open(ann_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    parts = line.strip().split()
                if len(parts) < 8:
                    continue

                try:
                    bx, by, bw, bh = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    cat_id = int(parts[5])
                except ValueError:
                    continue

                if cat_id not in VISDRONE_CAT_TO_CLASS:
                    ignored_boxes += 1
                    continue

                cls_id = VISDRONE_CAT_TO_CLASS[cat_id]

                # Convert to normalized xywh center
                xc = (bx + bw / 2.0) / w
                yc = (by + bh / 2.0) / h
                nw = bw / w
                nh = bh / h

                # Clamp values to [0, 1]
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))

                yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                total_boxes += 1

        out_file = labels_out / f"{stem}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

    print(f"Finished conversion. Valid boxes: {total_boxes:,}, Ignored boxes: {ignored_boxes:,}")


# ==============================================================================
# MODULE 2: MODEL TRAINING
# ==============================================================================
def train_yolov8(data_yaml, epochs=50, batch=8, imgsz=640, output_dir="runs/train_visdrone"):
    """
    Trains YOLOv8s on the VisDrone dataset using Windows-safe settings (workers=0).
    """
    print("=" * 65)
    print(" STARTING FULL YOLOV8S TRAINING ON VISDRONE")
    print(f" Config YAML   : {data_yaml}")
    print(f" Target Epochs : {epochs}")
    print(f" Batch Size    : {batch}")
    print(f" Image Size    : {imgsz}")
    print("=" * 65)

    model = YOLO("yolov8s.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=0,
        workers=0,      # Essential on Windows to avoid shared memory crashes
        amp=True,       # Mixed Precision for high throughput
        patience=15,
        project=output_dir,
        name="yolov8s_visdrone",
        exist_ok=True
    )
    print(f"Training complete. Weights saved to {output_dir}/yolov8s_visdrone/weights/best.pt")
    return results


# ==============================================================================
# MODULE 3: INFERENCE & VIDEO TRACKING
# ==============================================================================
def run_inference(weights, source, conf=0.25, save=True, output_dir="results/predictions"):
    """
    Runs high-speed inference on a single image, image directory, or MP4 video.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading trained model from {weights}...")
    model = YOLO(str(weights))

    t0 = time.time()
    results = model.predict(
        source=str(source),
        conf=conf,
        save=save,
        project=str(out_path),
        name="predict_run",
        exist_ok=True,
        device=0
    )
    elapsed = time.time() - t0
    num_items = len(results) if isinstance(results, list) else 1
    fps = num_items / max(elapsed, 1e-4)

    print(f"Inference complete: {num_items} frame(s) processed in {elapsed:.2f}s ({fps:.1f} FPS)")
    print(f"Visual detections saved in: {out_path / 'predict_run'}")
    return results


# ==============================================================================
# MODULE 4: AI-ASSISTED AUTO-ANNOTATION
# ==============================================================================
def auto_annotate(weights, source, conf=0.25, output_dir="results/auto_annotated", make_zip=True):
    """
    Generates standardized YOLO format .txt annotations and preview images for raw unlabeled data.
    """
    out_dir = Path(output_dir).resolve()
    labels_out = out_dir / "labels"
    images_out = out_dir / "images"
    visual_out = out_dir / "visual_previews"

    labels_out.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)
    visual_out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    print(f"Auto-annotating source {source} using {weights} at conf={conf}...")

    results = model.predict(
        source=str(source),
        conf=conf,
        save_txt=True,
        save_conf=True,
        save=True,
        project=str(out_dir / "ultra_run"),
        name="predict",
        exist_ok=True,
        device=0
    )

    class_counts = {name: 0 for name in VISDRONE_CLASSES}
    total_boxes = 0

    ultra_labels = out_dir / "ultra_run" / "predict" / "labels"
    ultra_visuals = out_dir / "ultra_run" / "predict"

    for r in results:
        p = Path(r.path)
        dest_img = images_out / p.name
        if p.is_file() and not dest_img.exists():
            shutil.copy2(p, dest_img)

        if r.boxes is not None and len(r.boxes) > 0:
            for c in r.boxes.cls.cpu().numpy():
                idx = int(c)
                if 0 <= idx < len(VISDRONE_CLASSES):
                    class_counts[VISDRONE_CLASSES[idx]] += 1
                    total_boxes += 1

        src_txt = ultra_labels / f"{p.stem}.txt"
        dest_txt = labels_out / f"{p.stem}.txt"
        if src_txt.exists():
            shutil.copy2(src_txt, dest_txt)
        else:
            dest_txt.touch()

        src_vis = ultra_visuals / p.name
        if src_vis.exists():
            shutil.copy2(src_vis, visual_out / p.name)

    # Write classes.txt and data.yaml
    with open(out_dir / "classes.txt", "w") as f:
        for c in VISDRONE_CLASSES:
            f.write(f"{c}\n")

    yaml_text = f"names:\n" + "\n".join([f"  {i}: {c}" for i, c in enumerate(VISDRONE_CLASSES)]) + "\nnc: 10\ntrain: ./images\nval: ./images\n"
    with open(out_dir / "data.yaml", "w") as f:
        f.write(yaml_text)

    print("-" * 45)
    print(f"Auto-annotation finished: {len(results)} images, {total_boxes} boxes.")
    for k, v in class_counts.items():
        if v > 0:
            print(f"  {k:<16}: {v:>5}")
    print("-" * 45)

    if make_zip:
        zip_file = out_dir / "roboflow_annotated_package.zip"
        print(f"Creating Roboflow upload package: {zip_file}")
        with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_dir / "data.yaml", arcname="data.yaml")
            zf.write(out_dir / "classes.txt", arcname="classes.txt")
            for img in images_out.glob("*.*"):
                zf.write(img, arcname=f"images/{img.name}")
            for txt in labels_out.glob("*.txt"):
                zf.write(txt, arcname=f"labels/{txt.name}")
        print("Roboflow ZIP package created successfully.")


# ==============================================================================
# MODULE 5: EVALUATION ENGINE (PR-Curve mAP@0.50)
# ==============================================================================
def compute_box_iou(box1, box2):
    """Computes IoU between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_ap_from_pr(recalls, precisions):
    """Computes Area Under the Precision-Recall Curve (AUC) via 101-point interpolation."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[ii + 1] - mrec[ii]) * mpre[ii + 1])


def evaluate_model_on_split(weights, data_yaml, split="val"):
    """
    Evaluates trained weights on real ground truth using Ultralytics standard validation protocol.
    Reports precision, recall, mAP@0.50, and mAP@0.5:0.95.
    """
    print(f"Evaluating {weights} on {data_yaml} (split: {split})...")
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        device=0,
        workers=0
    )

    print("\n" + "=" * 60)
    print(" OFFICIAL EVALUATION RESULTS")
    print("=" * 60)
    print(f" Overall Precision  : {metrics.box.mp * 100:.2f}%")
    print(f" Overall Recall     : {metrics.box.mr * 100:.2f}%")
    print(f" Overall mAP @ 0.50 : {metrics.box.map50 * 100:.2f}%")
    print(f" Overall mAP @ 50:95: {metrics.box.map * 100:.2f}%")
    print("=" * 60)
    return metrics


# ==============================================================================
# CLI DISPATCHER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="VisDrone All-in-One Vision Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # Sub-command: convert
    p_conv = subparsers.add_parser("convert", help="Convert VisDrone CSV annotations to YOLO format")
    p_conv.add_argument("--images-dir", required=True, help="Directory containing images")
    p_conv.add_argument("--annotations-dir", required=True, help="Directory containing CSV annotations")
    p_conv.add_argument("--output-labels-dir", required=True, help="Directory to save YOLO labels")

    # Sub-command: train
    p_train = subparsers.add_parser("train", help="Train YOLOv8s on full VisDrone dataset")
    p_train.add_argument("--data", default="configs/visdrone.yaml", help="Path to visdrone.yaml")
    p_train.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    p_train.add_argument("--batch", type=int, default=8, help="Batch size")
    p_train.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    p_train.add_argument("--output-dir", default="runs/train_visdrone", help="Output directory")

    # Sub-command: predict
    p_pred = subparsers.add_parser("predict", help="Run inference on image or video")
    p_pred.add_argument("--weights", default="weights/best.pt", help="Path to trained weights")
    p_pred.add_argument("--source", required=True, help="Path to image, folder, or video")
    p_pred.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p_pred.add_argument("--output-dir", default="results/predictions", help="Output directory")

    # Sub-command: auto-annotate
    p_ann = subparsers.add_parser("auto-annotate", help="Auto-annotate unlabeled images")
    p_ann.add_argument("--weights", default="weights/best.pt", help="Path to trained weights")
    p_ann.add_argument("--source", required=True, help="Path to raw unlabeled images")
    p_ann.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p_ann.add_argument("--output-dir", default="results/auto_annotated", help="Output directory")
    p_ann.add_argument("--zip", action="store_true", default=True, help="Create Roboflow zip package")

    # Sub-command: evaluate
    p_eval = subparsers.add_parser("evaluate", help="Benchmark model on test/val set")
    p_eval.add_argument("--weights", default="weights/best.pt", help="Path to model weights")
    p_eval.add_argument("--data", default="configs/visdrone.yaml", help="Path to visdrone.yaml")
    p_eval.add_argument("--split", default="val", help="Dataset split to evaluate on")

    args = parser.parse_args()

    if args.command == "convert":
        convert_visdrone_split(args.images_dir, args.annotations_dir, args.output_labels_dir)
    elif args.command == "train":
        train_yolov8(args.data, args.epochs, args.batch, args.imgsz, args.output_dir)
    elif args.command == "predict":
        run_inference(args.weights, args.source, args.conf, save=True, output_dir=args.output_dir)
    elif args.command == "auto-annotate":
        auto_annotate(args.weights, args.source, args.conf, args.output_dir, make_zip=args.zip)
    elif args.command == "evaluate":
        evaluate_model_on_split(args.weights, args.data, args.split)
    else:
        print("VisDrone Pipeline CLI: Specify a command: convert, train, predict, auto-annotate, evaluate.")
        print("Example: python all_in_one_pipeline.py predict --weights weights/best.pt --source ../img1.jpg")
        parser.print_help()


if __name__ == "__main__":
    main()
