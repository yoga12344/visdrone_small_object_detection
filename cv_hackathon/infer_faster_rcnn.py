"""
infer_faster_rcnn.py
====================
Faster R-CNN (ResNet50-FPN) Aerial Inference Script.
Maps pretrained COCO detections to target VisDrone aerial categories.
Runs on CUDA GPU with proper NMS and confidence filtering.
"""

import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image

# VisDrone class labels
VISDRONE_CLASSES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

# COCO pretrained class IDs to VisDrone class IDs
COCO_TO_VISDRONE = {
    1: 0,   # person -> pedestrian
    2: 2,   # bicycle -> bicycle
    3: 3,   # car -> car
    4: 9,   # motorcycle -> motor
    6: 8,   # bus -> bus
    8: 5    # truck -> truck
}

def run_faster_rcnn(image_path, model, device, conf_thresh=0.5):
    img = Image.open(image_path).convert("RGB")
    tensor_img = F.to_tensor(img).to(device)

    t0 = time.time()
    with torch.no_grad():
        preds = model([tensor_img])[0]
    dt = time.time() - t0

    boxes = preds['boxes'].cpu().numpy()
    labels = preds['labels'].cpu().numpy()
    scores = preds['scores'].cpu().numpy()

    # Filter by confidence and map to VisDrone classes
    keep_boxes = []
    keep_labels = []
    keep_scores = []

    for b, l, s in zip(boxes, labels, scores):
        if s >= conf_thresh and l in COCO_TO_VISDRONE:
            keep_boxes.append(b)
            keep_labels.append(COCO_TO_VISDRONE[l])
            keep_scores.append(s)

    return keep_boxes, keep_labels, keep_scores, dt

def main():
    parser = argparse.ArgumentParser(description="Faster R-CNN VisDrone Inference")
    parser.add_argument("--source", required=True, help="Path to image or directory of images")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--output-dir", default="results/faster_rcnn_output", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading Faster R-CNN (ResNet50-FPN) on {device}...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval().to(device)

    source_path = Path(args.source)
    if source_path.is_file():
        image_files = [source_path]
    else:
        image_files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Faster R-CNN inference on {len(image_files)} image(s)...")
    for img_path in image_files:
        boxes, labels, scores, dt = run_faster_rcnn(img_path, model, device, conf_thresh=args.conf)
        cv_img = cv2.imread(str(img_path))

        for b, l, s in zip(boxes, labels, scores):
            x1, y1, x2, y2 = map(int, b)
            cname = VISDRONE_CLASSES[l]
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cv_img, f"{cname} {s:.2f}", (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out_img_path = out_dir / img_path.name
        cv2.imwrite(str(out_img_path), cv_img)
        print(f"Processed {img_path.name}: {len(boxes)} detections in {dt * 1000:.1f}ms -> {out_img_path}")

if __name__ == "__main__":
    main()
