"""
infer_mask_rcnn.py
==================
Mask R-CNN (ResNet50-FPN) Aerial Instance Segmentation Script.
Generates segmentation masks and bounding boxes for aerial imagery.
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

VISDRONE_CLASSES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

COCO_TO_VISDRONE = {
    1: 0,   # person -> pedestrian
    2: 2,   # bicycle -> bicycle
    3: 3,   # car -> car
    4: 9,   # motorcycle -> motor
    6: 8,   # bus -> bus
    8: 5    # truck -> truck
}

def main():
    parser = argparse.ArgumentParser(description="Mask R-CNN VisDrone Instance Segmentation")
    parser.add_argument("--source", required=True, help="Path to image or image directory")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--output-dir", default="results/mask_rcnn_output", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading Mask R-CNN (ResNet50-FPN) on {device}...")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval().to(device)

    source_path = Path(args.source)
    if source_path.is_file():
        image_files = [source_path]
    else:
        image_files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_files:
        pil_img = Image.open(img_path).convert("RGB")
        tensor_img = F.to_tensor(pil_img).to(device)

        t0 = time.time()
        with torch.no_grad():
            preds = model([tensor_img])[0]
        dt = time.time() - t0

        cv_img = cv2.imread(str(img_path))
        boxes = preds['boxes'].cpu().numpy()
        labels = preds['labels'].cpu().numpy()
        scores = preds['scores'].cpu().numpy()
        masks = preds['masks'].cpu().numpy()

        det_count = 0
        for b, l, s, m in zip(boxes, labels, scores, masks):
            if s >= args.conf and l in COCO_TO_VISDRONE:
                det_count += 1
                cname = VISDRONE_CLASSES[COCO_TO_VISDRONE[l]]
                x1, y1, x2, y2 = map(int, b)

                # Overlay segmentation mask
                mask = m[0] > 0.5
                color = np.array([0, 255, 255], dtype=np.uint8)
                cv_img[mask] = cv2.addWeighted(cv_img[mask], 0.5, color, 0.5, 0)

                # Draw bounding box
                cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(cv_img, f"{cname} {s:.2f}", (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

        out_file = out_dir / img_path.name
        cv2.imwrite(str(out_file), cv_img)
        print(f"Processed {img_path.name}: {det_count} instances in {dt * 1000:.1f}ms -> {out_file}")

if __name__ == "__main__":
    main()
