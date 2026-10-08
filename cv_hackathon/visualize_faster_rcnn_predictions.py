"""
visualize_faster_rcnn_predictions.py
====================================
Generates visual qualitative comparison panels between Ground Truth annotations
and Fine-Tuned Faster R-CNN predictions on the locked 20 test images.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "data"))

import visdrone_parser

CLASS_NAMES = visdrone_parser.CLASS_NAMES

# Color palette for classes (0..9)
COLORS = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB"
]

def hex_to_rgb(hex_str):
    h = hex_str.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights_path = PROJECT_ROOT / "results" / "trained" / "faster_rcnn" / "weights" / "best.pt"
    out_dir = PROJECT_ROOT / "results" / "trained" / "faster_rcnn" / "qualitative"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 11)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    # Load test images & GT
    raw_val_dir = PROJECT_ROOT / "data" / "raw" / "VisDrone2019-DET-val"
    imgs_dir = raw_val_dir / "images"
    anns_dir = raw_val_dir / "annotations"
    test_ids_file = PROJECT_ROOT / "data" / "processed" / "splits" / "test_ids.txt"
    test_ids = [l.strip() for l in open(test_ids_file, encoding="utf-8") if l.strip()]

    all_gts, _, _ = visdrone_parser.parse_visdrone_dataset(str(imgs_dir), str(anns_dir))
    
    # Render first 6 test images as comparison side-by-sides
    for tid in test_ids[:6]:
        img_p = imgs_dir / f"{tid}.jpg"
        img_orig = Image.open(img_p).convert("RGB")
        W, H = img_orig.size

        # 1. Ground Truth Image
        img_gt = img_orig.copy()
        draw_gt = ImageDraw.Draw(img_gt)
        cur_gts = [g for g in all_gts if g["image_id"] == tid and 0 <= g["class_id"] <= 9]
        for g in cur_gts:
            b = g["box_xyxy"]
            cid = g["class_id"]
            cname = CLASS_NAMES[cid]
            color = hex_to_rgb(COLORS[cid % len(COLORS)])
            draw_gt.rectangle([b[0], b[1], b[2], b[3]], outline=color, width=2)
            draw_gt.text((b[0], max(0, b[1] - 10)), cname, fill=color)

        # 2. Predicted Image
        img_pred = img_orig.copy()
        draw_pred = ImageDraw.Draw(img_pred)
        img_t = torchvision.transforms.functional.to_tensor(img_orig).to(device)
        with torch.no_grad():
            outputs = model([img_t])[0]

        boxes = outputs["boxes"].detach().cpu().numpy()
        scores = outputs["scores"].detach().cpu().numpy()
        labels = outputs["labels"].detach().cpu().numpy()

        for b, s, l in zip(boxes, scores, labels):
            if s >= 0.25 and 1 <= l <= 10:
                cid = int(l - 1)
                cname = f"{CLASS_NAMES[cid]} {s:.2f}"
                color = hex_to_rgb(COLORS[cid % len(COLORS)])
                draw_pred.rectangle([b[0], b[1], b[2], b[3]], outline=color, width=2)
                draw_pred.text((b[0], max(0, b[1] - 10)), cname, fill=color)

        # Combine side-by-side
        combined = Image.new("RGB", (W * 2, H + 40), (240, 240, 240))
        combined.paste(img_gt, (0, 40))
        combined.paste(img_pred, (W, 40))

        draw_title = ImageDraw.Draw(combined)
        draw_title.text((20, 10), f"Ground Truth ({len(cur_gts)} objects) - {tid}", fill=(0, 0, 0))
        draw_title.text((W + 20, 10), f"Fine-Tuned Faster R-CNN Predictions (conf >= 0.25)", fill=(0, 0, 0))

        save_p = out_dir / f"compare_{tid}.jpg"
        combined.save(save_p, quality=90)

    print(f"Qualitative panels saved to {out_dir}")

if __name__ == "__main__":
    main()
