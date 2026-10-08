"""
generate_trained_qualitative.py
===============================
Generates side-by-side qualitative comparisons for representative scenes
on the LOCKED test set:
  Left: Authentic VisDrone Ground Truth
  Right: VisDrone-Fine-Tuned YOLOv8s (best.pt)
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "data"))

import visdrone_parser
from ultralytics import YOLO

def draw_boxes(image: Image.Image, boxes, labels, color="green"):
    draw = ImageDraw.Draw(image)
    for b, l in zip(boxes, labels):
        draw.rectangle([b[0], b[1], b[2], b[3]], outline=color, width=2)
        draw.text((b[0], max(0, b[1] - 10)), l, fill=color)
    return image

def main():
    test_ids_file = PROJECT_ROOT / "data" / "processed" / "splits" / "test_ids.txt"
    test_ids = [line.strip() for line in open(test_ids_file, encoding="utf-8") if line.strip()]

    raw_val_dir = PROJECT_ROOT / "data" / "raw" / "VisDrone2019-DET-val"
    imgs_dir = raw_val_dir / "images"
    anns_dir = raw_val_dir / "annotations"

    out_dir = PROJECT_ROOT / "results" / "trained" / "yolov8s" / "qualitative"
    os.makedirs(out_dir, exist_ok=True)

    all_gts, _, _ = visdrone_parser.parse_visdrone_dataset(str(imgs_dir), str(anns_dir))

    model_path = PROJECT_ROOT / "results" / "trained" / "yolov8s" / "weights" / "best.pt"
    model = YOLO(str(model_path))

    scenes = {
        "01_dense_traffic": test_ids[0],
        "02_many_small_objects": test_ids[1],
        "03_crowded_perspective": test_ids[4],
        "04_complex_intersection": test_ids[15],
    }

    thumb_w, thumb_h = 800, 600

    for scene_label, tid in scenes.items():
        img_path = str(imgs_dir / f"{tid}.jpg")
        raw_img = Image.open(img_path).convert("RGB")
        W, H = raw_img.size

        # GT boxes
        gt_boxes = [g["box_xyxy"] for g in all_gts if g["image_id"] == tid and 0 <= g["class_id"] <= 9]
        gt_labels = [g["class_name"] for g in all_gts if g["image_id"] == tid and 0 <= g["class_id"] <= 9]

        img_gt = draw_boxes(raw_img.copy(), gt_boxes, gt_labels, color="#00ff00")

        # Fine-tuned YOLO inference
        preds = model(img_path, conf=0.25, imgsz=640, device=0, verbose=False)[0]
        pred_boxes = []
        pred_labels = []
        if preds.boxes is not None and len(preds.boxes) > 0:
            for b, c, s in zip(preds.boxes.xyxy.cpu().numpy(), preds.boxes.cls.cpu().numpy().astype(int), preds.boxes.conf.cpu().numpy()):
                if 0 <= c <= 9:
                    pred_boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                    pred_labels.append(f"{visdrone_parser.CLASS_NAMES[c]} {s:.2f}")

        img_pred = draw_boxes(raw_img.copy(), pred_boxes, pred_labels, color="#00e5ff")

        # Side-by-side panel
        panel = Image.new("RGB", (thumb_w * 2, thumb_h), color="black")
        panel.paste(img_gt.resize((thumb_w, thumb_h)), (0, 0))
        panel.paste(img_pred.resize((thumb_w, thumb_h)), (thumb_w, 0))

        draw_panel = ImageDraw.Draw(panel)
        draw_panel.text((20, 20), "Authentic VisDrone Ground Truth (Green)", fill="white")
        draw_panel.text((thumb_w + 20, 20), "VisDrone-Fine-Tuned YOLOv8s (Cyan)", fill="white")

        save_p = out_dir / f"comparison_{scene_label}.png"
        panel.save(save_p)
        print(f"Saved qualitative panel: {save_p}")

if __name__ == "__main__":
    main()
