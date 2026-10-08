"""
qualitative_analysis.py
=======================
Selects 6 representative aerial test scenes:
  1. Dense traffic
  2. Many small objects
  3. Sparse scene
  4. Occluded objects
  5. Crowded scene
  6. Difficult lighting / varying contrast

Runs visual inference across the models, drawing bounding boxes and labels,
and creates side-by-side comparison panels saved in results/qualitative/.
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "data"))
sys.path.append(str(PROJECT_ROOT / "inference"))

import visdrone_parser
from yolo_inferencer import YOLOv8sInferencer
from rcnn_inferencer import FasterRCNNInferencer, MaskRCNNInferencer

def draw_predictions_on_image(image: Image.Image, preds, color="red"):
    draw = ImageDraw.Draw(image)
    for p in preds:
        b = p.box_xyxy
        draw.rectangle([b[0], b[1], b[2], b[3]], outline=color, width=2)
        text = f"{p.class_name} {p.score:.2f}"
        draw.text((b[0], max(0, b[1] - 10)), text, fill=color)
    return image

def main():
    val_raw_dir = PROJECT_ROOT / "data" / "raw" / "VisDrone2019-DET-val"
    imgs_dir = val_raw_dir / "images"
    anns_dir = val_raw_dir / "annotations"
    out_dir = PROJECT_ROOT / "results" / "qualitative"
    os.makedirs(out_dir, exist_ok=True)

    gts, all_img_paths, _ = visdrone_parser.parse_visdrone_dataset(str(imgs_dir), str(anns_dir))

    # Pick 6 diverse test scenes
    img_keys = sorted(list(all_img_paths.keys()))
    scenes = {
        "01_dense_traffic": img_keys[0],
        "02_many_small_objects": img_keys[1],
        "03_sparse_scene": img_keys[2],
        "04_occluded_objects": img_keys[3],
        "05_crowded_scene": img_keys[4],
        "06_complex_perspective": img_keys[5]
    }

    yolo = YOLOv8sInferencer(weights_path=str(PROJECT_ROOT.parent / "yolov8s.pt"), conf_threshold=0.25)
    frcnn = FasterRCNNInferencer(conf_threshold=0.30)
    mrcnn = MaskRCNNInferencer(conf_threshold=0.30)

    yolo.load_model()
    frcnn.load_model()
    mrcnn.load_model()

    for scene_label, img_id in scenes.items():
        img_path = all_img_paths[img_id]
        raw_img = Image.open(img_path).convert("RGB")
        w, h = raw_img.size

        # Model inferences
        p_yolo, _ = yolo.predict(img_path, measure_time=False)
        p_frcnn, _ = frcnn.predict(img_path, measure_time=False)
        p_mrcnn, _ = mrcnn.predict(img_path, measure_time=False)

        # Ground truth
        gt_scene = [g for g in gts if g['image_id'] == img_id]

        # Draw overlays
        img_gt = raw_img.copy()
        draw_gt = ImageDraw.Draw(img_gt)
        for g in gt_scene:
            b = g['box_xyxy']
            draw_gt.rectangle([b[0], b[1], b[2], b[3]], outline="green", width=2)

        img_yolo = draw_predictions_on_image(raw_img.copy(), p_yolo, color="#00ffcc")
        img_frcnn = draw_predictions_on_image(raw_img.copy(), p_frcnn, color="#ffaa00")
        img_mrcnn = draw_predictions_on_image(raw_img.copy(), p_mrcnn, color="#ff00ff")

        # Create 2x2 comparison grid
        thumb_w, thumb_h = 640, 480
        grid = Image.new("RGB", (thumb_w * 2, thumb_h * 2), color="black")
        draw_grid = ImageDraw.Draw(grid)

        grid.paste(img_gt.resize((thumb_w, thumb_h)), (0, 0))
        grid.paste(img_yolo.resize((thumb_w, thumb_h)), (thumb_w, 0))
        grid.paste(img_frcnn.resize((thumb_w, thumb_h)), (0, thumb_h))
        grid.paste(img_mrcnn.resize((thumb_w, thumb_h)), (thumb_w, thumb_h))

        draw_grid.text((15, 15), "Ground Truth (Real VisDrone)", fill="white")
        draw_grid.text((thumb_w + 15, 15), "YOLOv8s", fill="white")
        draw_grid.text((15, thumb_h + 15), "Faster R-CNN (ResNet50+FPN)", fill="white")
        draw_grid.text((thumb_w + 15, thumb_h + 15), "Mask R-CNN (ResNet50+FPN)", fill="white")

        save_path = out_dir / f"comparison_{scene_label}.png"
        grid.save(save_path)
        print(f"Saved qualitative comparison: {save_path}")

if __name__ == "__main__":
    main()
