"""
run_exp8_sam_hybrid.py
======================
Experiment 8: Hybrid Solution A (YOLOv8s + Segment Anything Model)
Combines fast object localization from YOLOv8s with zero-shot pixel-level
instance segmentation from SAM (ViT-B).

Pipeline:
  Full Aerial Image -> YOLOv8s (Bounding Box Detection) -> Box Prompts -> SAM -> Instance Masks

Outputs:
  - High-resolution qualitative comparison panels for 5 representative test scenes:
      Original Image | YOLOv8s Boxes | YOLOv8s + SAM Alpha-Blended Masks
  - Zoomed-in cluster views showing boundary delineation in crowded regions
  - Saved to results/hybrid/yolov8_sam/
  - Structured latency and qualitative analysis metadata in hybrid_sam_analysis.json
"""

import gc
import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "scripts"))
sys.path.append(str(PROJECT_ROOT / "data"))

import visdrone_parser
from ultralytics import YOLO

try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exp8_YOLO_SAM")

EXP_ROOT = PROJECT_ROOT / "results" / "hybrid" / "yolov8_sam"
EXP_ROOT.mkdir(parents=True, exist_ok=True)

SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_CHECKPOINT = PROJECT_ROOT / "sam_vit_b_01ec64.pth"

PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
    (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187)
]


def download_sam_if_needed():
    if not SAM_CHECKPOINT.exists():
        logger.info(f"Downloading SAM ViT-B checkpoint (~358MB) from {SAM_URL}...")
        urllib.request.urlretrieve(SAM_URL, str(SAM_CHECKPOINT))
        logger.info("SAM checkpoint downloaded successfully.")
    else:
        logger.info(f"SAM checkpoint already exists at {SAM_CHECKPOINT}")


def run_yolo_sam_pipeline(representative_scenes: list):
    download_sam_if_needed()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Loading SAM model (vit_b) on %s...", device)
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    yolo_weights = PROJECT_ROOT / "results" / "trained" / "yolov8s" / "weights" / "best.pt"
    if not yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")
    yolo_model = YOLO(str(yolo_weights))

    raw_val_dir = PROJECT_ROOT / "data" / "raw" / "VisDrone2019-DET-val" / "images"
    summary_records = []

    for idx, scene_info in enumerate(representative_scenes, 1):
        img_id = scene_info["image_id"]
        scene_name = scene_info["scene_name"]
        img_path = raw_val_dir / f"{img_id}.jpg"
        if not img_path.exists():
            logger.warning("Image %s not found, skipping.", img_path)
            continue

        logger.info(f"Processing Scene {idx}: {scene_name} ({img_id})...")
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W, _ = img_rgb.shape

        # Step 1: Fast YOLO detection
        t0 = time.perf_counter()
        yolo_res = yolo_model(str(img_path), imgsz=640, conf=0.25, device=0, verbose=False)[0]
        yolo_time = time.perf_counter() - t0

        boxes_list = []
        classes_list = []
        scores_list = []
        if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
            boxes = yolo_res.boxes.xyxy.cpu().numpy()
            classes = yolo_res.boxes.cls.cpu().numpy().astype(int)
            scores = yolo_res.boxes.conf.cpu().numpy()
            for b, c, s in zip(boxes, classes, scores):
                if 0 <= c <= 9:
                    boxes_list.append(b)
                    classes_list.append(c)
                    scores_list.append(s)

        # Step 2: SAM Segmentation
        t1 = time.perf_counter()
        predictor.set_image(img_rgb)

        mask_overlay = img_rgb.copy()
        mask_count = 0

        for b, c in zip(boxes_list, classes_list):
            box_tensor = np.array(b)[None, :]
            transformed_boxes = predictor.transform.apply_boxes(box_tensor, img_rgb.shape[:2])
            torch_boxes = torch.as_tensor(transformed_boxes, dtype=torch.float, device=device)

            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=torch_boxes,
                multimask_output=False
            )
            mask_np = masks[0, 0].cpu().numpy().astype(bool)
            color = PALETTE[c % len(PALETTE)]

            # Blend color on mask
            colored_mask = np.zeros_like(img_rgb, dtype=np.uint8)
            colored_mask[mask_np] = color
            mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored_mask, 0.45, 0)
            
            # Draw contour border
            contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_overlay, contours, -1, color, 2)
            mask_count += 1

        sam_time = time.perf_counter() - t1
        total_time = yolo_time + sam_time

        # Step 3: Draw YOLO Box Panel
        yolo_box_img = img_rgb.copy()
        for b, c, s in zip(boxes_list, classes_list, scores_list):
            color = PALETTE[c % len(PALETTE)]
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(yolo_box_img, (x1, y1), (x2, y2), color, 2)
            label = f"{visdrone_parser.CLASS_NAMES[c]} {s:.2f}"
            cv2.putText(yolo_box_img, label, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Step 4: Create Side-by-Side 3-Panel Comparison
        # Original | YOLO Boxes | YOLO + SAM Masks
        header_h = 60
        panel_w = W
        total_w = panel_w * 3
        total_h = H + header_h

        composite = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        composite[header_h:, 0:panel_w] = img_rgb
        composite[header_h:, panel_w:panel_w*2] = yolo_box_img
        composite[header_h:, panel_w*2:] = mask_overlay

        # Add Titles
        cv2.putText(composite, f"Scene {idx}: {scene_name} - Original Image", (40, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(composite, f"Stage 1: YOLOv8s Bounding Boxes ({len(boxes_list)} detections, {yolo_time*1000:.1f}ms)", (panel_w + 40, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(composite, f"Stage 2: YOLOv8s + SAM Segmentation ({mask_count} masks, {sam_time*1000:.1f}ms)", (panel_w*2 + 40, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        out_panel_file = EXP_ROOT / f"scene_{idx:02d}_{img_id}_yolo_sam_comparison.png"
        cv2.imwrite(str(out_panel_file), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        logger.info("Saved 3-panel comparison to %s", out_panel_file)

        # Create zoomed-in crop of dense region if boxes exist
        if len(boxes_list) > 0:
            center_x = int(np.mean([b[0] for b in boxes_list]))
            center_y = int(np.mean([b[1] for b in boxes_list]))
            crop_sz = min(300, min(W, H) // 2)
            cx1 = max(0, center_x - crop_sz // 2)
            cy1 = max(0, center_y - crop_sz // 2)
            cx2 = min(W, cx1 + crop_sz)
            cy2 = min(H, cy1 + crop_sz)

            zoom_box = cv2.resize(yolo_box_img[cy1:cy2, cx1:cx2], (600, 600))
            zoom_mask = cv2.resize(mask_overlay[cy1:cy2, cx1:cx2], (600, 600))
            zoom_composite = np.hstack([zoom_box, zoom_mask])
            zoom_header = np.zeros((50, 1200, 3), dtype=np.uint8)
            cv2.putText(zoom_header, "Zoom: YOLOv8s Boxes (Left) vs YOLO + SAM Precise Boundaries (Right)", (30, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            zoom_final = np.vstack([zoom_header, zoom_composite])
            zoom_file = EXP_ROOT / f"scene_{idx:02d}_{img_id}_cluster_zoom.png"
            cv2.imwrite(str(zoom_file), cv2.cvtColor(zoom_final, cv2.COLOR_RGB2BGR))

        summary_records.append({
            "scene_index": idx,
            "scene_name": scene_name,
            "image_id": img_id,
            "yolo_detections": len(boxes_list),
            "sam_masks": mask_count,
            "yolo_latency_ms": round(yolo_time * 1000.0, 2),
            "sam_latency_ms": round(sam_time * 1000.0, 2),
            "total_latency_ms": round(total_time * 1000.0, 2),
            "resolution": f"{W}x{H}"
        })

    # Save JSON report
    with open(EXP_ROOT / "hybrid_sam_analysis.json", "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)

    logger.info("Experiment 8 (YOLOv8 + SAM) completed successfully.")
    del sam, yolo_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def main():
    logger.info("==================================================================")
    logger.info("  EXPERIMENT 8: HYBRID SOLUTION A (YOLOv8s + SAM SEGMENTATION)    ")
    logger.info("==================================================================")

    # 5 Representative scenes selected from locked test set
    scenes = [
        {"image_id": "0000001_07999_d_0000012", "scene_name": "Dense_Pedestrian_Scene"},
        {"image_id": "0000001_08414_d_0000013", "scene_name": "Tiny_Object_Heavy_Scene"},
        {"image_id": "0000001_05249_d_0000009", "scene_name": "Vehicle_Heavy_Scene"},
        {"image_id": "0000001_02999_d_0000005", "scene_name": "Occluded_Crowded_Scene"},
        {"image_id": "0000021_00000_d_0000001", "scene_name": "Difficult_Aerial_Scene"}
    ]
    run_yolo_sam_pipeline(scenes)


if __name__ == "__main__":
    main()
