"""
infer_sam.py
============
YOLO + SAM (Segment Anything Model) Hybrid Aerial Segmentation Pipeline.
Uses YOLOv8 detections as bounding box prompts for SAM to generate high-precision masks.
"""

import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="YOLO + SAM Hybrid Inference")
    parser.add_argument("--source", required=True, help="Path to input image")
    parser.add_argument("--yolo-weights", default="weights/best.pt", help="Path to trained YOLO weights")
    parser.add_argument("--sam-weights", default=None, help="Path to SAM vit_b/vit_l checkpoint (optional)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--output-dir", default="results/sam_output", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO detector: {args.yolo_weights}...")
    yolo_model = YOLO(args.yolo_weights)

    img_path = Path(args.source)
    orig_img = cv2.imread(str(img_path))
    if orig_img is None:
        print(f"Error reading image {img_path}")
        return

    # Step 1: Detect bounding boxes with YOLO
    t0 = time.time()
    yolo_res = yolo_model.predict(str(img_path), conf=args.conf, device=0)[0]
    dt_yolo = time.time() - t0

    boxes = []
    classes = []
    scores = []
    if yolo_res.boxes is not None:
        boxes = yolo_res.boxes.xyxy.cpu().numpy()
        classes = yolo_res.boxes.cls.cpu().numpy()
        scores = yolo_res.boxes.conf.cpu().numpy()

    print(f"YOLO detected {len(boxes)} candidate aerial objects in {dt_yolo * 1000:.1f}ms")

    # Step 2: Prompt SAM or FastSAM
    has_sam = False
    predictor = None
    if args.sam_weights and Path(args.sam_weights).exists():
        try:
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry["vit_l"](checkpoint=args.sam_weights).to("cuda")
            predictor = SamPredictor(sam)
            predictor.set_image(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            has_sam = True
            print("Loaded Segment Anything Model (SAM) successfully on GPU.")
        except Exception as e:
            print(f"Warning: Could not initialize SAM ({e}). Using bounding box visualization.")

    annotated = orig_img.copy()
    names = yolo_model.names

    for i, (box, cls_id, conf) in enumerate(zip(boxes, classes, scores)):
        x1, y1, x2, y2 = map(int, box)
        cname = names.get(int(cls_id), str(int(cls_id)))

        if has_sam and predictor is not None:
            # Prompt SAM with bounding box
            masks, _, _ = predictor.predict(box=box[None, :], multimask_output=False)
            mask = masks[0]
            color = np.array([0, 255, 0], dtype=np.uint8)
            annotated[mask] = cv2.addWeighted(annotated[mask], 0.6, color, 0.4, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{cname} {conf:.2f}", (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out_file = out_dir / f"sam_{img_path.name}"
    cv2.imwrite(str(out_file), annotated)
    print(f"Hybrid SAM output saved to {out_file}")

if __name__ == "__main__":
    main()
