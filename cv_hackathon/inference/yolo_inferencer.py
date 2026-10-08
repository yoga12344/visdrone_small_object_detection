"""
yolo_inferencer.py
==================
YOLOv8s inferencer for VisDrone benchmark.

Maps YOLOv8/COCO class IDs to VisDrone class IDs.
Uses real VisDrone ground truth for evaluation (NOT its own predictions).
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional
from PIL import Image

import numpy as np

from base_inferencer import BaseInferencer, Prediction, VISDRONE_CLASSES, COCO_TO_VISDRONE

logger = logging.getLogger(__name__)


class YOLOv8sInferencer(BaseInferencer):
    """
    YOLOv8s one-stage detector.

    Architecture: Single-stage, anchor-free detector.
    Backbone: CSPDarknet + C2f modules.
    Neck: PAN-FPN.
    Head: Decoupled detection head.

    Key properties:
    - Very fast inference (~7-10ms on GPU)
    - Good for real-time applications
    - Struggles with very small objects (<16px)
    - Pretrained on COCO (80 classes) -> mapped to 10 VisDrone classes
    """

    def __init__(
        self,
        weights_path: str = 'yolov8s.pt',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        warmup_runs: int = 3,
        timing_runs: int = 5,
        device: str = 'cpu',
        image_size: int = 640,
    ):
        super().__init__(
            model_name='YOLOv8s',
            conf_threshold=conf_threshold,
            warmup_runs=warmup_runs,
            timing_runs=timing_runs,
            device=device,
            class_names=VISDRONE_CLASSES,
        )
        self.weights_path  = weights_path
        self.iou_threshold = iou_threshold
        self.image_size    = image_size
        self._model        = None

    def load_model(self):
        from ultralytics import YOLO
        logger.info(f"[YOLOv8s] Loading weights from {self.weights_path}")
        self._model = YOLO(self.weights_path)
        self._model_loaded = True
        self._measure_model_size(self.weights_path)
        # Count params
        try:
            import torch
            self._count_params(self._model.model)
        except Exception:
            pass
        logger.info(f"[YOLOv8s] Model loaded. Size: {self._model_size_mb:.1f}MB, "
                    f"Params: {self._param_count:,}" if self._param_count else
                    f"[YOLOv8s] Model loaded. Size: {self._model_size_mb:.1f}MB")

    def _predict_single(self, image_path: str, image: Image.Image) -> List[Prediction]:
        W, H = image.size
        results = self._model(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            verbose=False,
            device=self.device,
        )
        res = results[0]

        predictions = []
        if res.boxes is None or len(res.boxes) == 0:
            return predictions

        for cls_id, score, box in zip(
            res.boxes.cls.cpu().numpy().astype(int),
            res.boxes.conf.cpu().numpy(),
            res.boxes.xyxy.cpu().numpy(),
        ):
            # Map COCO class id to VisDrone class id
            vd_cls = COCO_TO_VISDRONE.get(int(cls_id), None)
            if vd_cls is None:
                continue  # Skip COCO classes not in VisDrone

            x1, y1, x2, y2 = box
            # Clamp to image bounds
            x1 = max(0.0, float(x1))
            y1 = max(0.0, float(y1))
            x2 = min(float(W), float(x2))
            y2 = min(float(H), float(y2))

            if x2 <= x1 or y2 <= y1:
                continue

            predictions.append(Prediction(
                image_id='',
                class_id=vd_cls,
                class_name=VISDRONE_CLASSES[vd_cls],
                score=float(score),
                box_xyxy=np.array([x1, y1, x2, y2]),
            ))

        return predictions

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            'weights_path':  self.weights_path,
            'iou_threshold': self.iou_threshold,
            'image_size':    self.image_size,
            'architecture':  'CSPDarknet + C2f + PAN-FPN + Decoupled Head',
            'type':          'one-stage anchor-free',
            'coco_classes_kept': list(COCO_TO_VISDRONE.keys()),
            'coco_classes_dropped': 'All COCO classes not in COCO_TO_VISDRONE mapping',
        })
        return info


if __name__ == '__main__':
    import json, sys
    logging.basicConfig(level=logging.INFO)

    weights = sys.argv[1] if len(sys.argv) > 1 else 'yolov8s.pt'
    test_image = sys.argv[2] if len(sys.argv) > 2 else None

    inferencer = YOLOv8sInferencer(weights_path=weights)

    if test_image:
        preds, timing = inferencer.predict_dataset([test_image])
        print(f"\n--- YOLOv8s Results ---")
        print(f"Predictions: {len(preds)}")
        print(f"Timing: {timing['mean_s']*1000:.1f}ms mean, {timing['fps']:.1f} FPS")
        for p in preds:
            print(f"  {p}")
        print(f"\nModel Info:")
        print(json.dumps(inferencer.get_model_info(), indent=2))
    else:
        print("Usage: python yolo_inferencer.py <weights.pt> <image_path>")
        print("Model info (no image):")
        inferencer.load_model()
        print(json.dumps(inferencer.get_model_info(), indent=2))
