"""
sam_inferencer.py
=================
YOLOv8 + SAM (Segment Anything Model) Pipeline for VisDrone Benchmark.

Pipeline:
  Image -> YOLOv8s -> Prompt Bounding Boxes -> SAM Predictor -> Fine Instance Masks -> Bounding Boxes

Computes:
  1. YOLO inference time
  2. SAM segmentation time
  3. Total pipeline inference time
"""

from __future__ import annotations
import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from PIL import Image
import numpy as np

from base_inferencer import BaseInferencer, Prediction, VISDRONE_CLASSES, COCO_TO_VISDRONE

logger = logging.getLogger(__name__)


class YOLOSAMInferencer(BaseInferencer):
    """
    Two-stage hybrid model:
      Stage 1: YOLOv8s detects objects and proposes candidate bounding boxes.
      Stage 2: SAM (Segment Anything Model) uses candidate boxes as spatial prompts
               to predict fine zero-shot segmentation masks.
    """

    MODEL_NAME = "YOLOv8 + SAM"

    def __init__(
        self,
        yolo_weights: str = "yolov8s.pt",
        sam_weights: Optional[str] = None,
        sam_model_type: str = "vit_b",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        warmup_runs: int = 1,
        timing_runs: int = 2,
        device: str = "cpu",
    ):
        super().__init__(
            model_name=self.MODEL_NAME,
            conf_threshold=conf_threshold,
            warmup_runs=warmup_runs,
            timing_runs=timing_runs,
            device=device,
            class_names=VISDRONE_CLASSES,
        )
        self.yolo_weights = yolo_weights
        self.sam_weights = sam_weights
        self.sam_model_type = sam_model_type
        self.iou_threshold = iou_threshold
        
        self.yolo_model = None
        self.sam_predictor = None
        self.sam_loaded = False
        
        # Detailed timing breakdown
        self.last_yolo_time = 0.0
        self.last_sam_time = 0.0

    def load_model(self):
        from ultralytics import YOLO
        logger.info(f"[{self.MODEL_NAME}] Loading YOLOv8s from {self.yolo_weights}...")
        self.yolo_model = YOLO(self.yolo_weights)
        
        # Check SAM weights
        if self.sam_weights and os.path.exists(self.sam_weights):
            try:
                from segment_anything import sam_model_registry, SamPredictor
                logger.info(f"[{self.MODEL_NAME}] Loading SAM ({self.sam_model_type}) from {self.sam_weights}...")
                sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_weights)
                sam.to(self.device)
                self.sam_predictor = SamPredictor(sam)
                self.sam_loaded = True
                self._measure_model_size(self.sam_weights)
                self._count_params(sam)
                logger.info(f"[{self.MODEL_NAME}] SAM loaded successfully.")
            except Exception as e:
                logger.warning(f"[{self.MODEL_NAME}] Failed to load SAM: {e}. Will operate in YOLO-fallback mode.")
                self.sam_loaded = False
        else:
            logger.info(f"[{self.MODEL_NAME}] SAM weights not found at {self.sam_weights}. Pipeline will document SAM architecture.")
            self.sam_loaded = False
            
        self._model_loaded = True

    def _predict_single(self, image_path: str, image: Image.Image) -> List[Prediction]:
        W, H = image.size
        img_np = np.array(image.convert("RGB"))
        
        # 1. YOLO Stage
        t0_yolo = time.perf_counter()
        results = self.yolo_model(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device
        )[0]
        self.last_yolo_time = time.perf_counter() - t0_yolo
        
        raw_boxes = []
        raw_classes = []
        raw_scores = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            for cls_id, score, box in zip(
                results.boxes.cls.cpu().numpy().astype(int),
                results.boxes.conf.cpu().numpy(),
                results.boxes.xyxy.cpu().numpy()
            ):
                vd_cls = COCO_TO_VISDRONE.get(int(cls_id), None)
                if vd_cls is None:
                    continue
                raw_boxes.append(box)
                raw_classes.append(vd_cls)
                raw_scores.append(float(score))
                
        if not raw_boxes:
            self.last_sam_time = 0.0
            return []
            
        predictions = []
        
        # 2. SAM Stage
        if self.sam_loaded and self.sam_predictor is not None:
            t0_sam = time.perf_counter()
            self.sam_predictor.set_image(img_np)
            
            for box, vd_cls, score in zip(raw_boxes, raw_classes, raw_scores):
                # Prompt SAM with YOLO box
                masks, _, _ = self.sam_predictor.predict(
                    box=box,
                    multimask_output=False
                )
                m = masks[0]
                ys, xs = np.where(m)
                if len(xs) == 0:
                    refined_box = box
                else:
                    refined_box = [xs.min(), ys.min(), xs.max(), ys.max()]
                    
                x1, y1, x2, y2 = refined_box
                x1 = max(0.0, float(x1))
                y1 = max(0.0, float(y1))
                x2 = min(float(W), float(x2))
                y2 = min(float(H), float(y2))
                
                if x2 > x1 and y2 > y1:
                    predictions.append(Prediction(
                        image_id="",
                        class_id=vd_cls,
                        class_name=VISDRONE_CLASSES[vd_cls],
                        score=score,
                        box_xyxy=np.array([x1, y1, x2, y2])
                    ))
            self.last_sam_time = time.perf_counter() - t0_sam
        else:
            # Fallback when SAM weights are absent
            self.last_sam_time = 0.0
            for box, vd_cls, score in zip(raw_boxes, raw_classes, raw_scores):
                x1, y1, x2, y2 = box
                predictions.append(Prediction(
                    image_id="",
                    class_id=vd_cls,
                    class_name=VISDRONE_CLASSES[vd_cls],
                    score=score,
                    box_xyxy=np.array([x1, y1, x2, y2])
                ))
                
        return predictions

    def get_model_info(self) -> Dict:
        info = super().get_model_info()
        info.update({
            "stage1": "YOLOv8s Box Proposal",
            "stage2": f"SAM ({self.sam_model_type}) Box-Prompted Segmentation",
            "sam_weights_present": self.sam_loaded,
            "architecture": "Two-stage cascading detector + segmenter",
            "timing_note": "Total latency = YOLO time + SAM mask generation time",
        })
        return info
