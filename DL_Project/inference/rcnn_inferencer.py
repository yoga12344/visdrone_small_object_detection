"""
rcnn_inferencer.py
==================
Faster R-CNN and Mask R-CNN inferencers for VisDrone benchmark.

Both models use ResNet50 + FPN backbone, pretrained on COCO.
COCO classes are mapped to VisDrone classes before evaluation.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional
from PIL import Image

import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as TF

from base_inferencer import BaseInferencer, Prediction, VISDRONE_CLASSES, COCO_TO_VISDRONE

logger = logging.getLogger(__name__)


class FasterRCNNInferencer(BaseInferencer):
    """
    Faster R-CNN with ResNet50 + FPN backbone.

    Architecture: Two-stage detector.
    Stage 1: Region Proposal Network (RPN) - generates candidate boxes.
    Stage 2: RoI Pooling + Classification head.

    Key properties:
    - Higher accuracy than one-stage at cost of speed
    - ~2-4 FPS on CPU for 640px images
    - Strong on medium/large objects
    - Pretrained on COCO (80 classes) -> mapped to VisDrone 10 classes
    """

    MODEL_NAME = 'Faster R-CNN (ResNet50+FPN)'

    def __init__(
        self,
        conf_threshold: float = 0.50,
        warmup_runs: int = 2,
        timing_runs: int = 3,
        device: str = 'cpu',
    ):
        super().__init__(
            model_name=self.MODEL_NAME,
            conf_threshold=conf_threshold,
            warmup_runs=warmup_runs,
            timing_runs=timing_runs,
            device=device,
            class_names=VISDRONE_CLASSES,
        )
        self._model = None

    def load_model(self):
        logger.info(f"[Faster R-CNN] Loading pretrained ResNet50-FPN...")
        self._model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        self._model.eval()
        self._model.to(self.device)
        self._model_loaded = True
        self._count_params(self._model)
        logger.info(f"[Faster R-CNN] Loaded. Params: {self._param_count:,}")

    def _predict_single(self, image_path: str, image: Image.Image) -> List[Prediction]:
        W, H = image.size
        tensor = TF.to_tensor(image).to(self.device)

        with torch.no_grad():
            outputs = self._model([tensor])[0]

        boxes   = outputs['boxes'].cpu().numpy()    # N x 4, xyxy
        labels  = outputs['labels'].cpu().numpy()   # N, COCO class ids (1-indexed!)
        scores  = outputs['scores'].cpu().numpy()   # N

        predictions = []
        for coco_label, score, box in zip(labels, scores, boxes):
            if score < self.conf_threshold:
                continue

            # torchvision returns 1-indexed COCO labels: label 1 = person, etc.
            # Subtract 1 to get 0-indexed COCO id matching our COCO_TO_VISDRONE map
            coco_id_0indexed = int(coco_label) - 1
            vd_cls = COCO_TO_VISDRONE.get(coco_id_0indexed, None)
            if vd_cls is None:
                continue

            x1, y1, x2, y2 = box
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
            'backbone':     'ResNet50',
            'neck':         'FPN',
            'architecture': 'Two-stage: RPN + RoI Pooling + Classification',
            'type':         'two-stage anchor-based',
            'pretrained':   'COCO',
            'note': (
                'COCO 1-indexed labels converted to 0-indexed before VisDrone mapping. '
                'Only 6 COCO classes map to VisDrone classes.'
            ),
        })
        return info


class MaskRCNNInferencer(BaseInferencer):
    """
    Mask R-CNN with ResNet50 + FPN backbone.

    Architecture: Extends Faster R-CNN with mask prediction branch.
    Stage 1: RPN - region proposals.
    Stage 2: RoI Align + Classification + Box Regression + Mask prediction.

    Key properties:
    - Instance segmentation in addition to detection
    - Slightly slower than Faster R-CNN due to mask head
    - VisDrone has NO segmentation GT -> detection evaluated, segmentation qualitative only
    - Pretrained on COCO (80 classes) -> mapped to VisDrone 10 classes

    IMPORTANT: VisDrone GT does NOT contain segmentation masks.
    Detection evaluation uses bounding boxes only.
    Segmentation output is shown qualitatively only.
    """

    MODEL_NAME = 'Mask R-CNN (ResNet50+FPN)'

    def __init__(
        self,
        conf_threshold: float = 0.50,
        warmup_runs: int = 2,
        timing_runs: int = 3,
        device: str = 'cpu',
        mask_threshold: float = 0.5,
    ):
        super().__init__(
            model_name=self.MODEL_NAME,
            conf_threshold=conf_threshold,
            warmup_runs=warmup_runs,
            timing_runs=timing_runs,
            device=device,
            class_names=VISDRONE_CLASSES,
        )
        self.mask_threshold = mask_threshold
        self._model = None

    def load_model(self):
        logger.info(f"[Mask R-CNN] Loading pretrained ResNet50-FPN...")
        self._model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        self._model.eval()
        self._model.to(self.device)
        self._model_loaded = True
        self._count_params(self._model)
        logger.info(f"[Mask R-CNN] Loaded. Params: {self._param_count:,}")

    def _predict_single(self, image_path: str, image: Image.Image) -> List[Prediction]:
        W, H = image.size
        tensor = TF.to_tensor(image).to(self.device)

        with torch.no_grad():
            outputs = self._model([tensor])[0]

        boxes   = outputs['boxes'].cpu().numpy()
        labels  = outputs['labels'].cpu().numpy()
        scores  = outputs['scores'].cpu().numpy()
        # masks available but not used for detection evaluation
        # masks = outputs['masks'].cpu().numpy()  # N x 1 x H x W

        predictions = []
        for coco_label, score, box in zip(labels, scores, boxes):
            if score < self.conf_threshold:
                continue

            coco_id_0indexed = int(coco_label) - 1  # torchvision is 1-indexed
            vd_cls = COCO_TO_VISDRONE.get(coco_id_0indexed, None)
            if vd_cls is None:
                continue

            x1, y1, x2, y2 = box
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

    def predict_with_masks(self, image_path: str) -> tuple:
        """
        Extended prediction that also returns raw segmentation masks.
        Used for qualitative visualization only (no GT masks to compare against).
        Returns (predictions, masks_list)
        """
        if not self._model_loaded:
            self.load_model()
        image = Image.open(image_path).convert('RGB')
        W, H = image.size
        tensor = TF.to_tensor(image).to(self.device)

        with torch.no_grad():
            outputs = self._model([tensor])[0]

        boxes   = outputs['boxes'].cpu().numpy()
        labels  = outputs['labels'].cpu().numpy()
        scores  = outputs['scores'].cpu().numpy()
        masks   = outputs['masks'].cpu().numpy()  # N x 1 x H x W

        predictions = []
        kept_masks = []
        image_id = Path(image_path).stem

        for coco_label, score, box, mask in zip(labels, scores, boxes, masks):
            if score < self.conf_threshold:
                continue
            coco_id_0indexed = int(coco_label) - 1
            vd_cls = COCO_TO_VISDRONE.get(coco_id_0indexed, None)
            if vd_cls is None:
                continue

            x1, y1, x2, y2 = box
            predictions.append(Prediction(
                image_id=image_id,
                class_id=vd_cls,
                class_name=VISDRONE_CLASSES[vd_cls],
                score=float(score),
                box_xyxy=np.array([x1, y1, x2, y2]),
            ))
            # Binarize mask
            kept_masks.append((mask[0] > self.mask_threshold).astype(np.uint8))

        return predictions, kept_masks

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            'backbone':     'ResNet50',
            'neck':         'FPN',
            'architecture': 'Two-stage: RPN + RoI Align + Cls + Box + Mask Head',
            'type':         'instance segmentation (two-stage)',
            'pretrained':   'COCO',
            'segmentation_note': (
                'VisDrone does NOT provide segmentation masks. '
                'Detection boxes are evaluated against VisDrone GT bounding boxes. '
                'Segmentation output is purely qualitative/demonstrative.'
            ),
        })
        return info
