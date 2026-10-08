"""
base_inferencer.py
==================
Abstract base class for all model inferencers in the CV hackathon benchmark.

All concrete inferencers must:
1. Implement predict(image_path) -> list[dict]
2. Use REAL VisDrone ground truth (never model predictions as GT)
3. Perform warmup before timing
4. Map outputs to VisDrone class IDs
"""

from __future__ import annotations
import time
import logging
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)


class Prediction:
    """Standardized prediction output from any model."""
    __slots__ = ['image_id', 'class_id', 'class_name', 'score', 'box_xyxy']

    def __init__(self, image_id: str, class_id: int, class_name: str,
                 score: float, box_xyxy: np.ndarray):
        self.image_id  = image_id
        self.class_id  = class_id
        self.class_name = class_name
        self.score     = float(score)
        self.box_xyxy  = np.asarray(box_xyxy, dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            'image_id':  self.image_id,
            'class_id':  self.class_id,
            'class_name': self.class_name,
            'score':     self.score,
            'box_xyxy':  self.box_xyxy.tolist(),
        }

    def __repr__(self):
        b = self.box_xyxy
        return (f"Prediction(id={self.image_id}, cls={self.class_name},"
                f" score={self.score:.3f}, box=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}])")


# VisDrone class definitions
VISDRONE_CLASSES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

# COCO class id -> VisDrone class id mapping
# Only classes with a reasonable VisDrone equivalent are kept
COCO_TO_VISDRONE: Dict[int, int] = {
    0: 0,   # person         -> pedestrian
    1: 2,   # bicycle        -> bicycle
    2: 3,   # car            -> car
    3: 9,   # motorcycle     -> motor
    5: 8,   # bus            -> bus
    7: 5,   # truck          -> truck
}


class BaseInferencer(ABC):
    """
    Abstract base for all model inferencers.

    Subclasses implement `_predict_single(image_path)` returning raw outputs.
    This class handles:
    - Warmup (mandatory before timing)
    - Timing (multiple runs, mean + std)
    - Standardized prediction format
    - Logging
    """

    def __init__(
        self,
        model_name: str,
        conf_threshold: float = 0.25,
        warmup_runs: int = 3,
        timing_runs: int = 5,
        device: str = 'cpu',
        class_names: Optional[List[str]] = None,
    ):
        self.model_name      = model_name
        self.conf_threshold  = conf_threshold
        self.warmup_runs     = warmup_runs
        self.timing_runs     = timing_runs
        self.device          = device
        self.class_names     = class_names or VISDRONE_CLASSES
        self._model_loaded   = False
        self._param_count: Optional[int] = None
        self._model_size_mb: Optional[float] = None

    @abstractmethod
    def load_model(self):
        """Load model weights. Called once before inference."""
        ...

    @abstractmethod
    def _predict_single(self, image_path: str, image: Image.Image) -> List[Prediction]:
        """
        Run inference on a single image.
        Return list of Prediction objects (already class-mapped if needed).
        Applied conf_threshold filtering should happen here.
        """
        ...

    def warmup(self, image_path: str):
        """Warmup the model with dummy/real inference to stabilize timing."""
        if not self._model_loaded:
            self.load_model()
        img = Image.open(image_path).convert('RGB')
        logger.info(f"[{self.model_name}] Warming up ({self.warmup_runs} runs)...")
        for i in range(self.warmup_runs):
            _ = self._predict_single(image_path, img)
        logger.info(f"[{self.model_name}] Warmup complete.")

    def predict(self, image_path: str, measure_time: bool = True) -> Tuple[List[Prediction], float]:
        """
        Run inference on one image.
        Returns (predictions, inference_time_seconds).
        If measure_time=True, runs `timing_runs` times and returns mean time.
        """
        if not self._model_loaded:
            self.load_model()
        img = Image.open(image_path).convert('RGB')
        image_id = Path(image_path).stem

        if measure_time:
            times = []
            preds = None
            for _ in range(self.timing_runs):
                t0 = time.perf_counter()
                preds = self._predict_single(image_path, img)
                times.append(time.perf_counter() - t0)
            # Assign image_id to all predictions
            for p in preds:
                p.image_id = image_id
            return preds, float(np.mean(times))
        else:
            t0 = time.perf_counter()
            preds = self._predict_single(image_path, img)
            t = time.perf_counter() - t0
            for p in preds:
                p.image_id = image_id
            return preds, t

    def predict_dataset(
        self,
        image_paths: List[str],
        warmup_image: Optional[str] = None,
    ) -> Tuple[List[Prediction], Dict[str, float]]:
        """
        Run inference on a list of images.
        Returns (all_predictions, timing_stats).

        timing_stats keys: mean_s, std_s, fps, all_times
        """
        if not self._model_loaded:
            self.load_model()

        # Warmup
        warmup_img = warmup_image or image_paths[0]
        self.warmup(warmup_img)

        all_preds = []
        all_times = []

        for img_path in image_paths:
            preds, t = self.predict(img_path, measure_time=True)
            all_preds.extend(preds)
            all_times.append(t)
            logger.debug(f"[{self.model_name}] {Path(img_path).name}: "
                         f"{len(preds)} predictions, {t*1000:.1f}ms")

        timing_stats = {
            'mean_s': float(np.mean(all_times)),
            'std_s':  float(np.std(all_times)),
            'fps':    float(1.0 / np.mean(all_times)) if np.mean(all_times) > 0 else 0.0,
            'all_times': all_times,
            'total_s': float(np.sum(all_times)),
        }
        logger.info(
            f"[{self.model_name}] Inference complete: "
            f"{len(all_preds)} total predictions, "
            f"mean={timing_stats['mean_s']*1000:.1f}ms, "
            f"fps={timing_stats['fps']:.1f}"
        )
        return all_preds, timing_stats

    def get_model_info(self) -> Dict:
        """Return model metadata (size, params, etc.)"""
        return {
            'model_name':    self.model_name,
            'param_count':   self._param_count,
            'size_mb':       self._model_size_mb,
            'device':        self.device,
            'conf_threshold': self.conf_threshold,
        }

    def _count_params(self, model) -> int:
        """Count trainable parameters."""
        try:
            n = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self._param_count = n
            return n
        except Exception:
            return 0

    def _measure_model_size(self, weight_path: str) -> float:
        """Measure checkpoint file size in MB."""
        try:
            size = Path(weight_path).stat().st_size / (1024 ** 2)
            self._model_size_mb = size
            return size
        except Exception:
            return 0.0

    @staticmethod
    def map_coco_to_visdrone(coco_class_id: int) -> Optional[int]:
        """Map COCO class id to VisDrone class id. Returns None if no mapping."""
        return COCO_TO_VISDRONE.get(coco_class_id, None)
