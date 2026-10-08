"""
roboflow_pipeline.py
Roboflow dataset management and preprocessing documentation for the
VisDrone CV hackathon project.

NOTE
----
This module documents the Roboflow configuration used for this project.
Actual Roboflow API calls require a valid API key set in the environment
variable ``ROBOFLOW_API_KEY`` or passed explicitly.

Install dependencies with::

    pip install roboflow pillow torchvision

"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Roboflow project configuration
# ---------------------------------------------------------------------------

ROBOFLOW_CONFIG: Dict[str, Any] = {
    # --- Project identifiers ---
    "workspace": "cv-hackathon",
    "project":   "visdrone-detection",
    "version":   1,

    # --- Preprocessing ---
    # Images are stretched to exactly 640×640 to match YOLOv8's default
    # input size.  Stretch (vs. fit-black or fit-grey) avoids border
    # artefacts that confuse small-object detectors.
    "resize": {
        "width":  640,
        "height": 640,
        "format": "Stretch to",
    },

    # --- Augmentations (applied online during training, not baked in) ---
    # Each augmentation is chosen specifically for aerial / drone imagery.
    # See reports/roboflow_preprocessing.md for full rationale.
    "augmentations": {
        # Horizontal flip: valid for aerial view (roads go both ways).
        "flip_horizontal": True,

        # Small rotation: camera pitch/roll variations in drone footage.
        "rotation": {"min": -15, "max": 15},

        # Brightness variation: simulates different times of day and
        # cloud cover in outdoor aerial scenes.
        "brightness": {"min": -25, "max": 25},

        # Exposure variation: compensates for auto-exposure differences
        # between drone cameras.
        "exposure": {"min": -15, "max": 15},

        # Mild Gaussian blur: simulates motion blur and lens defocus
        # at altitude.
        "blur": {"max_pixels": 1.5},

        # Low-level noise: simulates sensor noise on consumer-grade
        # drone cameras.
        "noise": {"max_percent": 2},
    },

    # --- Split ---
    # 70/20/10 gives a large training set while keeping a meaningful
    # held-out test set for final evaluation.
    "split": {
        "train": 0.70,
        "valid": 0.20,
        "test":  0.10,
    },

    # --- Export ---
    # YOLOv8 format: one .txt label file per image with normalised
    # cx cy w h coordinates, compatible with Ultralytics and RF-DETR.
    "export_format": "yolov8",
}


# ---------------------------------------------------------------------------
# Roboflow API wrapper
# ---------------------------------------------------------------------------


def get_roboflow_dataset(
    api_key:   str,
    workspace: str,
    project:   str,
    version:   int,
    location:  str,
) -> Any:
    """
    Download a dataset version from Roboflow.

    Requires ``roboflow`` package and a valid API key.

    Parameters
    ----------
    api_key : str
        Roboflow API key (obtain from https://app.roboflow.com → Settings →
        Roboflow API).  Can also be set via the ``ROBOFLOW_API_KEY``
        environment variable.
    workspace : str
        Roboflow workspace slug (e.g. ``"cv-hackathon"``).
    project : str
        Project slug (e.g. ``"visdrone-detection"``).
    version : int
        Dataset version number.
    location : str
        Local directory to download the dataset into.

    Returns
    -------
    roboflow.core.dataset.Dataset
        The downloaded dataset object with ``location`` attribute pointing
        to the local YAML path.

    Raises
    ------
    ImportError
        If the ``roboflow`` package is not installed.
    Exception
        On API or network errors.

    Example
    -------
    >>> import os
    >>> dataset = get_roboflow_dataset(
    ...     api_key   = os.environ["ROBOFLOW_API_KEY"],
    ...     workspace = "cv-hackathon",
    ...     project   = "visdrone-detection",
    ...     version   = 1,
    ...     location  = "./datasets/visdrone_rf",
    ... )
    >>> print(dataset.location)   # path to data.yaml
    """
    try:
        from roboflow import Roboflow  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "roboflow package not installed. Run: pip install roboflow"
        ) from exc

    rf      = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    dataset = project_obj.version(version).download(
        model_format=ROBOFLOW_CONFIG["export_format"],
        location=location,
        overwrite=True,
    )

    logger.info("Dataset downloaded to: %s", dataset.location)
    return dataset


# ---------------------------------------------------------------------------
# Configuration documenter
# ---------------------------------------------------------------------------


def document_preprocessing_config(config: Optional[Dict] = None) -> None:
    """
    Print the documented Roboflow preprocessing configuration to stdout.

    Useful for embedding in experiment logs to ensure reproducibility.

    Parameters
    ----------
    config : dict, optional
        Override the default :data:`ROBOFLOW_CONFIG`.
    """
    if config is None:
        config = ROBOFLOW_CONFIG

    sep = "=" * 60
    print(sep)
    print("ROBOFLOW PREPROCESSING CONFIGURATION")
    print(sep)
    print(f"  Workspace : {config['workspace']}")
    print(f"  Project   : {config['project']}")
    print(f"  Version   : {config['version']}")
    print()

    resize = config.get("resize", {})
    print(f"  Resize    : {resize.get('width')}×{resize.get('height')}  "
          f"[{resize.get('format')}]")
    print()

    split = config.get("split", {})
    print(f"  Split     : train={split.get('train', 0):.0%}  "
          f"valid={split.get('valid', 0):.0%}  "
          f"test={split.get('test', 0):.0%}")
    print(f"  Format    : {config.get('export_format', 'yolov8')}")
    print()

    augs = config.get("augmentations", {})
    print("  Augmentations:")
    for aug_name, aug_val in augs.items():
        print(f"    {aug_name:<20}: {aug_val}")

    print(sep)


# ---------------------------------------------------------------------------
# Local preprocessing (Roboflow-free alternative)
# ---------------------------------------------------------------------------


def apply_local_preprocessing(image: Any, config: Optional[Dict] = None) -> Any:
    """
    Apply the same preprocessing pipeline locally without the Roboflow API.

    Implements all augmentations from :data:`ROBOFLOW_CONFIG` using
    ``torchvision.transforms`` and ``PIL``.

    Parameters
    ----------
    image : PIL.Image.Image or torch.Tensor (C×H×W, uint8)
        Input image.
    config : dict, optional
        Override the default :data:`ROBOFLOW_CONFIG`.

    Returns
    -------
    torch.Tensor
        Preprocessed image tensor of shape ``(3, 640, 640)``, dtype ``float32``,
        values in ``[0, 1]``.

    Notes
    -----
    * Augmentations are applied **randomly** during training; during
      inference only the resize + to-tensor steps are applied.
    * Vertical flip is intentionally **excluded** (see
      ``reports/roboflow_preprocessing.md`` §7).
    * Rotation uses ``expand=False`` to maintain fixed 640×640 size.

    Example
    -------
    >>> from PIL import Image as PILImage
    >>> img = PILImage.open("frame_001.jpg")
    >>> tensor = apply_local_preprocessing(img)
    >>> tensor.shape
    torch.Size([3, 640, 640])
    """
    try:
        import torch
        import torchvision.transforms as T
        import torchvision.transforms.functional as TF
    except ImportError as exc:
        raise ImportError(
            "torchvision not installed. Run: pip install torchvision"
        ) from exc

    from PIL import Image as PILImage

    if config is None:
        config = ROBOFLOW_CONFIG

    resize_cfg = config.get("resize", {})
    aug_cfg    = config.get("augmentations", {})

    W = resize_cfg.get("width",  640)
    H = resize_cfg.get("height", 640)

    # --- Convert to PIL if necessary ---
    if not isinstance(image, PILImage.Image):
        # Assume torch Tensor C×H×W uint8
        image = TF.to_pil_image(image)

    # --- 1. Resize (stretch to target, no letterboxing) ---
    image = image.resize((W, H), PILImage.BILINEAR)

    # --- 2. Random horizontal flip ---
    if aug_cfg.get("flip_horizontal", False):
        import random
        if random.random() > 0.5:
            image = TF.hflip(image)

    # --- 3. Random rotation ---
    rot = aug_cfg.get("rotation", {})
    if rot:
        import random
        angle = random.uniform(rot.get("min", 0), rot.get("max", 0))
        image = TF.rotate(image, angle, expand=False)

    # --- 4. Colour jitter (brightness + exposure approximated as brightness) ---
    brightness_cfg = aug_cfg.get("brightness", {})
    exposure_cfg   = aug_cfg.get("exposure",   {})
    if brightness_cfg or exposure_cfg:
        b_lo = 1.0 + brightness_cfg.get("min", 0) / 100.0
        b_hi = 1.0 + brightness_cfg.get("max", 0) / 100.0
        e_lo = 1.0 + exposure_cfg.get("min", 0) / 100.0
        e_hi = 1.0 + exposure_cfg.get("max", 0) / 100.0
        jitter = T.ColorJitter(
            brightness=(max(0.0, b_lo), b_hi),
            contrast=(max(0.0, e_lo), e_hi),
        )
        image = jitter(image)

    # --- 5. Gaussian blur ---
    blur_cfg = aug_cfg.get("blur", {})
    if blur_cfg:
        import random
        max_px = blur_cfg.get("max_pixels", 0)
        if max_px > 0 and random.random() > 0.5:
            # kernel_size must be odd and >= 1
            k = max(1, int(max_px) * 2 + 1)
            image = TF.gaussian_blur(image, kernel_size=k)

    # --- 6. Gaussian noise ---
    noise_cfg = aug_cfg.get("noise", {})
    if noise_cfg:
        import random
        import torch as _torch
        max_pct = noise_cfg.get("max_percent", 0)
        if max_pct > 0 and random.random() > 0.5:
            tensor = TF.to_tensor(image)            # [0, 1]
            std    = random.uniform(0, max_pct / 100.0)
            noise  = _torch.randn_like(tensor) * std
            tensor = (tensor + noise).clamp(0.0, 1.0)
            return tensor

    # --- 7. Final to-tensor conversion ---
    return TF.to_tensor(image)
