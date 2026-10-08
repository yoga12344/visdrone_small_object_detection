"""
visdrone_parser.py
Parse VisDrone2019-DET annotation files into a standardised format and
optionally convert the entire dataset to YOLO label files.

VisDrone annotation row format (CSV, no header):
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,
    <score>,<category_id>,<truncation>,<occlusion>

Category IDs (1-indexed in the file):
    0  = ignored region  → SKIP
    1  = pedestrian
    2  = people
    3  = bicycle
    4  = car
    5  = van
    6  = truck
    7  = tricycle
    8  = awning-tricycle
    9  = bus
    10 = motor

0-indexed class labels used in training: subtract 1 from category_id,
skip category_id == 0.
"""

import os
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES: List[str] = [
    "pedestrian",    # 0  (VisDrone cat 1)
    "people",        # 1  (VisDrone cat 2)
    "bicycle",       # 2  (VisDrone cat 3)
    "car",           # 3  (VisDrone cat 4)
    "van",           # 4  (VisDrone cat 5)
    "truck",         # 5  (VisDrone cat 6)
    "tricycle",      # 6  (VisDrone cat 7)
    "awning-tricycle",# 7 (VisDrone cat 8)
    "bus",           # 8  (VisDrone cat 9)
    "motor",         # 9  (VisDrone cat 10)
]
NUM_CLASSES: int = len(CLASS_NAMES)

# COCO class index → 0-indexed VisDrone class index
COCO_TO_VISDRONE: Dict[int, int] = {
    0: 0,   # person      → pedestrian
    1: 2,   # bicycle     → bicycle
    2: 3,   # car         → car
    3: 9,   # motorcycle  → motor
    5: 8,   # bus         → bus
    7: 5,   # truck       → truck
}

# Object area thresholds (pixels²) – VisDrone benchmark definition
_SMALL_AREA_MAX  = 32 * 32
_MEDIUM_AREA_MAX = 96 * 96

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _image_id_from_path(path: Path) -> str:
    """Return the stem of a file path as the image ID."""
    return path.stem


def parse_visdrone_annotation(
    ann_path: str,
    image_id: Optional[str] = None,
) -> List[Dict]:
    """
    Parse one VisDrone annotation ``.txt`` file.

    Parameters
    ----------
    ann_path : str
        Absolute or relative path to the annotation file.
    image_id : str, optional
        Override the image ID.  Defaults to the file stem.

    Returns
    -------
    list of dict
        Each dict has the following keys:

        - ``image_id``   : str
        - ``class_id``   : int  0-indexed training label (-1 = ignored)
        - ``class_name`` : str  class name or "ignored"
        - ``box_xywh``   : [x, y, w, h]  absolute pixel coords
        - ``box_xyxy``   : [x1, y1, x2, y2]
        - ``truncation`` : int  (0 = no truncation, 1 = partial, 2 = heavy)
        - ``occlusion``  : int  (0 = none, 1 = partial, 2 = heavy, 3 = unknown)
        - ``score``      : int  annotation confidence (usually 1)
    """
    ann_path = Path(ann_path)
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    if image_id is None:
        image_id = _image_id_from_path(ann_path)

    annotations: List[Dict] = []

    with ann_path.open("r") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) < 8:
                logger.warning(
                    "%s line %d: expected 8 fields, got %d – skipping.",
                    ann_path.name, lineno, len(parts),
                )
                continue

            try:
                x      = int(parts[0])
                y      = int(parts[1])
                w      = int(parts[2])
                h      = int(parts[3])
                score  = int(parts[4])
                cat_id = int(parts[5])
                trunc  = int(parts[6])
                occ    = int(parts[7])
            except ValueError as exc:
                logger.warning(
                    "%s line %d: parse error (%s) – skipping.", ann_path.name, lineno, exc
                )
                continue

            if cat_id == 0:
                # Ignored region – still recorded with class_id = -1
                annotations.append(
                    {
                        "image_id":   image_id,
                        "class_id":   -1,
                        "class_name": "ignored",
                        "box_xywh":   [x, y, w, h],
                        "box_xyxy":   [x, y, x + w, y + h],
                        "truncation": trunc,
                        "occlusion":  occ,
                        "score":      score,
                    }
                )
                continue

            class_id   = cat_id - 1          # Convert to 0-indexed
            class_name = CLASS_NAMES[class_id] if 0 <= class_id < NUM_CLASSES else "unknown"

            annotations.append(
                {
                    "image_id":   image_id,
                    "class_id":   class_id,
                    "class_name": class_name,
                    "box_xywh":   [x, y, w, h],
                    "box_xyxy":   [x, y, x + w, y + h],
                    "truncation": trunc,
                    "occlusion":  occ,
                    "score":      score,
                }
            )

    return annotations


# ---------------------------------------------------------------------------
# Dataset-level parsing
# ---------------------------------------------------------------------------


def parse_visdrone_dataset(
    images_dir: str,
    annotations_dir: str,
) -> Tuple[List[Dict], Dict[str, str], Dict]:
    """
    Parse an entire VisDrone split (train / val / test-dev / test-challenge).

    Parameters
    ----------
    images_dir : str
        Directory containing the image files (.jpg).
    annotations_dir : str
        Directory containing the annotation files (.txt, same stem as images).

    Returns
    -------
    gt_annotations : list of dict
        All annotation dicts (see :func:`parse_visdrone_annotation`).
        Entries with ``class_id == -1`` (ignored regions) are included.
    image_paths : dict
        ``{image_id: absolute_path_str}``
    stats : dict
        Dataset statistics:

        - ``n_images``         : total images found
        - ``n_parsed``         : images successfully parsed
        - ``n_annotations``    : total annotation rows (incl. ignored)
        - ``n_valid``          : annotations with class_id >= 0
        - ``n_ignored``        : ignored-region annotations
        - ``per_class``        : ``{class_name: count}``
        - ``size_distribution``: ``{'small': N, 'medium': N, 'large': N}``
        - ``images_without_gt``: image IDs with zero valid GT boxes
    """
    images_dir      = Path(images_dir)
    annotations_dir = Path(annotations_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    image_paths: Dict[str, str] = {}
    gt_annotations: List[Dict]  = []

    per_class    = defaultdict(int)
    size_dist    = {"small": 0, "medium": 0, "large": 0}
    images_no_gt: List[str] = []
    n_parsed     = 0
    n_ignored    = 0

    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    for img_path in image_files:
        image_id = _image_id_from_path(img_path)
        ann_path = annotations_dir / (image_id + ".txt")

        image_paths[image_id] = str(img_path.resolve())

        if not ann_path.exists():
            logger.warning("No annotation file for image %s – skipping.", image_id)
            images_no_gt.append(image_id)
            continue

        anns = parse_visdrone_annotation(str(ann_path), image_id=image_id)
        gt_annotations.extend(anns)
        n_parsed += 1

        valid_anns = [a for a in anns if a["class_id"] >= 0]
        if not valid_anns:
            images_no_gt.append(image_id)

        for ann in anns:
            if ann["class_id"] == -1:
                n_ignored += 1
                continue
            per_class[ann["class_name"]] += 1
            w, h = ann["box_xywh"][2], ann["box_xywh"][3]
            area = w * h
            if area <= _SMALL_AREA_MAX:
                size_dist["small"] += 1
            elif area <= _MEDIUM_AREA_MAX:
                size_dist["medium"] += 1
            else:
                size_dist["large"] += 1

    n_valid = sum(per_class.values())
    stats = {
        "n_images":          len(image_files),
        "n_parsed":          n_parsed,
        "n_annotations":     len(gt_annotations),
        "n_valid":           n_valid,
        "n_ignored":         n_ignored,
        "per_class":         dict(per_class),
        "size_distribution": size_dist,
        "images_without_gt": images_no_gt,
    }

    logger.info(
        "Parsed %d images, %d valid annotations, %d ignored regions.",
        n_parsed, n_valid, n_ignored,
    )
    return gt_annotations, image_paths, stats


# ---------------------------------------------------------------------------
# YOLO conversion
# ---------------------------------------------------------------------------


def convert_to_yolo_format(
    ann_dict: Dict,
    image_width: int,
    image_height: int,
) -> Optional[str]:
    """
    Convert a single VisDrone annotation dict to a YOLO format string.

    Returns ``None`` for ignored-region annotations (class_id == -1).

    Parameters
    ----------
    ann_dict : dict
        A single dict as returned by :func:`parse_visdrone_annotation`.
    image_width : int
        Image width in pixels.
    image_height : int
        Image height in pixels.

    Returns
    -------
    str or None
        ``'<class_id> <xc> <yc> <w> <h>'`` with values normalised to [0, 1],
        or ``None`` if the annotation should be skipped.
    """
    if ann_dict["class_id"] < 0:
        return None

    x, y, w, h = ann_dict["box_xywh"]

    # Clamp and normalise
    x  = max(0, x)
    y  = max(0, y)
    w  = min(w, image_width  - x)
    h  = min(h, image_height - y)

    if w <= 0 or h <= 0:
        logger.warning(
            "Image %s: degenerate box [%d,%d,%d,%d] – skipped.",
            ann_dict.get("image_id", "?"), x, y, w, h,
        )
        return None

    xc = (x + w / 2.0) / image_width
    yc = (y + h / 2.0) / image_height
    wn = w / image_width
    hn = h / image_height

    # Clamp to [0, 1] in case of out-of-bound annotations
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    wn = min(max(wn, 0.0), 1.0)
    hn = min(max(hn, 0.0), 1.0)

    return f"{ann_dict['class_id']} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


def convert_dataset_to_yolo(
    images_dir: str,
    annotations_dir: str,
    output_labels_dir: str,
) -> Dict:
    """
    Convert an entire VisDrone annotation split to YOLO ``.txt`` label files.

    One output file is written per image.  Images with zero valid annotations
    receive an empty label file (required by some YOLO trainers).

    Parameters
    ----------
    images_dir : str
        Source image directory.
    annotations_dir : str
        Source VisDrone annotation directory.
    output_labels_dir : str
        Destination directory for YOLO ``.txt`` files.

    Returns
    -------
    dict
        Conversion statistics:

        - ``n_images``         : images processed
        - ``n_labels_written`` : label files written
        - ``n_boxes_written``  : total YOLO box lines written
        - ``n_skipped_boxes``  : degenerate / ignored boxes skipped
        - ``n_empty_labels``   : images with no valid annotations
        - ``per_class``        : ``{class_name: count}``
    """
    images_dir        = Path(images_dir)
    annotations_dir   = Path(annotations_dir)
    output_labels_dir = Path(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    per_class        = defaultdict(int)
    n_labels_written = 0
    n_boxes_written  = 0
    n_skipped_boxes  = 0
    n_empty_labels   = 0

    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    if not image_files:
        logger.warning("No images found in %s", images_dir)

    for img_path in image_files:
        image_id = _image_id_from_path(img_path)
        ann_path = annotations_dir / (image_id + ".txt")

        # Read image dimensions (needed for normalisation)
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception as exc:
            logger.error("Cannot open image %s: %s – skipping.", img_path, exc)
            continue

        yolo_lines: List[str] = []

        if ann_path.exists():
            anns = parse_visdrone_annotation(str(ann_path), image_id=image_id)
            for ann in anns:
                yolo_line = convert_to_yolo_format(ann, img_w, img_h)
                if yolo_line is None:
                    n_skipped_boxes += 1
                else:
                    yolo_lines.append(yolo_line)
                    per_class[ann["class_name"]] += 1
                    n_boxes_written += 1
        else:
            logger.warning("No annotation file for %s.", image_id)

        out_path = output_labels_dir / (image_id + ".txt")
        out_path.write_text("\n".join(yolo_lines), encoding="utf-8")
        n_labels_written += 1

        if not yolo_lines:
            n_empty_labels += 1

    stats = {
        "n_images":         len(image_files),
        "n_labels_written": n_labels_written,
        "n_boxes_written":  n_boxes_written,
        "n_skipped_boxes":  n_skipped_boxes,
        "n_empty_labels":   n_empty_labels,
        "per_class":        dict(per_class),
    }
    logger.info(
        "YOLO conversion: %d images → %d labels (%d boxes, %d skipped).",
        len(image_files), n_labels_written, n_boxes_written, n_skipped_boxes,
    )
    return stats


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def print_dataset_stats(stats: Dict) -> None:
    """
    Print formatted dataset statistics to stdout.

    Covers:
    - Total images and parsed images
    - Total annotations (valid + ignored)
    - Per-class object counts with bar chart
    - Object size distribution (small / medium / large)
    - Any images missing ground-truth annotations
    """
    sep = "=" * 60
    print(sep)
    print("VISDRONE DATASET STATISTICS")
    print(sep)
    print(f"  Total images found : {stats.get('n_images', 0)}")
    print(f"  Images parsed      : {stats.get('n_parsed', 0)}")
    print(f"  Total annotations  : {stats.get('n_annotations', 0)}")
    print(f"    Valid objects    : {stats.get('n_valid', 0)}")
    print(f"    Ignored regions  : {stats.get('n_ignored', 0)}")

    per_class = stats.get("per_class", {})
    if per_class:
        print()
        print("  Per-class object counts:")
        total = max(sum(per_class.values()), 1)
        max_count = max(per_class.values(), default=1)
        bar_width = 30
        for cls_name in CLASS_NAMES:
            count = per_class.get(cls_name, 0)
            bar = "█" * int(count / max_count * bar_width)
            pct = count / total * 100
            print(f"    {cls_name:<20s}: {count:6d}  ({pct:5.1f}%)  {bar}")

    size_dist = stats.get("size_distribution", {})
    if size_dist:
        print()
        print("  Object size distribution (by area):")
        size_total = max(sum(size_dist.values()), 1)
        for label, count in size_dist.items():
            pct = count / size_total * 100
            print(f"    {label:<8s}: {count:6d}  ({pct:5.1f}%)")

    missing = stats.get("images_without_gt", [])
    if missing:
        print()
        print(f"  Images without valid GT: {len(missing)}")
        if len(missing) <= 10:
            for mid in missing:
                print(f"    - {mid}")
        else:
            for mid in missing[:5]:
                print(f"    - {mid}")
            print(f"    ... and {len(missing) - 5} more")

    print(sep)
