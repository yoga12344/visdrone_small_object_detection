"""
small_objects.py — Small-object analysis utilities.

Provides dataset-level size distribution reporting and per-size AP
computation.  Zero counts are never silently reported as 0.0 — an explicit
explanation is always attached.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

try:
    from .metrics import size_category, compute_ap, match_predictions
except ImportError:
    from metrics import size_category, compute_ap, match_predictions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset distribution
# ---------------------------------------------------------------------------

def analyze_dataset_sizes(gt_annotations: List[Dict]) -> Dict:
    """
    Compute the size distribution of all ground-truth boxes in the dataset.

    Parameters
    ----------
    gt_annotations : list of dict
        Each entry: ``{image_id, class_id, box_xyxy}``.

    Returns
    -------
    dict with keys:
        ``'counts'`` – ``{size_category: int}``
        ``'fractions'`` – ``{size_category: float}``
        ``'total'`` – total number of GT boxes
        ``'size_per_class'`` – ``{class_id: {size: count}}``
        ``'report'`` – human-readable string summary
    """
    if not gt_annotations:
        return {
            "counts": {},
            "fractions": {},
            "total": 0,
            "size_per_class": {},
            "report": "No GT annotations provided.",
        }

    sizes = [size_category(ann["box_xyxy"]) for ann in gt_annotations]
    total = len(sizes)
    counts: Dict[str, int] = dict(Counter(sizes))
    for sz in ("small", "medium", "large"):
        counts.setdefault(sz, 0)

    fractions = {sz: counts[sz] / total for sz in ("small", "medium", "large")}

    # Per-class breakdown
    size_per_class: Dict[int, Dict[str, int]] = {}
    for ann in gt_annotations:
        cls = ann["class_id"]
        sz  = size_category(ann["box_xyxy"])
        if cls not in size_per_class:
            size_per_class[cls] = {"small": 0, "medium": 0, "large": 0}
        size_per_class[cls][sz] += 1

    # Report string
    lines = [
        f"Dataset Size Distribution (total boxes = {total})",
        "-" * 50,
        f"  Small  (w<32  & h<32 ) : {counts['small']:5d}  ({fractions['small']*100:5.1f}%)",
        f"  Medium (w<96  & h<96 ) : {counts['medium']:5d}  ({fractions['medium']*100:5.1f}%)",
        f"  Large  (otherwise    ) : {counts['large']:5d}  ({fractions['large']*100:5.1f}%)",
    ]
    report = "\n".join(lines)
    logger.info(report)

    return {
        "counts": counts,
        "fractions": fractions,
        "total": total,
        "size_per_class": size_per_class,
        "report": report,
    }


# ---------------------------------------------------------------------------
# Per-size AP
# ---------------------------------------------------------------------------

def compute_small_object_ap(
    gt_annotations: List[Dict],
    pred_annotations: List[Dict],
    size: str = "small",
    iou_thresh: float = 0.5,
) -> Dict:
    """
    Compute AP restricted to objects of a given size category.

    Parameters
    ----------
    gt_annotations : list of dict
        Full GT set.
    pred_annotations : list of dict
        Full prediction set.
    size : str
        One of ``'small'``, ``'medium'``, ``'large'``.
    iou_thresh : float
        IoU threshold for matching.

    Returns
    -------
    dict with keys:
        ``'ap'`` (float or None), ``'n_gt'``, ``'n_tp'``, ``'n_fp'``,
        ``'precision'``, ``'recall'``, ``'f1'``, and optionally ``'note'``
        when n_gt == 0.

    Notes
    -----
    When ``n_gt == 0`` the function returns ``ap=None`` and a ``'note'``
    key explaining the absence — it **never** silently returns ``0.0``.
    """
    if size not in ("small", "medium", "large"):
        raise ValueError(f"size must be 'small', 'medium', or 'large'; got {size!r}")

    from collections import defaultdict

    # Filter GT to the requested size
    gt_filtered = [a for a in gt_annotations if size_category(a["box_xyxy"]) == size]
    n_gt = len(gt_filtered)

    if n_gt == 0:
        note = (
            f"No '{size}' ground-truth objects exist in this annotation set.  "
            f"AP cannot be computed.  This is a dataset property, not a model "
            f"failure.  To evaluate small-object detection, use a dataset that "
            f"contains objects with w < 32 and h < 32 pixels."
        )
        logger.warning("compute_small_object_ap: n_gt=0 for size='%s'", size)
        return {
            "ap": None,
            "n_gt": 0,
            "n_tp": None,
            "n_fp": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "note": note,
        }

    # Build GT index for the filtered subset
    gt_index: Dict = defaultdict(list)
    n_gt_per_class: Dict[int, int] = defaultdict(int)
    for ann in gt_filtered:
        gt_index[(ann["image_id"], ann["class_id"])].append(ann["box_xyxy"])
        n_gt_per_class[ann["class_id"]] += 1

    pred_index: Dict = defaultdict(list)
    for ann in pred_annotations:
        pred_index[(ann["image_id"], ann["class_id"])].append(ann)

    tp_all: List[np.ndarray] = []
    fp_all: List[np.ndarray] = []
    sc_all: List[np.ndarray] = []

    all_keys = set(gt_index.keys()) | set(pred_index.keys())
    for key in all_keys:
        gt_boxes = gt_index.get(key, [])
        preds = pred_index.get(key, [])
        if not preds:
            continue
        scores = [p["score"] for p in preds]
        boxes  = [p["box_xyxy"] for p in preds]
        if not gt_boxes:
            n = len(preds)
            tp_all.append(np.zeros(n))
            fp_all.append(np.ones(n))
            sc_all.append(np.array(sorted(scores, reverse=True)))
            continue
        tp, fp, sc = match_predictions(gt_boxes, boxes, scores, iou_thresh)
        tp_all.append(tp)
        fp_all.append(fp)
        sc_all.append(sc)

    if tp_all:
        tp_cat = np.concatenate(tp_all)
        fp_cat = np.concatenate(fp_all)
        sc_cat = np.concatenate(sc_all)
        sort_idx = np.argsort(-sc_cat)
        tp_s = tp_cat[sort_idx]
        fp_s = fp_cat[sort_idx]
    else:
        tp_s = np.array([])
        fp_s = np.array([])

    ap, _, _ = compute_ap(tp_s, fp_s, n_gt)
    n_tp = int(tp_s.sum()) if len(tp_s) else 0
    n_fp = int(fp_s.sum()) if len(fp_s) else 0
    n_fn = n_gt - n_tp
    precision = n_tp / (n_tp + n_fp + 1e-9)
    recall    = n_tp / (n_gt + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    logger.info(
        "Size '%s' AP=%.4f P=%.4f R=%.4f n_gt=%d n_tp=%d n_fp=%d",
        size, ap, precision, recall, n_gt, n_tp, n_fp,
    )

    return {
        "ap": round(float(ap), 6),
        "n_gt": n_gt,
        "n_tp": n_tp,
        "n_fp": n_fp,
        "n_fn": n_fn,
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
    }


# ---------------------------------------------------------------------------
# Explanation helper
# ---------------------------------------------------------------------------

def explain_zero_small_map(
    gt_annotations: List[Dict],
    size: str = "small",
) -> str:
    """
    Produce a transparent explanation when small-object mAP is 0 or N/A.

    This function **always** diagnoses the root cause — it never returns an
    explanation-free zero.

    Parameters
    ----------
    gt_annotations : list of dict
        GT annotation set (used to count objects of each size).
    size : str
        Size category to diagnose.

    Returns
    -------
    str
        A human-readable explanation of why the metric is absent or zero,
        with exact counts.
    """
    dist = analyze_dataset_sizes(gt_annotations)
    n_size = dist["counts"].get(size, 0)
    total  = dist["total"]

    if n_size == 0:
        return (
            f"[{size.upper()} OBJECT mAP = N/A]\n"
            f"Root cause: The test set contains 0 '{size}' ground-truth boxes "
            f"out of {total} total GT boxes.\n"
            f"  - Small  : {dist['counts'].get('small',  0)}\n"
            f"  - Medium : {dist['counts'].get('medium', 0)}\n"
            f"  - Large  : {dist['counts'].get('large',  0)}\n"
            f"A metric of 0 or N/A is correct and expected — it is NOT an "
            f"indicator of poor model performance.  To benchmark small-object "
            f"detection, acquire a dataset with objects smaller than 32×32 pixels."
        )

    fraction = n_size / max(total, 1) * 100
    return (
        f"[{size.upper()} OBJECT mAP = 0.0]\n"
        f"The test set contains {n_size} '{size}' GT boxes ({fraction:.1f}% of "
        f"{total} total).  A true AP of 0.0 means the model made zero correct "
        f"detections at any confidence threshold for this size category.\n"
        f"Possible causes:\n"
        f"  1. Model never outputs high-confidence predictions for {size} objects.\n"
        f"  2. {size.capitalize()}-object images are under-represented in training.\n"
        f"  3. IoU threshold too strict for tiny boxes (consider lowering iou_thresh).\n"
        f"Recommendation: inspect per-image predictions for {size} GT boxes."
    )
