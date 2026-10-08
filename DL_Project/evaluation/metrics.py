"""
metrics.py — Proper object detection metrics.

mAP is computed correctly via precision-recall curve (not precision*recall).
IoU matching uses greedy assignment (score-descending) so that each GT box
is matched at most once — equivalent to the PASCAL VOC / COCO protocol.

No sklearn dependency: every metric is implemented from scratch.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. IoU helper
# ---------------------------------------------------------------------------

def iou_box(box_a: List[float], box_b: List[float]) -> float:
    """
    Compute Intersection-over-Union for two axis-aligned bounding boxes.

    Parameters
    ----------
    box_a, box_b : array-like of length 4
        Boxes in ``[x1, y1, x2, y2]`` (xyxy) format.
        It is assumed x2 >= x1 and y2 >= y1.

    Returns
    -------
    float
        IoU in [0, 1].  Returns 0.0 when either box has zero area.
    """
    ax1, ay1, ax2, ay2 = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
    bx1, by1, bx2, by2 = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])

    # Areas
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if area_a == 0.0 or area_b == 0.0:
        return 0.0

    # Intersection
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


# ---------------------------------------------------------------------------
# 2. Greedy matching of predictions to GT boxes
# ---------------------------------------------------------------------------

def match_predictions(
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    pred_scores: List[float],
    iou_thresh: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match predictions to ground-truth boxes for a single (image, class) pair.

    Algorithm
    ---------
    Predictions are processed in descending score order.  A prediction is a
    **true positive** if it overlaps with an unmatched GT box at IoU >=
    ``iou_thresh``; otherwise it is a **false positive**.

    Parameters
    ----------
    gt_boxes : list of [x1, y1, x2, y2]
        Ground-truth boxes for this image/class.
    pred_boxes : list of [x1, y1, x2, y2]
        Predicted boxes, parallel to ``pred_scores``.
    pred_scores : list of float
        Confidence scores, parallel to ``pred_boxes``.
    iou_thresh : float
        Minimum IoU to count as a match.

    Returns
    -------
    tp : np.ndarray, shape (N,)
        1 at index i if prediction i is a true positive, else 0.
    fp : np.ndarray, shape (N,)
        1 at index i if prediction i is a false positive, else 0.
    scores : np.ndarray, shape (N,)
        Sorted scores (descending) corresponding to tp/fp.
    """
    pred_scores = np.asarray(pred_scores, dtype=np.float64)
    sort_idx = np.argsort(-pred_scores)          # descending
    sorted_scores = pred_scores[sort_idx]
    sorted_pred_boxes = [pred_boxes[i] for i in sort_idx]

    n_preds = len(sorted_pred_boxes)
    tp = np.zeros(n_preds, dtype=np.float64)
    fp = np.zeros(n_preds, dtype=np.float64)
    matched_gt = set()

    for i, pb in enumerate(sorted_pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = iou_box(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_thresh and best_gt_idx >= 0:
            tp[i] = 1.0
            matched_gt.add(best_gt_idx)
        else:
            fp[i] = 1.0

    return tp, fp, sorted_scores


# ---------------------------------------------------------------------------
# 3. Average Precision from TP/FP arrays
# ---------------------------------------------------------------------------

def compute_ap(
    tp: np.ndarray,
    fp: np.ndarray,
    n_gt: int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Average Precision (AP) from cumulative TP and FP arrays.

    Two AP estimates are computed and both are returned:
    * **Area-Under-Curve (AUC)**: the integral of the precision-recall curve
      after monotone envelope (right-to-left max).
    * **11-point interpolation**: the mean precision sampled at recall levels
      [0.0, 0.1, …, 1.0] (PASCAL VOC 2007 style).

    The function returns the AUC AP as the primary value because it is more
    stable, but also exposes the 11-point value for reference.

    Parameters
    ----------
    tp : np.ndarray, shape (N,)
        Binary TP indicators, sorted by descending score.
    fp : np.ndarray, shape (N,)
        Binary FP indicators, sorted by descending score.
    n_gt : int
        Total number of ground-truth boxes for this class.

    Returns
    -------
    ap : float
        Area-under-curve AP (primary).
    precision_curve : np.ndarray
        Precision at each threshold.
    recall_curve : np.ndarray
        Recall at each threshold.

    Notes
    -----
    If ``n_gt == 0`` the function returns ``(0.0, empty, empty)`` and logs a
    warning — callers should not report this class in mAP averaging.
    """
    if n_gt == 0:
        logger.warning("compute_ap called with n_gt=0; returning 0.0")
        return 0.0, np.array([]), np.array([])

    if len(tp) == 0:
        return 0.0, np.array([0.0]), np.array([0.0])

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall_curve = tp_cum / float(n_gt)
    precision_curve = tp_cum / (tp_cum + fp_cum + 1e-9)

    # Prepend (0, 1) sentinel so the curve starts at recall=0
    rec = np.concatenate(([0.0], recall_curve, [recall_curve[-1]]))
    prec = np.concatenate(([1.0], precision_curve, [0.0]))

    # Monotone envelope (right-to-left max)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])

    # AUC AP
    ap_auc = float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))

    # 11-point interpolated AP (for reference)
    ap_11pt = 0.0
    for thr in np.linspace(0.0, 1.0, 11):
        p_at_r = prec[rec >= thr]
        ap_11pt += (p_at_r.max() if len(p_at_r) > 0 else 0.0)
    ap_11pt /= 11.0

    logger.debug("AP (AUC)=%.4f  AP (11-pt)=%.4f  n_gt=%d  n_pred=%d",
                 ap_auc, ap_11pt, n_gt, len(tp))

    return ap_auc, precision_curve, recall_curve


# ---------------------------------------------------------------------------
# 4. mAP over all classes
# ---------------------------------------------------------------------------

def compute_map(
    all_results_by_class: Dict[int, Tuple[np.ndarray, np.ndarray]],
    n_gt_by_class: Dict[int, int],
    iou_thresh: float = 0.5,
) -> Dict:
    """
    Compute mean Average Precision (mAP) over all classes.

    Parameters
    ----------
    all_results_by_class : dict
        ``{class_id: (tp_array, fp_array)}`` where arrays are already sorted
        by descending score (as returned by :func:`match_predictions`).
    n_gt_by_class : dict
        ``{class_id: n_gt}`` total GT count per class.
    iou_thresh : float
        The IoU threshold used (stored in the output for bookkeeping).

    Returns
    -------
    dict with keys:
        * ``mAP`` – scalar mean AP over classes with at least one GT box.
        * ``per_class_AP`` – ``{class_id: ap_float}``.
        * ``per_class_precision_curve`` – ``{class_id: np.ndarray}``.
        * ``per_class_recall_curve`` – ``{class_id: np.ndarray}``.
        * ``iou_thresh`` – echoed back for traceability.
    """
    per_class_ap: Dict[int, float] = {}
    per_class_prec: Dict[int, np.ndarray] = {}
    per_class_rec: Dict[int, np.ndarray] = {}

    ap_values = []
    for cls_id, (tp_arr, fp_arr) in all_results_by_class.items():
        n_gt = n_gt_by_class.get(cls_id, 0)
        ap, prec_c, rec_c = compute_ap(tp_arr, fp_arr, n_gt)
        per_class_ap[cls_id] = ap
        per_class_prec[cls_id] = prec_c
        per_class_rec[cls_id] = rec_c
        if n_gt > 0:
            ap_values.append(ap)

    # Classes present in GT but absent from predictions get AP=0
    for cls_id, n_gt in n_gt_by_class.items():
        if cls_id not in per_class_ap and n_gt > 0:
            per_class_ap[cls_id] = 0.0
            ap_values.append(0.0)

    mean_ap = float(np.mean(ap_values)) if ap_values else 0.0
    logger.info("mAP@%.2f = %.4f over %d classes", iou_thresh, mean_ap, len(ap_values))

    return {
        "mAP": mean_ap,
        "per_class_AP": per_class_ap,
        "per_class_precision_curve": per_class_prec,
        "per_class_recall_curve": per_class_rec,
        "iou_thresh": iou_thresh,
    }


# ---------------------------------------------------------------------------
# 5. Full metric computation for one model
# ---------------------------------------------------------------------------

def compute_metrics_for_model(
    gt_annotations: List[Dict],
    pred_annotations: List[Dict],
    iou_thresh: float = 0.5,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict:
    """
    Compute the full detection metric suite for a single model's predictions.

    Parameters
    ----------
    gt_annotations : list of dict
        Each entry: ``{image_id, class_id, box_xyxy}``
        where ``box_xyxy`` is ``[x1, y1, x2, y2]``.
    pred_annotations : list of dict
        Each entry: ``{image_id, class_id, score, box_xyxy}``.
    iou_thresh : float
        IoU threshold for TP/FP assignment.
    class_names : dict or None
        Optional ``{class_id: name}`` map used for human-readable output.

    Returns
    -------
    dict with keys:
        ``mAP``, ``per_class_AP``, ``precision``, ``recall``, ``f1``,
        ``mean_iou``, ``matched_ious``, ``n_gt``, ``n_pred``,
        ``n_tp``, ``n_fp``, ``n_fn``, ``iou_thresh``,
        ``per_class_precision_curve``, ``per_class_recall_curve``.
    """
    if not gt_annotations:
        raise ValueError("gt_annotations is empty — cannot evaluate.")

    # ------------------------------------------------------------------ index
    # GT: {(image_id, class_id): [box, …]}
    gt_index: Dict[Tuple, List] = defaultdict(list)
    n_gt_by_class: Dict[int, int] = defaultdict(int)
    for ann in gt_annotations:
        key = (ann["image_id"], ann["class_id"])
        gt_index[key].append(ann["box_xyxy"])
        n_gt_by_class[ann["class_id"]] += 1

    # Pred: {(image_id, class_id): [(score, box), …]}
    pred_index: Dict[Tuple, List] = defaultdict(list)
    for ann in pred_annotations:
        key = (ann["image_id"], ann["class_id"])
        pred_index[key].append((ann["score"], ann["box_xyxy"]))

    # -------------------------------------------------------------- matching
    # Accumulate per-class TP/FP lists across all images
    tp_by_class: Dict[int, List[np.ndarray]] = defaultdict(list)
    fp_by_class: Dict[int, List[np.ndarray]] = defaultdict(list)
    sc_by_class: Dict[int, List[np.ndarray]] = defaultdict(list)

    matched_ious: List[float] = []
    n_tp_total = 0
    n_fp_total = 0

    all_keys = set(gt_index.keys()) | set(pred_index.keys())
    for (img_id, cls_id) in all_keys:
        gt_boxes = gt_index.get((img_id, cls_id), [])
        preds = pred_index.get((img_id, cls_id), [])

        if not preds:
            # All GT are FN — no predictions to evaluate here
            continue

        scores_list = [p[0] for p in preds]
        boxes_list = [p[1] for p in preds]

        if not gt_boxes:
            # All predictions are FP
            n = len(preds)
            tp_by_class[cls_id].append(np.zeros(n))
            fp_by_class[cls_id].append(np.ones(n))
            sc_by_class[cls_id].append(np.array(sorted(scores_list, reverse=True)))
            n_fp_total += n
            continue

        tp, fp, sc = match_predictions(gt_boxes, boxes_list, scores_list, iou_thresh)
        tp_by_class[cls_id].append(tp)
        fp_by_class[cls_id].append(fp)
        sc_by_class[cls_id].append(sc)

        # Collect IoU of each TP for mean_iou computation
        sort_idx = np.argsort(-np.asarray(scores_list))
        matched_gt: set = set()
        for i in sort_idx:
            pb = boxes_list[i]
            best_iou = 0.0
            best_j = -1
            for j, gb in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = iou_box(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_thresh and best_j >= 0:
                matched_ious.append(best_iou)
                matched_gt.add(best_j)

        n_tp_total += int(tp.sum())
        n_fp_total += int(fp.sum())

    # --------------------------------- merge per-class results across images
    merged_by_class: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for cls_id in set(list(tp_by_class.keys()) + list(n_gt_by_class.keys())):
        if cls_id not in tp_by_class:
            merged_by_class[cls_id] = (np.array([]), np.array([]))
            continue
        tp_cat = np.concatenate(tp_by_class[cls_id])
        fp_cat = np.concatenate(fp_by_class[cls_id])
        sc_cat = np.concatenate(sc_by_class[cls_id])
        sort_idx = np.argsort(-sc_cat)
        merged_by_class[cls_id] = (tp_cat[sort_idx], fp_cat[sort_idx])

    # ------------------------------------------------------- mAP computation
    map_results = compute_map(merged_by_class, dict(n_gt_by_class), iou_thresh)

    # ----------------------------------------- global precision / recall / f1
    n_gt_total = sum(n_gt_by_class.values())
    n_fn_total = n_gt_total - n_tp_total
    precision = n_tp_total / (n_tp_total + n_fp_total + 1e-9)
    recall    = n_tp_total / (n_gt_total + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    mean_iou  = float(np.mean(matched_ious)) if matched_ious else 0.0

    # ------------------------------------------------ human-readable AP names
    per_class_ap_named: Dict = {}
    for cls_id, ap in map_results["per_class_AP"].items():
        if isinstance(class_names, list) and 0 <= cls_id < len(class_names):
            name = class_names[cls_id]
        elif isinstance(class_names, dict):
            name = class_names.get(cls_id, str(cls_id))
        else:
            name = str(cls_id)
        per_class_ap_named[name] = round(ap, 6)

    results = {
        "mAP": round(map_results["mAP"], 6),
        "per_class_AP": per_class_ap_named,
        "per_class_AP_by_id": map_results["per_class_AP"],
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "mean_iou": round(float(mean_iou), 6),
        "matched_ious": [round(v, 4) for v in matched_ious],
        "n_gt": n_gt_total,
        "n_pred": len(pred_annotations),
        "n_tp": n_tp_total,
        "n_fp": n_fp_total,
        "n_fn": n_fn_total,
        "iou_thresh": iou_thresh,
        "per_class_precision_curve": {
            (class_names[k] if (isinstance(class_names, list) and 0 <= k < len(class_names)) else (class_names.get(k, str(k)) if isinstance(class_names, dict) else str(k))): v.tolist()
            for k, v in map_results["per_class_precision_curve"].items()
        },
        "per_class_recall_curve": {
            (class_names[k] if (isinstance(class_names, list) and 0 <= k < len(class_names)) else (class_names.get(k, str(k)) if isinstance(class_names, dict) else str(k))): v.tolist()
            for k, v in map_results["per_class_recall_curve"].items()
        },
    }
    logger.info(
        "Metrics: mAP=%.4f P=%.4f R=%.4f F1=%.4f mIoU=%.4f "
        "TP=%d FP=%d FN=%d GT=%d Pred=%d",
        results["mAP"], results["precision"], results["recall"],
        results["f1"], results["mean_iou"],
        n_tp_total, n_fp_total, n_fn_total, n_gt_total, len(pred_annotations),
    )
    return results


# ---------------------------------------------------------------------------
# 6. Size category helper
# ---------------------------------------------------------------------------

def size_category(box_xyxy: List[float]) -> str:
    """
    Classify a bounding box into a COCO-like size category.

    Parameters
    ----------
    box_xyxy : array-like of length 4
        Box in ``[x1, y1, x2, y2]`` format.

    Returns
    -------
    str
        One of ``'small'``, ``'medium'``, or ``'large'``.

    Notes
    -----
    * **small** : w < 32 *and* h < 32
    * **medium**: (w < 96 *and* h < 96) but not small
    * **large** : everything else
    """
    w = float(box_xyxy[2]) - float(box_xyxy[0])
    h = float(box_xyxy[3]) - float(box_xyxy[1])
    if w < 32 and h < 32:
        return "small"
    if w < 96 and h < 96:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# 7. Size-stratified metrics
# ---------------------------------------------------------------------------

def compute_size_metrics(
    gt_annotations: List[Dict],
    pred_annotations: List[Dict],
    iou_thresh: float = 0.5,
) -> Dict:
    """
    Compute precision, recall, and AP separately for small / medium / large
    objects.

    Object size is determined by the GT box dimensions via
    :func:`size_category`.

    Parameters
    ----------
    gt_annotations : list of dict
        ``{image_id, class_id, box_xyxy}``
    pred_annotations : list of dict
        ``{image_id, class_id, score, box_xyxy}``
    iou_thresh : float
        IoU threshold for matching.

    Returns
    -------
    dict with keys ``'small'``, ``'medium'``, ``'large'``.  Each sub-dict
    contains: ``ap``, ``precision``, ``recall``, ``f1``, ``n_gt``,
    ``n_pred``, ``n_tp``, ``n_fp``, ``n_fn``.  When ``n_gt == 0`` the metrics
    are replaced by a ``'note'`` key so that zeros are never silently reported.
    """
    sizes = ("small", "medium", "large")

    # Partition GT by size
    gt_by_size: Dict[str, List[Dict]] = {s: [] for s in sizes}
    gt_image_class_size: Dict[Tuple, str] = {}   # (img_id, cls_id, gt_idx) -> size

    # Build a per-(image, class) gt list with size tags
    gt_index: Dict[Tuple, List[Tuple[int, str]]] = defaultdict(list)
    for ann in gt_annotations:
        key = (ann["image_id"], ann["class_id"])
        sz = size_category(ann["box_xyxy"])
        idx = len(gt_index[key])
        gt_index[key].append((idx, sz, ann["box_xyxy"]))
        gt_by_size[sz].append(ann)

    # Build pred index
    pred_index: Dict[Tuple, List] = defaultdict(list)
    for ann in pred_annotations:
        key = (ann["image_id"], ann["class_id"])
        pred_index[key].append(ann)

    results: Dict[str, Dict] = {}

    for size in sizes:
        n_gt = len(gt_by_size[size])
        if n_gt == 0:
            results[size] = {
                "ap": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "n_gt": 0,
                "n_pred": "N/A",
                "n_tp": "N/A",
                "n_fp": "N/A",
                "n_fn": "N/A",
                "note": (
                    f"No '{size}' ground-truth objects found in the test set. "
                    "Metrics cannot be computed. This is NOT a model failure — "
                    "the dataset simply does not contain objects of this size."
                ),
            }
            logger.warning(
                "Size stratum '%s': 0 GT objects — metrics set to N/A (not 0.0).", size
            )
            continue

        # For each (image, class) key, filter GT boxes to those of `size`
        tp_all: List[np.ndarray] = []
        fp_all: List[np.ndarray] = []
        sc_all: List[np.ndarray] = []
        n_pred_size = 0

        all_img_cls_keys = set(gt_index.keys()) | set(pred_index.keys())
        for key in all_img_cls_keys:
            gt_entries = gt_index.get(key, [])
            gt_boxes_size = [e[2] for e in gt_entries if e[1] == size]
            if not gt_boxes_size and key not in pred_index:
                continue

            preds = pred_index.get(key, [])
            if not preds:
                continue

            scores_list = [p["score"] for p in preds]
            boxes_list  = [p["box_xyxy"] for p in preds]
            n_pred_size += len(preds)

            if not gt_boxes_size:
                n = len(preds)
                tp_all.append(np.zeros(n))
                fp_all.append(np.ones(n))
                sc_all.append(np.array(sorted(scores_list, reverse=True)))
                continue

            tp, fp, sc = match_predictions(
                gt_boxes_size, boxes_list, scores_list, iou_thresh
            )
            tp_all.append(tp)
            fp_all.append(fp)
            sc_all.append(sc)

        if tp_all:
            tp_cat = np.concatenate(tp_all)
            fp_cat = np.concatenate(fp_all)
            sc_cat = np.concatenate(sc_all)
            sort_idx = np.argsort(-sc_cat)
            tp_sorted = tp_cat[sort_idx]
            fp_sorted = fp_cat[sort_idx]
        else:
            tp_sorted = np.array([])
            fp_sorted = np.array([])

        ap, _, _ = compute_ap(tp_sorted, fp_sorted, n_gt)
        n_tp = int(tp_sorted.sum()) if len(tp_sorted) else 0
        n_fp = int(fp_sorted.sum()) if len(fp_sorted) else 0
        n_fn = n_gt - n_tp
        precision = n_tp / (n_tp + n_fp + 1e-9)
        recall    = n_tp / (n_gt + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)

        results[size] = {
            "ap": round(float(ap), 6),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "n_gt": n_gt,
            "n_pred": n_pred_size,
            "n_tp": n_tp,
            "n_fp": n_fp,
            "n_fn": n_fn,
        }
        logger.info(
            "Size '%s': AP=%.4f P=%.4f R=%.4f n_gt=%d n_pred=%d",
            size, ap, precision, recall, n_gt, n_pred_size,
        )

    return results
