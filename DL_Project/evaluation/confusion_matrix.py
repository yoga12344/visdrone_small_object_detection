"""
confusion_matrix.py
===================
Generates, normalizes, plots, and interprets class-wise confusion matrices
using REAL ground-truth VisDrone annotations.

Handles unmatched GTs as Background False Negatives (FN),
and unmatched Predictions as Background False Positives (FP).
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def build_confusion_matrix(
    gt_annotations: List[Dict],
    pred_annotations: List[Dict],
    class_names: List[str],
    iou_thresh: float = 0.50
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a class-wise confusion matrix comparing real GT against predictions.
    
    Matrix layout:
      Shape: (num_classes + 1, num_classes + 1)
      Rows: Ground Truth class (0..num_classes-1), last row = Background (FP predictions)
      Cols: Predicted class (0..num_classes-1), last col = Background (FN missed GT)
    """
    num_classes = len(class_names)
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    
    # Group by image_id
    gt_by_img = {}
    for g in gt_annotations:
        img_id = g['image_id']
        gt_by_img.setdefault(img_id, []).append(g)
        
    pred_by_img = {}
    for p in pred_annotations:
        img_id = p['image_id']
        pred_by_img.setdefault(img_id, []).append(p)
        
    all_imgs = set(gt_by_img.keys()).union(set(pred_by_img.keys()))
    
    for img_id in all_imgs:
        gts = gt_by_img.get(img_id, [])
        preds = pred_by_img.get(img_id, [])
        
        # Sort preds by score descending
        preds_sorted = sorted(preds, key=lambda x: x.get('score', 1.0), reverse=True)
        matched_gt = set()
        
        for p in preds_sorted:
            p_box = np.array(p['box_xyxy'])
            p_cls = p['class_id']
            
            best_iou = 0.0
            best_gt_idx = -1
            
            for g_idx, g in enumerate(gts):
                if g_idx in matched_gt:
                    continue
                g_box = np.array(g['box_xyxy'])
                
                # Compute IoU
                xA = max(p_box[0], g_box[0])
                yA = max(p_box[1], g_box[1])
                xB = min(p_box[2], g_box[2])
                yB = min(p_box[3], g_box[3])
                inter = max(0.0, xB - xA) * max(0.0, yB - yA)
                areaA = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
                areaB = (g_box[2] - g_box[0]) * (g_box[3] - g_box[1])
                union = areaA + areaB - inter
                iou = inter / (union + 1e-6)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
            
            if best_iou >= iou_thresh and best_gt_idx >= 0:
                # Match found!
                g_cls = gts[best_gt_idx]['class_id']
                if 0 <= g_cls < num_classes and 0 <= p_cls < num_classes:
                    matrix[g_cls, p_cls] += 1
                matched_gt.add(best_gt_idx)
            else:
                # Prediction with no matching GT -> False Positive (from Background)
                if 0 <= p_cls < num_classes:
                    matrix[num_classes, p_cls] += 1
                    
        # All unmatched GTs are False Negatives (predicted as Background)
        for g_idx, g in enumerate(gts):
            if g_idx not in matched_gt:
                g_cls = g['class_id']
                if 0 <= g_cls < num_classes:
                    matrix[g_cls, num_classes] += 1
                    
    names_with_bg = list(class_names) + ['background']
    return matrix, names_with_bg


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names_with_bg: List[str],
    title: str,
    output_path: str,
    normalize: bool = True
):
    """Plot and save confusion matrix as high-resolution PNG."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        norm_matrix = np.divide(
            matrix.astype(float),
            row_sums,
            out=np.zeros_like(matrix, dtype=float),
            where=row_sums > 0
        )
        display_mat = norm_matrix
    else:
        display_mat = matrix.astype(float)
        
    n = len(class_names_with_bg)
    fig, ax = plt.subplots(figsize=(max(8, n), max(7, n - 1)))
    cax = ax.matshow(display_mat, cmap='Blues')
    fig.colorbar(cax)
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names_with_bg, rotation=45, ha='left', fontsize=9)
    ax.set_yticklabels(class_names_with_bg, fontsize=9)
    
    ax.set_xlabel('Predicted Class', fontsize=11, labelpad=10)
    ax.set_ylabel('True Class', fontsize=11)
    ax.set_title(title, fontsize=13, pad=20)
    
    # Text annotations in cells
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val > 0:
                if normalize:
                    text = f"{display_mat[i, j]:.2f}\n({val})"
                else:
                    text = f"{val}"
                color = "white" if display_mat[i, j] > 0.5 else "black"
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=7)
                
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved confusion matrix plot: {output_path}")


def interpret_confusion_matrix(matrix: np.ndarray, class_names: List[str]) -> str:
    """
    Produce a rigorous written technical analysis of the confusion matrix.
    """
    num_classes = len(class_names)
    lines = []
    lines.append("=" * 60)
    lines.append("  CONFUSION MATRIX TECHNICAL INTERPRETATION")
    lines.append("=" * 60)
    
    recalls = {}
    precisions = {}
    
    for i, cname in enumerate(class_names):
        tp = matrix[i, i]
        total_gt = matrix[i, :].sum()
        total_pred = matrix[:, i].sum()
        
        recall = tp / total_gt if total_gt > 0 else 0.0
        precision = tp / total_pred if total_pred > 0 else 0.0
        
        recalls[cname] = (recall, tp, total_gt)
        precisions[cname] = (precision, tp, total_pred)
        
    sorted_recall = sorted(recalls.items(), key=lambda x: x[1][0], reverse=True)
    
    lines.append("\n1. Class Sensitivity / Recall (Sorted Best -> Worst):")
    for cname, (r, tp, tot) in sorted_recall:
        lines.append(f"   - {cname:<16}: Recall = {r*100:5.1f}% ({tp}/{tot} GT objects detected)")
        
    lines.append("\n2. Background and False Alarm Analysis:")
    total_fp = matrix[num_classes, :num_classes].sum()
    total_fn = matrix[:num_classes, num_classes].sum()
    lines.append(f"   - False Positives (Detections matching no GT): {total_fp}")
    lines.append(f"   - False Negatives (GT objects missed completely): {total_fn}")
    
    lines.append("\n3. Notable Inter-Class Confusions:")
    found_confusion = False
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and matrix[i, j] > 0:
                found_confusion = True
                lines.append(
                    f"   - True '{class_names[i]}' misclassified as '{class_names[j]}': {matrix[i, j]} instances"
                )
    if not found_confusion:
        lines.append("   - No inter-class confusion observed among matched detections.")
        
    lines.append("=" * 60)
    return "\n".join(lines)
