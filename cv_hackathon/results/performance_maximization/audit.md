# Phase A: Complete Current-Result Audit & Convergence Analysis
**Date:** October 5, 2026  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8.59 GB VRAM, CUDA 12.1)  
**Dataset Partitions:** Train: 422 images (29,428 targets) | Val: 106 images (8,036 targets) | Test: 20 images (1,295 targets, LOCKED)

---

## 1. Convergence & Overfitting Audit

An analysis of `results/final_optimization/training/yolov8s_100ep_640/results.csv` reveals the following training dynamics:
- **Total Epochs Trained:** 100 epochs.
- **Initial Metrics (Epoch 1):** Train box loss: 2.036, Val box loss: 1.941, Val mAP50: 0.0371.
- **Peak Performance:** Reached at **Epoch 87** with a peak validation mAP50 of **0.3355** and val box loss of **1.3994**.
- **Minimum Validation Loss:** Reached at **Epoch 82** (val box loss: 1.3990).
- **Post-Peak Dynamics (Epochs 88–100):** Validation box loss slowly crept up from 1.399 to 1.448 while train box loss continued dropping to 1.353, and validation mAP fluctuated between 0.302 and 0.307.
- **Convergence Conclusion:** The model converged between epochs 80 and 90. Extending standard 640px training to 200 epochs without higher resolution or spatial preservation would yield diminishing returns and risk further mild overfitting.

---

## 2. Model Undertraining Audit

- **YOLOv8s (100ep):** Fully converged (best weights saved at epoch 87 with 0.334 validation mAP).
- **YOLOv8m (Phase 3 run, 25ep):** Severely undertrained at 25 epochs. Despite undertraining, it attained 0.2017 val mAP at 640px. Full training (up to 150–200 epochs) will test whether larger parameter capacity (25.86M params vs 11.14M) enables richer high-frequency feature representations for tiny objects.
- **YOLOv8n / YOLO11n (25ep):** Constrained by backbone channel capacity (2.59M–3.01M params), yielding poor small-object AP (<0.013).

---

## 3. Scale-Stratified and Class Bottleneck Analysis

From validation metrics on the 8,036 authentic ground truth targets:
- **Scale Distribution:**
  - Small objects (<32×32 px): 4,312 targets (**53.7% of all annotations**)
  - Medium objects (32–96 px): 3,281 targets (**40.8% of all annotations**)
  - Large objects (>96 px): 443 targets (**5.5% of all annotations**)
- **Small-Object Recall Bottleneck:**
  - Native 640px achieves only **23.01% recall on small objects** (AP = 0.0470).
  - Native 1024px elevates small-object recall to **37.24%** (AP = 0.0998).
  - Tiled 2×2 elevates small-object recall to **43.46%** (AP = 0.1094).
  - Tiled 3×2 elevates small-object recall to **48.21%** (AP = 0.1262).
  - **Conclusion:** More than 50% of the false negatives in aerial imagery arise directly from downsampling small objects below the detector's anchor/stride resolution.

- **Class-Level Vulnerability:**
  - **Easiest Class:** `car` (val mAP50 = 0.759, recall = 0.766).
  - **Hardest Small Classes:**
    - `bicycle`: val mAP50 = 0.0782, recall = 0.0702 (extreme aspect ratio, thin frame, tiny footprint).
    - `awning-tricycle`: val mAP50 = 0.0784, recall = 0.1200 (rare class, distinct silhouette often occluded).
    - `people` vs `pedestrian`: High inter-class confusion due to semantic ambiguity in VisDrone labeling.

---

## 4. Post-Processing & Threshold Optimization Opportunities

- **Confidence Threshold:** Currently fixed at `conf=0.25`. VisDrone has thousands of small objects with marginal visibility where lower confidence thresholds (0.15–0.20) may dramatically increase recall without catastrophic precision collapse.
- **NMS Threshold:** Currently fixed at `iou=0.45`. Dense crowds and vehicle queues in aerial imagery frequently exhibit high physical bounding box overlap; tuning NMS between 0.50 and 0.65 will prevent suppression of closely packed valid objects.

---

## 5. Prioritized Execution Plan for Performance Maximization

1. **Phase G & H:** Conduct systematic sweeps of Confidence Threshold ($[0.05 \dots 0.50]$) and NMS IoU ($[0.45 \dots 0.70]$) on the validation set using the converged YOLOv8s and Tiled models.
2. **Phase C:** Train YOLOv8m (high-capacity) with early stopping to test capacity limits.
3. **Phase D:** Evaluate high-resolution training (768px, 1024px) against Native and Tiled representations.
4. **Phase I & J:** Refine advanced and adaptive tiling parameters (overlap optimization, adaptive density triggering).
5. **Phase O & P:** Freeze the winning configuration in `FINAL_CONFIGURATION.yaml` based strictly on validation rankings.
6. **Phase Q:** Execute single, leakage-free evaluation on the locked 20-image test set.
