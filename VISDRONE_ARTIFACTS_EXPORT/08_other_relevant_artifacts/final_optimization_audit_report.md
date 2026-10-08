# Phase 0: Full Project Audit Report
**Date:** October 5, 2026  
**Hardware Platform:** NVIDIA GeForce RTX 4060 Laptop GPU (8.59 GB VRAM, Driver 595.71, CUDA 12.1)  
**Software Platform:** Python 3.11, PyTorch 2.3.0+cu121, TorchVision 0.18.0+cu121, Ultralytics 8.4.5, OpenCV 4.11.0  
**Research Project:** *Comparative Analysis of Advanced Object Detection and Segmentation Models for Aerial Drone Imagery using VisDrone2019-DET*

---

## 1. Executive Summary & Verification of Scientific Integrity

A comprehensive audit was performed across the dataset, annotations, split partitions, model checkpoints, inference pipelines, and evaluation routines. The project maintains rigorous scientific boundaries:
1. **Zero Pseudo-Ground-Truth in Final Evaluation:** Ground-truth annotations are parsed strictly from the authentic VisDrone annotations (`VisDrone2019-DET-val/annotations`).
2. **Locked Test Partition:** An immutable, disjoint test set of 20 images (`data/processed/splits/test_ids.txt`) containing 1,295 valid ground-truth targets is maintained across all experiments.
3. **Hardware & Multiprocessing Stability:** Under Windows OS, PyTorch `DataLoader` with `workers > 0` triggers memory map and IPC errors (`error code: 1455`). All scripts are verified and configured to execute deterministically with `workers=0`.

---

## 2. Dataset Split Verification

The dataset partitioning adheres strictly to the benchmark protocol:
- **Training Set (`train_ids.txt`):** 422 images (29,428 foreground bounding boxes)
- **Validation Set (`val_ids.txt`):** 106 images (8,036 foreground bounding boxes)
- **Test Set (`test_ids.txt`):** 20 images (1,295 foreground bounding boxes, **LOCKED**)
- **Total Images:** 548 images (38,791 valid targets, 1,378 category-0 ignored annotations appropriately filtered)
- **Set Disjointness:**
  - $\text{Train} \cap \text{Val} = \emptyset$ (0 overlap)
  - $\text{Train} \cap \text{Test} = \emptyset$ (0 overlap)
  - $\text{Val} \cap \text{Test} = \emptyset$ (0 overlap)

All class mappings conform to the 10 VisDrone categories (0 to 9): `pedestrian`, `people`, `bicycle`, `car`, `van`, `truck`, `tricycle`, `awning-tricycle`, `bus`, `motor`.

---

## 3. Existing Baseline Models & Checkpoints Audit

| Model | Checkpoint Path | Parameters / Size | Locked Test mAP@0.5 | Small-Object AP | Test FPS (RTX 4060) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8s** | `results/trained/yolov8s/weights/best.pt` | 11.14M / 22.50 MB | 0.1322 | 0.0028 | 51.20 | 25 epochs initial run, strong vehicle precision, high miss rate on tiny objects |
| **Faster R-CNN** | `results/trained/faster_rcnn/weights/best.pt` | 41.35M / 165.89 MB | 0.2749 | 0.0365 | 11.89 | ResNet-50-FPN, 2-stage anchor density provides better small-scale localization |
| **Mask R-CNN** | `results/trained/mask_rcnn/weights/best.pt` | 43.92M / 176.40 MB | 0.2587 | 0.0401 | 0.88 | High mask computation overhead, competitive small-object detection |
| **RF-DETR Nano** | `results/trained/rf_detr_nano/weights/best.pth` | 30.2M / 121.03 MB | 0.1995 | 0.0063 | 23.47 | Deformable attention DETR architecture |

---

## 4. Evaluation Engine and Metric Methodology Audit

- **IoU Calculation:** Hungarian greedy assignment sorted by descending score matching authentic VisDrone bounding boxes.
- **Scale Stratification:**
  - **Small:** $\text{area} < 32 \times 32\text{ px}$ ($\text{side} < 32\text{ px}$)
  - **Medium:** $32 \times 32 \le \text{area} < 96 \times 96\text{ px}$
  - **Large:** $\text{area} \ge 96 \times 96\text{ px}$
- **Latency Measurement:** Synchronized CUDA events (`torch.cuda.Event`) with proper warmup cycles (minimum 3 warmups) to prevent CPU dispatch overhead and initialization jitter.

---

## 5. Planned Final Optimization Roadmap

1. **Phase 1:** Full 100-Epoch Training of YOLOv8s with early stopping, AMP enabled, learning rate warmups, and evaluated exclusively against validation metrics.
2. **Phase 2 & 3:** System capacity and input resolution ablation (640, 768, 1024; v8n, v8s, v8m, yolo11).
3. **Phase 4:** Controlled aerial augmentation ablation.
4. **Phase 5 & 6:** Spatial tiling and high-resolution inference validation.
5. **Phase 7-12:** Two-stage hybrid evaluation, qualitative generation, single final pass on locked test set, and compilation into `FINAL_OPTIMIZATION_REPORT.md`.
