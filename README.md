# VisDrone Aerial Small Object Detection & Instance Segmentation

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3%2BCUDA12.1-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-green.svg)](https://github.com/ultralytics/ultralytics)
[![VisDrone Benchmark](https://img.shields.io/badge/Dataset-VisDrone2019--DET-orange.svg)](https://github.com/VisDrone/VisDrone-Dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end computer vision research framework and production deployment suite for **tiny aerial object detection**, **instance segmentation**, and **automated annotation** on the **VisDrone-DET** benchmark. 

Includes trained weights, multi-model benchmarks (YOLOv8, Faster R-CNN, Mask R-CNN, YOLO+SAM, RF-DETR), resolution & tiling ablations (SAHI 3×2), and an AI auto-annotation engine.

---

## 📌 Key Highlights & Results

* **Full 50-Epoch Convergence:** Trained on all 6,471 images (343,204 annotations) of VisDrone2019-DET on an NVIDIA GeForce RTX 4060 GPU.
* **Accuracy Breakthrough:** Lifted mAP@0.50 from **14.2%** (untrained baseline) to **37.40%** (+163% gain).
* **Real-Time Edge Throughput:** Runs at **4.5 ms per frame (~140+ FPS)** on RTX 4060 GPU at 640×640.
* **High-Accuracy Tiling (SAHI):** Spatial preservation via 3×2 slicing yields **29.56% mAP** and **6.01% Small-Object AP** on tiny objects ($< 32 \times 32$ pixels).
* **AI-Assisted Auto-Annotation:** Automatically labels raw drone footage into normalized YOLO `.txt` files in **144 ms per image**.

---

## 📊 Benchmark Results

### 1. 50-Epoch YOLOv8s Performance on Full VisDrone

| Metric | Epoch 1 Baseline | **Final Result (Epoch 50 / Best)** | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **mAP @ 0.50** | 14.21% | **37.40%** | **+163%** |
| **mAP @ 0.50:0.95** | 8.15% | **21.90%** | **+168%** |
| **Precision** | 28.83% | **48.00%** | **+19.17%** |
| **Recall** | 17.35% | **36.30%** | **+18.95%** |
| **Train Box Loss** | 1.818 | **1.311** | **-27.9% reduction** |
| **Train Class Loss** | 2.767 | **0.844** | **-69.5% reduction** |
| **Inference Speed** | 4.5 ms | **4.5 ms** | **~140+ FPS on RTX 4060** |

### 2. Per-Class Detection Breakdown (Val Split: 548 Images, 38,759 Objects)

| Class | GT Count | Precision | Recall | **mAP @ 0.50** | **mAP @ 0.5:0.95** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Car** | 14,064 | 67.0% | 77.1% | **78.3%** | **54.3%** |
| **Bus** | 251 | 65.5% | 44.6% | **52.3%** | **37.0%** |
| **Motor (Motorcycle)** | 4,886 | 50.7% | 41.3% | **41.9%** | **17.5%** |
| **Van** | 1,975 | 45.8% | 42.1% | **41.4%** | **28.9%** |
| **Pedestrian** | 8,844 | 49.8% | 39.8% | **41.1%** | **18.3%** |
| **Truck** | 750 | 50.0% | 34.7% | **35.6%** | **23.4%** |
| **People (Gathering)** | 5,125 | 51.2% | 29.3% | **31.9%** | **11.8%** |
| **Tricycle** | 1,045 | 41.2% | 24.3% | **25.1%** | **13.8%** |
| **Awning-Tricycle** | 532 | 32.5% | 15.0% | **14.3%** | **8.8%** |
| **Bicycle** | 1,287 | 26.8% | 15.2% | **12.6%** | **5.1%** |
| **ALL CLASSES** | **38,759** | **48.0%** | **36.3%** | **37.4%** | **21.9%** |

### 3. Multi-Model Architecture Comparison

| Model Architecture | Paradigm | Precision | Recall | **mAP @ 0.50** | Latency | FPS | Model Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8s (Full Trained)** | Anchor-Free Single-Stage | **48.0%** | **36.3%** | **37.40%** | **4.5 ms** | **140.8** | **22.5 MB** |
| **Faster R-CNN** | Two-Stage (ResNet-50-FPN) | 42.1% | 29.5% | 27.49% | 84.1 ms | 11.9 | 158.0 MB |
| **Mask R-CNN** | Instance Segmentation | 39.4% | 26.8% | 24.81% | 98.6 ms | 10.1 | 168.0 MB |
| **YOLOv8 + SAM** | Detector-Prompted Foundation Masking | 48.0% | 36.3% | 37.40% | 450.0 ms | 2.2 | 2.5 GB |
| **RF-DETR (Nano)** | Transformer-Based Query Detector | 34.2% | 23.1% | 21.05% | 31.2 ms | 32.0 | 42.0 MB |

---

## 📁 Repository Structure

```
visdrone_small_object_detection/
├── full_code_submission/         # Clean, standalone submission package
│   ├── all_in_one_pipeline.py    # Master pipeline (Train, Predict, Auto-Annotate, Eval)
│   ├── infer_yolo.py             # High-speed YOLOv8s inference script
│   ├── auto_annotate.py          # AI-assisted annotation tool (labels + Roboflow export)
│   ├── evaluate_benchmark.py     # Scientific evaluation engine (PR-curve mAP)
│   ├── infer_faster_rcnn.py      # Faster R-CNN with VisDrone class mapping
│   ├── infer_mask_rcnn.py        # Mask R-CNN instance segmentation
│   ├── infer_sam.py              # YOLO + SAM hybrid segmentation pipeline
│   ├── requirements.txt          # Python dependencies
│   ├── README.md                 # Package guide
│   ├── configs/                  # Dataset configs (visdrone.yaml, classes.txt)
│   └── weights/
│       └── best.pt               # Trained YOLOv8s weights (22.5 MB, 37.40% mAP)
│
├── VISDRONE_ARTIFACTS_EXPORT/    # Complete audited experiment outputs (210 files)
│   ├── 01_training_outputs/      # Training curves, PR/F1 plots, confusion matrices
│   ├── 02_benchmark_figures/     # Publication graphs (mAP, FPS, Pareto, Ablation)
│   ├── 03_qualitative_predictions/# Multi-model comparison grids & scene zooms
│   ├── 04_failure_analysis/      # Failure taxonomy & dense crowd cluster zooms
│   ├── 05_sam_and_hybrid/        # SAM masks, hybrid flowchart, quantitative data
│   ├── 06_results_tables/        # CSV, JSON, and Markdown ablation tables
│   ├── 07_experiment_configs/    # Hyperparameter configs for reproducibility
│   ├── 08_other_relevant_artifacts/ # Audit reports & model weight manifests
│   ├── ARTIFACT_INVENTORY.md     # Detailed markdown catalog of all artifacts
│   └── ARTIFACT_INVENTORY.csv    # Machine-readable inventory of all outputs
│
├── DL_Project/                    # Full research codebase & experimental testbed
│   ├── configs/                  # Benchmark configurations
│   ├── data/                     # Annotation parsers, split manifests & packages
│   ├── evaluation/               # Metric modules & Hungarian matching
│   ├── inference/                # Model inferencers
│   ├── models/                   # Architecture wrappers
│   ├── reports/                  # Detailed experiment notes & Roboflow guides
│   ├── results/                  # Experimental results & plots
│   └── scripts/                  # Training and evaluation runner scripts
│
├── dataset/                      # Dataset splits and sample aerial images
├── AUDIT.md                      # Comprehensive codebase audit & bug remediation log
└── README.md                     # This file
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/yoga12344/visdrone_small_object_detection.git
cd visdrone_small_object_detection
pip install -r full_code_submission/requirements.txt
```

### 2. Run Inference on Images or Videos
Run detection using the trained VisDrone model weights:
```bash
# Using the master all-in-one pipeline:
python full_code_submission/all_in_one_pipeline.py predict \
  --weights full_code_submission/weights/best.pt \
  --source "path/to/image_or_video.mp4"

# Using the dedicated YOLO inference script:
python full_code_submission/infer_yolo.py \
  --source "path/to/image.jpg" \
  --save-txt
```

### 3. AI-Assisted Auto-Annotation
Automatically annotate raw unlabeled drone imagery into standard YOLO `.txt` labels and export a Roboflow-ready ZIP package:
```bash
python full_code_submission/auto_annotate.py \
  --source "path/to/unlabeled_images" \
  --conf 0.25 \
  --save-visual \
  --create-roboflow-zip \
  --output-dir "results/auto_annotated"
```

### 4. Evaluate Benchmark Performance
Evaluate on the validation or test partition with true Precision-Recall curve mAP:
```bash
python full_code_submission/evaluate_benchmark.py \
  --weights full_code_submission/weights/best.pt \
  --data full_code_submission/configs/visdrone.yaml \
  --split val
```

### 5. Multi-Model Baselines
```bash
# Faster R-CNN (ResNet50-FPN):
python full_code_submission/infer_faster_rcnn.py --source "path/to/image.jpg" --conf 0.50

# Mask R-CNN Instance Segmentation:
python full_code_submission/infer_mask_rcnn.py --source "path/to/image.jpg" --conf 0.50

# YOLO + SAM Hybrid Masking:
python full_code_submission/infer_sam.py --source "path/to/image.jpg"
```

---

## 📈 Key Visual Artifacts

All figures and plots are tracked in [`VISDRONE_ARTIFACTS_EXPORT/`](VISDRONE_ARTIFACTS_EXPORT/):
* 📈 **50-Epoch Loss & Convergence:** `VISDRONE_ARTIFACTS_EXPORT/01_training_outputs/yolov8s_full_50ep_results.png`
* 🎯 **10-Class Normalized Confusion Matrix:** `VISDRONE_ARTIFACTS_EXPORT/01_training_outputs/yolov8s_full_50ep_confusion_matrix_normalized.png`
* 📊 **Precision-Recall Curve:** `VISDRONE_ARTIFACTS_EXPORT/01_training_outputs/yolov8s_full_50ep_BoxPR_curve.png`
* ⚡ **FPS Throughput Comparison:** `VISDRONE_ARTIFACTS_EXPORT/02_benchmark_figures/fig05_fps_comparison.png`
* ⚖️ **Pareto Frontier (Accuracy vs FPS):** `VISDRONE_ARTIFACTS_EXPORT/02_benchmark_figures/fig09_accuracy_vs_fps.png`
* 🔍 **Resolution vs Small Object AP Ablation:** `VISDRONE_ARTIFACTS_EXPORT/02_benchmark_figures/fig12_resolution_vs_small_ap.png`
* 🖼️ **Multi-Model Qualitative Comparison Grid:** `VISDRONE_ARTIFACTS_EXPORT/03_qualitative_predictions/fig16_same_image_qualitative.png`
* ⚠️ **Empirical Failure Analysis Grid:** `VISDRONE_ARTIFACTS_EXPORT/04_failure_analysis/fig17_failure_analysis.png`
* 🎨 **YOLO + SAM Instance Masks:** `VISDRONE_ARTIFACTS_EXPORT/05_sam_and_hybrid/fig18_yolo_sam_qualitative.png`

---

## 📜 License & Citation

This project is licensed under the MIT License. If you use this work in your research, please cite:
```bibtex
@misc{visdrone_small_object_detection_2026,
  author = {Yoga},
  title = {VisDrone Aerial Small Object Detection & Instance Segmentation Benchmark},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/yoga12344/visdrone_small_object_detection}}
}
```
