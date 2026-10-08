# DL_Project — VisDrone Aerial Object Detection

This folder contains the complete, self-contained codebase for aerial object detection and auto-annotation on the **VisDrone-DET** benchmark.

---

## 📁 Package Contents

| File / Folder | Purpose |
| :--- | :--- |
| **`all_in_one_pipeline.py`** | **Master all-in-one script** supporting Dataset Conversion, Training, Inference, Auto-Annotation, and Benchmark Evaluation in a single file. |
| **`infer_yolo.py`** | Fast YOLOv8s inference on images, folders, or MP4 videos using trained weights. |
| **`auto_annotate.py`** | Automated AI-assisted annotation engine generating normalized YOLO `.txt` labels and Roboflow upload packages. |
| **`evaluate_benchmark.py`**| Scientific evaluator computing PR-curve mAP@0.50 and mAP@0.5:0.95 against genuine ground truth. |
| **`infer_faster_rcnn.py`** | Faster R-CNN (ResNet50-FPN) inference with COCO-to-VisDrone class mapping and CUDA acceleration. |
| **`infer_mask_rcnn.py`**   | Mask R-CNN (ResNet50-FPN) instance segmentation overlay on aerial images. |
| **`infer_sam.py`**         | YOLO + SAM (Segment Anything Model) bounding box-prompted segmentation pipeline. |
| **`weights/best.pt`**      | **Trained YOLOv8s Model Weights** (22.5 MB, trained 50 epochs on 6,471 full VisDrone images, **37.40% mAP@0.50**). |
| **`configs/`**             | Contains `visdrone.yaml` and `classes.txt` definitions. |
| **`requirements.txt`**     | Dependencies list. |

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```powershell
pip install -r requirements.txt
```

---

### 2. Run Inference (Images & Video)
To detect objects on any image or MP4 video:
```powershell
# Using the master script:
python all_in_one_pipeline.py predict --weights weights/best.pt --source "path/to/image_or_video.mp4"

# Or using dedicated YOLO script:
python infer_yolo.py --source "path/to/image.jpg" --save-txt
```

---

### 3. AI Auto-Annotation (Instant YOLO Labels + Roboflow Package)
To automatically annotate a folder of unlabeled drone images:
```powershell
python auto_annotate.py --source "path/to/unlabeled_images" --conf 0.25 --save-visual --create-roboflow-zip
```
This produces:
- `labels/`: Normalized YOLO `.txt` annotation files.
- `visual_previews/`: Image previews with detection boxes.
- `auto_annotated_roboflow.zip`: One-click upload package for Roboflow.

---

### 4. Evaluate Benchmark Performance
```powershell
python evaluate_benchmark.py --weights weights/best.pt --data configs/visdrone.yaml --split val
```

---

### 5. Other Models
```powershell
# Faster R-CNN
python infer_faster_rcnn.py --source "path/to/image.jpg" --conf 0.50

# Mask R-CNN Instance Segmentation
python infer_mask_rcnn.py --source "path/to/image.jpg" --conf 0.50

# YOLO + SAM Hybrid Segmentation
python infer_sam.py --source "path/to/image.jpg" --yolo-weights weights/best.pt
```

---

## 📊 Trained Model Benchmark Summary (50 Epochs)

* **Dataset:** Full VisDrone2019-DET (6,471 Train Images / 548 Val Images)
* **Overall Precision:** 48.0%
* **Overall Recall:** 36.3%
* **Overall mAP @ 0.50:** **37.40%** (Car: 78.3%, Bus: 52.3%, Motor: 41.9%, Pedestrian: 41.1%, Van: 41.4%)
* **Latency:** **4.5 ms** (~140 FPS on RTX 4060 GPU)
