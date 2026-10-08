# Training Summary & Convergence Analysis
**Project:** VisDrone2019-DET Performance Maximization Protocol  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8.59 GB VRAM, Driver 595.71, CUDA 12.1)  
**Dataset:** 422 Train, 106 Validation, 20 Locked Test

---

## 1. Primary Model Convergence: YOLOv8s 100-Epoch Run

- **Architecture:** YOLOv8s (11.14M parameters, 28.5 GFLOPs at 640px)
- **Pretrained Initialization:** COCO backbone (`yolov8s.pt`)
- **Optimization Settings:**
  - Optimizer: MuSGD (lr0=0.01, lrf=0.01, momentum=0.9, weight_decay=0.0005)
  - Batch size: 8 (optimal balance of gradient signal stability and 8GB VRAM headroom)
  - AMP: Enabled
  - Dataloader Workers: 0 (configured for Windows IPC memory map stability)
  - Epochs: 100 with Early Stopping `patience=25`
  - Seed: 42 (deterministic)

### Epoch Dynamics & Metric Progression:
- **Epoch 1:** Train Box Loss = 2.0362 | Val Box Loss = 1.9406 | Val mAP@0.5 = 0.0371
- **Epoch 25:** Train Box Loss = 1.5412 | Val Box Loss = 1.5218 | Val mAP@0.5 = 0.2280
- **Epoch 50:** Train Box Loss = 1.4390 | Val Box Loss = 1.4421 | Val mAP@0.5 = 0.2814
- **Epoch 82:** Minimum Validation Box Loss reached at **1.3990**
- **Epoch 87 (Peak Val Metric):** Peak Validation mAP@0.5 achieved at **0.3355**
- **Epoch 100:** Train Box Loss = 1.3814 | Val Box Loss = 1.4483 | Val mAP@0.5 = 0.3068

The model completed 100 epochs in 0.721 hours. Best checkpoint was cleanly preserved at epoch 87 (`best.pt`, 22.5 MB).

---

## 2. Capacity Scaling Observations

- **YOLOv8n (3.01M params):** Rapidly converged by epoch 15, but severely underfit VisDrone's dense spatial patterns, achieving a validation mAP@0.5 ceiling of only **0.0606**.
- **YOLO11n (2.59M params):** Marginally outperformed v8n at **0.0747 val mAP@0.5**, but similarly lacked the feature pyramid depth to distinguish overlapping small targets.
- **YOLOv8s (11.14M params):** Proved to be the optimal capacity architecture, delivering **0.3355 peak validation mAP@0.5** and maintaining strong generalization without catastrophic overfitting.
- **YOLOv8m (25.86M params):** Exhibited higher training capacity but showed overfitting tendencies under limited aerial training samples (422 images), saturating at 0.2017 val mAP.
