# Comprehensive Research Benchmark Report: Comparative Analysis of Advanced Object Detection and Segmentation Models for Aerial Drone Imagery

**Project:** Computer Vision Hackathon — Aerial Drone Imagery Benchmark  
**Dataset:** VisDrone2019-DET  
**Hardware Platform:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM, Compute Capability 8.9)  
**Execution Environment:** CUDA 12.1 (`torch==2.3.0+cu121`, `torchvision==0.18.0+cu121`)  
**Execution Mode:** Sequential GPU Training & Evaluation (`cuda:0` enforced; zero CPU fallback)  

---

## 1. Executive Summary & Experimental Methodology

This research report presents a scientifically grounded, reproducible comparative analysis of four contemporary vision architectures on challenging aerial drone imagery from the **VisDrone2019-DET** benchmark:
1. **YOLOv8s** (Single-stage, anchor-free CNN)
2. **Faster R-CNN** (Two-stage ResNet-50-FPN)
3. **Mask R-CNN** (Two-stage ResNet-50-FPN instance segmentation detector)
4. **RF-DETR Nano** (End-to-end Real-Time Feature Detection Transformer with DINOv2 backbone)

### Critical Methodological Safeguards
- **Zero Pseudo-Ground-Truth:** Unlike preliminary project implementations that evaluated models against model predictions, **100% of the metrics in this benchmark are computed against authentic VisDrone human-annotated ground-truth labels**.
- **Locked Test Isolation:** Exactly **20 test images** containing **1,295 valid ground-truth foreground objects** (`data/processed/splits/test_ids.txt`) were strictly isolated. Neither training nor checkpoint selection accessed this test partition.
- **Identical Scale Definitions:** Object scales strictly adhere to benchmark standards:
  - **Small:** $\text{area} < 32 \times 32\text{ px}$ (493 GT instances in locked test set)
  - **Medium:** $32 \times 32 \le \text{area} \le 96 \times 96\text{ px}$ (722 GT instances in locked test set)
  - **Large:** $\text{area} > 96 \times 96\text{ px}$ (80 GT instances in locked test set)
- **Synchronized Hardware Timing:** Latency and FPS were measured exclusively on genuine, warmed GPU forward passes (`torch.cuda.synchronize()` on RTX 4060 Laptop GPU).

---

## 2. Dataset & Split Partitioning

- **Total Images:** 548 images
- **Train Split:** 422 images (29,428 valid foreground annotations)
- **Validation Split:** 106 images (8,036 valid foreground annotations)
- **Test Split (Locked):** 20 images (**1,295 valid foreground annotations**, 21 ignored Category-0 regions excluded)

### 10-Class Foreground Mapping:
`0: pedestrian, 1: people, 2: bicycle, 3: car, 4: van, 5: truck, 6: tricycle, 7: awning-tricycle, 8: bus, 9: motor`

---

## 3. Comprehensive Master Benchmark Results

All four models were trained for 25 epochs on `cuda:0` and evaluated on the **exact same locked 20 test images** at $IoU = 0.50$:

| Model | Architecture Paradigm | Parameters | Checkpoint Size | Test mAP@0.50 | Test Precision | Test Recall | Test F1 | Small AP ($<32\text{px}$) | Medium AP ($32-96\text{px}$) | Large AP ($>96\text{px}$) | Mean IoU | GPU Latency | GPU FPS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8s** | 1-Stage Anchor-Free CNN | **11.1M** | **21.5 MB** | 0.1322 | **0.5803** | 0.2456 | 0.3451 | 0.0028 | 0.1699 | 0.3917 | **0.7831** | **19.53 ms** | **51.20** |
| **Faster R-CNN** | 2-Stage ResNet-50-FPN | 41.3M | 158.2 MB | **0.2749** | 0.5119 | 0.4656 | 0.4877 | 0.0365 | 0.3064 | 0.4020 | 0.7701 | 84.12 ms | 11.89 |
| **Mask R-CNN** | 2-Stage Multi-Task FPN | 44.0M | 168.2 MB | 0.2587 | 0.5216 | **0.4672** | **0.4929** | **0.0401** | **0.3134** | 0.3938 | 0.7606 | 1141.14 ms | 0.88 |
| **RF-DETR Nano** | Vision Transformer (DINOv2) | 30.5M | 115.4 MB | 0.1995 | 0.3981 | 0.3653 | 0.3810 | 0.0063 | 0.2578 | **0.4906** | 0.7298 | 42.60 ms | 23.47 |

---

## 4. Zero-Shot Baseline vs. Fine-Tuned Progression

To scientifically assess the benefit of domain adaptation on drone imagery, fine-tuned performance is contrasted against COCO zero-shot baselines:

| Model | Zero-Shot mAP@0.5 | Fine-Tuned mAP@0.5 | Absolute Gain ($\Delta$) | Relative Gain (%) | Zero-Shot Small AP | Fine-Tuned Small AP | Small AP Relative Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8s** | 0.0587 | 0.1322 | +0.0735 | **+125.2%** | 0.0005 | 0.0028 | **+460.0%** |
| **Faster R-CNN** | 0.0888 | 0.2749 | +0.1861 | **+209.6%** | 0.0084 | 0.0365 | **+334.5%** |
| **Mask R-CNN** | 0.0887 | 0.2587 | +0.1700 | **+191.7%** | 0.0089 | 0.0401 | **+350.6%** |
| **RF-DETR Nano** | 0.0612 | 0.1995 | +0.1383 | **+226.0%** | 0.0011 | 0.0063 | **+472.7%** |

*Takeaway:* Aerial drone imagery features acute nadir angles, extreme object scale miniaturization, and heavy background clutter. Domain-specific fine-tuning delivered between **125% and 226% relative mAP gains** across all model architectures.

---

## 5. In-Depth Small-Object & Scale-Stratified Analysis

VisDrone is dominated by small objects. In the locked test partition:
- **Small Targets ($<32\text{px}$):** 493 objects (38.1% of test set)
- **Medium Targets ($32-96\text{px}$):** 722 objects (55.8% of test set)
- **Large Targets ($>96\text{px}$):** 80 objects (6.1% of test set)

### Key Findings by Scale:
1. **Small-Object Superiority of Two-Stage Detectors:**
   - **Mask R-CNN ($AP_{small} = 0.0401$)** and **Faster R-CNN ($AP_{small} = 0.0365$)** dramatically outperform single-stage YOLOv8s ($AP_{small} = 0.0028$) by **13x to 14x**.
   - Two-stage architectures preserve higher spatial resolution across multi-scale feature pyramids (FPN) and employ class-agnostic Region Proposal Networks (RPN) that generate dense, localized proposals that RoIAlign crops at sub-pixel precision.
2. **Large-Object Superiority of Transformer Attention:**
   - **RF-DETR Nano achieves the highest Large-Object AP (0.4906)** across all models, outperforming Faster R-CNN (0.4020) and YOLOv8s (0.3917).
   - Global self-attention mechanisms in the DINOv2 backbone and deformable transformer decoder capture global semantic context across the entire frame, giving DETR models an advantage on larger, unoccluded vehicles and infrastructure.

---

## 6. Computational Efficiency, Latency & FPS Trade-Offs

All latency measurements represent warm, synchronized GPU inference per frame ($640 \times 640$ or $560 \times 560$ resolution) on the NVIDIA RTX 4060 Laptop GPU:

1. **Ultra-Real-Time Tier (>30 FPS):**
   - **YOLOv8s:** **19.53 ms latency (51.20 FPS)** with only **11.1M parameters (21.46 MB)**.
   - Ideal for onboard drone edge devices (e.g., Jetson Orin Nano / Xavier NX) running at video framerate.
2. **Interactive Real-Time Tier (20–30 FPS):**
   - **RF-DETR Nano:** **42.60 ms latency (23.47 FPS)** with **30.5M parameters (115.42 MB)**.
   - Achieves near-real-time streaming without requiring Non-Maximum Suppression (NMS) post-processing.
3. **High-Accuracy Server Tier (10–15 FPS):**
   - **Faster R-CNN:** **84.12 ms latency (11.89 FPS)** with **41.3M parameters (158.21 MB)**.
   - Strikes the optimal balance for post-processing flight data where detection accuracy and small-scale recall are prioritized over ultra-low latency.
4. **Heavy Multi-Task Tier (<2 FPS):**
   - **Mask R-CNN:** **1141.14 ms latency (0.88 FPS)** with **44.0M parameters (168.23 MB)**.
   - The simultaneous prediction of high-resolution pixel masks for every proposal creates significant computational overhead, making it unsuitable for real-time edge streaming.

---

## 7. Architectural Observations & Methodological Limitations

### A. Mask R-CNN Segmentation Limitation
- The VisDrone2019-DET dataset provides bounding-box coordinates only (no ground-truth polygon segmentation masks).
- In accordance with academic integrity guidelines, **no synthetic segmentation ground truth was fabricated**. Mask R-CNN was trained using box-level supervision.
- Predicted instance masks are generated for qualitative visual assessment and are explicitly documented as qualitative predictions rather than ground-truth-validated mask mAP.

### B. RF-DETR Nano Integration & Verification
- Prior baseline attempts experienced environment fallback issues resulting in artificial 0 metrics or unrealistic 10,000+ FPS timings.
- In this verified experiment:
  - The official 349 MB checkpoint was successfully integrated.
  - Upstream dependencies (`pytorch_lightning`, `faster_coco_eval`, `torch_linear_assignment`) and Windows console character encoding handlers were resolved.
  - The model underwent **genuine 25-epoch GPU training** (27.3 minutes) and genuine timed inference on the locked test set, yielding valid, non-zero detection results.

---

## 8. Real-Time Deployment Recommendation

Based on empirical performance on the NVIDIA GeForce RTX 4060 GPU:

### 🏆 Top Recommendation for Real-Time Onboard Drone Flight: **YOLOv8s**
- **Rationale:** Operates at **51.20 FPS** (19.5 ms) with a featherweight **21.5 MB footprint**, consuming minimal VRAM and power on embedded compute modules. It delivers the highest precision (**0.5803**) and cleanest localization quality (**Mean IoU: 0.7831**) for vehicles and roads.

### 🥈 Top Recommendation for High-Precision Server Post-Flight Inspection: **Faster R-CNN**
- **Rationale:** Achieves the **highest overall mAP@0.50 (27.49%)** and **13x greater small-object detection accuracy (0.0365)** than YOLOv8s. At 11.9 FPS on RTX 4060, it is well-suited for ground-station analysis and aerial search-and-rescue post-processing where detecting tiny pedestrians is critical.

### 🥉 Top Recommendation for Modern Transformer Deployments: **RF-DETR Nano**
- **Rationale:** At **23.47 FPS** and **0.1995 mAP@0.50**, RF-DETR Nano offers an NMS-free transformer pipeline with superior large-object detection (**Large AP: 0.4906**), making it a viable option for modern drone setups with dedicated GPU acceleration.
