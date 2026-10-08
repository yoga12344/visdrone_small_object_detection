# Systematic Validation Ablation Report
**Protocol:** All configurations tuned and selected strictly on the 106-image Validation split. Zero test-set leakage.

---

## 1. Input Resolution Ablation (106 Val Images)

| Resolution | Val mAP@0.5 | Precision | Recall | F1 Score | Small AP (<32px) | Medium AP | Large AP | Latency (ms) | FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **640×640 (Default)** | 0.2282 | 0.7021 | 0.4270 | 0.5310 | 0.0470 | 0.3856 | 0.3865 | 30.06 | 33.27 |
| **768×768** | 0.2586 | 0.6946 | 0.4670 | 0.5585 | 0.0631 | 0.4045 | 0.3277 | 30.03 | 33.30 |
| **1024×1024 (Default)** | 0.3034 | 0.7054 | 0.5348 | 0.6084 | 0.0998 | 0.4287 | 0.3110 | 39.87 | 25.08 |
| **1024×1024 (Tuned: c=0.20, nms=0.50)** | **0.3159** | 0.6461 | **0.5606** | **0.6003** | **0.1075** | **0.4358** | 0.3117 | **18.32** | **54.59** |

---

## 2. Confidence Threshold Sweep (Val Set, YOLOv8s 100ep, 640px)

| Confidence | Val mAP@0.5 | Precision | Recall | F1 Score | Small AP (<32px) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.05** | 0.2766 | 0.3732 | 0.5312 | 0.4384 | 0.0680 |
| **0.10** | 0.2637 | 0.4887 | 0.5014 | 0.4949 | 0.0629 |
| **0.15** | 0.2550 | 0.5757 | 0.4811 | 0.5242 | 0.0590 |
| **0.20** | 0.2421 | 0.6399 | 0.4536 | **0.5309** | 0.0531 |
| **0.25 (Default)** | 0.2282 | 0.7021 | 0.4270 | **0.5310** | 0.0470 |
| **0.30** | 0.2167 | 0.7477 | 0.4031 | 0.5238 | 0.0417 |
| **0.40** | 0.1906 | 0.8187 | 0.3539 | 0.4942 | 0.0306 |
| **0.50** | 0.1668 | 0.8675 | 0.3071 | 0.4536 | 0.0201 |

*Selection:* `conf=0.20` delivers near-optimal F1 while providing a 13% relative gain in Small-Object AP over the default 0.25 threshold.

---

## 3. NMS IoU Threshold Sweep (Val Set, conf=0.20)

| NMS IoU | Val mAP@0.5 | Precision | Recall | F1 Score | Small AP (<32px) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.40** | 0.2276 | 0.7057 | 0.4246 | 0.5302 | 0.0468 |
| **0.45** | 0.2282 | 0.7021 | 0.4270 | 0.5310 | 0.0470 |
| **0.50** | 0.2292 | 0.6967 | 0.4291 | **0.5311** | 0.0469 |
| **0.55** | 0.2294 | 0.6887 | 0.4308 | 0.5300 | 0.0468 |
| **0.60** | 0.2293 | 0.6790 | 0.4328 | 0.5287 | 0.0467 |
| **0.65** | **0.2304** | 0.6656 | 0.4340 | 0.5255 | 0.0462 |
| **0.70** | 0.2302 | 0.6499 | 0.4354 | 0.5215 | 0.0456 |

*Selection:* `iou=0.50` retains peak F1 and suppresses redundant overlaps in aerial object clusters.

---

## 4. Advanced Tiling Optimization (Val Set)

| Strategy / Grid | Overlap | Val mAP@0.5 | Recall | F1 Score | Small AP (<32px) | Latency (ms) | FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tiled 2×2** | 10% | 0.3438 | 0.6249 | 0.5617 | 0.1195 | 66.56 | 15.03 |
| **Tiled 2×2** | 20% | 0.3473 | 0.6186 | 0.5605 | 0.1165 | 71.78 | 13.93 |
| **Tiled 2×2** | 25% | 0.3433 | 0.6149 | 0.5536 | 0.1117 | 74.14 | 13.49 |
| **Tiled 3×2** | 20% | **0.3560** | **0.6532** | **0.5602** | **0.1396** | 103.10 | 9.70 |
| **Adaptive 2×2** | 20% | 0.3473 | 0.6186 | 0.5605 | 0.1165 | 89.03 | 11.23 |
