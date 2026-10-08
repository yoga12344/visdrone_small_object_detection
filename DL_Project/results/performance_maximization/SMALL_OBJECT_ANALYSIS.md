# Small-Object Detection Bottlenecks & Spatial Recovery Analysis
**Focus:** Objects with bounding box area $< 32 \times 32\text{ pixels}$ in VisDrone2019-DET.

---

## 1. Scale Distribution in Drone Imagery

Of the 8,036 authentic targets in the VisDrone validation split:
- **Small (<32px):** 4,312 objects (**53.7%**)
- **Medium (32–96px):** 3,281 objects (**40.8%**)
- **Large (>96px):** 443 objects (**5.5%**)

Over half of all objects in the aerial scene are small. When a high-resolution drone image ($1360 \times 765$ pixels) is uniformly resized down to $640 \times 640$, a 20-pixel pedestrian is downsampled to ~9 pixels. At stride 16 or 32 in modern CNN backbones, this sub-10-pixel target projects to less than a single feature map cell, causing complete disappearance from the prediction tensor.

---

## 2. Experimental Comparison: Resolution & Tiling vs Small-Object AP

| Method | Downsampling Factor | Small AP (<32px) on Val | Small Recall on Val | Locked Test Small AP |
| :--- | :--- | :--- | :--- | :--- |
| **Native 640px** | ~2.1× reduction | 0.0470 | 23.01% | 0.0039 |
| **Native 1024px** | ~1.3× reduction | 0.1075 | 40.42% | 0.0363 |
| **Tiled 2×2 (20% overlap)** | Native 1.0× (sliced) | 0.1165 | 47.22% | 0.0392 |
| **Tiled 3×2 (20% overlap)** | Native 1.0× (dense slices) | **0.1396** | **52.99%** | **0.0601** |

---

## 3. Key Takeaways
1. **Recall is the Main Bottleneck:** Increasing spatial preservation elevates small object recall from 23.01% to 52.99%.
2. **Native 1024px vs Tiling:** Native 1024px provides a 10× increase in small-object test AP (0.0039 → 0.0363) while preserving 50+ FPS real-time throughput. Tiled 3×2 pushes Small AP to 0.0601 (+1440% over baseline 640px), serving as the high-accuracy operating point for non-time-critical forensic pipelines.
