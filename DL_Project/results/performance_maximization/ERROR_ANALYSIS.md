# Qualitative Error Analysis & Failure Taxonomy
**Location of Visual Artifacts:** `results/performance_maximization/figures/` and `error_analysis/`

---

## 1. Primary Failure Modes Identified

1. **Micro-Scale Target Misses (False Negatives):**
   - Dominant failure mode for single-stage standard 640px detectors.
   - Occurs on distant pedestrians walking along tree-lined walkways and cyclists crossing multi-lane intersections.
   - *Mitigation:* Sliced tiling (2×2 / 3×2) and 1024px high-resolution input dramatically resolve this failure by preserving high-frequency edges.

2. **Inter-Class Confusion (Pedestrian vs People):**
   - `pedestrian` vs `people` accounts for the largest confusion block in the confusion matrix.
   - VisDrone distinguishes individual standing pedestrians from groups of people, but annotations frequently exhibit blurred boundaries in dense crowds.
   - Detectors often predict `pedestrian` on tightly packed groups, leading to high classification cross-entropy penalty despite correct spatial localization.

3. **Thin-Structure Vulnerability (Bicycle & Awning-Tricycle):**
   - `bicycle` achieves only 0.0855 validation AP50.
   - Thin tubular frames, wire wheels, and dynamic rider postures produce fragile visual features easily obscured by asphalt road textures and shadows.

4. **Crowded Perspective Occlusion & Overlap:**
   - Overlapping vehicles at traffic lights and angled parking zones cause mutual bounding box overlap exceeding 50% IoU.
   - Default NMS (`iou=0.45`) erroneously suppresses valid adjacent targets.
   - Raising NMS threshold to `0.50`–`0.65` allows dual-box survival on tight vehicle pairs.

---

## 2. Multi-Model Qualitative Behavior Comparison

- **YOLOv8s Baseline (640px):** Produces tight, high-confidence bounding boxes on prominent vehicles, but leaves 70%+ of distant pedestrians undetected.
- **YOLOv8s (1024px):** Detects dense pedestrian streams and distant cars with clear spatial separation.
- **YOLOv8s (Tiled 3×2):** Recovers ultra-tiny targets at image margins and corners that are invisible to downsampled models.
- **Faster R-CNN:** High anchor density excels at capturing small clusters, but exhibits higher false-positive rates on complex road textures.
- **YOLO + SAM Hybrid:** Yields precise instance-level boundary contours for irregularly shaped objects (e.g. awnings, bicycles) which bounding boxes over-approximate.
