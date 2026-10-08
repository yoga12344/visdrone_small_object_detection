# CV Hackathon — Code & Methodology Audit

> **Audited:** `cv_hackthon-main/cv_hackthon-main/`  
> **Auditor:** Automated static analysis + manual review  
> **Date:** 2026-10-04

---

## Executive Summary

The existing codebase contains **10 critical bugs and methodological problems** that invalidate all reported metrics.
The most severe issue is that **run_all.py explicitly uses YOLOv8 predictions as the ground truth** for evaluating every model — meaning every model is measured against a different set of "ground truth" boxes per run, and a perfect YOLO run trivially yields mAP@0.5 = 1.0.
Additionally, mAP is computed as `precision × recall` (scalar, not the area under the precision-recall curve), which is mathematically incorrect.
No real VisDrone annotation files are used anywhere. All COCO pre-trained class IDs (1–90) are compared directly against YOLO class IDs (0-indexed) with no remapping.

**Confidence in any published metric from this codebase: ZERO.**

---

## Existing Architecture

```
cv_hackthon-main/
├── yolo_infer.py        ← standalone YOLOv8 eval (hardcoded paths)
├── fastrcnn_infer.py    ← standalone Faster R-CNN eval (hardcoded paths)
├── maskrcnn_infer.py    ← standalone Mask R-CNN eval (hardcoded paths)
├── sam_infer.py         ← standalone YOLO+SAM eval (hardcoded paths)
├── run_all.py           ← unified runner (pseudo-GT bug present)
├── yolov8s.pt           ← YOLOv8s COCO-pretrained weights (21 MB)
├── dataset/
│   ├── images/          ← img1.jpg, img2.jpg
│   └── labels/          ← pseudo-GT .txt files (YOLO format from predictions)
└── *.png / *.mp4        ← output artefacts
```

Each inference script follows the same pattern:
1. Hardcode absolute paths to images and label files from a different machine (`C:\Users\bomma\...`)
2. Load a pre-trained model
3. Run predictions
4. Match predictions to "GT" using greedy IoU
5. Compute `mAP50 = precision * recall` ← **wrong**

---

## Existing Models

| Script | Model | Backbone | Source |
|---|---|---|---|
| `yolo_infer.py` | YOLOv8s | CSPDarkNet | Ultralytics |
| `fastrcnn_infer.py` | Faster R-CNN | ResNet50-FPN | torchvision |
| `maskrcnn_infer.py` | Mask R-CNN | ResNet50-FPN | torchvision |
| `sam_infer.py` | YOLOv8 + SAM ViT-L | CSPDarkNet + ViT-L | Ultralytics + Meta |
| *(missing)* | RF-DETR | — | — |

### Code Snippets Showing Key Bugs

**Bug A — mAP computed as precision × recall (all scripts):**
```python
# fastrcnn_infer.py  line 109
# maskrcnn_infer.py  line 110
# yolo_infer.py      line 133
# run_all.py         line 57
map50 = precision * recall   # ← WRONG: this is not mAP
```

**Bug B — Pseudo-GT generation in run_all.py (lines 94–111):**
```python
# run_all.py  lines 94-111
print("[INFO] Generating pseudo ground-truth labels with YOLOv8s...")
...
for img_path in image_files:
    res = yolo_model(img_path, verbose=False)[0]
    with open(lab_path, "w") as f:
        for cls, box in zip(res.boxes.cls.cpu().numpy(),
                             res.boxes.xyxy.cpu().numpy()):
            xc, yc, w, h = xyxy_to_yolo(*box, W, H)
            f.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
label_files.append(lab_path)
# These YOLO predictions are then used as GT for ALL subsequent models!
```

**Bug C — No confidence filtering for COCO models (fastrcnn_infer.py / maskrcnn_infer.py):**
```python
# fastrcnn_infer.py  lines 76-79
preds = list(zip(
    out["labels"].detach().cpu().numpy(),
    out["boxes"].detach().cpu().numpy()
))
# scores = out["scores"] is NEVER used → all boxes kept regardless of score
```

**Bug D — Small object mAP is TP/total (recall), not mAP (all scripts):**
```python
# run_all.py  line 59
small_map  = small_tp / (small_total+1e-6)   # this is small-object recall, not mAP
```

**Bug E — No warmup before timing (all scripts):**
```python
# yolo_infer.py  lines 84-87
t0 = time.time()
pred = model(img_path)[0]   # first call includes JIT/GPU warmup overhead
t1 = time.time()
times.append(t1-t0)
```

**Bug F — SAM: no confusion matrix, class ID stripped (sam_infer.py):**
```python
# sam_infer.py  lines 65, 68
_,xc,yc,w,h = map(float,l.split())   # class column silently discarded
gt.append((b,area))                   # no class stored → confusion impossible
```

---

## Existing Evaluation Pipeline

```
Image → Model → Predictions
             ↓
          IoU Match against "GT"
             ↓
       Greedy best-match (no Hungarian assignment)
             ↓
   precision = TP/(TP+FP)
   recall    = TP/(TP+FN)
   mAP50     = precision * recall   ← WRONG
   small_mAP = small_tp/small_total ← this is recall, not mAP
```

**What correct mAP50 requires:**
1. Collect all (confidence, TP/FP) pairs across the entire dataset
2. Sort by confidence descending
3. Compute cumulative precision and recall at each threshold
4. Integrate (area under precision-recall curve) using the 11-point or all-point interpolation
5. Average over all classes → mean Average Precision

---

## Critical Bugs Found

### Bug 1 — Pseudo-Ground-Truth Contamination (Severity: CRITICAL)
- **Bug:** `run_all.py` lines 94–111 generate YOLO labels by running YOLOv8 predictions, then all subsequent models are evaluated against these pseudo-GT labels.
- **Location:** `run_all.py:94-111`, propagated to evaluation of Faster R-CNN, Mask R-CNN, and SAM.
- **Impact:** Every metric is measuring agreement with YOLO, not detection accuracy on real annotations. YOLOv8 trivially achieves near-perfect scores. COCO-class predictions from Faster/Mask R-CNN are compared to YOLO-class predictions — class IDs differ, so class matching is also broken.
- **Fix:** Replace pseudo-GT generation with a proper VisDrone annotation loader (`scripts/download_visdrone.py` → `data/raw/`). Use `evaluation/` module with real GT boxes and class IDs.

### Bug 2 — mAP Computed as precision × recall (Severity: CRITICAL)
- **Bug:** `map50 = precision * recall` (scalar F1-like product) is used in place of mean Average Precision.
- **Location:** `yolo_infer.py:133`, `fastrcnn_infer.py:109`, `maskrcnn_infer.py:110`, `sam_infer.py:113`, `run_all.py:57`.
- **Impact:** The reported "mAP@0.5" is not comparable to any published benchmark. It underestimates mAP for models with imbalanced precision/recall and is not a standard metric.
- **Fix:** Implement proper AP via precision-recall curve integration: collect `(score, is_tp)` tuples across all images per class, sort descending by score, compute cumulative precision/recall, use `np.trapz` or 11-point interpolation, then average over classes.

### Bug 3 — Small Object mAP Reported as Recall (Severity: HIGH)
- **Bug:** `small_map = small_tp / (small_total + 1e-6)` is small-object recall, not small-object mAP.
- **Location:** All scripts.
- **Impact:** The metric labelled "Small Object mAP" is not mAP. Misleads performance analysis on VisDrone's dominant small-object challenge.
- **Fix:** Filter the per-detection list to small objects only, then compute AP over that subset using the same PR-curve integration.

### Bug 4 — Greedy IoU Matching (Severity: HIGH)
- **Bug:** For each GT box, the code finds the single best-IoU prediction. However, a prediction can be "matched" to multiple GT boxes if it happens to have highest IoU for each — the `matched` set only prevents double-counting FP, not double-counting TP.
- **Location:** `run_all.py:129-144`, all standalone scripts.
- **Impact:** TP can be inflated. Proper COCO evaluation uses the Hungarian algorithm (optimal bipartite matching) or the standard "greedy by confidence, mark used" approach from Pascal VOC.
- **Fix:** Sort predictions by confidence, iterate in descending order, match each prediction to the best unmatched GT at IoU ≥ threshold.

### Bug 5 — No COCO-to-VisDrone Class Remapping (Severity: HIGH)
- **Bug:** Faster R-CNN and Mask R-CNN output COCO class IDs (1-indexed, 1=person, 3=car, etc.). These are compared directly against the label file class IDs (which are either YOLO predictions or VisDrone IDs, both 0-indexed). No mapping is applied.
- **Location:** `fastrcnn_infer.py:94`, `maskrcnn_infer.py:95`, `run_all.py:188,212`.
- **Impact:** All class-level metrics (confusion matrix, per-class AP) are completely wrong. A car detected as COCO class 3 will never match a VisDrone class 3 (which is "bicycle").
- **Fix:** Apply `COCO_TO_VISDRONE` mapping from `configs/benchmark_config.yaml` before matching. Filter out unmapped COCO classes.

### Bug 6 — Inference Timing Not Warmed Up (Severity: MEDIUM)
- **Bug:** The first inference call includes JIT compilation, CUDA kernel loading, and model caching overhead. No warmup runs are performed before recording timing.
- **Location:** All scripts — `t0 = time.time()` immediately before first `model()` call.
- **Impact:** Reported inference times are inflated by 5–100× for the first image, biasing the mean heavily when only 2–5 images are evaluated.
- **Fix:** Run `model(dummy_input)` for `warmup_runs` (configured in `benchmark_config.yaml`) before starting the timing loop.

### Bug 7 — SAM Has No Confusion Matrix (Severity: MEDIUM)
- **Bug:** `sam_infer.py` strips the class column from GT labels (line 65: `_,xc,yc,...`) and assigns all SAM predictions class 0 (line 246 in `run_all.py`: `preds = [(0, np.array(b)) for b in sam_boxes]`). No confusion matrix is generated.
- **Location:** `sam_infer.py:65`, `run_all.py:246`.
- **Impact:** Inconsistent reporting — all other models have confusion matrices; SAM does not. Class-level analysis is impossible.
- **Fix:** Propagate YOLO class predictions through to SAM matching. Rebuild confusion matrix for SAM predictions against GT classes.

### Bug 8 — No Confidence Score Filtering for COCO Models (Severity: MEDIUM)
- **Bug:** Faster R-CNN and Mask R-CNN return `out["scores"]`, but these are never used to threshold predictions. All proposals (even those with score 0.01) are included in evaluation.
- **Location:** `fastrcnn_infer.py:76-79`, `maskrcnn_infer.py:76-79`.
- **Impact:** Massive FP count from low-confidence proposals → precision collapses → reported "mAP" (already wrong) is artificially low.
- **Fix:** Apply `conf_threshold = 0.50` (from config) to filter predictions before IoU matching.

### Bug 9 — No RF-DETR Implementation (Severity: MEDIUM)
- **Bug:** RF-DETR (Roboflow Detection Transformer) is listed as a model to benchmark but no implementation exists.
- **Location:** Missing — no `rf_detr` script anywhere.
- **Impact:** Benchmark is incomplete; comparative analysis lacks the transformer-based detector.
- **Fix:** Implement `cv_hackathon/models/rf_detr/rf_detr_infer.py` using the `rfdetr` pip package with the nano variant as configured.

### Bug 10 — Hardcoded Absolute Paths from Another Machine (Severity: LOW, but blocks reproducibility)
- **Bug:** All standalone scripts reference `C:\Users\bomma\Downloads\...` which does not exist on any other machine.
- **Location:** `yolo_infer.py:12-27`, `fastrcnn_infer.py:12-25`, `maskrcnn_infer.py:12-25`, `sam_infer.py:10-26`.
- **Impact:** All standalone scripts fail immediately on any machine other than the original developer's.
- **Fix:** Use `argparse` or config files with relative paths. `run_all.py` already uses relative paths (correct pattern to follow).

---

## Methodological Problems

1. **Dataset leakage:** Using the same model's predictions as GT means the evaluation set is derived from the model being tested — this is a form of circular evaluation. True GT must be independent of model predictions.

2. **No class-level AP:** Real mAP averages AP over all classes. These scripts accumulate TP/FP globally, conflating all classes into a single score. A model that detects only one class perfectly would score identically to one that detects all classes correctly.

3. **Only 2–5 images evaluated:** Statistically meaningless. VisDrone-DET test set has 1,610 images. Evaluation on 2 images produces extreme variance and cannot represent model performance.

4. **Small object threshold:** `SMALL_AREA = 32 * 32` compares area in pixels, but VisDrone defines small objects by the *shorter side* (< 32px), not area. A 1×1024 object would be classified as large but is effectively a 1-pixel-wide line.

5. **No NMS applied to Faster/Mask R-CNN before evaluation:** torchvision's detection models do apply NMS internally, but the confidence threshold interaction with NMS means many low-quality boxes survive without score filtering.

---

## Evaluation Risks

| Risk | Likelihood | Impact |
|---|---|---|
| Reported mAP is uncorrelated with true mAP | Certain | All conclusions invalid |
| Confusion matrices use wrong class IDs | Certain | All class-level analysis invalid |
| Timing results misleading (no warmup) | Certain | Model speed comparisons wrong |
| Small object mAP is actually small recall | Certain | Performance on VisDrone main challenge metric wrong |
| Results not reproducible (hardcoded paths) | Certain | Cannot reproduce on another machine |
| 2-image evaluation has >50% variance | Very High | Cannot draw any statistical conclusion |

---

## Recommended Fixes

### Priority 1 — Fix the Ground Truth (blocks everything else)
1. Download VisDrone2019-DET-test (`scripts/download_visdrone.py`)
2. Convert VisDrone annotations → YOLO format with class remapping (category 0 = ignored, skip it)
3. Store in `data/raw/` and `data/processed/`
4. Verify with a sample visualization

### Priority 2 — Fix mAP Computation
Replace `map50 = precision * recall` with proper PR-curve integration:
```python
# evaluation/metrics.py  (new file)
def compute_ap(recalls, precisions):
    """11-point interpolation AP (Pascal VOC style)."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap
```

### Priority 3 — Fix Class Mapping
Load `coco_to_visdrone` from `configs/benchmark_config.yaml`. Apply before IoU matching:
```python
COCO_TO_VD = {0:0, 1:2, 2:3, 3:9, 5:8, 7:5}  # person, bicycle, car, motor, bus, truck
mapped_preds = [(COCO_TO_VD[c], b) for c, b in preds if c in COCO_TO_VD]
```

### Priority 4 — Add Warmup
```python
# warmup
for _ in range(cfg.warmup_runs):
    _ = model(dummy_tensor)

# timed runs
for img in eval_images:
    t0 = time.perf_counter()
    out = model(img)
    times.append(time.perf_counter() - t0)
```

### Priority 5 — Implement RF-DETR
```bash
pip install rfdetr
```
See `cv_hackathon/models/rf_detr/` for implementation stub.

### Priority 6 — Expand Dataset
Evaluate on at least the full VisDrone test set (1,610 images). Statistical error bars require minimum ~100 images.

---

## Reproducibility Issues

| Item | Status |
|---|---|
| Random seed set | ✗ — not set anywhere |
| Dependency versions pinned | ✗ — no `requirements.txt` |
| Model weights versioned | Partial — `yolov8s.pt` present, SAM weight absent |
| Dataset version specified | ✗ — no dataset provenance |
| Absolute paths to another machine | ✗ — 4 of 5 scripts broken on other machines |
| Deterministic PyTorch ops | ✗ — `torch.use_deterministic_algorithms(True)` not set |

---

## Dataset Status

| Split | Images | Annotations | Format |
|---|---|---|---|
| `dataset/images/` | 2 (img1.jpg, img2.jpg) | pseudo-GT from YOLO | YOLO .txt |
| VisDrone2019-DET train | 6,471 | Real VisDrone | `.txt` per image |
| VisDrone2019-DET val | 548 | Real VisDrone | `.txt` per image |
| VisDrone2019-DET test | 1,610 | Real VisDrone | `.txt` per image |

**VisDrone download:** See `cv_hackathon/scripts/download_visdrone.py`

VisDrone annotation format (comma-separated per line):
```
<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>
```
- category 0 = ignored region (must be excluded)
- categories 1–10 map to VisDrone classes (subtract 1 to get 0-indexed)

---

## Environment

```
Python:        3.8+ required
PyTorch:       ≥ 2.0
torchvision:   ≥ 0.15
ultralytics:   ≥ 8.0  (YOLOv8)
segment-anything: latest (Meta SAM)
rfdetr:        latest (Roboflow RF-DETR)
numpy:         ≥ 1.24
Pillow:        ≥ 9.0
PyYAML:        ≥ 6.0
scipy:         ≥ 1.10  (Hungarian matching)
matplotlib:    ≥ 3.7
seaborn:       ≥ 0.12
```

Recommended install:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics segment-anything rfdetr scipy matplotlib seaborn pyyaml
```

> **Note:** SAM `vit_l` weights (`sam_vit_l_0b3195.pth`, ~2.4 GB) must be downloaded separately:
> `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth`
