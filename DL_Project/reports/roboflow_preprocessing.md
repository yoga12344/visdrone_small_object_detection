# Roboflow Preprocessing Pipeline — VisDrone2019 Hackathon

> **Status legend used in this document**
> - ✅ **Executed** — step has been run and its outputs are committed to the repo
> - 📝 **Documented** — step has been defined and is reproducible, but was run externally or requires credentials

---

## 1. What is Roboflow and Why We Use It

[Roboflow](https://roboflow.com) is a managed computer-vision dataset platform that provides:

| Feature | Benefit for this project |
|---|---|
| Annotation viewer & editor | Quick visual sanity-check of VisDrone bounding boxes |
| Version-controlled datasets | Reproducible dataset snapshots tied to model runs |
| Preprocessing pipeline | Deterministic resize + augmentation config stored as metadata |
| YAML export | One-click YOLOv8/RF-DETR compatible `data.yaml` |
| Hosted storage | Team members pull the same binary data via API |

> [!NOTE]
> Roboflow is used for **dataset management and preprocessing documentation only**.  
> Model training is done locally / on Colab with the exported dataset.

---

## 2. Dataset Upload Process

> **Status: 📝 Documented** (upload requires Roboflow workspace credentials)

### Step-by-step

```
Step 1 – Download raw VisDrone2019-DET
   └─ Source: https://github.com/VisDrone/VisDrone-Dataset
      ├─ VisDrone2019-DET-train.zip   (6 333 images)
      ├─ VisDrone2019-DET-val.zip     (  548 images)
      └─ VisDrone2019-DET-test-dev.zip(  548 images)

Step 2 – Convert to YOLO format locally (executed ✅)
   └─ Script: data/visdrone_parser.py → convert_dataset_to_yolo()
      Output: labels/{train,val,test}/*.txt

Step 3 – Upload to Roboflow
   └─ roboflow.com → "Create New Project" → Object Detection
   └─ Project name : visdrone-detection
   └─ Upload images + label folder pairs (drag-and-drop or CLI):
        rfcli upload --workspace cv-hackathon \
                     --project visdrone-detection \
                     --folder ./datasets/VisDrone2019-DET-train/images \
                     --labels ./datasets/VisDrone2019-DET-train/labels

Step 4 – Verify upload counts
   └─ Expected: 6 333 train / 548 val / 548 test images
```

---

## 3. Annotation Verification Steps

> **Status: 📝 Documented**

After upload, the following checks are performed in the Roboflow UI:

1. **Bounding box overlay** — sample 20 random images; confirm boxes align with vehicles/pedestrians.
2. **Class histogram** — `car` and `pedestrian` should be the two most frequent classes (VisDrone GT).
3. **Null-annotation images** — Roboflow flags images with no boxes; review each one (some VisDrone images genuinely have no visible objects).
4. **Out-of-bounds boxes** — Roboflow highlights boxes extending outside the image boundary; these are corrected by the clamp logic in `visdrone_parser.py → convert_to_yolo_format`.
5. **Duplicate images** — Roboflow's deduplication hash check is run before versioning.

Programmatic validation is also available via [`data/dataset_validator.py`](../data/dataset_validator.py):

```python
from data.dataset_validator import check_no_pseudo_gt, check_class_distribution

is_ok, msg = check_no_pseudo_gt("./labels/train")
print(msg)
```

---

## 4. Class Mapping Configuration

VisDrone uses 1-indexed category IDs. Category 0 is an **ignored region** and must be skipped during training and evaluation.

| VisDrone cat ID | Class name | 0-indexed label | COCO equivalent |
|---|---|---|---|
| 0 | ignored region | — | — |
| 1 | pedestrian | **0** | person (0) |
| 2 | people | **1** | person (0) |
| 3 | bicycle | **2** | bicycle (1) |
| 4 | car | **3** | car (2) |
| 5 | van | **4** | — |
| 6 | truck | **5** | truck (7) |
| 7 | tricycle | **6** | — |
| 8 | awning-tricycle | **7** | — |
| 9 | bus | **8** | bus (5) |
| 10 | motor | **9** | motorcycle (3) |

In Roboflow the classes are named exactly as in the table above (column "Class name").  
When using COCO-pretrained weights the mapping `COCO_TO_VISDRONE` in [`visdrone_parser.py`](../data/visdrone_parser.py) is applied at inference time.

> [!IMPORTANT]
> **Never** pass raw VisDrone category IDs (1–10) as training labels.  
> Always subtract 1 and skip category 0.

---

## 5. Train / Val / Test Split Strategy

> **Status: ✅ Executed** (VisDrone's official splits are respected)

### Strategy: Respect the Official VisDrone Splits

VisDrone2019-DET provides pre-defined splits.  We use them **as-is** rather than re-splitting, because:

- The official split was stratified by scene type (urban, suburban, rural) and drone altitude.
- Re-splitting risks altitude or scene leakage between train and test.
- Results reported on the official test set are directly comparable to the VisDrone leaderboard.

| Split | Images | Objects (approx.) | Purpose |
|---|---|---|---|
| train | 6 333 | ~540 000 | Model training |
| val | 548 | ~38 000 | Hyperparameter tuning, early stopping |
| test-dev | 548 | ~39 000 | Final evaluation reported in paper |

> [!WARNING]
> **Data leakage check:** `data/dataset_validator.py → check_train_test_overlap()` is run after every dataset modification. Zero overlap confirmed ✅.

The 70/20/10 `ROBOFLOW_CONFIG["split"]` values document Roboflow's **display split** only; the actual binary data respects the official VisDrone partition.

---

## 6. Augmentation Rationale

All augmentations in `ROBOFLOW_CONFIG["augmentations"]` are chosen for **aerial drone imagery** characteristics.

### ✅ Horizontal Flip
```
"flip_horizontal": True
```
- Aerial scenes have no inherent left/right orientation.  
- Roads, vehicles, and pedestrians appear equally valid when mirrored.  
- **Doubles** effective training set size at zero cost.

### ✅ Rotation (±15°)
```
"rotation": {"min": -15, "max": 15}
```
- Drone cameras experience pitch/roll drift (±10–20° typical).  
- Rotation invariance is critical for a robust detector.  
- Capped at ±15° to avoid introducing unrealistic views.

### ✅ Brightness Variation (±25%)
```
"brightness": {"min": -25, "max": 25}
```
- Outdoor scenes vary from golden-hour (dim) to midday (bright).  
- VisDrone images were captured across different lighting conditions; augmentation bridges the gap for underrepresented conditions.

### ✅ Exposure Variation (±15%)
```
"exposure": {"min": -15, "max": 15}
```
- Consumer drone cameras use auto-exposure which can over/under-expose.  
- Complements brightness augmentation for camera-response variability.

### ✅ Gaussian Blur (max 1.5 px)
```
"blur": {"max_pixels": 1.5}
```
- Motion blur occurs when the drone is moving fast or panning.  
- 1.5 px is mild — enough to improve robustness without degrading features.

### ✅ Noise (max 2%)
```
"noise": {"max_percent": 2}
```
- Simulates sensor noise from small drone CMOS sensors in low light.  
- Kept very low (2%) to avoid destroying fine-grained small-object features.

---

## 7. Augmentations Avoided and Why

| Augmentation | Reason for exclusion |
|---|---|
| **Vertical flip** | Aerial images always have sky-up / ground-down orientation. Vertical flip produces physically impossible scenes (buildings upside down). |
| **Mosaic (YOLOv8 default)** | Can be used in training but is not configured in Roboflow; applied in the Ultralytics training config directly. |
| **Shear** | Drone cameras use near-telecentric lenses — shear distortion is not realistic for this sensor type. |
| **Saturation / Hue** | Hue shift can change vehicle colour in ways that confuse the detector (e.g., red car → green). Brightness/exposure are sufficient. |
| **Cutout / Mosaic-mix** | Effective but requires careful implementation to preserve small-object GT boxes; deferred to model-level augmentation. |
| **Heavy JPEG compression** | VisDrone images are already compressed; adding more compression destroys the fine texture cues needed for tiny objects. |

---

## 8. Dataset Version Management

> **Status: 📝 Documented**

| Version | Date | Changes |
|---|---|---|
| v1 | 2026-10-01 | Initial upload, all 10 classes, 640×640 stretch resize |

Version bump triggers:
- Any change to the class list or mapping
- Any change to preprocessing parameters
- Bug fix in annotation parsing

Old versions remain accessible via the Roboflow API (`version=N`) to reproduce historical runs.

```python
# Always pin a specific version in training scripts:
dataset = rf.workspace("cv-hackathon") \
             .project("visdrone-detection") \
             .version(1) \          # <-- pinned
             .download("yolov8")
```

---

## 9. Export Format

**Format: YOLOv8**

Each split is exported as:
```
visdrone-detection-1/
├── data.yaml           # class names + train/val/test paths
├── train/
│   ├── images/*.jpg
│   └── labels/*.txt    # class_id cx cy w h  (normalised to [0,1])
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

`data.yaml` content (generated by Roboflow):
```yaml
names:
- pedestrian
- people
- bicycle
- car
- van
- truck
- tricycle
- awning-tricycle
- bus
- motor
nc: 10
roboflow:
  license: CC BY 4.0
  project: visdrone-detection
  url: https://universe.roboflow.com/cv-hackathon/visdrone-detection/dataset/1
  version: 1
  workspace: cv-hackathon
train: train/images
val: valid/images
test: test/images
```

> [!TIP]
> The `data.yaml` path is passed directly to `ultralytics.YOLO.train(data=...)` and to RF-DETR's config without modification.

---

## 10. Reproducing Without Roboflow (Local Pipeline)

All Roboflow steps can be replicated locally using the modules in `data/`.

### Step 1 — Convert VisDrone annotations to YOLO

```python
from data.visdrone_parser import convert_dataset_to_yolo

stats = convert_dataset_to_yolo(
    images_dir      = "./VisDrone2019-DET-train/images",
    annotations_dir = "./VisDrone2019-DET-train/annotations",
    output_labels_dir = "./datasets/visdrone_yolo/train/labels",
)
from data.visdrone_parser import print_dataset_stats
print_dataset_stats(stats)
```

### Step 2 — Validate labels

```python
from data.dataset_validator import check_no_pseudo_gt, check_class_distribution
from data.visdrone_parser import parse_visdrone_dataset

is_ok, msg = check_no_pseudo_gt("./datasets/visdrone_yolo/train/labels")
print(msg)

gt_anns, _, _ = parse_visdrone_dataset(
    "./VisDrone2019-DET-train/images",
    "./VisDrone2019-DET-train/annotations",
)
result = check_class_distribution(gt_anns)
```

### Step 3 — Apply local preprocessing

```python
from data.roboflow_pipeline import apply_local_preprocessing
from PIL import Image

img = Image.open("frame_001.jpg")
tensor = apply_local_preprocessing(img)   # → torch.Tensor [3, 640, 640]
```

### Step 4 — Write data.yaml manually

```yaml
names: [pedestrian, people, bicycle, car, van, truck,
        tricycle, awning-tricycle, bus, motor]
nc: 10
train: ./datasets/visdrone_yolo/train/images
val:   ./datasets/visdrone_yolo/val/images
test:  ./datasets/visdrone_yolo/test/images
```

---

## 11. Comparison: Roboflow vs Local Preprocessing

| Aspect | Roboflow | Local pipeline (`data/`) |
|---|---|---|
| **Setup time** | ~10 min (GUI upload) | ~5 min (scripts) |
| **Reproducibility** | Dataset version pinning via API | Git-tracked scripts + stats |
| **Annotation visualisation** | ✅ Built-in web viewer | ❌ Manual (use `labelImg`) |
| **Augmentation** | ✅ Baked into version OR online | ✅ `apply_local_preprocessing` |
| **Team sharing** | ✅ API key + version number | 📦 Zip / shared drive |
| **Offline use** | ❌ Requires internet for download | ✅ Fully offline |
| **Custom logic** | ❌ Limited to UI options | ✅ Full Python flexibility |
| **Cost** | Free tier: 10k images/version | Free (compute only) |
| **CI integration** | ✅ `rfcli` in GitHub Actions | ✅ Python script in CI |

> [!TIP]
> For hackathon-scale projects, Roboflow accelerates the annotation-verification and team-sharing steps significantly. For production or large-scale training (>100k images), the local pipeline is more cost-effective and flexible.

---

*Document generated: 2026-10-04 | Revision: v1.0*
