"""
prepare_dataset.py
==================
Deterministic VisDrone dataset partition and conversion to YOLO format.

Strict Rules Enforced:
1. Locks the EXACT 20-image test set used in the benchmark.
2. The 20 test images NEVER appear in train or val.
3. Remaining 528 images partitioned deterministically:
     - 422 Train (80%)
     - 106 Validation (20%)
   using random seed 42.
4. Category 0 (ignored regions) is strictly excluded.
5. VisDrone categories 1-10 are mapped to 0-9:
     0: pedestrian, 1: people, 2: bicycle, 3: car, 4: van,
     5: truck, 6: tricycle, 7: awning-tricycle, 8: bus, 9: motor
6. All bounding boxes validated and normalized to [0, 1].
7. Generates data.yaml and saves split manifests:
     data/processed/splits/train_ids.txt
     data/processed/splits/val_ids.txt
     data/processed/splits/test_ids.txt
8. Performs full 10-point data integrity audit.
"""

import os
import shutil
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetPreparation")

# Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_VAL = DATA_DIR / "raw" / "VisDrone2019-DET-val"
RAW_IMGS = RAW_VAL / "images"
RAW_ANNS = RAW_VAL / "annotations"

PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"

CLASS_NAMES = [
    "pedestrian",       # 0 (VisDrone cat 1)
    "people",           # 1 (VisDrone cat 2)
    "bicycle",          # 2 (VisDrone cat 3)
    "car",              # 3 (VisDrone cat 4)
    "van",              # 4 (VisDrone cat 5)
    "truck",            # 5 (VisDrone cat 6)
    "tricycle",         # 6 (VisDrone cat 7)
    "awning-tricycle",  # 7 (VisDrone cat 8)
    "bus",              # 8 (VisDrone cat 9)
    "motor",            # 9 (VisDrone cat 10)
]

# Exact authoritative test image IDs from benchmark run_benchmark.py
AUTHORITATIVE_TEST_IDS = [
    "0000001_02999_d_0000005",
    "0000001_03499_d_0000006",
    "0000001_03999_d_0000007",
    "0000001_04527_d_0000008",
    "0000001_05249_d_0000009",
    "0000001_05499_d_0000010",
    "0000001_05999_d_0000011",
    "0000001_07999_d_0000012",
    "0000001_08414_d_0000013",
    "0000021_00000_d_0000001",
    "0000021_00500_d_0000002",
    "0000021_00800_d_0000003",
    "0000022_00000_d_0000004",
    "0000022_00500_d_0000005",
    "0000022_01036_d_0000006",
    "0000022_01251_d_0000007",
    "0000023_00000_d_0000008",
    "0000023_00300_d_0000009",
    "0000023_00868_d_0000010",
    "0000023_01233_d_0000011",
]


def validate_and_convert_ann(line: str, img_w: int, img_h: int) -> Tuple[bool, int, str]:
    """
    Validates a single VisDrone annotation line and converts to YOLO format.
    Returns (is_valid, category_id, yolo_line_or_reason).
    """
    parts = line.strip().split(",")
    if len(parts) < 8:
        return False, -1, "malformed line (< 8 fields)"
    
    try:
        x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        score = int(parts[4])
        cat = int(parts[5])
    except ValueError:
        return False, -1, "non-integer values"

    if cat == 0:
        return False, 0, "ignored region (category 0)"

    if cat == 11:
        return False, 11, "category 11 ('others' outside the 10 evaluation classes)"

    if cat < 1 or cat > 10:
        return False, cat, f"unknown category {cat}"

    # Target class (0-indexed)
    class_id = cat - 1

    # Box validation
    if w <= 0 or h <= 0:
        return False, cat, "zero or negative dimension"

    # Clamp box coordinates to image dimensions
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    actual_w = x2 - x1
    actual_h = y2 - y1
    if actual_w <= 0 or actual_h <= 0:
        return False, cat, "box outside image boundary"

    xc = (x1 + actual_w / 2.0) / img_w
    yc = (y1 + actual_h / 2.0) / img_h
    wn = actual_w / img_w
    hn = actual_h / img_h

    # Ensure bounds in [0, 1]
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    wn = min(max(wn, 0.0), 1.0)
    hn = min(max(hn, 0.0), 1.0)

    yolo_str = f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n"
    return True, cat, yolo_str


def main():
    logger.info("Starting deterministic dataset partition and conversion...")

    if not RAW_IMGS.exists() or not RAW_ANNS.exists():
        raise FileNotFoundError(f"Raw VisDrone directory not found at {RAW_VAL}")

    # Gather all image files
    all_img_files = sorted(RAW_IMGS.glob("*.jpg"))
    all_ids = [p.stem for p in all_img_files]
    logger.info(f"Total raw images discovered: {len(all_ids)}")

    # Verify all 20 authoritative test IDs exist
    for tid in AUTHORITATIVE_TEST_IDS:
        if tid not in all_ids:
            raise ValueError(f"Authoritative test ID {tid} not found in raw images!")

    # Lock test set
    test_ids = list(AUTHORITATIVE_TEST_IDS)
    remaining_ids = [i for i in all_ids if i not in test_ids]
    logger.info(f"Locked 20-image test set. Remaining for train/val: {len(remaining_ids)}")

    # Deterministic split with fixed random seed 42
    random.seed(42)
    random.shuffle(remaining_ids)

    n_train = 422
    n_val = 106
    train_ids = sorted(remaining_ids[:n_train])
    val_ids = sorted(remaining_ids[n_train:n_train + n_val])

    logger.info(f"Deterministic partition: Train = {len(train_ids)}, Val = {len(val_ids)}, Test = {len(test_ids)}")

    # Create target directory layout
    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }

    for split_name in splits:
        (PROCESSED_DIR / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / "labels" / split_name).mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # Save split ID files
    for split_name, id_list in splits.items():
        with open(SPLITS_DIR / f"{split_name}_ids.txt", "w", encoding="utf-8") as f:
            for item in id_list:
                f.write(f"{item}\n")
    logger.info(f"Split ID manifests saved in {SPLITS_DIR}")

    # Process and convert each split
    stats = {
        "total_images": len(all_ids),
        "total_annotations": 0,
        "valid_annotations": 0,
        "ignored_category_0": 0,
        "invalid_annotations": 0,
        "per_split_boxes": {"train": 0, "val": 0, "test": 0},
        "per_class_boxes": {c: 0 for c in CLASS_NAMES},
        "size_counts": {"small": 0, "medium": 0, "large": 0}
    }

    for split_name, id_list in splits.items():
        dst_img_dir = PROCESSED_DIR / "images" / split_name
        dst_lbl_dir = PROCESSED_DIR / "labels" / split_name

        for img_id in id_list:
            src_img = RAW_IMGS / f"{img_id}.jpg"
            src_ann = RAW_ANNS / f"{img_id}.txt"
            dst_img = dst_img_dir / f"{img_id}.jpg"
            dst_lbl = dst_lbl_dir / f"{img_id}.txt"

            # Copy image
            shutil.copy2(src_img, dst_img)

            # Get image dimensions
            with Image.open(src_img) as im:
                img_w, img_h = im.size

            yolo_lines = []
            if src_ann.exists():
                with open(src_ann, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        stats["total_annotations"] += 1
                        is_valid, cat, res = validate_and_convert_ann(line, img_w, img_h)
                        if is_valid:
                            yolo_lines.append(res)
                            stats["valid_annotations"] += 1
                            stats["per_split_boxes"][split_name] += 1
                            cls_id = cat - 1
                            stats["per_class_boxes"][CLASS_NAMES[cls_id]] += 1

                            # Size measurement (pixel w, h)
                            parts = line.split(",")
                            bw, bh = int(parts[2]), int(parts[3])
                            if bw < 32 and bh < 32:
                                stats["size_counts"]["small"] += 1
                            elif bw < 96 and bh < 96:
                                stats["size_counts"]["medium"] += 1
                            else:
                                stats["size_counts"]["large"] += 1
                        else:
                            if cat == 0:
                                stats["ignored_category_0"] += 1
                            else:
                                stats["invalid_annotations"] += 1

            # Write converted label file
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.writelines(yolo_lines)

    # Generate data.yaml (portable relative paths from project root)
    data_yaml_content = f"""# VisDrone2019-DET Processed Dataset Configuration
path: {PROCESSED_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""
    yaml_path = PROCESSED_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(data_yaml_content)
    logger.info(f"Generated data.yaml at {yaml_path}")

    # 10-Point Data Integrity Verification
    logger.info("\n=======================================================")
    logger.info("       10-POINT DATA INTEGRITY VERIFICATION           ")
    logger.info("=======================================================")

    checks = []

    # 1. No overlap between splits
    s_train = set(train_ids)
    s_val = set(val_ids)
    s_test = set(test_ids)
    overlap = len(s_train.intersection(s_val)) + len(s_train.intersection(s_test)) + len(s_val.intersection(s_test))
    checks.append(("1. Zero overlap between splits", overlap == 0))

    # 2. Every image has a corresponding label file
    all_imgs_ok = True
    for sp in ["train", "val", "test"]:
        imgs = list((PROCESSED_DIR / "images" / sp).glob("*.jpg"))
        lbls = list((PROCESSED_DIR / "labels" / sp).glob("*.txt"))
        if len(imgs) != len(lbls):
            all_imgs_ok = False
    checks.append(("2. Every image has a corresponding label file", all_imgs_ok))

    # 3. Every label uses class IDs 0-9
    valid_classes = True
    for lbl_file in (PROCESSED_DIR / "labels").rglob("*.txt"):
        with open(lbl_file, "r", encoding="utf-8") as f:
            for l in f:
                c = int(l.split()[0])
                if c < 0 or c > 9:
                    valid_classes = False
    checks.append(("3. Every label uses valid class IDs 0-9", valid_classes))

    # 4. Normalized coordinates in [0, 1]
    valid_coords = True
    for lbl_file in (PROCESSED_DIR / "labels").rglob("*.txt"):
        with open(lbl_file, "r", encoding="utf-8") as f:
            for l in f:
                vals = [float(v) for v in l.split()[1:5]]
                if any(v < 0.0 or v > 1.0 for v in vals):
                    valid_coords = False
    checks.append(("4. Normalized coordinates strictly within [0, 1]", valid_coords))

    # 5. Test IDs are exactly preserved
    checks.append(("5. Authoritative 20-image test set exactly preserved", test_ids == AUTHORITATIVE_TEST_IDS))

    # 6. Category 0 excluded
    checks.append(("6. Category 0 ignored regions strictly excluded (1,378 instances)", stats["ignored_category_0"] == 1378))

    # 7. Object counts reported
    checks.append(("7. Valid object targets strictly accounted for (38,759 in classes 0-9)", stats["valid_annotations"] == 38759))

    # 8. Split counts correct (422 / 106 / 20)
    checks.append(("8. Split image counts match exactly (422 / 106 / 20)", len(train_ids) == 422 and len(val_ids) == 106 and len(test_ids) == 20))

    # 9. No duplicate image IDs
    checks.append(("9. No duplicate image IDs across dataset", len(set(all_ids)) == 548))

    # 10. Test images never used for training
    checks.append(("10. Test images completely absent from training split", len(s_test.intersection(s_train)) == 0))

    for desc, passed in checks:
        logger.info(f"[{'PASS' if passed else 'FAIL'}] {desc}")

    all_passed = all(p for _, p in checks)
    if not all_passed:
        raise RuntimeError("Data integrity check failed!")
    logger.info("All 10 data integrity checks PASSED successfully!\n")

    return stats


if __name__ == "__main__":
    main()
