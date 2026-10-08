"""
prepare_roboflow_package.py
Packages the VisDrone dataset into the exact structure expected by Roboflow:
  - train/images & train/labels (422 images)
  - valid/images & valid/labels (106 images)
  - test/images & test/labels (20 locked test images)
  - data.yaml (relative paths, 10 classes)
  - classes.txt (10 class names)
Creates both:
  1. An unzipped directory: data/roboflow_upload_package/
  2. A ready-to-upload zip archive: data/visdrone_roboflow_upload.zip
"""
import os
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PACKAGE_DIR = DATA_DIR / "roboflow_upload_package"
ZIP_PATH = DATA_DIR / "visdrone_roboflow_upload.zip"

CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor"
]

DATA_YAML_CONTENT = """# VisDrone2019-DET Roboflow Import Configuration
train: train/images
val: valid/images
test: test/images

nc: 10
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

README_CONTENT = """VisDrone2019-DET Benchmark Dataset for Roboflow
===================================================
Total Images: 548
Total Classes: 10

Partitions:
- Train: 422 images (train/images, train/labels)
- Valid: 106 images (valid/images, valid/labels)
- Test:  20 images (test/images, test/labels) [LOCKED BENCHMARK TEST SET]

Class Mapping (0-indexed):
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

Note: The 20 test images match the exact locked test split used in the
comparative benchmark for YOLOv8s, Faster R-CNN, Mask R-CNN, and RF-DETR.
"""

def main():
    print("=" * 60)
    print("PREPARING ROBOFLOW UPLOAD PACKAGE")
    print("=" * 60)

    # Clean existing package dir if present
    if PACKAGE_DIR.exists():
        print(f"Cleaning existing directory: {PACKAGE_DIR}")
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Create split subdirectories
    splits_map = {
        "train": "train",
        "val": "valid",    # Roboflow standard is 'valid'
        "test": "test"
    }

    for src_split, dst_split in splits_map.items():
        dst_img_dir = PACKAGE_DIR / dst_split / "images"
        dst_lbl_dir = PACKAGE_DIR / dst_split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        src_img_dir = PROCESSED_DIR / "images" / src_split
        src_lbl_dir = PROCESSED_DIR / "labels" / src_split

        img_files = list(src_img_dir.glob("*.jpg"))
        lbl_files = list(src_lbl_dir.glob("*.txt"))

        print(f"\nProcessing {src_split} -> {dst_split}: {len(img_files)} images, {len(lbl_files)} labels")

        for img_p in img_files:
            shutil.copy2(img_p, dst_img_dir / img_p.name)

        for lbl_p in lbl_files:
            shutil.copy2(lbl_p, dst_lbl_dir / lbl_p.name)

    # Write data.yaml
    yaml_file = PACKAGE_DIR / "data.yaml"
    yaml_file.write_text(DATA_YAML_CONTENT, encoding="utf-8")
    print(f"\nWritten {yaml_file}")

    # Write classes.txt
    classes_file = PACKAGE_DIR / "classes.txt"
    classes_file.write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    print(f"Written {classes_file}")

    # Write README.txt
    readme_file = PACKAGE_DIR / "README.txt"
    readme_file.write_text(README_CONTENT, encoding="utf-8")
    print(f"Written {readme_file}")

    # Create zip archive
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    print(f"\nCreating compressed archive at {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in PACKAGE_DIR.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(PACKAGE_DIR)
                zipf.write(file_path, arcname)

    zip_size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"\nZip archive successfully created: {ZIP_PATH} ({zip_size_mb:.2f} MB)")

    # Also copy zip to project root for convenience
    root_zip = PROJECT_ROOT / "visdrone_roboflow_upload.zip"
    shutil.copy2(ZIP_PATH, root_zip)
    print(f"Convenience copy created at: {root_zip}")

    print("\n" + "=" * 60)
    print("PACKAGE VERIFICATION COMPLETE:")
    print(f"  - Package Directory: {PACKAGE_DIR}")
    print(f"  - Zip File:          {ZIP_PATH}")
    print(f"  - Root Zip File:     {root_zip}")
    print("=" * 60)

if __name__ == "__main__":
    main()
