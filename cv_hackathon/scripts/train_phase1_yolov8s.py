"""
train_phase1_yolov8s.py
Phase 1: Full 100-Epoch YOLOv8s Training on VisDrone2019-DET with early stopping.
"""
import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    print("=" * 60)
    print("PHASE 1: FULL 100-EPOCH YOLOv8s TRAINING (VISDRONE)")
    print("=" * 60)
    
    assert torch.cuda.is_available(), "CUDA GPU is required but not available."
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Device: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    
    data_yaml = Path("data/processed/data.yaml").resolve()
    assert data_yaml.exists(), f"data.yaml not found at {data_yaml}"
    
    output_dir = Path("results/final_optimization/training").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize YOLOv8s pretrained on COCO as backbone feature extractor
    model = YOLO("yolov8s.pt")
    
    # Training configuration with early stopping (patience=25)
    train_args = {
        "data": str(data_yaml),
        "epochs": 100,
        "patience": 25,
        "batch": 8,          # Stable for 8GB VRAM
        "imgsz": 640,
        "device": 0,
        "workers": 0,        # Critical for Windows stability
        "optimizer": "auto",
        "lr0": 0.01,
        "lrf": 0.01,
        "seed": 42,
        "deterministic": True,
        "amp": True,
        "project": str(output_dir),
        "name": "yolov8s_100ep_640",
        "exist_ok": True,
        "save": True,
        "plots": True,
        "verbose": True
    }
    
    print("\nTraining configuration:")
    for k, v in train_args.items():
        print(f"  {k}: {v}")
        
    print("\nStarting model training...")
    results = model.train(**train_args)
    print("\nTraining completed successfully.")

if __name__ == "__main__":
    main()
