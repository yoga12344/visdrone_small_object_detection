"""
benchmark_plots.py
==================
Generates all 11 scientific benchmark charts requested for the technical report:
 1. mAP@0.5 Comparison
 2. Precision Comparison
 3. Recall Comparison
 4. F1 Comparison
 5. Small-Object mAP Comparison
 6. Mean IoU Comparison
 7. Inference Time Comparison (latency)
 8. FPS Comparison
 9. Model Size & Parameter Count Comparison
10. Accuracy vs. FPS (Trade-off Frontier)
11. Accuracy vs. Model Size (Efficiency Frontier)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any


def plot_all_benchmark_charts(summary_metrics: Dict[str, Dict[str, Any]], output_dir: str):
    """
    Given summary metrics dict:
      {
         'YOLOv8s': {'mAP50': ..., 'Precision': ..., 'Recall': ..., 'F1': ..., 'Small_mAP': ..., 'MeanIoU': ..., 'Time_ms': ..., 'FPS': ..., 'Params_M': ..., 'Size_MB': ...},
         ...
      }
    Generates high-resolution publication-quality PNG charts in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    models = list(summary_metrics.keys())
    
    # Palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    model_colors = {m: colors[i % len(colors)] for i, m in enumerate(models)}
    
    # Helper for simple bar charts
    def make_bar_chart(metric_key, ylabel, title, filename, multiplier=1.0, is_percent=False):
        fig, ax = plt.subplots(figsize=(8, 5))
        vals = [summary_metrics[m].get(metric_key, 0.0) * multiplier for m in models]
        bars = ax.bar(models, vals, color=[model_colors[m] for m in models], width=0.55, edgecolor='black', linewidth=0.8)
        
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Add values on top of bars
        for bar in bars:
            yval = bar.get_height()
            fmt = f"{yval:.1f}%" if is_percent else f"{yval:.3f}" if yval < 10 else f"{yval:.1f}"
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + (max(vals)*0.02 if max(vals)>0 else 0.01),
                    fmt, ha='center', va='bottom', fontsize=9, fontweight='bold')
            
        plt.xticks(rotation=15, ha='right', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=200)
        plt.close()

    # 1. mAP@0.5
    make_bar_chart('mAP50', 'mAP@0.5', 'mAP@0.5 on VisDrone Aerial Imagery', '01_map50_comparison.png')

    # 2. Precision
    make_bar_chart('Precision', 'Precision', 'Detection Precision Comparison', '02_precision_comparison.png')

    # 3. Recall
    make_bar_chart('Recall', 'Recall', 'Detection Recall Comparison', '03_recall_comparison.png')

    # 4. F1 Score
    make_bar_chart('F1', 'F1 Score', 'F1 Harmonic Mean Comparison', '04_f1_comparison.png')

    # 5. Small-Object mAP
    make_bar_chart('Small_mAP', 'Small-Object mAP@0.5', 'Small-Object (<32x32px) Detection Accuracy', '05_small_object_map.png')

    # 6. Mean IoU
    make_bar_chart('MeanIoU', 'Mean Bounding Box IoU', 'Localization Quality (Mean IoU on Matched Targets)', '06_mean_iou_comparison.png')

    # 7. Inference Latency (ms)
    make_bar_chart('Time_ms', 'Latency per Frame (ms)', 'Inference Latency per Frame (Lower is Better)', '07_inference_time_ms.png')

    # 8. FPS
    make_bar_chart('FPS', 'Frames Per Second (FPS)', 'Throughput (FPS, Higher is Better)', '08_fps_comparison.png')

    # 9. Model Size
    make_bar_chart('Size_MB', 'Model Checkpoint Size (MB)', 'Model Footprint (Disk / Checkpoint Size)', '09_model_size_mb.png')

    # 10. Accuracy vs FPS Scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in models:
        fps = summary_metrics[m].get('FPS', 0.0)
        acc = summary_metrics[m].get('mAP50', 0.0)
        ax.scatter(fps, acc, color=model_colors[m], s=180, edgecolors='black', linewidth=1.2, zorder=5)
        ax.annotate(f" {m}", (fps, acc), fontsize=10, fontweight='bold', xytext=(6, 4), textcoords='offset points')
    ax.set_xlabel('Throughput (Frames Per Second, FPS)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (mAP@0.5)', fontsize=11, fontweight='bold')
    ax.set_title('Accuracy vs. Speed Frontier (VisDrone)', fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '10_accuracy_vs_fps.png'), dpi=200)
    plt.close()

    # 11. Accuracy vs Model Size Scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in models:
        sz = summary_metrics[m].get('Size_MB', 0.0)
        acc = summary_metrics[m].get('mAP50', 0.0)
        ax.scatter(sz, acc, color=model_colors[m], s=180, edgecolors='black', linewidth=1.2, zorder=5)
        ax.annotate(f" {m}", (sz, acc), fontsize=10, fontweight='bold', xytext=(6, 4), textcoords='offset points')
    ax.set_xlabel('Model Size (MB)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (mAP@0.5)', fontsize=11, fontweight='bold')
    ax.set_title('Accuracy vs. Model Size (Footprint Efficiency)', fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '11_accuracy_vs_size.png'), dpi=200)
    plt.close()

    print(f"Generated all 11 scientific benchmark plots in: {output_dir}")
