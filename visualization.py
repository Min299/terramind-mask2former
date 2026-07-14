"""
visualization.py

Visualization utilities for semantic segmentation benchmarking.
Updated to include Ground Truth alongside Predictions.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

def validate_palette(palette: Dict[int, Tuple[int, int, int]], class_names: Dict[int, str]):
    if palette is None: raise ValueError("Palette cannot be None.")
    if class_names is None: raise ValueError("Class names cannot be None.")
    for cls in class_names:
        if cls not in palette: raise ValueError(f"Missing palette entry for class {cls}")
        if len(palette[cls]) != 3: raise ValueError(f"Palette color for class {cls} must be RGB.")

def tensor_to_numpy(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 3: image = image.permute(1, 2, 0)
        image = image.numpy()
    elif isinstance(image, Image.Image):
        image = np.array(image)
    return image

def colorize_mask(prediction: np.ndarray, palette: Dict[int, Tuple[int, int, int]]):
    prediction = np.asarray(prediction)
    h, w = prediction.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in palette.items():
        colored[prediction == cls] = color
    return colored

def overlay_prediction(image, prediction, palette, alpha=0.40):
    image = tensor_to_numpy(image).astype(np.float32)
    prediction = colorize_mask(prediction, palette).astype(np.float32)
    if image.max() <= 1.0: image *= 255
    overlay = ((1 - alpha) * image + alpha * prediction)
    return np.clip(overlay, 0, 255).astype(np.uint8)

def build_legend(palette, class_names):
    handles = []
    for cls in sorted(class_names.keys()):
        color = np.array(palette[cls]) / 255.0
        handles.append(mpatches.Patch(color=color, label=class_names[cls]))
    return handles

def metrics_to_text(metrics):
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float): lines.append(f"{key:<12}: {value:.4f}")
        else: lines.append(f"{key:<12}: {value}")
    return "\n".join(lines)

def ensure_output_dir(save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path

def save_prediction_figure(
    image, prediction, metrics, palette, class_names, save_path, 
    ground_truth=None, title="Segmentation Benchmark", 
    image_name=None, model_name=None, task_name=None
):
    validate_palette(palette, class_names)
    save_path = ensure_output_dir(save_path)
    
    image_np = tensor_to_numpy(image)
    if image_np.max() <= 1.0: image_np = (image_np * 255).astype(np.uint8)
    
    pred_overlay = overlay_prediction(image_np, prediction, palette, alpha=0.40)
    
    # Determine columns based on whether Ground Truth is provided
    cols = 3 if ground_truth is not None else 2
    
    fig = plt.figure(figsize=(6 * cols, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, cols, height_ratios=[5, 1])

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(image_np)
    ax_orig.set_title("Original Image", fontsize=13, fontweight="bold")
    ax_orig.axis("off")

    if ground_truth is not None:
        gt_overlay = overlay_prediction(image_np, ground_truth, palette, alpha=0.40)
        ax_gt = fig.add_subplot(gs[0, 1])
        ax_gt.imshow(gt_overlay)
        ax_gt.set_title("Ground Truth", fontsize=13, fontweight="bold")
        ax_gt.axis("off")
        ax_pred = fig.add_subplot(gs[0, 2])
    else:
        ax_pred = fig.add_subplot(gs[0, 1])

    ax_pred.imshow(pred_overlay)
    ax_pred.set_title("Prediction", fontsize=13, fontweight="bold")
    ax_pred.axis("off")

    ax_bottom = fig.add_subplot(gs[1, :])
    ax_bottom.axis("off")
    ax_bottom.legend(handles=build_legend(palette, class_names), loc="upper left", fontsize=11, frameon=True, title="Classes")
    
    ax_bottom.text(
        0.45, 0.95, metrics_to_text(metrics), fontsize=11, va="top", family="monospace",
        transform=ax_bottom.transAxes, bbox=dict(facecolor="whitesmoke", edgecolor="gray", boxstyle="round,pad=0.5")
    )

    info = []
    if image_name: info.append(f"Image : {image_name}")
    if model_name: info.append(f"Model : {model_name}")
    if task_name: info.append(f"Task  : {task_name}")
    ax_bottom.text(0.80, 0.95, "\n".join(info), fontsize=11, va="top", ha="left", transform=ax_bottom.transAxes)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def visualize_prediction(**kwargs):
    save_prediction_figure(**kwargs)