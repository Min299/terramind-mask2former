"""
metrics.py

Centralized semantic segmentation metrics tracking using Confusion Matrices.
"""

import numpy as np
import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.detach().cpu().numpy().flatten()
        targets = targets.detach().cpu().numpy().flatten()
        
        valid_mask = targets != self.ignore_index
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        
        bins = self.num_classes * targets + preds
        hist = np.bincount(bins, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += hist

    def compute(self, reset: bool = False, return_confusion: bool = False) -> dict:
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        
        eps = 1e-6
        iou = tp / (tp + fp + fn + eps)
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        class_acc = tp / (tp + fn + eps)  # Class-wise accuracy
        
        valid = (tp + fp + fn) > 0
        
        miou = np.mean(iou[valid]) if np.any(valid) else 0.0
        mdice = np.mean(dice[valid]) if np.any(valid) else 0.0
        mprecision = np.mean(precision[valid]) if np.any(valid) else 0.0
        mrecall = np.mean(recall[valid]) if np.any(valid) else 0.0
        macc = np.mean(class_acc[valid]) if np.any(valid) else 0.0
        
        pixel_acc = tp.sum() / (self.confusion_matrix.sum() + eps)
        
        results = {
            "iou": iou.tolist(),
            "miou": float(miou),
            "dice": dice.tolist(),
            "mdice": float(mdice),
            "precision": precision.tolist(),
            "mean_precision": float(mprecision),
            "recall": recall.tolist(),
            "mean_recall": float(mrecall),
            "class_accuracy": class_acc.tolist(),
            "mean_accuracy": float(macc),
            "pixel_accuracy": float(pixel_acc),
        }
        
        if return_confusion:
            results["confusion_matrix"] = self.confusion_matrix.copy()
            
        if reset:
            self.reset()
            
        return results

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)