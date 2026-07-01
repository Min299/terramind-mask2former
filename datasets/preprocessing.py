"""
Preprocessing utilities for datasets.
"""

import numpy as np
import torch


def normalize_image(image, mean=None, std=None):
    """
    Normalize image with mean and std.
    
    Args:
        image: np.ndarray of shape [C, H, W]
        mean: Mean values per channel
        std: Std values per channel
    
    Returns:
        Normalized image
    """
    if mean is not None:
        mean = np.array(mean).reshape(-1, 1, 1)
        image = image - mean
    
    if std is not None:
        std = np.array(std).reshape(-1, 1, 1)
        image = image / std
    
    return image


def prepare_multiclass_mask(mask, ignore_index=255):
    """
    Ensure mask is ready for training.
    
    Args:
        mask: np.ndarray of shape [H, W]
        ignore_index: Index to ignore
    
    Returns:
        Processed mask
    """
    mask = mask.astype(np.int64)
    mask[mask == ignore_index] = 0  # Map ignore to background
    return mask
