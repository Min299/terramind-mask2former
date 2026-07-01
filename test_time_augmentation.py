"""
Test Time Augmentation for semantic segmentation.
"""

import copy
import torch
import torch.nn.functional as F
from torch import nn


class SemanticSegmentorWithTTA(nn.Module):
    """
    A SemanticSegmentor with test-time augmentation enabled.
    
    Supports:
    - Horizontal flip
    - Multi-scale inference
    """

    def __init__(self, model, tta_transforms=None, batch_size=1):
        """
        Args:
            model: The segmentation model to apply TTA on.
            tta_transforms: List of TTA transforms. Defaults to horizontal flip.
            batch_size: Batch size for inference.
        """
        super().__init__()
        self.model = model
        self.tta_transforms = tta_transforms or ['hflip']
        self.batch_size = batch_size

    def forward(self, x):
        """
        Forward pass with TTA.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            predictions: Averaged predictions from all augmentations
        """
        original_shape = x.shape[-2:]
        predictions = []
        
        for transform in self.tta_transforms:
            x_aug = self._apply_transform(x, transform)
            pred = self.model(x_aug)
            
            # Reverse transform on predictions
            pred = self._reverse_transform(pred, transform, original_shape)
            predictions.append(pred)
        
        # Average predictions
        final_pred = torch.stack(predictions).mean(dim=0)
        return final_pred

    def _apply_transform(self, x, transform):
        """Apply a transform to the input."""
        if transform == 'hflip':
            return x.flip(-1)
        elif transform == 'vflip':
            return x.flip(-2)
        elif isinstance(transform, tuple) and transform[0] == 'scale':
            # Scale transform
            scale_factor = transform[1]
            h, w = x.shape[-2:]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            return F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        return x

    def _reverse_transform(self, pred, transform, original_shape):
        """Reverse a transform on predictions."""
        if transform == 'hflip':
            return pred.flip(-1)
        elif transform == 'vflip':
            return pred.flip(-2)
        elif isinstance(transform, tuple) and transform[0] == 'scale':
            # Scale back to original size
            return F.interpolate(pred, size=original_shape, mode='bilinear', align_corners=False)
        return pred


def get_tta_transforms(config):
    """
    Get TTA transforms from config.
    
    Args:
        config: Configuration dict
    
    Returns:
        List of TTA transforms
    """
    transforms = []
    
    if config.get('tta_hflip', False):
        transforms.append('hflip')
    
    if config.get('tta_vflip', False):
        transforms.append('vflip')
    
    if config.get('tta_scales', None):
        scales = config['tta_scales']
        transforms.extend([('scale', s) for s in scales])
    
    return transforms if transforms else ['hflip']  # Default to hflip
