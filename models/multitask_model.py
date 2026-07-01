"""
MultiTask Mask2Former Model.


Wraps shared encoder, neck, pixel decoder with task-specific decoders.
"""

from typing import Dict

import torch
import torch.nn as nn


class MultiTaskMask2Former(nn.Module):
    """
    Multi-task Mask2Former model.
    
    Architecture:
    1. Frozen TerraMind encoder
    2. TerraMind neck (tokens → feature maps)
    3. Shared MSDeformAttn pixel decoder
    4. Task-specific transformer decoders
    """

    def __init__(
        self,
        encoder,
        neck,
        pixel_decoder,
        decoders: Dict[str, nn.Module],
        task_default: str = "flood",
    ):
        super().__init__()
        
        # Encoder (frozen)
        self.encoder = encoder
        
        # Neck
        self.neck = neck
        
        # Shared pixel decoder
        self.pixel_decoder = pixel_decoder
        
        # Task-specific decoders
        self.decoders = nn.ModuleDict(decoders)
        
        self.task_default = task_default

    def forward(self, x, task=None):
        """
        Forward pass.
        
        Args:
            x: Input image tensor [B, C, H, W]
            task: Task name (str). If None, uses default.
        
        Returns:
            Dictionary with predictions and active decoder name:
            {
                "pred_logits": ...,
                "pred_masks": ...,
                "aux_outputs": [...],
                "active_decoder": task_name,
            }
        """
        if task is None:
            task = self.task_default
        
        # Encode with TerraMind
        encoder_outputs = self.encoder(x)
        
        # Convert tokens to feature maps
        features = self.neck(encoder_outputs)
        
        # Pixel decoder (shared)
        mask_features, multi_scale_features = self.pixel_decoder(features)
        
        # Task-specific decoder
        if task not in self.decoders:
            raise ValueError(f"Unknown task: {task}. Available: {list(self.decoders.keys())}")
        
        decoder = self.decoders[task]
        outputs = decoder(multi_scale_features, mask_features)
        
        # Add active decoder name for debugging
        outputs["active_decoder"] = task
        
        return outputs

    def get_trainable_params(self):
        """Get trainable parameters (exclude frozen encoder)."""
        trainable = []
        for name, param in self.named_parameters():
            if param.requires_grad and not name.startswith("encoder."):
                trainable.append(param)
        return trainable

    def freeze_encoder(self):
        """Freeze the encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
