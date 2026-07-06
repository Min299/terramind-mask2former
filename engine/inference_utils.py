"""
inference_utils.py

Centralized utilities for TerraMind Mask2Former.
Ensures identical model reconstruction, target formatting, and post-processing 
across training, testing, and prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from encoder import TerraMindEncoder               
from neck import TerraMindNeck                     
from pixel_decoder import MSDeformAttnPixelDecoder 
from transformer_decoder import MultiScaleMaskedTransformerDecoder 
from multitask_model import MultiTaskMask2Former             


def build_model(config: dict) -> nn.Module:
    """
    Strict, config-driven model reconstruction. 
    Used identically by train.py, test.py, and predict.py.
    """
    model_cfg = config["MODEL"]
    tasks_cfg = config["TASKS"]

    # 1. Encoder
    encoder = TerraMindEncoder(**model_cfg.get("encoder", {}))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
        
    # A1/B1: No fallback values. Strict interface check.
    embed_dim = getattr(encoder, "embed_dim", None) or getattr(encoder, "out_channels", None)
    if embed_dim is None:
        raise AttributeError("Encoder must expose 'embed_dim' or 'out_channels'.")

    # 2. Neck
    neck = TerraMindNeck(embed_dim=embed_dim, **model_cfg.get("neck", {}))
    
    # 3. Pixel Decoder
    pixel_decoder = MSDeformAttnPixelDecoder(conv_dim=neck.hidden_dim, **model_cfg.get("pixel_decoder", {}))
    
    # 4. Task Decoders
    decoders = {}
    td_cfg = model_cfg.get("transformer_decoder", {})
    for task, t_cfg in tasks_cfg.items():
        decoders[task] = MultiScaleMaskedTransformerDecoder(
            in_channels=pixel_decoder.conv_dim, 
            num_classes=t_cfg["num_classes"], 
            mask_dim=pixel_decoder.mask_dim,
            **td_cfg
        )
        
    return MultiTaskMask2Former(encoder, neck, pixel_decoder, decoders)


def semantic_to_mask2former_targets(
    semantic_masks: torch.Tensor, 
    ignore_index: int,
    background_id: Optional[int] = None
) -> List[Dict[str, torch.Tensor]]:
    """
    A5: Standardized target converter. 
    Guarantees output dtypes: labels -> torch.long, masks -> torch.float32.
    """
    if semantic_masks.dtype != torch.long:
        raise ValueError(f"Semantic masks must be torch.long, got {semantic_masks.dtype}")

    targets = []
    for mask_i in semantic_masks:
        classes = torch.unique(mask_i)
        
        classes = classes[classes != ignore_index]
        if background_id is not None:
            classes = classes[classes != background_id]

        if len(classes) == 0:
            labels = torch.zeros(0, dtype=torch.long, device=mask_i.device)
            masks = torch.zeros((0, mask_i.shape[0], mask_i.shape[1]), dtype=torch.float32, device=mask_i.device)
        else:
            masks = [(mask_i == c) for c in classes]
            masks = torch.stack(masks).to(mask_i.device).float()
            labels = classes.to(torch.long)
            
        targets.append({"labels": labels, "masks": masks})
    return targets


def postprocess_mask2former_outputs(pred_logits: torch.Tensor, pred_masks: torch.Tensor, target_size: tuple) -> torch.Tensor:
    """Mask2Former inference post-processing."""
    pred_masks = F.interpolate(pred_masks, size=target_size, mode="bilinear", align_corners=False)
    prob = F.softmax(pred_logits, dim=-1)[..., :-1] 
    mask_pred = pred_masks.sigmoid()
    
    sem_seg = torch.einsum("bqc,bqhw->bchw", prob, mask_pred)
    return sem_seg.argmax(dim=1).to(torch.long)