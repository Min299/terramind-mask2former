import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from models.terramind_encoder import TerraMindEncoder               
from models.terramind_neck import TerraMindNeck                     
from models.pixel_decoder import MSDeformAttnPixelDecoder 
from models.transformer_decoder import MultiScaleMaskedTransformerDecoder 
from models.multitask_model import MultiTaskMask2Former             


def build_model(config: dict) -> nn.Module:
    model_cfg = config.get("MODEL", config.get("model", config))
    tasks_cfg = config.get("TASKS", config.get("tasks", []))

    encoder = TerraMindEncoder(**model_cfg.get("encoder", {}))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
        
    embed_dim = getattr(encoder, "out_channels", None)
    if isinstance(embed_dim, (list, tuple)) or type(embed_dim).__name__ == "ListConfig":
        embed_dim = embed_dim[0]
    if embed_dim is None or embed_dim <= 0:
        raise AttributeError("Encoder must expose canonical 'out_channels' > 0.")

    # Grab the global dimensions from MODEL config
    hidden_dim = model_cfg.get("hidden_dim", model_cfg.get("neck", {}).get("hidden_dim", 256))
    mask_dim = model_cfg["mask_dim"]

    # Explicitly pass hidden_dim to neck
    neck = TerraMindNeck(embed_dim=embed_dim, hidden_dim=hidden_dim, **model_cfg.get("neck", {}))
    
    # Explicitly pass hidden_dim as conv_dim
    pixel_decoder = MSDeformAttnPixelDecoder(conv_dim=hidden_dim, mask_dim=mask_dim, **model_cfg.get("pixel_decoder", {}))
    
    decoders = {}
    td_cfg = model_cfg.get("transformer_decoder", {})
    for task, t_cfg in tasks_cfg.items():
        decoders[task] = MultiScaleMaskedTransformerDecoder(
            in_channels=hidden_dim,  # Tied explicitly to the global hidden_dim
            num_classes=t_cfg["num_classes"], 
            mask_dim=mask_dim,       # Tied explicitly to the global mask_dim
            **td_cfg
        )
        
    return MultiTaskMask2Former(encoder, neck, pixel_decoder, decoders)

def semantic_to_mask2former_targets(
    semantic_masks: torch.Tensor, 
    num_classes: int,
    ignore_index: int,
    background_id: Optional[int] = None
) -> List[Dict[str, torch.Tensor]]:
    """
    Standardized target converter. 
    Guarantees output dtypes: labels -> torch.long, masks -> torch.float32.
    """
    if semantic_masks.dtype != torch.long:
        raise TypeError(f"Semantic masks must be torch.long, got {semantic_masks.dtype}")

    # FIX: Prevent illogical config setups
    if background_id is not None and background_id == ignore_index:
        raise ValueError(f"background_id ({background_id}) cannot equal ignore_index ({ignore_index}).")

    targets = []
    for mask_i in semantic_masks:
        classes = torch.unique(mask_i)
        
        classes = classes[classes != ignore_index]
        if background_id is not None:
            classes = classes[classes != background_id]

        # FIX: Validate class IDs don't exceed the model's prediction head
        if len(classes) > 0 and classes.max() >= num_classes:
            raise ValueError(f"Found Class ID {classes.max()} in mask, but num_classes is {num_classes}")

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
    # FIX: Validate Query dimensions match
    if pred_logits.shape[1] != pred_masks.shape[1]:
        raise ValueError(f"Query mismatch: logits have {pred_logits.shape[1]} queries, masks have {pred_masks.shape[1]}")

    pred_masks = F.interpolate(pred_masks, size=target_size, mode="bilinear", align_corners=False)
    prob = F.softmax(pred_logits, dim=-1)[..., :-1] 
    mask_pred = pred_masks.sigmoid()
    
    sem_seg = torch.einsum("bqc,bqhw->bchw", prob, mask_pred)
    return sem_seg.argmax(dim=1).to(torch.long)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str, current_config: dict = None) -> torch.nn.Module:
    """Loads model weights with DDP compatibility and config fingerprinting."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Config fingerprint validation
    if current_config is not None and "config" in checkpoint:
        saved_config = checkpoint["config"]
        saved_tasks = {k: v["num_classes"] for k, v in saved_config.get("TASKS", {}).items()}
        curr_tasks = {k: v["num_classes"] for k, v in current_config.get("TASKS", {}).items()}
        if saved_tasks != curr_tasks:
            raise RuntimeError(f"Checkpoint tasks {saved_tasks} do not match current config {curr_tasks}")

    state_dict = checkpoint.get("model_state", checkpoint)
    
    # FIX: Handle Multi-GPU (DDP) saved checkpoints by stripping "module." prefix
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k[7:] if k.startswith("module.") else k
        clean_state_dict[clean_key] = v

    model.load_state_dict(clean_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model