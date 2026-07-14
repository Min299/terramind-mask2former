"""
model_registry.py

Unified model loading and inference registry.
Strictly for TerraMind Mask2Former (Kaggle Environment).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict
import torch
import yaml
from unittest.mock import MagicMock

# ============================================================
# BYPASS TRAINING DEPENDENCIES (DETECTRON2)
# ============================================================
# We fake the module in memory so the training losses don't crash the inference script
sys.modules['detectron2'] = MagicMock()
sys.modules['detectron2.projects'] = MagicMock()
sys.modules['detectron2.projects.point_rend'] = MagicMock()
sys.modules['detectron2.projects.point_rend.point_features'] = MagicMock()

# Now we can safely import TerraMind utilities
from engine.inference_utils import build_model, load_checkpoint, postprocess_mask2former_outputs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Dynamic File Finders
# ============================================================
def find_file(base_dir: str, pattern: str) -> str:
    """Recursively search for a file matching the pattern."""
    matches = list(Path(base_dir).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find '{pattern}' inside {base_dir}")
    return str(matches[0])

# ============================================================
# TerraMind Loader
# ============================================================
def load_terramind(spec: Dict):
    base_dir = "/kaggle/working/terramind-mask2former"
    
    # 1. Dynamically find the best model checkpoint
    checkpoint_path = find_file(f"{base_dir}/checkpoints", "best_model.pth")
    print(f"[*] Found checkpoint: {checkpoint_path}")
    
    # 2. Dynamically find the YAML config
    try:
        config_path = find_file(base_dir, "config.yaml")
    except FileNotFoundError:
        config_path = find_file(base_dir, "*.yaml") # Fallback to any yaml
    print(f"[*] Found config: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = build_model(config)
    model = load_checkpoint(
        model=model, 
        checkpoint_path=checkpoint_path, 
        device=str(DEVICE), 
        current_config=config
    )
    model.eval()
    return model, config

@torch.inference_mode()
def predict_terramind(model, images, task, config, **kwargs):
    # Map friendly task names to Kaggle YAML keys
    task_map = {"flood": "sen1floods11", "burn": "fire_scars", "landcover": "m_chesapeake_landcover"}
    yaml_task_name = task_map.get(task, task)
    
    task_cfg = config["TASKS"][yaml_task_name]
    
    # Scale image tensors based on your Kaggle config (e.g. 10000.0)
    images = images.float() / task_cfg.get("scale_factor", 1.0)

    use_amp = (DEVICE.type == "cuda")
    with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
        outputs = model(images, task=yaml_task_name)

    prediction = postprocess_mask2former_outputs(
        pred_logits=outputs["pred_logits"], 
        pred_masks=outputs["pred_masks"], 
        target_size=images.shape[-2:]
    )
    return prediction.cpu()

# ============================================================
# Registry
# ============================================================
MODEL_REGISTRY = {
    "terramind_model": {
        "loader": load_terramind, 
        "predict": predict_terramind,
        "config": "",       # Resolved dynamically in load_terramind
        "checkpoint": "",   # Resolved dynamically in load_terramind
    }
}