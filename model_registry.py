"""
model_registry.py

Unified model loading and inference registry.
Strictly for TerraMind Mask2Former (Kaggle Environment).
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict
import torch
import yaml

# Safely import TerraMind utilities
# (Since detectron2 is installed in your Kaggle env, this will work natively)
from engine.inference_utils import build_model, load_checkpoint, postprocess_mask2former_outputs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Dynamic File Finder
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
    # Dynamically find the config in your cloned repo
    repo_dir = "/kaggle/working/terramind-mask2former"
    try:
        config_path = find_file(repo_dir, "config.yaml")
    except FileNotFoundError:
        config_path = find_file(repo_dir, "*.yaml")

    print(f"[*] Found config: {config_path}")
    print(f"[*] Loading weights: {spec['checkpoint']}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = build_model(config)
    model = load_checkpoint(
        model=model, 
        checkpoint_path=spec["checkpoint"], 
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
    "terramind_tiny": {
        "loader": load_terramind, 
        "predict": predict_terramind,
        "config": "", # (Handled dynamically)
        "checkpoint": "/kaggle/input/datasets/mintumushahary/terramind-mask2former-models/tiny_model.pth", 
    },
    "terramind_base": {
        "loader": load_terramind, 
        "predict": predict_terramind,
        "config": "", # (Handled dynamically)
        "checkpoint": "/kaggle/input/datasets/mintumushahary/terramind-mask2former-models/base_model.pth", 
    }
}
