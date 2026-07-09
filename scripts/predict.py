"""
predict.py

Inference script for TerraMind Mask2Former.
Supports standard RGB formats and multi-band GeoTIFFs.
Ensures preprocessing exactly matches the training transformations.
"""

import os
import glob
import argparse
import yaml
import torch
import numpy as np
from PIL import Image

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

import torchvision.transforms.functional as TF

# FIX 1: Updated imports to match inference_utils.py
from engine.inference_utils import build_model, load_checkpoint, postprocess_mask2former_outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Predict with TerraMind Mask2Former")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml (from checkpoint dir)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--task", type=str, required=True, help="Task to predict (e.g., 'flood')")
    parser.add_argument("--input", type=str, required=True, help="Path to image or folder of images")
    parser.add_argument("--output", type=str, required=True, help="Directory to save predictions")
    parser.add_argument("--overlay", action="store_true", help="Save blended overlay images")
    return parser.parse_args()


def validate_prediction_config(task_config: dict):
    """Fails fast if visualization metadata is invalid."""
    if "palette" not in task_config:
        raise ValueError("Config missing 'palette' for visualization.")
        
    palette = task_config["palette"]
    num_classes = task_config["num_classes"]
    
    for c in range(num_classes):
        if c not in palette:
            raise ValueError(f"Palette missing color mapping for class index {c}")
        if len(palette[c]) != 3:
            raise ValueError(f"Palette color for class {c} must be an RGB triplet.")
            
    if "rgb_bands" in task_config:
        rgb_bands = task_config["rgb_bands"]
        if len(set(rgb_bands)) != len(rgb_bands):
            raise ValueError(f"RGB bands must be unique, got {rgb_bands}")


def load_image(image_path: str, task_config: dict):
    """Robust image loader handling standard formats and GeoTIFFs."""
    ext = os.path.splitext(image_path)[1].lower()
    
    if ext in [".tif", ".tiff"]:
        if not HAS_RASTERIO:
            raise ImportError("Please install rasterio (`pip install rasterio`) to process GeoTIFFs.")
        with rasterio.open(image_path) as src:
            img_array = src.read()  # [C, H, W]
            
        scale_factor = task_config.get("scale_factor", 1.0)
        tensor = torch.from_numpy(img_array.astype(np.float32)) / scale_factor
        
        rgb_bands = task_config.get("rgb_bands", [0, 1, 2])
        if max(rgb_bands) >= img_array.shape[0]:
            raise IndexError(f"RGB band index {max(rgb_bands)} exceeds image channels {img_array.shape[0]}")
            
        vis_array = np.transpose(img_array[rgb_bands, :, :], (1, 2, 0))
        if vis_array.max() > 255:
            vis_array = (vis_array / vis_array.max() * 255).astype(np.uint8)
        img_pil = Image.fromarray(vis_array.astype(np.uint8))
        
    else:
        img_pil = Image.open(image_path).convert("RGB")
        tensor = TF.to_tensor(img_pil)  
        
    return img_pil, tensor


def preprocess(image_tensor: torch.Tensor, device: str, task_config: dict):
    """Configurable normalization handling. (Duplicate removed)"""
    image_tensor = image_tensor.float()
    
    if task_config.get("normalize", False):
        if "mean" not in task_config or "std" not in task_config:
            raise ValueError("Predict config requires 'mean' and 'std' when normalize=True.")
            
        # Validate channel lengths
        channels = image_tensor.shape[0]
        if len(task_config["mean"]) != channels or len(task_config["std"]) != channels:
            raise ValueError(f"Mean/Std lengths must match image channels ({channels}).")
            
        mean = torch.tensor(task_config["mean"]).view(-1, 1, 1)
        std = torch.tensor(task_config["std"]).view(-1, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
    return image_tensor.unsqueeze(0).to(device)


def colorize_mask(semantic_mask: np.ndarray, task_config: dict) -> Image.Image:
    """Applies the task-specific RGB palette defined in the Config YAML."""
    h, w = semantic_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    palette = task_config.get("palette", {0: [0, 0, 0], 1: [255, 255, 255]})
    for class_id, color in palette.items():
        colored[semantic_mask == class_id] = color
        
    return Image.fromarray(colored)


def overlay_prediction(img: Image.Image, colored_mask: Image.Image, alpha: float = 0.5) -> Image.Image:
    mask_rgba = colored_mask.convert("RGBA")
    mask_data = np.array(mask_rgba)
    mask_data[..., 3] = np.where(np.any(mask_data[..., :3] > 0, axis=-1), int(255 * alpha), 0)
    mask_rgba = Image.fromarray(mask_data)
    
    img_rgba = img.convert("RGBA")
    overlay = Image.alpha_composite(img_rgba, mask_rgba)
    return overlay.convert("RGB")


def save_prediction(semantic_mask: np.ndarray, img: Image.Image, output_path: str, task_config: dict, save_overlay: bool):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base_name = os.path.splitext(output_path)[0]
    
    np.save(f"{base_name}_raw.npy", semantic_mask)
    
    colored_mask = colorize_mask(semantic_mask.astype(np.uint8), task_config)
    colored_mask.save(f"{base_name}_mask.png")
    
    if save_overlay:
        overlay = overlay_prediction(img, colored_mask)
        overlay.save(f"{base_name}_overlay.jpg")


@torch.no_grad()
def predict_single_image(model, image_path, task, task_config, output_dir, device, save_overlay):
    img_pil, img_tensor = load_image(image_path, task_config)
    img_input = preprocess(img_tensor, device, task_config)
    
    use_amp = "cuda" in device
    device_type = "cuda" if use_amp else "cpu"
    
    with torch.autocast(device_type=device_type, enabled=use_amp):
        outputs = model(img_input, task=task)
    
    target_size = img_input.shape[-2:]
    semantic_map = postprocess_mask2former_outputs(outputs["pred_logits"], outputs["pred_masks"], target_size)
    
    assert semantic_map.dtype == torch.long
    semantic_array = semantic_map.squeeze(0).cpu().numpy()
    
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    save_prediction(semantic_array, img_pil, output_path, task_config, save_overlay)
    print(f"Processed: {filename}")


def predict_folder(model, input_folder, task, task_config, output_dir, device, save_overlay):
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
        
    image_paths = sorted(image_paths)
    print(f"Found {len(image_paths)} images in {input_folder}")
    for path in image_paths:
        predict_single_image(model, path, task, task_config, output_dir, device, save_overlay)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading configuration from {args.config}...")
    with open(args.config, "r") as f:
        full_config = yaml.safe_load(f)
        
    TASKS = full_config["TASKS"]
    
    if args.task not in TASKS:
        raise ValueError(f"Task '{args.task}' not found in configuration. Available: {list(TASKS.keys())}")
        
    task_config = TASKS[args.task]
    
    # FIX 5: Actually call the palette/config validator!
    validate_prediction_config(task_config)
        
    print(f"Building Model & Loading Checkpoint: {args.checkpoint}...")
    model = build_model(full_config)
    model = load_checkpoint(model, args.checkpoint, device, current_config=full_config)
    
    if os.path.isdir(args.input):
        predict_folder(model, args.input, args.task, task_config, args.output, device, args.overlay)
    else:
        predict_single_image(model, args.input, args.task, task_config, args.output, device, args.overlay)
        
    print(f"Inference complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()