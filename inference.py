"""
Inference script for TerraMind + Mask2Former.


Usage:
    python inference.py --checkpoint checkpoints/best.pth --input image.tif --output prediction.tif
    python inference.py --checkpoint checkpoints/best.pth --input_dir images/ --output_dir predictions/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets import get_val_transforms
from test_time_augmentation import SemanticSegmentorWithTTA, get_tta_transforms


def load_image(path, transform):
    """Load and preprocess image."""
    import rasterio
    
    with rasterio.open(path) as src:
        image = src.read().astype(np.float32)
    
    if transform:
        image_dict = transform({"image": image, "mask": np.zeros((image.shape[1], image.shape[2]))})
        image = image_dict["image"]
    
    return torch.from_numpy(image).unsqueeze(0)


def predict_image(model, image, task, device):
    """Predict segmentation mask for a single image."""
    model.eval()
    
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image, task=task)
        pred_masks = outputs["pred_masks"]
        pred_logits = outputs["pred_logits"]
        
        # Get predicted classes
        pred_classes = pred_logits.argmax(dim=-1)
        
        # Combine masks with classes
        pred_mask = torch.zeros(
            (image.shape[0], image.shape[2], image.shape[3]),
            device=device,
            dtype=torch.long,
        )
        
        for i in range(image.shape[0]):
            for j in range(pred_masks.shape[1]):
                if pred_classes[i, j] < 10:  # Not background
                    mask = pred_masks[i, j] > 0
                    pred_mask[i, mask] = pred_classes[i, j]
    
    return pred_mask


def save_prediction(pred_mask, output_path):
    """Save prediction mask as GeoTIFF."""
    import rasterio
    from rasterio.transform import from_bounds
    
    pred_mask = pred_mask.cpu().numpy().astype(np.uint8)
    
    # Create a dummy geotransform (will need to be updated with real georeference)
    transform = from_bounds(0, 0, pred_mask.shape[2], pred_mask.shape[1], pred_mask.shape[2], pred_mask.shape[1])
    
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=pred_mask.shape[1],
        width=pred_mask.shape[2],
        count=1,
        dtype=pred_mask.dtype,
        transform=transform,
    ) as dst:
        dst.write(pred_mask, 1)


def predict_directory(model, input_dir, output_dir, task, device, use_tta=False):
    """Predict masks for all images in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff")) + list(input_dir.glob("*.npy"))
    
    if use_tta:
        from config import ModelConfig
        cfg = ModelConfig()
        tta_transforms = get_tta_transforms({
            "tta_hflip": cfg.tta_hflip,
            "tta_vflip": cfg.tta_vflip,
            "tta_scales": cfg.tta_scales,
        })
        model = SemanticSegmentorWithTTA(model, tta_transforms)
    
    for img_path in tqdm(image_files, desc="Predicting"):
        image = load_image(img_path, get_val_transforms(image_size=224))
        pred = predict_image(model, image, task, device)
        save_prediction(pred, output_dir / img_path.name)
    
    print(f"Predictions saved to {output_dir}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Import model components
    from models import (
        TerraMindEncoder,
        TerraMindNeck,
        MSDeformAttnPixelDecoder,
        MultiScaleMaskedTransformerDecoder,
    )
    from models.multitask_model import MultiTaskMask2Former
    
    # Rebuild model (placeholder - implement multitask_model.py separately)
    # For now, just load state dict if available
    print("Note: Full model reconstruction requires multitask_model.py implementation")
    
    if args.input:
        # Single image inference
        image = load_image(args.input, get_val_transforms(image_size=224))
        print(f"Running inference on {args.input}...")
        # Placeholder - actual inference requires full model
        print("Note: Full inference requires multitask_model.py implementation")
    
    elif args.input_dir:
        # Directory inference
        predict_directory(None, args.input_dir, args.output_dir, args.task, device, args.use_tta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for TerraMind + Mask2Former")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--input", type=str, default=None, help="Path to input image")
    parser.add_argument("--output", type=str, default=None, help="Path to output prediction")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory with input images")
    parser.add_argument("--output_dir", type=str, default="./predictions", help="Output directory")
    parser.add_argument("--task", type=str, default="flood", choices=["flood", "burnscar", "lulc"])
    parser.add_argument("--use_tta", action="store_true", help="Use test-time augmentation")
    
    args = parser.parse_args()
    main(args)
