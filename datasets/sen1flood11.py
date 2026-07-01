"""
Sen1Flood11 Dataset.


Flood detection from Sentinel-1 SAR imagery.


Dataset: https://arxiv.org/abs/2009.00255
"""


from pathlib import Path
from typing import List, Dict


import numpy as np
import torch


from .base_dataset import BaseSegmentationDataset


class Sen1Flood11Dataset(BaseSegmentationDataset):
    """
    Sen1Flood11 for flood detection.
    
    Binary segmentation: flood (1) vs no flood (0)
    """
    
    # Class definitions
    CLASSES = ["no_flood", "flood"]
    NUM_CLASSES = 2
    
    # Label mapping (raw label -> class index)
    LABEL_MAP = {
        0: 0,   # No Flood
        255: 1, # Flood
    }
    
    # Bands to load from SAR imagery
    BANDS = ["VV", "VH"]  # Sentinel-1 dual-pol
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        normalize=None,
    ):
        super().__init__(root, split, transform, normalize)
    
    @property
    def task_name(self) -> str:
        return "flood_detection"
    
    def _build_index(self) -> List[Dict]:
        """
        Build index of image-mask pairs.
        
        Customize paths based on your data organization.
        """
        samples = []
        
        # Example structure - adjust to your data layout
        images_dir = self.root / self.split / "images"
        masks_dir = self.root / self.split / "masks"
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Find all tif files
        for img_path in sorted(images_dir.glob("*.tif")):
            mask_path = masks_dir / img_path.name
            
            if mask_path.exists():
                samples.append({
                    "image": str(img_path),
                    "mask": str(mask_path),
                    "id": img_path.stem,
                })
        
        return samples
    
    def _load_image(self, path: str) -> np.ndarray:
        """
        Load SAR image with VV and VH bands.
        
        Returns:
            np.ndarray of shape [2, H, W]
        """
        image = self.read_raster(path)
        
        # Ensure we have the right bands
        # Adjust based on actual band ordering in your data
        if image.shape[0] > 2:
            image = image[:2]  # Take first 2 bands
        
        return image
    
    def _load_mask(self, path: str) -> np.ndarray:
        """
        Load flood mask.
        
        Returns:
            np.ndarray of shape [H, W] with values 0 or 1
        """
        mask = self.read_raster(path).squeeze().astype(np.int64)
        
        # Remap labels
        for orig_label, new_label in self.LABEL_MAP.items():
            mask[mask == orig_label] = new_label
        
        return mask
