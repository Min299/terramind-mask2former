"""
HLS Burn Scar Dataset.


Burn scar detection from Harmonized Landsat Sentinel-2 data.


Dataset: https://burnscar.vito.be/
"""


from pathlib import Path
from typing import List, Dict


import numpy as np


from .base_dataset import BaseSegmentationDataset


class BurnScarDataset(BaseSegmentationDataset):
    """
    HLS Burn Scar for burn scar detection.
    
    Binary segmentation: burn scar (1) vs unburned (0)
    """
    
    # Class definitions
    CLASSES = ["unburned", "burned"]
    NUM_CLASSES = 2
    
    # Label mapping (raw label -> class index)
    LABEL_MAP = {
        0: 0,    # Unburned
        1: 1,    # Burned
        255: 0,  # No data -> unburned
    }
    
    # Bands to load (HLS provides 15 spectral bands)
    BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]
    
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
        return "burn_scar_detection"
    
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
        
        # Find all files (could be .tif, .npy, etc.)
        for ext in ["*.tif", "*.tiff", "*.npy"]:
            for img_path in sorted(images_dir.glob(ext)):
                mask_path = masks_dir / f"{img_path.stem}.tif"
                
                if mask_path.exists():
                    samples.append({
                        "image": str(img_path),
                        "mask": str(mask_path),
                        "id": img_path.stem,
                    })
        
        return samples
    
    def _load_image(self, path: str) -> np.ndarray:
        """
        Load HLS multispectral image.
        
        Returns:
            np.ndarray of shape [C, H, W] where C is number of bands
        """
        image = self.read_raster(path)
        
        return image
    
    def _load_mask(self, path: str) -> np.ndarray:
        """
        Load burn scar mask.
        
        Returns:
            np.ndarray of shape [H, W] with values 0 or 1
        """
        mask = self.read_raster(path).squeeze().astype(np.int64)
        
        # Remap labels
        for orig_label, new_label in self.LABEL_MAP.items():
            mask[mask == orig_label] = new_label
        
        return mask
