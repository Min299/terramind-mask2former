"""
LoveDA Dataset.


Land cover mapping from aerial imagery.


Dataset: https://github.com/JiauZhang/LoveDA
"""


from pathlib import Path
from typing import List, Dict


import numpy as np


from .base_dataset import BaseSegmentationDataset


class LoveDADataset(BaseSegmentationDataset):
    """
    LoveDA for land cover semantic segmentation.
    
    Multi-class segmentation: 7 land cover classes
    """
    
    # Class definitions (LoveDA standard)
    CLASSES = [
        "background",       # 0
        "building",         # 1
        "road",             # 2
        "water",            # 3
        "barren",           # 4
        "forest",           # 5
        "agriculture",      # 6
    ]
    NUM_CLASSES = 7
    
    # Label mapping (if different from class index)
    LABEL_MAP = {}  # No remapping needed for LoveDA standard
    
    # Bands for RGB aerial imagery
    BANDS = ["R", "G", "B"]
    
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
        return "land_cover"
    
    def _build_index(self) -> List[Dict]:
        """
        Build index of image-mask pairs.
        
        Customize paths based on your data organization.
        """
        samples = []
        
        # Example structure - adjust to your data layout
        # LoveDA has separate directories for rural/urban scenes
        for scene in ["rural", "urban"]:
            images_dir = self.root / self.split / scene / "images"
            masks_dir = self.root / self.split / scene / "masks"
            
            if not images_dir.exists():
                continue
            
            # Find all files
            for ext in ["*.png", "*.tif"]:
                for img_path in sorted(images_dir.glob(ext)):
                    # LoveDA naming convention: xxx.png -> xxx.png
                    mask_path = masks_dir / img_path.name
                    
                    if mask_path.exists():
                        samples.append({
                            "image": str(img_path),
                            "mask": str(mask_path),
                            "id": f"{scene}_{img_path.stem}",
                        })
        
        return samples
    
    def _load_image(self, path: str) -> np.ndarray:
        """
        Load RGB aerial image.
        
        Returns:
            np.ndarray of shape [3, H, W]
        """
        image = self.read_raster(path)
        
        # Handle different formats
        if image.ndim == 3:
            # CHW format
            pass
        elif image.ndim == 2:
            # Single band - expand to 3 channels
            image = np.stack([image, image, image], axis=0)
        
        return image
    
    def _load_mask(self, path: str) -> np.ndarray:
        """
        Load land cover mask.
        
        Returns:
            np.ndarray of shape [H, W] with class indices
        """
        mask = self.read_raster(path).squeeze().astype(np.int64)
        
        # Remap labels if needed
        for orig_label, new_label in self.LABEL_MAP.items():
            mask[mask == orig_label] = new_label
        
        return mask
