"""
Base Dataset for TerraMind-Mask2Former.


Every downstream dataset should inherit from this class.


Returned sample format
----------------------


{
    "image": Tensor[C,H,W],
    "mask": Tensor[H,W],
    "task": str,
    "image_id": str
}
"""


from __future__ import annotations


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional


import numpy as np
import torch
from torch.utils.data import Dataset


try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class BaseSegmentationDataset(Dataset, ABC):
    """
    Base class for all semantic segmentation datasets.


    Child classes only need to implement:


        _build_index()
        _load_image()
        _load_mask()
    """


    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        normalize: Optional[Callable] = None,
    ):
        super().__init__()


        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.normalize = normalize


        self.samples = self._build_index()


    # -------------------------------------------------------------
    # Methods child datasets MUST implement
    # -------------------------------------------------------------


    @abstractmethod
    def _build_index(self):
        """
        Returns


        [
            {
                "image": "...",
                "mask": "...",
                "id": "...",
            },
            ...
        ]
        """
        raise NotImplementedError


    @abstractmethod
    def _load_image(self, path: str) -> np.ndarray:
        raise NotImplementedError


    @abstractmethod
    def _load_mask(self, path: str) -> np.ndarray:
        raise NotImplementedError


    @property
    @abstractmethod
    def task_name(self) -> str:
        raise NotImplementedError


    # -------------------------------------------------------------
    # Common utilities
    # -------------------------------------------------------------


    def read_raster(self, path):
        """Read raster file using rasterio."""
        if not HAS_RASTERIO:
            raise ImportError("rasterio is required to read raster files. Install with: pip install rasterio")
        
        with rasterio.open(path) as src:
            img = src.read()


        return img.astype(np.float32)


    def image_to_tensor(self, image):


        if image.ndim == 2:
            image = image[None]


        return torch.from_numpy(image).float()


    def mask_to_tensor(self, mask):


        return torch.from_numpy(mask).long()


    # -------------------------------------------------------------
    # Dataset interface
    # -------------------------------------------------------------


    def __len__(self):


        return len(self.samples)


    def __getitem__(self, idx):


        sample = self.samples[idx]


        image = self._load_image(sample["image"])


        mask = self._load_mask(sample["mask"])


        if self.transform is not None:


            transformed = self.transform(
                image=image,
                mask=mask,
            )


            image = transformed["image"]
            mask = transformed["mask"]


        if isinstance(image, np.ndarray):
            image = self.image_to_tensor(image)


        if isinstance(mask, np.ndarray):
            mask = self.mask_to_tensor(mask)


        if self.normalize is not None:
            image = self.normalize(image)


        return {
            "image": image,
            "mask": mask,
            "task": self.task_name,
            "image_id": sample["id"],
        }
