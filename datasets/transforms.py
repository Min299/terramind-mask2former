"""
Augmentation transforms shared across all datasets.


This module provides augmentation pipelines for training and validation.


Usage
-----


    from datasets.transforms import get_train_transforms, get_val_transforms


    train_dataset = SomeDataset(
        root="...",
        split="train",
        transform=get_train_transforms(image_size=224),
    )
"""


from typing import Dict, Optional, Tuple


import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


__all__ = [
    "get_train_transforms",
    "get_val_transforms",
    "ValTransforms",
    "TrainTransforms",
]


class TrainTransforms:
    """
    Training augmentations.


    Applied in order:


        1. Random horizontal flip
        2. Random vertical flip
        3. Random rotation (90° increments)
        4. Color jitter (multispectral)
        5. Random crop
        6. Normalize


    Note: mask transformations use nearest neighbor to avoid class bleeding.
    """


    def __init__(
        self,
        image_size: int = 224,
        crop_ratio: float = 1.0,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate_prob: float = 0.5,
        jitter_kwargs: Optional[Dict] = None,
    ):
        self.image_size = image_size
        self.crop_ratio = crop_ratio
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotate_prob = rotate_prob
        
        # Default jitter for multispectral
        self.jitter_kwargs = jitter_kwargs or {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
        }


    def __call__(self, image, mask):
        """
        Apply augmentations.


        Args:
            image: np.ndarray of shape [C, H, W]
            mask: np.ndarray of shape [H, W]


        Returns:
            Dict with "image" and "mask"
        """
        C, H, W = image.shape


        #
        # Random horizontal flip
        #


        if np.random.random() < self.hflip_prob:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()


        #
        # Random vertical flip
        #


        if np.random.random() < self.vflip_prob:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()


        #
        # Random rotation (90, 180, 270)
        #


        if np.random.random() < self.rotate_prob:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(0, 1)).copy()


        #
        # Color jitter (per-band for multispectral)
        #


        for key, delta in self.jitter_kwargs.items():
            if key == "brightness":
                factor = 1.0 + np.random.uniform(-delta, delta)
                image = image * factor
            elif key == "contrast":
                factor = 1.0 + np.random.uniform(-delta, delta)
                mean = image.mean(axis=(1, 2), keepdims=True)
                image = (image - mean) * factor + mean


        #
        # Resize / crop
        #


        if self.crop_ratio < 1.0:
            crop_h = int(H * self.crop_ratio)
            crop_w = int(W * self.crop_ratio)
            
            top = np.random.randint(0, H - crop_h + 1)
            left = np.random.randint(0, W - crop_w + 1)
            
            image = image[:, top:top+crop_h, left:left+crop_w]
            mask = mask[top:top+crop_h, left:left+crop_w]


        #
        # Resize to target size
        #


        if image.shape[1:] != (self.image_size, self.image_size):
            image = self._resize_image(image, self.image_size)
            mask = self._resize_mask(mask, self.image_size)


        return {
            "image": image,
            "mask": mask,
        }


    def _resize_image(self, image, size):
        """Resize image using bilinear interpolation."""
        tensor = torch.from_numpy(image).unsqueeze(0)
        resized = F.interpolate(
            tensor,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0).numpy()


    def _resize_mask(self, mask, size):
        """Resize mask using nearest neighbor."""
        tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        resized = F.interpolate(
            tensor,
            size=(size, size),
            mode="nearest",
        )
        return resized.squeeze().numpy().astype(np.int64)


class ValTransforms:
    """
    Validation / inference transforms.


    Simple resize to target size and normalize.
    """


    def __init__(self, image_size: int = 224):
        self.image_size = image_size


    def __call__(self, image, mask):
        """
        Apply transforms.


        Args:
            image: np.ndarray of shape [C, H, W]
            mask: np.ndarray of shape [H, W]


        Returns:
            Dict with "image" and "mask"
        """
        #
        # Resize to target size
        #


        if image.shape[1:] != (self.image_size, self.image_size):
            image = self._resize_image(image, self.image_size)
            mask = self._resize_mask(mask, self.image_size)


        return {
            "image": image,
            "mask": mask,
        }


    def _resize_image(self, image, size):
        """Resize image using bilinear interpolation."""
        tensor = torch.from_numpy(image).unsqueeze(0)
        resized = F.interpolate(
            tensor,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0).numpy()


    def _resize_mask(self, mask, size):
        """Resize mask using nearest neighbor."""
        tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        resized = F.interpolate(
            tensor,
            size=(size, size),
            mode="nearest",
        )
        return resized.squeeze().numpy().astype(np.int64)


def get_train_transforms(
    image_size: int = 224,
    crop_ratio: float = 1.0,
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.5,
    rotate_prob: float = 0.5,
    jitter_kwargs: Optional[Dict] = None,
) -> TrainTransforms:
    """
    Get training transforms.


    Args:
        image_size: Target image size (H=W)
        crop_ratio: Random crop ratio (1.0 = no crop)
        hflip_prob: Probability of horizontal flip
        vflip_prob: Probability of vertical flip
        rotate_prob: Probability of 90° rotation
        jitter_kwargs: Color jitter parameters


    Returns:
        TrainTransforms instance
    """
    return TrainTransforms(
        image_size=image_size,
        crop_ratio=crop_ratio,
        hflip_prob=hflip_prob,
        vflip_prob=vflip_prob,
        rotate_prob=rotate_prob,
        jitter_kwargs=jitter_kwargs,
    )


def get_val_transforms(image_size: int = 224) -> ValTransforms:
    """
    Get validation transforms.


    Args:
        image_size: Target image size (H=W)


    Returns:
        ValTransforms instance
    """
    return ValTransforms(image_size=image_size)
