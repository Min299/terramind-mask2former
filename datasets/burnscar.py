"""
HLS Burn Scar Dataset Wrapper
"""


from pathlib import Path


import numpy as np
import rasterio


from .base_dataset import BaseSegmentationDataset




class BurnScarDataset(BaseSegmentationDataset):


    NUM_CLASSES = 2


    IGNORE_INDEX = 255


    def __init__(
        self,
        root,
        split="train",
        transform=None,
        normalize=None,
    ):


        super().__init__(
            root=root,
            split=split,
            transform=transform,
            normalize=normalize,
        )


    @property
    def task_name(self):


        return "burnscar"


    def _build_index(self):


        image_dir = self.root / self.split / "images"
        mask_dir = self.root / self.split / "masks"


        image_files = sorted(
            list(image_dir.glob("*.tif"))
            + list(image_dir.glob("*.tiff"))
            + list(image_dir.glob("*.npy"))
        )


        samples = []


        for image_path in image_files:


            stem = image_path.stem


            mask_path = None


            for ext in [
                ".tif",
                ".tiff",
                ".png",
                ".npy",
            ]:


                candidate = mask_dir / (stem + ext)


                if candidate.exists():
                    mask_path = candidate
                    break


            if mask_path is None:
                continue


            samples.append(
                {
                    "image": str(image_path),
                    "mask": str(mask_path),
                    "id": stem,
                }
            )


        if len(samples) == 0:
            raise RuntimeError(
                f"No samples found in {image_dir}"
            )


        return samples


    def _load_image(self, path):


        path = Path(path)


        if path.suffix == ".npy":


            image = np.load(path).astype(np.float32)


        else:


            with rasterio.open(path) as src:
                image = src.read().astype(np.float32)


        return image


    def _load_mask(self, path):


        path = Path(path)


        if path.suffix == ".npy":


            mask = np.load(path)


        else:


            with rasterio.open(path) as src:
                mask = src.read(1)


        mask = (mask > 0).astype(np.int64)


        return mask


    @property
    def class_names(self):


        return [
            "background",
            "burnscar",
        ]


    @property
    def palette(self):


        return [
            (0, 0, 0),
            (255, 0, 0),
        ]
