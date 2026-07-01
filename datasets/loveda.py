"""
LoveDA Dataset Wrapper


Expected directory structure
--------------------------


LoveDA/
│
├── train/
│   ├── rural/
│   │   ├── images/
│   │   └── masks/
│   └── urban/
│       ├── images/
│       └── masks/
│
├── val/
│   └── ...
│
└── test/
    └── ...


Dataset: https://github.com/JiauZhang/LoveDA
"""


from pathlib import Path


import numpy as np
import rasterio


from .base_dataset import BaseSegmentationDataset




class LoveDADataset(BaseSegmentationDataset):


    NUM_CLASSES = 7


    IGNORE_INDEX = 255


    # LoveDA class definitions
    CLASSES = [
        "background",       # 0
        "building",         # 1
        "road",             # 2
        "water",            # 3
        "barren",           # 4
        "forest",           # 5
        "agriculture",       # 6
    ]


    def __init__(
        self,
        root,
        split="train",
        transform=None,
        normalize=None,
        scenes=None,
    ):
        self.scenes = scenes or ["rural", "urban"]
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            normalize=normalize,
        )


    @property
    def task_name(self):


        return "land_cover"


    def _build_index(self):


        samples = []


        for scene in self.scenes:
            image_dir = self.root / self.split / scene / "images"
            mask_dir = self.root / self.split / scene / "masks"


            if not image_dir.exists():
                continue


            image_files = sorted(
                list(image_dir.glob("*.tif"))
                + list(image_dir.glob("*.tiff"))
                + list(image_dir.glob("*.png"))
                + list(image_dir.glob("*.npy"))
            )


            for image_path in image_files:
                stem = f"{scene}_{image_path.stem}"


                mask_path = None


                for ext in [".tif", ".tiff", ".png", ".npy"]:
                    candidate = mask_dir / (image_path.stem + ext)


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
                f"No samples found in {self.root / self.split}"
            )


        return samples


    def _load_image(self, path):


        path = Path(path)


        if path.suffix == ".npy":


            image = np.load(path).astype(np.float32)


        elif path.suffix == ".png":


            import PIL.Image
            img = PIL.Image.open(path)
            image = np.array(img).transpose(2, 0, 1).astype(np.float32)


        else:


            with rasterio.open(path) as src:


                image = src.read().astype(np.float32)


        return image


    def _load_mask(self, path):


        path = Path(path)


        if path.suffix == ".npy":


            mask = np.load(path)


        elif path.suffix == ".png":


            import PIL.Image
            img = PIL.Image.open(path)
            mask = np.array(img)


        else:


            with rasterio.open(path) as src:


                mask = src.read(1)


        mask = mask.astype(np.int64)


        return mask


    @property
    def class_names(self):


        return self.CLASSES


    @property
    def palette(self):


        return [
            (0, 0, 0),           # background - black
            (255, 0, 0),         # building - red
            (128, 128, 128),      # road - gray
            (0, 0, 255),         # water - blue
            (139, 69, 19),       # barren - brown
            (0, 128, 0),         # forest - green
            (255, 255, 0),       # agriculture - yellow
        ]
