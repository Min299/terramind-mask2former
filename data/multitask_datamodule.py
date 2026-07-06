"""
multitask_datamodule.py

Config-driven MultiTask DataModule for TerraMind.
Dynamically instantiates datasets from a registry, handles flexible 
epoch definitions, and guarantees balanced round-robin sampling.
"""

import itertools
from typing import Any, Dict, Iterator, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Mocking TerraTorch imports (Replace with actual imports)
class Sen1Floods11DataModule(pl.LightningDataModule): pass
class HLSBurnScarsDataModule(pl.LightningDataModule): pass
class ESRILULCDataModule(pl.LightningDataModule): pass

# 🔴 1. TASK REGISTRY: No more hardcoded datasets
TASK_REGISTRY = {
    "flood": Sen1Floods11DataModule,
    "burn": HLSBurnScarsDataModule,
    "lulc": ESRILULCDataModule,
}


class MultiTaskTrainDataLoader:
    """Infinite round-robin DataLoader with flexible epoch definitions."""
    
    def __init__(
        self, 
        dataloaders: Dict[str, DataLoader], 
        epoch_mode: str = "fixed_steps", 
        fixed_steps: int = 1000
    ):
        self.dataloaders = dataloaders
        self.task_names = list(dataloaders.keys())
        self.epoch_mode = epoch_mode
        self.fixed_steps = fixed_steps
        
        self.iterators = {
            task: self._infinite_iterator(dl)
            for task, dl in dataloaders.items()
        }
        self.task_cycle = itertools.cycle(self.task_names)

        # 🟡 4. Flexible Epoch Definition
        if self.epoch_mode == "sum":
            self.total_batches = sum(len(dl) for dl in self.dataloaders.values())
        elif self.epoch_mode == "max":
            self.total_batches = max(len(dl) for dl in self.dataloaders.values()) * len(self.task_names)
        elif self.epoch_mode == "fixed_steps":
            self.total_batches = fixed_steps
        else:
            raise ValueError(f"Unknown epoch_mode: {epoch_mode}")

    def _infinite_iterator(self, dataloader: DataLoader) -> Iterator:
        while True:
            for batch in dataloader:
                yield batch

    def __iter__(self):
        self.current_step = 0
        return self

    def __next__(self) -> Tuple[str, Any]:
        if self.current_step >= self.total_batches:
            raise StopIteration
            
        self.current_step += 1
        task = next(self.task_cycle)
        batch = next(self.iterators[task])
        
        # 🟡 3. Metadata Pass-through: We yield the raw batch dict blindly.
        return task, batch

    def __len__(self) -> int:
        return self.total_batches


class MultiTaskDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Dict[str, Any]], epoch_mode: str = "fixed_steps", fixed_steps: int = 1000):
        super().__init__()
        self.config = config
        self.epoch_mode = epoch_mode
        self.fixed_steps = fixed_steps
        self.datamodules = {}

        # 🔴 2. Config-driven instantiation & 🟢 5. Sanity Checks
        for task, kwargs in config.items():
            if task not in TASK_REGISTRY:
                raise KeyError(f"Task '{task}' not found in TASK_REGISTRY.")
            
            assert "batch_size" in kwargs, f"Missing batch_size for task {task}"
            
            # Instantiate dynamically
            self.datamodules[task] = TASK_REGISTRY[task](**kwargs)

    def prepare_data(self):
        for dm in self.datamodules.values():
            dm.prepare_data()

    def setup(self, stage: str = None):
        for dm in self.datamodules.values():
            dm.setup(stage)

    def train_dataloader(self) -> MultiTaskTrainDataLoader:
        train_loaders = {task: dm.train_dataloader() for task, dm in self.datamodules.items()}
        return MultiTaskTrainDataLoader(train_loaders, self.epoch_mode, self.fixed_steps)

    def val_dataloader(self) -> Dict[str, DataLoader]:
        return {task: dm.val_dataloader() for task, dm in self.datamodules.items()}

    def test_dataloader(self) -> Dict[str, DataLoader]:
        return {task: dm.test_dataloader() for task, dm in self.datamodules.items()}