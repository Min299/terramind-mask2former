"""
multitask_datamodule.py

Config-driven MultiTask DataModule for TerraMind.
Acts as a Meta-DataModule that wraps and orchestrates genuine TerraTorch
DataModules, preserving all of TerraTorch's native loading and streaming utilities.
"""

import inspect
import itertools
from typing import Any, Dict, Iterator, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from terratorch.datamodules import (
    Sen1Floods11NonGeoDataModule,
    FireScarsNonGeoDataModule,
    MChesapeakeLandcoverNonGeoDataModule,
)

# Maps our internal task name -> the real TerraTorch DataModule class that
# actually loads/streams that task's data.
TASK_REGISTRY = {
    "sen1floods11": Sen1Floods11NonGeoDataModule,
    "fire_scars": FireScarsNonGeoDataModule,
    "m_chesapeake_landcover": MChesapeakeLandcoverNonGeoDataModule,
}

# Task-bookkeeping keys that our own code (trainer/criterion/metrics) needs
# from tasks.yaml but that are NOT constructor arguments of the TerraTorch
# DataModules themselves. These must never be forwarded to TASK_REGISTRY[...](**kwargs)
# or the DataModule constructor will raise TypeError("unexpected keyword argument").
_NON_DATAMODULE_KEYS = {"dataset", "num_classes", "ignore_index", "background_id"}


def _filter_datamodule_kwargs(datamodule_cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only the kwargs that `datamodule_cls.__init__` actually accepts.

    First drops the known non-datamodule bookkeeping keys, then (defensively)
    also drops anything not present in the target class's __init__ signature,
    unless that signature itself accepts **kwargs.
    """
    kwargs = {k: v for k, v in kwargs.items() if k not in _NON_DATAMODULE_KEYS}

    sig = inspect.signature(datamodule_cls.__init__)
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_var_kwargs:
        return kwargs

    accepted_names = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in accepted_names}


class MultiTaskTrainDataLoader:
    """Infinite round-robin DataLoader with flexible epoch definitions."""

    def __init__(
        self,
        dataloaders: Dict[str, DataLoader],
        epoch_mode: str = "fixed_steps",
        fixed_steps: int = 1000,
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

        # Raw batch dict is passed through unchanged.
        # TerraTorch's standard format (e.g., {"image": tensor, "mask": tensor}) is maintained.
        return task, batch

    def __len__(self) -> int:
        return self.total_batches


class MultiTaskDataModule(pl.LightningDataModule):
    """
    The Meta-DataModule.
    Takes a configuration dictionary and dynamically instantiates real TerraTorch DataModules.
    Delegates all PyTorch Lightning lifecycle hooks (prepare_data, setup) to them.
    """

    def __init__(self, config: Dict[str, Dict[str, Any]], epoch_mode: str = "fixed_steps", fixed_steps: int = 1000):
        super().__init__()
        self.config = config
        self.epoch_mode = epoch_mode
        self.fixed_steps = fixed_steps
        self.datamodules = {}

        for task, kwargs in config.items():
            if task not in TASK_REGISTRY:
                raise KeyError(f"Task '{task}' not found in TASK_REGISTRY. Available: {list(TASK_REGISTRY.keys())}")

            if "batch_size" not in kwargs:
                raise ValueError(f"Missing 'batch_size' for task {task}")

            datamodule_cls = TASK_REGISTRY[task]
            dm_kwargs = _filter_datamodule_kwargs(datamodule_cls, kwargs)
            self.datamodules[task] = datamodule_cls(**dm_kwargs)

    def prepare_data(self):
        """Delegates downloading/preparation to TerraTorch."""
        for dm in self.datamodules.values():
            dm.prepare_data()

    def setup(self, stage: str = None):
        """Delegates train/val/test splitting to TerraTorch."""
        for dm in self.datamodules.values():
            dm.setup(stage)

    def train_dataloader(self) -> MultiTaskTrainDataLoader:
        """Collects TerraTorch's native DataLoaders and wraps them in our multi-task round-robin mixer."""
        train_loaders = {task: dm.train_dataloader() for task, dm in self.datamodules.items()}
        return MultiTaskTrainDataLoader(train_loaders, self.epoch_mode, self.fixed_steps)

    def val_dataloader(self) -> Dict[str, DataLoader]:
        """Returns a dictionary of TerraTorch native val dataloaders for independent evaluation."""
        return {task: dm.val_dataloader() for task, dm in self.datamodules.items()}

    def test_dataloader(self) -> Dict[str, DataLoader]:
        """Returns a dictionary of TerraTorch native test dataloaders for independent evaluation."""
        return {task: dm.test_dataloader() for task, dm in self.datamodules.items()}
