"""
trainer.py

Production-grade Multi-task training loop for TerraMind Mask2Former.
Features AMP, Linear-Warmup-Cosine Scheduling, Early Stopping, 
Full-State Checkpointing, and dynamic task routing.
"""

import os
import math
import random
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from criterion import SetCriterion
from matcher import HungarianMatcher
from metrics import SegmentationMetrics


def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup -> Cosine decay scheduler"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def semantic_to_mask2former_targets(
    semantic_masks: torch.Tensor, 
    ignore_index: int = 255,
    background_id: Optional[int] = None
) -> List[Dict[str, torch.Tensor]]:
    """
    Converts standard [B, H, W] semantic masks to Set Prediction format.
    Filters out ignore_index and optionally skips background_id.
    """
    targets = []
    for i in range(semantic_masks.shape[0]):
        mask_i = semantic_masks[i]
        classes = torch.unique(mask_i)
        
        # Filter out ignored/background pixels
        classes = classes[classes != ignore_index]
        if background_id is not None:
            classes = classes[classes != background_id]

        if len(classes) == 0:
            labels = torch.zeros(0, dtype=torch.int64, device=mask_i.device)
            masks = torch.zeros((0, mask_i.shape[0], mask_i.shape[1]), dtype=torch.float32, device=mask_i.device)
        else:
            masks = [(mask_i == c) for c in classes]
            masks = torch.stack(masks).to(mask_i.device).float()
            labels = classes.to(torch.int64)
            
        targets.append({"labels": labels, "masks": masks})
    return targets


class MultiTaskTrainer:
    def __init__(
        self,
        model: nn.Module,
        datamodule,
        tasks_config: Dict[str, Dict],
        train_config: Dict[str, Any],
        device: str = "cuda",
        save_dir: str = "./checkpoints",
        seed: int = 42,
    ):
        # 1. Deterministic Seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model = model.to(device)
        self.datamodule = datamodule
        self.device = device
        
        # Apply configurations
        self.tasks_config = tasks_config
        self.train_config = train_config
        self.epochs = train_config["epochs"]
        self.save_dir = save_dir
        self.patience = train_config.get("patience", 15)
        os.makedirs(save_dir, exist_ok=True)

        # 2. Optimizer & AMP Scaler
        self.optimizer = AdamW(
            self.model.get_trainable_params(), 
            lr=train_config["lr"], 
            weight_decay=train_config.get("weight_decay", 0.05)
        )
        self.scaler = GradScaler()

        # 3. Loss Criteria Setup (Deep Supervision weights)
        weight_dict = {"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0}
        aux_weight_dict = {f"{k}_{i}": v for i in range(9) for k, v in weight_dict.items()}
        weight_dict.update(aux_weight_dict)

        self.criteria = {}
        matcher = HungarianMatcher()
        for task, cfg in self.tasks_config.items():
            self.criteria[task] = SetCriterion(
                num_classes=cfg["num_classes"], 
                matcher=matcher, 
                weight_dict=weight_dict, 
                losses=["labels", "masks"], 
                eos_coef=0.1
            ).to(device)

        # 4. Data Loaders & Schedulers
        self.train_loader = self.datamodule.train_dataloader()
        self.val_loaders = self.datamodule.val_dataloader()
        
        total_steps = self.epochs * len(self.train_loader)
        warmup_steps = train_config.get("warmup_epochs", 5) * len(self.train_loader)
        self.scheduler = get_warmup_cosine_scheduler(self.optimizer, warmup_steps, total_steps)

        # State tracking
        self.start_epoch = 1
        self.best_loss = float("inf")
        self.epochs_without_improvement = 0

    def save_checkpoint(self, epoch: int, is_best: bool = False, filename: str = "latest.pth"):
        """Full State Checkpointing"""
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "best_loss": self.best_loss,
        }
        torch.save(state, os.path.join(self.save_dir, filename))
        if is_best:
            torch.save(state, os.path.join(self.save_dir, "best.pth"))
        
        if epoch % 10 == 0:
            torch.save(state, os.path.join(self.save_dir, f"epoch_{epoch}.pth"))

    def resume(self, checkpoint_path: str):
        """Resume Support"""
        print(f"Resuming from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_loss = checkpoint["best_loss"]

    def train_epoch(self, epoch: int):
        self.model.train() 
        self.model.encoder.eval() # Keep encoder strictly frozen

        metrics = defaultdict(float)
        task_counts = defaultdict(int)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")
        
        for step, (task, batch) in enumerate(pbar):
            images = batch["image"].to(self.device)
            semantic_masks = batch["mask"].to(self.device)
            
            # Use task-specific configs for target conversion
            task_cfg = self.tasks_config[task]
            targets = semantic_to_mask2former_targets(
                semantic_masks, 
                ignore_index=task_cfg.get("ignore_index", 255),
                background_id=task_cfg.get("background_id", None)
            )
            
            # Handle Optional Metadata
            metadata = batch.get("metadata", None)
            if metadata is not None and isinstance(metadata, dict):
                metadata = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in metadata.items()}

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with autocast():
                outputs = self.model(images, task=task, metadata=metadata)
                loss_dict = self.criteria[task](outputs, targets)
                weight_dict = self.criteria[task].weight_dict
                total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # Mixed Precision Backward
            self.scaler.scale(total_loss).backward()
            
            # Gradient Clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # Detailed Logging
            metrics[f"{task}_loss"] += total_loss.item()
            metrics[f"{task}_ce"] += loss_dict.get("loss_ce", torch.tensor(0)).item()
            metrics[f"{task}_mask"] += loss_dict.get("loss_mask", torch.tensor(0)).item()
            task_counts[task] += 1
            
            pbar.set_postfix(
                task=task, 
                loss=f"{total_loss.item():.3f}", 
                lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                grad=f"{grad_norm:.2f}"
            )

        return {k: v / task_counts[k.split('_')[0]] for k, v in metrics.items()}

    @torch.no_grad()
    def validate(self):
        """Runs validation and computes losses and metrics for each task."""
        self.model.eval()
        val_losses = {}
        
        for task, loader in self.val_loaders.items():
            task_cfg = self.tasks_config[task]
            task_loss = 0.0
            
            # Initialize our SegmentationMetrics for this specific task
            metric_tracker = SegmentationMetrics(
                num_classes=task_cfg["num_classes"], 
                ignore_index=task_cfg.get("ignore_index", 255)
            )
            
            pbar = tqdm(loader, desc=f"Validation [{task}]", leave=False)
            
            for batch in pbar:
                images = batch["image"].to(self.device)
                semantic_masks = batch["mask"].to(self.device)
                
                targets = semantic_to_mask2former_targets(
                    semantic_masks, 
                    ignore_index=task_cfg.get("ignore_index", 255),
                    background_id=task_cfg.get("background_id", None)
                )

                metadata = batch.get("metadata", None)
                if metadata is not None and isinstance(metadata, dict):
                    metadata = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in metadata.items()}

                with autocast():
                    outputs = self.model(images, task=task, metadata=metadata)
                    loss_dict = self.criteria[task](outputs, targets)
                    weight_dict = self.criteria[task].weight_dict
                    total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                
                task_loss += total_loss.item()
                
                # --- Quick Semantic Mask Recovery for Metrics ---
                # Mask2Former outputs masks of shape [B, Q, H, W] and class logits [B, Q, C+1]
                # We do a quick argmax over classes, then multiply to get a dense semantic map
                pred_masks = outputs["pred_masks"]
                pred_logits = outputs["pred_logits"]
                
                # Convert to probability and find most likely class per query
                prob = pred_logits.softmax(-1)
                scores, labels = prob.max(-1)
                
                # Simplified inference merging for metrics evaluation
                pred_semantic = torch.zeros_like(semantic_masks)
                for b in range(images.size(0)):
                    # Get masks for this image, sigmoid them
                    b_masks = pred_masks[b].sigmoid()
                    # Filter out the 'no object' class predictions (the last index)
                    keep = labels[b] != self.criteria[task].num_classes
                    if keep.any():
                        b_masks = b_masks[keep]
                        b_labels = labels[b][keep]
                        
                        # Assign each pixel to the class with the highest mask probability
                        # (This is a simplified metric inference approximation)
                        mask_probs, mask_idx = b_masks.max(0)
                        pred_semantic[b] = b_labels[mask_idx]
                
                metric_tracker.update(pred_semantic, semantic_masks)
                
            val_losses[task] = task_loss / len(loader)
            
            # Print task-specific metrics cleanly
            results = metric_tracker.compute()
            print(f"\n[{task}] Val Loss: {val_losses[task]:.4f} | mIoU: {results['miou']:.4f} | mAcc: {results['mean_accuracy']:.4f}")
            
        return val_losses

    def fit(self, resume_path: Optional[str] = None):
        print(f"Starting Multi-Task Training on {self.device}")
        if resume_path:
            self.resume(resume_path)
            
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_losses = self.validate() 
            
            avg_val_loss = sum(val_losses.values()) / len(val_losses)
            
            print(f"\n--- Epoch {epoch} Summary ---")
            for task in val_losses.keys():
                print(f"Task '{task}': Train Loss = {train_metrics[f'{task}_loss']:.4f}")
            print(f"Average Val Loss: {avg_val_loss:.4f}\n")

            is_best = avg_val_loss < self.best_loss
            if is_best:
                self.best_loss = avg_val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self.save_checkpoint(epoch, is_best=is_best, filename="latest.pth")

            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break