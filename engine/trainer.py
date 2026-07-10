"""
trainer.py

Production-grade Multi-task training loop for TerraMind Mask2Former.
Features AMP, Linear-Warmup-Cosine Scheduling, Early Stopping, 
History Tracking (CSV/JSON), and unified Checkpointing.
"""

import os
import csv
import json
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

from losses.criterion import SetCriterion
from losses.matcher import HungarianMatcher
from engine.metrics import SegmentationMetrics
# Import BOTH shared utilities from inference_utils
from engine.inference_utils import semantic_to_mask2former_targets, postprocess_mask2former_outputs


def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup -> Cosine decay scheduler"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cap progress at 1.0 so the LR stays at 0 if we train past total_steps
        progress = min(1.0, float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps)))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
    return LambdaLR(optimizer, lr_lambda)


class MultiTaskTrainer:
    def __init__(
        self,
        model: nn.Module,
        datamodule,
        full_config: dict,
        device: str = "cuda",
        save_dir: str = "./checkpoints",
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model = model.to(device)
        self.datamodule = datamodule
        self.device = device
        self.config = full_config
        
        self.tasks_config = full_config.get("TASKS", {})
        self.train_config = full_config.get("TRAIN", {})
        self.epochs = self.train_config.get("epochs", 50)
        
        self.save_dir = save_dir
        self.patience = self.train_config.get("patience", 15)
        self.gradient_clip = self.train_config.get("gradient_clip", 1.0)
        os.makedirs(save_dir, exist_ok=True)

        # Optimizer & AMP Scaler
        trainable_params = self.model.get_trainable_params()
        encoder_param_ids = {id(p) for p in self.model.encoder.parameters()}
        if any(id(p) in encoder_param_ids for p in trainable_params):
            raise RuntimeError("Encoder parameters leaked into the optimizer!")

        self.optimizer = AdamW(
            trainable_params, 
            lr=self.train_config["lr"], 
            weight_decay=self.train_config.get("weight_decay", 0.05)
        )
        self.scaler = GradScaler(enabled=self.train_config.get("amp", True))

        # Loss Criteria Setup
        weight_dict = {"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0}

        # Number of auxiliary (deep-supervision) losses equals dec_layers:
        # the decoder emits one prediction before the loop plus one per
        # decoder layer, and all but the last are "aux" outputs. This MUST
        # match models.transformer_decoder.MultiScaleMaskedTransformerDecoder's
        # actual `dec_layers`, not be hardcoded, or SetCriterion's strict
        # weight-validation will raise a ValueError on the first training step.
        dec_layers = (
            full_config.get("MODEL", {})
            .get("transformer_decoder", {})
            .get("dec_layers", 9)
        )
        aux_weight_dict = {f"{k}_{i}": v for i in range(dec_layers) for k, v in weight_dict.items()}
        weight_dict.update(aux_weight_dict)

        self.criteria = {}
        matcher = HungarianMatcher(**full_config.get("MATCHER", {}))
        for task, cfg in self.tasks_config.items():
            self.criteria[task] = SetCriterion(
                num_classes=cfg["num_classes"], 
                matcher=matcher, 
                weight_dict=weight_dict, 
                losses=["labels", "masks"], 
                eos_coef=0.1
            ).to(device)

        # Data Loaders & Schedulers
        self.train_loader = self.datamodule.train_dataloader()
        self.val_loaders = self.datamodule.val_dataloader()
        
        total_steps = self.epochs * len(self.train_loader)
        warmup_steps = self.train_config.get("warmup_epochs", 5) * len(self.train_loader)
        self.scheduler = get_warmup_cosine_scheduler(self.optimizer, warmup_steps, total_steps)

        # State tracking
        self.start_epoch = 1
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0

        self.history = {
            "epoch": [], "train_loss": [], "train_loss_ce": [], "train_loss_mask": [], "train_loss_dice": [],
            "val_loss": [], "val_loss_ce": [], "val_loss_mask": [], "val_loss_dice": [],
            "miou": [], "dice": [], "precision": [], "recall": [], "pixel_acc": [], "lr": []
        }

        with open(os.path.join(self.save_dir, "training.log"), "w") as f:
            f.write("--- Training Log Initialized ---\n\n")

    def save_checkpoint(self, epoch: int, filename: str = "best_model.pth"):
        """Saves the training state. (Simplified to prevent filename mixups)"""
        state = {
            "epoch": epoch,
            "best_epoch": self.best_epoch,
            "config": self.config,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "best_loss": self.best_loss,
        }
        torch.save(state, os.path.join(self.save_dir, filename))

    def resume(self, checkpoint_path: str):
        print(f"Resuming from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.best_loss = checkpoint["best_loss"]

    def train_epoch(self, epoch: int) -> dict:
        self.model.train() 
        self.model.encoder.eval() 

        metrics = {"loss": 0.0, "loss_ce": 0.0, "loss_mask": 0.0, "loss_dice": 0.0}
        steps = 0
        use_amp = "cuda" in self.device
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")
        
        for step, (task, batch) in enumerate(pbar):
            images = batch["image"].to(self.device)
            semantic_masks = batch["mask"].to(self.device)
            
            task_cfg = self.tasks_config[task]
            targets = semantic_to_mask2former_targets(
                semantic_masks, 
                num_classes=task_cfg["num_classes"],
                ignore_index=task_cfg.get("ignore_index", 255),
                background_id=task_cfg.get("background_id", None)
            )

            self.optimizer.zero_grad()

            with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                outputs = self.model(images, task=task)
                
                if outputs["pred_logits"].device != images.device:
                    raise RuntimeError("Device mismatch between inputs and outputs")

                loss_dict = self.criteria[task](outputs, targets)
                weight_dict = self.criteria[task].weight_dict
                total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            if not torch.isfinite(total_loss):
                raise RuntimeError(f"Loss is {total_loss.item()}. Stopping training to prevent corruption.")

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Gradient norm is {grad_norm.item()}. Stopping training.")
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            metrics["loss"] += total_loss.item()
            metrics["loss_ce"] += loss_dict.get("loss_ce", torch.tensor(0.0)).item()
            metrics["loss_mask"] += loss_dict.get("loss_mask", torch.tensor(0.0)).item()
            metrics["loss_dice"] += loss_dict.get("loss_dice", torch.tensor(0.0)).item()
            steps += 1
            
            pbar.set_postfix(loss=f"{total_loss.item():.3f}", lr=f"{self.scheduler.get_last_lr()[0]:.2e}")

        return {k: v / steps for k, v in metrics.items()}

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        
        metrics = {
            "loss": 0.0, "loss_ce": 0.0, "loss_mask": 0.0, "loss_dice": 0.0,
            "miou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "pixel_acc": 0.0
        }
        num_tasks = len(self.val_loaders)
        use_amp = "cuda" in self.device
        
        for task, loader in self.val_loaders.items():
            task_cfg = self.tasks_config[task]
            metric_tracker = SegmentationMetrics(
                num_classes=task_cfg["num_classes"], 
                ignore_index=task_cfg.get("ignore_index", 255)
            )
            
            task_loss, task_ce, task_mask, task_dice = 0.0, 0.0, 0.0, 0.0
            steps = 0
            
            pbar = tqdm(loader, desc=f"Validation [{task}]", leave=False)
            for batch in pbar:
                images = batch["image"].to(self.device)
                semantic_masks = batch["mask"].to(self.device)
                targets = semantic_to_mask2former_targets(
                    semantic_masks, 
                    num_classes=task_cfg["num_classes"],
                    ignore_index=task_cfg.get("ignore_index", 255),
                    background_id=task_cfg.get("background_id", None)
                )

                # We extract metadata safely but don't pass it to the model 
                # to strictly obey the multi-task model's forward signature.
                metadata = batch.get("metadata", None)
                if metadata is not None and isinstance(metadata, dict):
                    metadata = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in metadata.items()}

                with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                    outputs = self.model(images, task=task)
                    loss_dict = self.criteria[task](outputs, targets)
                    weight_dict = self.criteria[task].weight_dict
                    total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                
                task_loss += total_loss.item()
                task_ce += loss_dict.get("loss_ce", torch.tensor(0.0)).item()
                task_mask += loss_dict.get("loss_mask", torch.tensor(0.0)).item()
                task_dice += loss_dict.get("loss_dice", torch.tensor(0.0)).item()
                steps += 1
                
                target_size = semantic_masks.shape[-2:]
                pred_semantic = postprocess_mask2former_outputs(outputs["pred_logits"], outputs["pred_masks"], target_size)
                
                metric_tracker.update(pred_semantic, semantic_masks)
                
            # FIX: Prevent ZeroDivisionError if a validation dataloader is empty!
            metrics["loss"] += task_loss / steps if steps > 0 else 0.0
            metrics["loss_ce"] += task_ce / steps if steps > 0 else 0.0
            metrics["loss_mask"] += task_mask / steps if steps > 0 else 0.0
            metrics["loss_dice"] += task_dice / steps if steps > 0 else 0.0
            
            res = metric_tracker.compute()
            metrics["miou"] += res["miou"]
            metrics["dice"] += res["mdice"]
            metrics["precision"] += res["mean_precision"]
            metrics["recall"] += res["mean_recall"]
            metrics["pixel_acc"] += res["pixel_accuracy"]

        # Average across tasks
        return {k: v / num_tasks for k, v in metrics.items()}

    def fit(self, resume_path: Optional[str] = None):
        print(f"Starting Multi-Task Training on {self.device}")
        if resume_path:
            self.resume(resume_path)
            
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate() 
            current_lr = self.scheduler.get_last_lr()[0]
            
            # 1. Update History
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_loss_ce"].append(train_metrics["loss_ce"])
            self.history["train_loss_mask"].append(train_metrics["loss_mask"])
            self.history["train_loss_dice"].append(train_metrics["loss_dice"])
            
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_loss_ce"].append(val_metrics["loss_ce"])
            self.history["val_loss_mask"].append(val_metrics["loss_mask"])
            self.history["val_loss_dice"].append(val_metrics["loss_dice"])
            
            self.history["miou"].append(val_metrics["miou"])
            self.history["dice"].append(val_metrics["dice"])
            self.history["precision"].append(val_metrics["precision"])
            self.history["recall"].append(val_metrics["recall"])
            self.history["pixel_acc"].append(val_metrics["pixel_acc"])
            self.history["lr"].append(current_lr)

            # 2. Save JSON and CSV
            with open(os.path.join(self.save_dir, "history.json"), "w") as f:
                json.dump(self.history, f, indent=4)
                
            with open(os.path.join(self.save_dir, "history.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                keys = list(self.history.keys())
                writer.writerow(keys)
                writer.writerows(zip(*[self.history[k] for k in keys]))

            # 3. Write and Print Text Logger
            log_msg = (
                f"Epoch {epoch}\n"
                f"Train Loss : {train_metrics['loss']:.4f}\n"
                f"Validation Loss : {val_metrics['loss']:.4f}\n"
                f"mIoU : {val_metrics['miou']:.4f}\n"
                f"Dice : {val_metrics['dice']:.4f}\n"
                f"Precision : {val_metrics['precision']:.4f}\n"
                f"Recall : {val_metrics['recall']:.4f}\n"
                f"Pixel Accuracy : {val_metrics['pixel_acc']:.4f}\n"
                f"Learning Rate : {current_lr:.6f}\n"
                f"----------------------------------\n"
            )
            with open(os.path.join(self.save_dir, "training.log"), "a") as f:
                f.write(log_msg)
            print(log_msg)

            # 4. Checkpoint Logic (Save ONLY best_model.pth)
            if val_metrics["loss"] < self.best_loss:
                self.best_loss = val_metrics["loss"]
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, filename="best_model.pth")
                print(f"*** New best model saved! (Loss: {self.best_loss:.4f}) ***\n")
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break