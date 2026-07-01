"""
Training engine for TerraMind + Mask2Former.

Responsibilities
----------------
• Build optimizer
• Build scheduler
• Mixed precision training
• Round-robin task routing
• Training
• Validation with per-task metrics
• Checkpointing
"""


from __future__ import annotations


import time
from pathlib import Path


import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


def compute_iou(pred, target, num_classes, ignore_index=255):
    """Compute IoU per class."""
    pred = pred.view(-1)
    target = target.view(-1)
    
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    
    ious = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union > 0:
            ious.append((intersection / union).item())
        else:
            ious.append(float('nan'))
    
    return ious


def compute_dice(pred, target, num_classes, ignore_index=255):
    """Compute Dice score per class."""
    pred = pred.view(-1)
    target = target.view(-1)
    
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    
    dices = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        
        intersection = (pred_cls & target_cls).sum().float()
        total = pred_cls.sum() + target_cls.sum()
        
        if total > 0:
            dices.append((2 * intersection / total).item())
        else:
            dices.append(float('nan'))
    
    return dices


class Trainer:


    def __init__(
        self,
        model,
        criterion,
        flood_train_loader,
        burn_train_loader,
        lulc_train_loader,
        flood_val_loader,
        burn_val_loader,
        lulc_val_loader,
        device,
        epochs=100,
        lr=1e-4,
        weight_decay=0.05,
        output_dir="./checkpoints",
        amp=True,
        task_schedule=None,
        num_classes_per_task=None,
    ):
        self.model = model.to(device)
        self.criterion = criterion

        # Store dataloaders per task
        self.flood_train_loader = flood_train_loader
        self.burn_train_loader = burn_train_loader
        self.lulc_train_loader = lulc_train_loader

        self.flood_val_loader = flood_val_loader
        self.burn_val_loader = burn_val_loader
        self.lulc_val_loader = lulc_val_loader

        # Number of classes per task
        self.num_classes = num_classes_per_task or {
            "flood": 2,
            "burn": 2,
            "lulc": 7,
        }
        
        # Task loaders map
        self.val_loaders = {
            "flood": flood_val_loader,
            "burn": burn_val_loader,
            "lulc": lulc_val_loader,
        }
        
        self.train_loaders = {
            "flood": flood_train_loader,
            "burn": burn_train_loader,
            "lulc": lulc_train_loader,
        }

        self.device = device
        self.epochs = epochs

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_amp = amp
        self.scaler = GradScaler(enabled=amp)

        self.optimizer = self.build_optimizer(lr, weight_decay)
        self.scheduler = self.build_scheduler()

        self.best_score = -1.0
        self.start_epoch = 0

        # Configurable task schedule
        self.task_schedule = task_schedule or ["flood", "burn", "lulc"]

    def build_optimizer(self, lr, weight_decay):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        return AdamW(parameters, lr=lr, weight_decay=weight_decay)

    def build_scheduler(self):
        return CosineAnnealingLR(self.optimizer, T_max=self.epochs)

    def save_checkpoint(self, epoch, is_best=False, metrics=None):
        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_score": self.best_score,
            "metrics": metrics,
        }
        latest = self.output_dir / "latest.pth"
        torch.save(checkpoint, latest)
        if is_best:
            best = self.output_dir / "best.pth"
            torch.save(checkpoint, best)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_score = checkpoint["best_score"]
        self.start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {self.start_epoch}")

    def forward_batch(self, batch, task):
        images = batch["image"].to(self.device)

        # Targets are already prepared by collate_fn
        targets = []
        for target in batch["targets"]:
            targets.append({
                "labels": target["labels"].to(self.device),
                "masks": target["masks"].to(self.device),
            })

        with autocast(enabled=self.use_amp):
            outputs = self.model(images, task=task)
            loss_dict = self.criterion(outputs, targets)
            total_loss = sum(loss_dict.values())

        return total_loss, loss_dict, outputs

    def train_one_epoch(self, epoch):
        self.model.train()

        epoch_steps = max(
            len(self.flood_train_loader),
            len(self.burn_train_loader),
            len(self.lulc_train_loader),
        )

        iters = {task: iter(loader) for task, loader in self.train_loaders.items()}
        running_loss = 0.0
        total_updates = 0

        progress = tqdm(range(epoch_steps), desc=f"Epoch {epoch}")

        for step in progress:
            for task in self.task_schedule:
                try:
                    batch = next(iters[task])
                except StopIteration:
                    iters[task] = iter(self.train_loaders[task])
                    batch = next(iters[task])

                self.optimizer.zero_grad(set_to_none=True)
                loss, loss_dict, _ = self.forward_batch(batch, task)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += loss.item()
                total_updates += 1

                progress.set_postfix(
                    task=task,
                    loss=f"{loss.item():.4f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )

        return running_loss / total_updates

    @torch.no_grad()
    def validate_task(self, loader, task):
        """Validate a single task with IoU and Dice metrics."""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch in tqdm(loader, desc=f"Val {task}"):
            loss, _, outputs = self.forward_batch(batch, task)
            total_loss += loss.item()
            
            # Get predictions
            pred_masks = outputs["pred_masks"]
            pred_logits = outputs["pred_logits"]
            pred_classes = pred_logits.argmax(dim=-1)
            
            # Get ground truth
            gt_masks = batch["mask"].to(self.device)
            
            # Convert predictions to semantic masks (simplified)
            pred_semantic = pred_classes[:, :, None, None].expand(-1, -1, gt_masks.shape[-2], gt_masks.shape[-1])
            
            all_preds.append(pred_semantic.cpu())
            all_targets.append(gt_masks.cpu())
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        num_classes = self.num_classes[task]
        ious = compute_iou(all_preds, all_targets, num_classes)
        dices = compute_dice(all_preds, all_targets, num_classes)
        
        # Compute means (ignoring NaN)
        valid_ious = [x for x in ious if not (x != x)]  # filter NaN
        valid_dices = [x for x in dices if not (x != x)]
        
        miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
        mean_dice = sum(valid_dices) / len(valid_dices) if valid_dices else 0.0
        
        avg_loss = total_loss / len(loader)
        
        return {
            "loss": avg_loss,
            "miou": miou,
            "dice": mean_dice,
            "ious_per_class": ious,
            "dices_per_class": dices,
        }

    def validate(self):
        """Validate all tasks."""
        results = {}
        
        for task in self.task_schedule:
            loader = self.val_loaders[task]
            results[task] = self.validate_task(loader, task)
        
        # Compute overall score (average of all task mIoUs)
        overall_miou = sum(r["miou"] for r in results.values()) / len(results)
        
        return results, overall_miou

    def train(self):
        print("\nStarting Training...\n")

        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()

            train_loss = self.train_one_epoch(epoch)
            results, overall_miou = self.validate()

            self.scheduler.step()

            elapsed = time.time() - start

            # Print per-task metrics
            print(f"\nEpoch [{epoch+1}/{self.epochs}] - Time: {elapsed:.2f}s")
            print("-" * 60)
            
            for task, metrics in results.items():
                print(f"{task.upper():>8} | Loss: {metrics['loss']:.4f} | mIoU: {metrics['miou']:.4f} | Dice: {metrics['dice']:.4f}")
            
            print("-" * 60)
            print(f"{'OVERALL':>8} | mIoU: {overall_miou:.4f}")
            print()

            is_best = overall_miou > self.best_score
            if is_best:
                self.best_score = overall_miou

            self.save_checkpoint(epoch, is_best=is_best, metrics=results)

        print("\nTraining Complete.")

    @torch.no_grad()
    def predict(self, images, task):
        self.model.eval()
        images = images.to(self.device)
        outputs = self.model(images, task=task)
        return outputs
