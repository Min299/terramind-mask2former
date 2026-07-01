"""
Training engine for TerraMind + Mask2Former.

Responsibilities
----------------
• Build optimizer
• Build scheduler
• Mixed precision training
• Round-robin task routing
• Training
• Validation
• Checkpointing
"""


from __future__ import annotations


import time
from pathlib import Path


import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


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

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_score": self.best_score,
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

        return total_loss, loss_dict

    def train_one_epoch(self, epoch):
        self.model.train()

        # Create iterators
        loaders = {
            "flood": self.flood_train_loader,
            "burn": self.burn_train_loader,
            "lulc": self.lulc_train_loader,
        }

        iters = {
            "flood": iter(self.flood_train_loader),
            "burn": iter(self.burn_train_loader),
            "lulc": iter(self.lulc_train_loader),
        }

        max_steps = max(
            len(self.flood_train_loader),
            len(self.burn_train_loader),
            len(self.lulc_train_loader),
        )

        running_loss = 0.0
        total_updates = 0

        progress = tqdm(range(max_steps), desc=f"Epoch {epoch}")

        for _ in progress:
            for task in self.task_schedule:
                try:
                    batch = next(iters[task])
                except StopIteration:
                    iters[task] = iter(loaders[task])
                    batch = next(iters[task])

                self.optimizer.zero_grad(set_to_none=True)
                loss, loss_dict = self.forward_batch(batch, task)
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
    def validate(self, loader, task):
        self.model.eval()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Val {task}"):
            loss, _ = self.forward_batch(batch, task)
            total_loss += loss.item()
        return total_loss / len(loader)

    def train(self):
        print("\nStarting Training...\n")

        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()

            train_loss = self.train_one_epoch(epoch)

            # Validate all tasks
            flood_loss = self.validate(self.flood_val_loader, "flood")
            burn_loss = self.validate(self.burn_val_loader, "burn")
            lulc_loss = self.validate(self.lulc_val_loader, "lulc")
            val_loss = (flood_loss + burn_loss + lulc_loss) / 3.0

            self.scheduler.step()

            elapsed = time.time() - start

            print(
                f"Epoch [{epoch+1}/{self.epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} (f: {flood_loss:.4f}, b: {burn_loss:.4f}, l: {lulc_loss:.4f}) | "
                f"Time: {elapsed:.2f}s"
            )

            score = -val_loss
            is_best = score > self.best_score
            if is_best:
                self.best_score = score

            self.save_checkpoint(epoch, is_best=is_best)

        print("\nTraining Complete.")

    @torch.no_grad()
    def predict(self, images, task):
        self.model.eval()
        images = images.to(self.device)
        outputs = self.model(images, task=task)
        return outputs
