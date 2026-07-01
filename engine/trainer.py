"""
Training engine for TerraMind + Mask2Former.


Responsibilities
----------------
• Build optimizer
• Build scheduler
• Mixed precision training
• Task routing
• Training
• Validation
• Checkpointing
"""


from __future__ import annotations


import os
import time
from pathlib import Path


import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm




class Trainer:


    def __init__(
        self,
        model,
        criterion,
        train_loader,
        val_loader,
        device,
        epochs=100,
        lr=1e-4,
        weight_decay=0.05,
        output_dir="./checkpoints",
        amp=True,
    ):


        self.model = model.to(device)


        self.criterion = criterion


        self.train_loader = train_loader
        self.val_loader = val_loader


        self.device = device


        self.epochs = epochs


        self.output_dir = Path(output_dir)


        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        self.use_amp = amp


        self.scaler = GradScaler(
            enabled=amp,
        )


        self.optimizer = self.build_optimizer(
            lr,
            weight_decay,
        )


        self.scheduler = self.build_scheduler()


        self.best_score = -1.0


        self.start_epoch = 0


    ###########################################################


    def build_optimizer(
        self,
        lr,
        weight_decay,
    ):


        parameters = [
            p
            for p in self.model.parameters()
            if p.requires_grad
        ]


        optimizer = AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
        )


        return optimizer


    ###########################################################


    def build_scheduler(self):


        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
        )


        return scheduler


    ###########################################################


    def save_checkpoint(
        self,
        epoch,
        is_best=False,
    ):


        checkpoint = {


            "epoch": epoch,


            "model": self.model.state_dict(),


            "optimizer": self.optimizer.state_dict(),


            "scheduler": self.scheduler.state_dict(),


            "best_score": self.best_score,


        }


        latest = self.output_dir / "latest.pth"


        torch.save(
            checkpoint,
            latest,
        )


        if is_best:


            best = self.output_dir / "best.pth"


            torch.save(
                checkpoint,
                best,
            )


    ###########################################################


    def load_checkpoint(
        self,
        path,
    ):


        checkpoint = torch.load(
            path,
            map_location=self.device,
        )


        self.model.load_state_dict(
            checkpoint["model"]
        )


        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )


        self.scheduler.load_state_dict(
            checkpoint["scheduler"]
        )


        self.best_score = checkpoint["best_score"]


        self.start_epoch = checkpoint["epoch"] + 1


        print(
            f"Resumed from epoch {self.start_epoch}"
        )


    ###########################################################
    # Forward Pass
    ###########################################################


    def forward_batch(self, batch):


        images = batch["image"].to(self.device)


        task = batch["task"]


        if isinstance(task, (list, tuple)):
            task = task[0]


        targets = []


        masks = batch["mask"]


        if isinstance(masks, torch.Tensor):


            masks = masks.to(self.device)


            for mask in masks:


                classes = torch.unique(mask)


                classes = classes[classes != 255]


                gt_masks = []
                gt_classes = []


                for cls in classes:


                    gt_masks.append(mask == cls)
                    gt_classes.append(cls)


                if len(gt_masks) == 0:


                    gt_masks = torch.zeros(
                        (0, mask.shape[0], mask.shape[1]),
                        device=self.device,
                        dtype=torch.bool,
                    )


                    gt_classes = torch.zeros(
                        (0,),
                        device=self.device,
                        dtype=torch.long,
                    )


                else:


                    gt_masks = torch.stack(gt_masks)


                    gt_classes = torch.stack(gt_classes)


                targets.append(
                    {
                        "labels": gt_classes,
                        "masks": gt_masks.float(),
                    }
                )


        with autocast(enabled=self.use_amp):


            outputs = self.model(
                images,
                task=task,
            )


            loss_dict = self.criterion(
                outputs,
                targets,
            )


            total_loss = sum(loss_dict.values())


        return total_loss, loss_dict


    ###########################################################
    # Train One Epoch
    ###########################################################


    def train_one_epoch(
        self,
        epoch,
    ):


        self.model.train()


        running_loss = 0.0


        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
        )


        for step, batch in enumerate(progress):


            self.optimizer.zero_grad(
                set_to_none=True,
            )


            loss, loss_dict = self.forward_batch(
                batch,
            )


            self.scaler.scale(loss).backward()


            self.scaler.unscale_(self.optimizer)


            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=0.1,
            )


            self.scaler.step(
                self.optimizer,
            )


            self.scaler.update()


            running_loss += loss.item()


            progress.set_postfix(


                loss=f"{loss.item():.4f}",


                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",


            )


        epoch_loss = running_loss / len(
            self.train_loader
        )


        return epoch_loss


    ###########################################################
    # Validation
    ###########################################################


    @torch.no_grad()
    def validate(self):


        self.model.eval()


        total_loss = 0.0


        for batch in tqdm(
            self.val_loader,
            desc="Validation",
        ):


            loss, _ = self.forward_batch(batch)


            total_loss += loss.item()


        return total_loss / len(self.val_loader)


    ###########################################################
    # Main Training Loop
    ###########################################################


    def train(self):


        print(
            "\nStarting Training...\n"
        )


        for epoch in range(
            self.start_epoch,
            self.epochs,
        ):


            start = time.time()


            train_loss = self.train_one_epoch(
                epoch,
            )


            val_loss = self.validate()


            self.scheduler.step()


            elapsed = time.time() - start


            print(
                f"Epoch [{epoch+1}/{self.epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Time: {elapsed:.2f}s"
            )


            score = -val_loss


            is_best = score > self.best_score


            if is_best:
                self.best_score = score


            self.save_checkpoint(
                epoch,
                is_best=is_best,
            )


        print(
            "\nTraining Complete."
        )


    ###########################################################
    # Prediction
    ###########################################################


    @torch.no_grad()
    def predict(
        self,
        images,
        task,
    ):


        self.model.eval()


        images = images.to(self.device)


        outputs = self.model(
            images,
            task=task,
        )


        return outputs
