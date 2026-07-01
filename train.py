"""
Training script for TerraMind + Mask2Former.


Usage:
    python train.py --config config.yaml
    python train.py --model_path checkpoints/best.pth
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import ModelConfig
from datasets import (
    Sen1Flood11Dataset,
    BurnScarDataset,
    LoveDADataset,
    get_train_transforms,
    get_val_transforms,
)
from losses import HungarianMatcher, SetCriterion
from models import (
    TerraMindEncoder,
    TerraMindNeck,
    MSDeformAttnPixelDecoder,
    MultiScaleMaskedTransformerDecoder,
)
from engine import Trainer


def build_dataloader(dataset_name, root, batch_size, num_workers, split="train"):
    """Build dataloader for a dataset."""
    
    transform = get_train_transforms(image_size=224) if split == "train" else get_val_transforms(image_size=224)
    
    if dataset_name == "sen1flood11":
        dataset = Sen1Flood11Dataset(root=root, split=split, transform=transform)
    elif dataset_name == "burnscar":
        dataset = BurnScarDataset(root=root, split=split, transform=transform)
    elif dataset_name == "loveda":
        dataset = LoveDADataset(root=root, split=split, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return loader


def build_model(cfg):
    """Build the full model."""
    
    # Encoder (frozen)
    encoder = TerraMindEncoder(
        backbone_name=cfg.backbone_name,
        pretrained=cfg.pretrained,
        modalities=cfg.modalities,
        merge_method=cfg.merge_method,
        freeze=cfg.freeze_backbone,
    )
    
    # Neck
    neck = TerraMindNeck(
        embed_dim=encoder.out_channels[0] if isinstance(encoder.out_channels, list) else encoder.out_channels,
        hidden_dim=cfg.hidden_dim,
    )
    
    # Pixel Decoder
    pixel_decoder = MSDeformAttnPixelDecoder(
        in_channels=neck.out_channels,
        conv_dim=cfg.conv_dim,
        mask_dim=cfg.mask_dim,
        transformer_enc_layers=cfg.transformer_enc_layers,
    )
    
    # Task-specific decoders
    decoders = {}
    for task_name, num_classes in [("flood", 2), ("burnscar", 2), ("lulc", 7)]:
        decoders[task_name] = MultiScaleMaskedTransformerDecoder(
            in_channels=cfg.hidden_dim,
            num_classes=num_classes,
            hidden_dim=cfg.hidden_dim,
            num_queries=cfg.num_queries,
            nheads=cfg.nheads,
            dim_feedforward=cfg.dim_feedforward,
            dec_layers=cfg.dec_layers,
            mask_dim=cfg.mask_dim,
        )
    
    # Build model wrapper (placeholder - implement multitask_model.py separately)
    from models.multitask_model import MultiTaskMask2Former
    model = MultiTaskMask2Former(
        encoder=encoder,
        neck=neck,
        pixel_decoder=pixel_decoder,
        decoders=decoders,
    )
    
    return model


def build_criterion(cfg):
    """Build loss criterion."""
    matcher = HungarianMatcher(
        cost_class=cfg.cost_class,
        cost_mask=cfg.cost_mask,
        cost_dice=cfg.cost_dice,
        num_points=cfg.num_points,
    )
    
    weight_dict = {
        "loss_ce": cfg.class_weight,
        "loss_mask": cfg.mask_weight,
        "loss_dice": cfg.dice_weight,
    }
    
    criterion = SetCriterion(
        num_classes=cfg.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=cfg.no_object_weight,
        losses=["labels", "masks"],
        num_points=cfg.num_points,
        oversample_ratio=cfg.oversample_ratio,
        importance_sample_ratio=cfg.importance_sample_ratio,
    )
    
    return criterion


def main(args):
    # Load config
    cfg = ModelConfig()
    
    # Override with args if provided
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.lr = args.lr
    
    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    # Build model
    print("Building model...")
    model = build_model(cfg)
    model = model.to(device)
    
    # Build criterion
    criterion = build_criterion(cfg)
    criterion = criterion.to(device)
    
    # Build dataloaders
    print("Building dataloaders...")
    train_loader = build_dataloader(
        args.dataset,
        args.data_root,
        cfg.batch_size,
        cfg.workers,
        split="train",
    )
    val_loader = build_dataloader(
        args.dataset,
        args.data_root,
        cfg.batch_size,
        cfg.workers,
        split="val",
    )
    
    # Build trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        output_dir=args.output_dir,
        amp=cfg.amp,
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        print(f"Resuming from {args.resume}...")
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()
    
    print(f"Best model saved to {args.output_dir}/best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TerraMind + Mask2Former")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset")
    parser.add_argument("--dataset", type=str, required=True, choices=["sen1flood11", "burnscar", "loveda"])
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    
    args = parser.parse_args()
    main(args)
