"""
Training script for TerraMind + Mask2Former.

Usage:
    python train.py --flood_data ROOT --burn_data ROOT --lulc_data ROOT
"""

import argparse
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
    multitask_collate_fn,
)
from losses import HungarianMatcher, SetCriterion
from models import (
    TerraMindEncoder,
    TerraMindNeck,
    MSDeformAttnPixelDecoder,
    MultiScaleMaskedTransformerDecoder,
    MultiTaskMask2Former,
)
from engine import Trainer


def build_dataloader(dataset_class, root, batch_size, num_workers, split="train"):
    """Build dataloader with collate function."""
    transform = get_train_transforms(image_size=224) if split == "train" else get_val_transforms(image_size=224)
    
    dataset = dataset_class(root=root, split=split, transform=transform)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=multitask_collate_fn,
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
    
    # Get encoder output channels
    enc_channels = encoder.out_channels
    if isinstance(enc_channels, list):
        enc_channels = enc_channels[0]
    
    # Neck
    neck = TerraMindNeck(
        embed_dim=enc_channels,
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
    decoders = {
        "flood": MultiScaleMaskedTransformerDecoder(
            in_channels=cfg.hidden_dim,
            num_classes=2,  # Sen1Flood11: background, flood
            hidden_dim=cfg.hidden_dim,
            num_queries=cfg.num_queries,
            nheads=cfg.nheads,
            dim_feedforward=cfg.dim_feedforward,
            dec_layers=cfg.dec_layers,
            mask_dim=cfg.mask_dim,
        ),
        "burn": MultiScaleMaskedTransformerDecoder(
            in_channels=cfg.hidden_dim,
            num_classes=2,  # BurnScar: background, burned
            hidden_dim=cfg.hidden_dim,
            num_queries=cfg.num_queries,
            nheads=cfg.nheads,
            dim_feedforward=cfg.dim_feedforward,
            dec_layers=cfg.dec_layers,
            mask_dim=cfg.mask_dim,
        ),
        "lulc": MultiScaleMaskedTransformerDecoder(
            in_channels=cfg.hidden_dim,
            num_classes=7,  # LoveDA: 7 classes
            hidden_dim=cfg.hidden_dim,
            num_queries=cfg.num_queries,
            nheads=cfg.nheads,
            dim_feedforward=cfg.dim_feedforward,
            dec_layers=cfg.dec_layers,
            mask_dim=cfg.mask_dim,
        ),
    }
    
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
    cfg = ModelConfig()
    
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.lr = args.lr
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    print("Building model...")
    model = build_model(cfg)
    model = model.to(device)
    
    print("Building criterion...")
    criterion = build_criterion(cfg)
    criterion = criterion.to(device)
    
    print("Building dataloaders...")
    # Flood dataloaders
    flood_train = build_dataloader(
        Sen1Flood11Dataset, args.flood_data, cfg.batch_size, cfg.workers, "train"
    )
    flood_val = build_dataloader(
        Sen1Flood11Dataset, args.flood_data, cfg.batch_size, cfg.workers, "val"
    )
    
    # Burn dataloaders
    burn_train = build_dataloader(
        BurnScarDataset, args.burn_data, cfg.batch_size, cfg.workers, "train"
    )
    burn_val = build_dataloader(
        BurnScarDataset, args.burn_data, cfg.batch_size, cfg.workers, "val"
    )
    
    # LULC dataloaders
    lulc_train = build_dataloader(
        LoveDADataset, args.lulc_data, cfg.batch_size, cfg.workers, "train"
    )
    lulc_val = build_dataloader(
        LoveDADataset, args.lulc_data, cfg.batch_size, cfg.workers, "val"
    )
    
    print("Building trainer...")
    trainer = Trainer(
        model=model,
        criterion=criterion,
        flood_train_loader=flood_train,
        burn_train_loader=burn_train,
        lulc_train_loader=lulc_train,
        flood_val_loader=flood_val,
        burn_val_loader=burn_val,
        lulc_val_loader=lulc_val,
        device=device,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        output_dir=args.output_dir,
        amp=cfg.amp,
        task_schedule=args.schedule,
    )
    
    if args.resume:
        print(f"Resuming from {args.resume}...")
        trainer.load_checkpoint(args.resume)
    
    trainer.train()
    print(f"Best model saved to {args.output_dir}/best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TerraMind + Mask2Former (Multi-Task)")
    parser.add_argument("--flood_data", type=str, required=True, help="Path to Sen1Flood11 dataset")
    parser.add_argument("--burn_data", type=str, required=True, help="Path to HLS Burn Scar dataset")
    parser.add_argument("--lulc_data", type=str, required=True, help="Path to LoveDA dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--schedule",
        type=str,
        nargs="+",
        default=["flood", "burn", "lulc"],
        help="Task schedule (e.g., flood burn lulc or flood flood burn lulc)",
    )
    
    args = parser.parse_args()
    main(args)
