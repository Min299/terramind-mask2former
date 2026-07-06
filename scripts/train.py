"""
train.py

Main entry point for multi-task TerraMind Mask2Former training.
Features a clean three-block config (MODEL, TASKS, TRAIN) for reproducibility.
"""

import os
import yaml
import argparse
import logging
import torch

from multitask_datamodule import MultiTaskDataModule
from collate import MultiTaskCollate
from trainer import MultiTaskTrainer

from encoder import TerraMindEncoder               
from neck import TerraMindNeck                     
from pixel_decoder import MSDeformAttnPixelDecoder 
from transformer_decoder import MultiScaleMaskedTransformerDecoder 
from model import MultiTaskMask2Former             


def parse_args():
    parser = argparse.ArgumentParser(description="Train TerraMind Mask2Former")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save weights/configs")
    return parser.parse_args()


def log_model_params(model, logger):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(f"Model Parameters -> Trainable: {trainable:,} | Frozen: {frozen:,} | Total: {trainable + frozen:,}")


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    # =========================================================
    # 1. CONFIGURATION
    # =========================================================
    MODEL_CONFIG = {
        "hidden_dim": 256,
        "mask_dim": 256,
    }
    
    TRAIN_CONFIG = {
        "epochs": 50,
        "lr": 1e-4,
        "weight_decay": 0.05,
        "warmup_epochs": 5,
        "epoch_mode": "fixed_steps",
        "fixed_steps": 2000,
    }
    
    TASKS = {
        "flood": {
            "num_classes": 2,
            "ignore_index": 255,
            "background_id": 0,    # Treat 0 as implicit background
            "batch_size": 4,
            "num_workers": 4,
            "data_root": "./data/sen1floods11",
        },
        "burn": {
            "num_classes": 2,
            "ignore_index": 255,
            "background_id": None, # E.g., burn dataset models 0 as a strict class
            "batch_size": 4,
            "num_workers": 4,
            "data_root": "./data/hls_burn_scars",
        },
        "lulc": {
            "num_classes": 10,
            "ignore_index": 255,
            "background_id": None, 
            "batch_size": 4,
            "num_workers": 4,
            "data_root": "./data/esri_lulc",
        },
    }
    
    # Save reproducible configuration
    config_path = os.path.join(args.save_dir, "training_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump({"TRAIN": TRAIN_CONFIG, "MODEL": MODEL_CONFIG, "TASKS": TASKS}, f)
    logger.info(f"Saved training configuration to {config_path}")

    # =========================================================
    # 2. DATAMODULES
    # =========================================================
    logger.info("Initializing MultiTask DataModule...")
    datamodule_config = {}
    for task_name, task_cfg in TASKS.items():
        datamodule_config[task_name] = task_cfg.copy()
        datamodule_config[task_name]["collate_fn"] = MultiTaskCollate()

    datamodule = MultiTaskDataModule(
        config=datamodule_config, 
        epoch_mode=TRAIN_CONFIG["epoch_mode"], 
        fixed_steps=TRAIN_CONFIG["fixed_steps"]
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    # =========================================================
    # 3. ARCHITECTURE
    # =========================================================
    logger.info("Building Model Architecture...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    encoder = TerraMindEncoder() 
    
    # Explicitly freeze the encoder immediately after creation
    for p in encoder.parameters():
        p.requires_grad = False
    
    # Direct access, fail fast if the encoder does not expose embed_dim
    embed_dim = encoder.embed_dim  
    
    neck = TerraMindNeck(embed_dim=embed_dim, hidden_dim=MODEL_CONFIG["hidden_dim"])
    pixel_decoder = MSDeformAttnPixelDecoder(conv_dim=MODEL_CONFIG["hidden_dim"], mask_dim=MODEL_CONFIG["mask_dim"])
    
    decoders = {}
    for task, cfg in TASKS.items():
        decoders[task] = MultiScaleMaskedTransformerDecoder(
            in_channels=pixel_decoder.conv_dim,  # Coupled to pixel_decoder
            num_classes=cfg["num_classes"], 
            activation="gelu",
            mask_dim=pixel_decoder.mask_dim      # Coupled to pixel_decoder
        )
        
    model = MultiTaskMask2Former(encoder, neck, pixel_decoder, decoders)
    log_model_params(model, logger)

    # =========================================================
    # 4. TRAINING
    # =========================================================
    logger.info("Initializing MultiTask Trainer...")
    trainer = MultiTaskTrainer(
        model=model,
        datamodule=datamodule,
        tasks_config=TASKS,
        train_config=TRAIN_CONFIG,
        device=device,
        save_dir=args.save_dir
    )
    
    logger.info(f"Starting training loop... (Resume: {args.resume})")
    trainer.fit(resume_path=args.resume)


if __name__ == "__main__":
    main()