"""
train.py

Main entry point for multi-task TerraMind Mask2Former training.
Features a clean three-block config (MODEL, TASKS, TRAIN) for reproducibility.
Initializes the save_dir, config.yaml, and loggers.
"""

import os
import yaml
import argparse
import logging
import torch

from data.multitask_datamodule import MultiTaskDataModule
from data.collate import MultiTaskCollate
from engine.trainer import MultiTaskTrainer
from engine.inference_utils import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train TerraMind Mask2Former")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save weights/configs")
    return parser.parse_args()


def validate_config_schema(config: dict):
    """
    Validates that the provided configuration dictionary contains all 
    architecturally critical keys before training begins.
    """
    required_blocks = {"MODEL", "TRAIN", "TASKS"}
    missing_blocks = required_blocks - set(config.keys())
    if missing_blocks:
        raise KeyError(f"Config is missing required top-level blocks: {missing_blocks}")

    # ---------------------------------------------------------
    # 1. Validate MODEL
    # ---------------------------------------------------------
    model_cfg = config["MODEL"]
    for key in ["hidden_dim", "mask_dim"]:
        if key not in model_cfg:
            raise KeyError(f"MODEL block is missing required key: '{key}'")
        if model_cfg[key] <= 0:
            raise ValueError(f"MODEL['{key}'] must be > 0, got {model_cfg[key]}")

    # ---------------------------------------------------------
    # 2. Validate TRAIN
    # ---------------------------------------------------------
    train_cfg = config["TRAIN"]
    required_train_keys = ["epochs", "lr"]
    for key in required_train_keys:
        if key not in train_cfg:
            raise KeyError(f"TRAIN block is missing required key: '{key}'")
            
    if train_cfg["epochs"] <= 0:
        raise ValueError("TRAIN['epochs'] must be > 0")
    if train_cfg["lr"] <= 0:
        raise ValueError("TRAIN['lr'] must be > 0")

    # ---------------------------------------------------------
    # 3. Validate TASKS
    # ---------------------------------------------------------
    tasks_cfg = config["TASKS"]
    if not isinstance(tasks_cfg, dict) or len(tasks_cfg) == 0:
        raise ValueError("TASKS block must contain at least one task dictionary.")

    for task_name, t_cfg in tasks_cfg.items():
        # Critically required to build the model decoder and criterion
        if "num_classes" not in t_cfg:
            raise KeyError(f"Task '{task_name}' is missing 'num_classes'")
        if t_cfg["num_classes"] <= 0:
            raise ValueError(f"Task '{task_name}' num_classes must be > 0")
            
        # Critically required to initialize the Dataloaders
        if "batch_size" not in t_cfg:
            raise KeyError(f"Task '{task_name}' is missing 'batch_size'")
        if t_cfg["batch_size"] <= 0:
            raise ValueError(f"Task '{task_name}' batch_size must be > 0")
            
        # Notice: ignore_index, background_id, and data_root are INTENTIONALLY excluded.
        # Downstream code uses .get() fallbacks for them or handles kwargs dynamically.


def log_model_params(model, logger):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(f"Model Parameters -> Trainable: {trainable:,} | Frozen: {frozen:,} | Total: {trainable + frozen:,}")


def main():
    args = parse_args()
    
    # 1. Initialize Directory Structure
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup Logger (Prints to console, trainer.py will handle the training.log file)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    # =========================================================
    # 2. CONFIGURATION
    # =========================================================
    logger.info(f"Loading configuration from {args.config}...")
    with open(args.config, "r") as f:
        full_config = yaml.safe_load(f)

    validate_config_schema(full_config)

    # 3. Copy YAML config to save_dir (Reproducibility)
    config_path = os.path.join(args.save_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(full_config, f)
    logger.info(f"Saved backup training configuration to {config_path}")

    TRAIN_CONFIG = full_config["TRAIN"]
    TASKS = full_config["TASKS"]

    # =========================================================
    # 4. DATAMODULES
    # =========================================================
    logger.info("Initializing MultiTask DataModule...")
    datamodule_config = {}
    for task_name, task_cfg in TASKS.items():
        datamodule_config[task_name] = task_cfg.copy()
        datamodule_config[task_name]["collate_fn"] = MultiTaskCollate()

    datamodule = MultiTaskDataModule(
        config=datamodule_config, 
        epoch_mode=TRAIN_CONFIG.get("epoch_mode", "fixed_steps"), 
        fixed_steps=TRAIN_CONFIG.get("fixed_steps", 2000)
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    # =========================================================
    # 5. ARCHITECTURE
    # =========================================================
    logger.info("Building Model Architecture...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = build_model(full_config)
    log_model_params(model, logger)

    # =========================================================
    # 6. TRAINING
    # =========================================================
    logger.info("Initializing MultiTask Trainer...")
    trainer = MultiTaskTrainer(
        model=model,
        datamodule=datamodule,
        full_config=full_config,
        device=device,
        save_dir=args.save_dir
    )
    
    logger.info(f"Starting training loop... (Resume: {args.resume})")
    trainer.fit(resume_path=args.resume)


if __name__ == "__main__":
    main()