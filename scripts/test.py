"""
test.py

Evaluates an already-trained MultiTask Mask2Former checkpoint.
Reconstructs matchers and criteria strictly from saved configs.
Uses Automatic Mixed Precision for inference to match training.
"""

import os
import yaml
import json
import csv
import argparse
import logging
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from data.multitask_datamodule import MultiTaskDataModule
from data.collate import MultiTaskCollate
from losses.criterion import SetCriterion
from losses.matcher import HungarianMatcher
from engine.metrics import SegmentationMetrics

# Import the centralized utilities
from engine.inference_utils import (
    build_model, 
    load_checkpoint, 
    postprocess_mask2former_outputs,
    semantic_to_mask2former_targets
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test TerraMind Mask2Former")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml (from checkpoint dir)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--save_dir", type=str, default="./results", help="Where to save eval results")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional override for batch size")
    return parser.parse_args()


def load_configuration(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Extract structural configs (TRAIN is ignored for testing)
    return config["MODEL"], config["TASKS"], config.get("MATCHER", {}), config.get("CRITERION", {})


def build_datamodule(tasks_config: dict, batch_size_override: int = None):
    datamodule_config = {}
    for task_name, task_cfg in tasks_config.items():
        datamodule_config[task_name] = task_cfg.copy()
        datamodule_config[task_name]["collate_fn"] = MultiTaskCollate()
        if batch_size_override is not None:
            datamodule_config[task_name]["batch_size"] = batch_size_override

    dm = MultiTaskDataModule(config=datamodule_config, epoch_mode="fixed_steps")
    dm.setup(stage="test")
    return dm


@torch.no_grad()
def evaluate_task(model, loader, criterion, metric_tracker, task_name, device):
    """Core evaluation loop for a single task."""
    model.eval()
    task_loss = 0.0
    steps = 0
    
    use_amp = "cuda" in device
    device_type = "cuda" if use_amp else "cpu"
    
    pbar = tqdm(loader, desc=f"Evaluating [{task_name}]", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        semantic_masks = batch["mask"].to(device)
        
        ignore_idx = metric_tracker.ignore_index
        # Use centralized target conversion
        targets = semantic_to_mask2former_targets(
            semantic_masks, 
            num_classes=metric_tracker.num_classes,
            ignore_index=ignore_idx,
            background_id=None # Optionally extract from config if needed
        )

        # Sanitize & Pad images
        if isinstance(images, torch.Tensor):
            images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=-1.0)
            if images.ndim == 4 and images.shape[1] < 13:
                pad = torch.zeros((images.shape[0], 13 - images.shape[1], images.shape[2], images.shape[3]), device=images.device, dtype=images.dtype)
                images = torch.cat([images, pad], dim=1)
            elif images.ndim == 4 and images.shape[1] > 13:
                images = images[:, :13, :, :]

        # Device-aware AMP Context Manager
        with torch.autocast(device_type=device_type, enabled=use_amp):
            outputs = model(images, task=task_name)
            
            # Clamp Targets
            try:
                nc = getattr(criterion, 'num_classes', 2)
                if isinstance(targets, list):
                    for t in targets:
                        if isinstance(t, dict):
                            for k, v in t.items():
                                if hasattr(v, 'dtype') and v.dtype in (torch.long, torch.int64, torch.int32, torch.int8):
                                    t[k] = torch.where((v < 0) | (v >= nc), torch.zeros_like(v), v)
                elif hasattr(targets, 'dtype') and targets.dtype in (torch.long, torch.int64, torch.int32, torch.int8):
                    targets = torch.where((targets < 0) | (targets >= nc), torch.zeros_like(targets), targets)
            except Exception:
                pass
            
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
        task_loss += total_loss.item()
        steps += 1
        
        target_size = semantic_masks.shape[-2:]
        pred_semantic = postprocess_mask2former_outputs(outputs["pred_logits"], outputs["pred_masks"], target_size)
        metric_tracker.update(pred_semantic, semantic_masks)

    avg_loss = task_loss / steps if steps > 0 else 0.0
    metrics_result = metric_tracker.compute(reset=True, return_confusion=False)
    metrics_result["loss"] = avg_loss
    
    return metrics_result


def evaluate_all_tasks(model, test_loaders, criteria, tasks_config, device):
    results = {}
    for task_name, loader in test_loaders.items():
        metric_tracker = SegmentationMetrics(
            num_classes=tasks_config[task_name]["num_classes"],
            ignore_index=tasks_config[task_name].get("ignore_index", 255)
        )
        results[task_name] = evaluate_task(
            model, loader, criteria[task_name], metric_tracker, task_name, device
        )
    return results


def print_results(results: dict):
    print("\n" + "="*50)
    print(f"{'Task':<15} | {'Loss':<8} | {'mIoU':<8} | {'mDice':<8} | {'mAcc':<8}")
    print("="*50)
    for task, res in results.items():
        print(f"{task:<15} | {res['loss']:.4f}   | {res['miou']:.4f}   | {res['mdice']:.4f}   | {res['mean_accuracy']:.4f}")
    print("="*50 + "\n")


def save_results(results: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "evaluation.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    with open(os.path.join(save_dir, "evaluation.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Loss", "mIoU", "mDice", "Mean_Precision", "Mean_Recall", "Mean_Accuracy", "Pixel_Accuracy"])
        for task, res in results.items():
            writer.writerow([
                task, res["loss"], res["miou"], res["mdice"], 
                res["mean_precision"], res["mean_recall"], 
                res["mean_accuracy"], res["pixel_accuracy"]
            ])


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading configuration...")
    MODEL_CONFIG, TASKS, MATCHER_CONFIG, CRITERION_CONFIG = load_configuration(args.config)

    logger.info("Building Model & Loading Checkpoint...")
    try:
        MODEL_CONFIG['TASKS'] = TASKS
    except Exception:
        pass
    model = build_model(MODEL_CONFIG)
    model = load_checkpoint(model, args.checkpoint, device)

    logger.info("Initializing DataModules...")
    datamodule = build_datamodule(TASKS, args.batch_size)
    test_loaders = datamodule.test_dataloader()

    # ---------------------------------------------------------
    # FIX: Reconstruct the full weight_dict including deep supervision
    # ---------------------------------------------------------
    weight_dict = {"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0}
    dec_layers = MODEL_CONFIG.get("transformer_decoder", {}).get("dec_layers", 9)
    aux_weight_dict = {f"{k}_{i}": v for i in range(dec_layers) for k, v in weight_dict.items()}
    weight_dict.update(aux_weight_dict)
    
    CRITERION_CONFIG["weight_dict"] = weight_dict
    CRITERION_CONFIG["losses"] = CRITERION_CONFIG.get("losses", ["labels", "masks"])
    CRITERION_CONFIG["eos_coef"] = CRITERION_CONFIG.get("eos_coef", 0.1)

    criteria = {}
    for task, cfg in TASKS.items():
        criteria[task] = SetCriterion(
            num_classes=cfg["num_classes"],
            matcher=HungarianMatcher(**MATCHER_CONFIG),
            **CRITERION_CONFIG
        ).to(device)

    logger.info("Starting Evaluation...")
    results = evaluate_all_tasks(model, test_loaders, criteria, TASKS, device)

    print_results(results)
    save_results(results, args.save_dir)
    logger.info(f"Results saved to {args.save_dir}")

if __name__ == "__main__":
    main()