"""
predict.py

Benchmarking suite for TerraMind semantic segmentation models on Kaggle.
"""
from __future__ import annotations

import os
import glob
import random
import gc
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tifffile
from torch.utils.data import Dataset, DataLoader, Subset

from visualization import visualize_prediction
from model_registry import MODEL_REGISTRY

# ============================================================
# Palettes and Class Maps 
# ============================================================
TASK_PALETTES = {
    "flood": {
        "class_names": {0: "Background / No Water", 1: "Water", 255: "Cloud / No Data"},
        "palette": {0: (0, 0, 0), 1: (0, 150, 255), 255: (128, 128, 128)}
    },
    "burn": {
        "class_names": {0: "Background / Unburned", 1: "Burn Scar", 255: "Unlabeled / No Data"},
        "palette": {0: (0, 0, 0), 1: (255, 50, 50), 255: (128, 128, 128)}
    },
    "landcover": {
        "class_names": {
            0: "No Data", 1: "Water", 2: "Tree Canopy", 
            3: "Low Vegetation", 4: "Barren", 5: "Impervious", 6: "Other", 255: "Unlabeled"
        },
        "palette": {
            0: (0, 0, 0), 1: (0, 0, 255), 2: (0, 128, 0), 
            3: (144, 238, 144), 4: (210, 180, 140), 5: (105, 105, 105), 
            6: (128, 128, 128), 255: (128, 128, 128)
        }
    }
}

# ============================================================
# Configuration
# ============================================================
@dataclass
class InferenceConfig:
    model: str = "terramind_tiny"
    task: str = "flood" 
    
    dataset_root: str = "/kaggle/working/data"
    output_dir: str = "/kaggle/working/outputs" 
    
    num_classes: int = 2
    samples: int = 10
    batch_size: int = 2
    shuffle: bool = True
    seed: int = 42
    workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    image_dir: str = ""
    mask_dir: str = ""
    image_suffix: str = ""
    mask_suffix: str = ""

    def __post_init__(self):
        root = Path(self.dataset_root)
        if self.task == "flood":
            base = root / "sen1floods11" / "v1.1" / "data" / "flood_events" / "HandLabeled"
            self.image_dir, self.mask_dir = str(base / "S2Hand"), str(base / "LabelHand")
            self.image_suffix, self.mask_suffix = "_S2Hand", "_LabelHand"
            self.num_classes = 2
        elif self.task == "burn":
            base = root / "fire_scars" 
            if (base / "validation").exists(): base = base / "validation"
            
            # Robust fallback for images/masks folders
            self.image_dir = str(base / "images") if (base / "images").exists() else str(base)
            self.mask_dir = str(base / "masks") if (base / "masks").exists() else str(base)
            self.image_suffix, self.mask_suffix = "", ""
            self.num_classes = 2
        elif self.task == "landcover":
            base = root / "chesapeake" / "segmentation_v1.0" 
            self.image_dir, self.mask_dir = str(base / "images"), str(base / "labels")
            self.image_suffix, self.mask_suffix = "", ""
            self.num_classes = 7

# ============================================================
# Smart Data Loader
# ============================================================
class GenericEODataset(Dataset):
    def __init__(self, cfg: InferenceConfig):
        valid_exts = ('*.tif', '*.tiff', '*.png')
        self.image_paths, self.mask_paths = [], []
        
        for ext in valid_exts:
            self.image_paths.extend(glob.glob(os.path.join(cfg.image_dir, ext)))
            self.mask_paths.extend(glob.glob(os.path.join(cfg.mask_dir, ext)))
            
        mask_dict = {}
        for p in self.mask_paths:
            stem = Path(p).stem
            base_id = stem[:-len(cfg.mask_suffix)] if cfg.mask_suffix and stem.endswith(cfg.mask_suffix) else stem
            mask_dict[base_id] = p
            
        self.samples = []
        for img_p in sorted(self.image_paths):
            stem = Path(img_p).stem
            base_id = stem[:-len(cfg.image_suffix)] if cfg.image_suffix and stem.endswith(cfg.image_suffix) else stem
            if base_id in mask_dict: self.samples.append((img_p, mask_dict[base_id]))

        if not self.samples: print(f"\n[WARNING] No matching pairs found in {cfg.image_dir}!")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_p, mask_p = self.samples[idx]
        img = tifffile.imread(img_p)
        mask = np.squeeze(tifffile.imread(mask_p))
            
        if img.ndim == 2: img = np.expand_dims(img, axis=0)
        elif img.ndim == 3 and img.shape[-1] <= 12: img = np.transpose(img, (2, 0, 1))
            
        return {
            "image": torch.from_numpy(img.astype(np.float32)), "mask": torch.from_numpy(mask.astype(np.int64)),
            "filename": Path(img_p).name, "path": img_p
        }

def build_loader(cfg):
    dataset = GenericEODataset(cfg)
    random.seed(cfg.seed)
    samples = min(cfg.samples, len(dataset))
    indices = random.sample(range(len(dataset)), samples) if cfg.shuffle else list(range(samples))
    return DataLoader(Subset(dataset, indices), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers, pin_memory=torch.cuda.is_available())

# ============================================================
# Metrics
# ============================================================
def calculate_metrics(prediction, target, num_classes=2, ignore_index=255):
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    pred_valid, target_valid = prediction[valid], target[valid]
    if len(target_valid) == 0: return {"mIoU": 0.0, "Accuracy": 0.0}
        
    intersection = np.bincount(target_valid[pred_valid == target_valid], minlength=num_classes)
    target_count = np.bincount(target_valid, minlength=num_classes)
    pred_count = np.bincount(pred_valid, minlength=num_classes)
    
    iou = intersection / (target_count + pred_count - intersection + 1e-10)
    acc = np.sum(intersection) / (np.sum(target_count) + 1e-10)
    
    valid_classes = (target_count + pred_count) > 0
    return {"mIoU": np.mean(iou[valid_classes]) if np.any(valid_classes) else 0.0, "Accuracy": acc}

# ============================================================
# Inference
# ============================================================
@torch.inference_mode()
def inference(model, model_cfg, spec, loader, cfg, output_dir):
    model.eval()
    predictor = spec["predict"]
    all_results = []

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(cfg.device, non_blocking=True)
        masks, filenames, paths = batch["mask"], batch["filename"], batch["path"]

        prediction = predictor(model=model, images=images, task=cfg.task, config=model_cfg)
        batch_metrics = {"mIoU": [], "Accuracy": []}

        for i in range(len(prediction)):
            pred_np, mask_np = prediction[i].cpu().numpy(), masks[i].cpu().numpy()
            metrics = calculate_metrics(pred_np, mask_np, num_classes=cfg.num_classes)
            batch_metrics["mIoU"].append(metrics["mIoU"])
            batch_metrics["Accuracy"].append(metrics["Accuracy"])
            
            all_results.append({"Filename": filenames[i], "mIoU": round(metrics["mIoU"], 4), "Accuracy": round(metrics["Accuracy"], 4)})
            
            raw_img = tifffile.imread(paths[i])
            if raw_img.ndim == 3:
                if raw_img.shape[0] < raw_img.shape[-1]: raw_img = np.transpose(raw_img, (1, 2, 0))
                rgb_idx = [3, 2, 1] if raw_img.shape[-1] >= 4 else [0, 1, 2]
                rgb_img = raw_img[..., rgb_idx].astype(np.float32)
                for b in range(rgb_img.shape[-1]):
                    p2, p98 = np.percentile(rgb_img[..., b], (2, 98))
                    rgb_img[..., b] = np.clip((rgb_img[..., b] - p2) / (p98 - p2 + 1e-8), 0, 1)
            else:
                rgb_img = raw_img.astype(np.float32)
                p2, p98 = np.percentile(rgb_img, (2, 98))
                rgb_img = np.clip((rgb_img - p2) / (p98 - p2 + 1e-8), 0, 1)

            save_path = output_dir / f"{cfg.model}_{cfg.task}_{Path(filenames[i]).stem}.png"
            visualize_prediction(
                image=rgb_img, prediction=pred_np, ground_truth=mask_np, metrics=metrics, 
                palette=TASK_PALETTES[cfg.task]["palette"], class_names=TASK_PALETTES[cfg.task]["class_names"],
                save_path=str(save_path), image_name=filenames[i], model_name=cfg.model, task_name=cfg.task
            )
            print(f"  -> Processed {filenames[i]:<30} | mIoU: {metrics['mIoU']:.4f} | Acc: {metrics['Accuracy']:.4f}")
            
        print(f"\n[Batch {batch_idx+1}/{len(loader)}] Avg mIoU: {np.mean(batch_metrics['mIoU']):.4f} | Avg Acc: {np.mean(batch_metrics['Accuracy']):.4f}\n")

    if all_results:
        csv_path = output_dir / f"benchmark_metrics_{cfg.model}_{cfg.task}.csv"
        avg_iou = np.mean([r["mIoU"] for r in all_results])
        avg_acc = np.mean([r["Accuracy"] for r in all_results])
        all_results.append({"Filename": "", "mIoU": "", "Accuracy": ""})
        all_results.append({"Filename": "OVERALL AVERAGE", "mIoU": round(avg_iou, 4), "Accuracy": round(avg_acc, 4)})

        with open(csv_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Filename", "mIoU", "Accuracy"])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"[*] Report saved to: {csv_path}\n")

# ============================================================
# Main - TerraMind Benchmarking
# ============================================================
def main():
    # Evaluate BOTH models across ALL 3 TASKS
    BENCHMARK_QUEUE = [
        ("terramind_tiny", "flood"),
        ("terramind_tiny", "burn"),
        ("terramind_tiny", "landcover"),
        ("terramind_base", "flood"),
        ("terramind_base", "burn"),
        ("terramind_base", "landcover"),
    ]

    print("=" * 70)
    print("🚀 TerraMind Comprehensive Benchmark Suite (Kaggle)")
    print("=" * 70)

    out_dir = Path("/kaggle/working/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    for step, (model_name, task_name) in enumerate(BENCHMARK_QUEUE, 1):
        print(f"\n[{step}/{len(BENCHMARK_QUEUE)}] Executing: {model_name} on {task_name.upper()}")
        print("-" * 70)
        
        cfg = InferenceConfig(model=model_name, task=task_name)
        spec = MODEL_REGISTRY[model_name]
        
        try:
            model, model_cfg = spec["loader"](spec)
            loader = build_loader(cfg)
            inference(model, model_cfg, spec, loader, cfg, out_dir)
        except Exception as e:
            print(f"[FAILED] Error running {model_name} on {task_name}: {e}")
        finally:
            if 'model' in locals(): del model
            torch.cuda.empty_cache()
            gc.collect()

    print("=" * 70)
    print(f"✅ Full Benchmarking Suite Complete. All outputs saved to {out_dir.absolute()}")

if __name__ == "__main__":
    main()
