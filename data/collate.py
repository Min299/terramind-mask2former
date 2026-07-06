from typing import Any, Dict, List
import torch

class MultiTaskCollate:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch: return {}

        first_shape = batch[0]["image"].shape
        
        # A3 / C1: Strict validation with helpful error messages
        for i, item in enumerate(batch):
            img, mask = item["image"], item["mask"]
            
            if img.dtype != torch.float32:
                raise TypeError(f"Batch[{i}] image dtype {img.dtype} != torch.float32")
            if mask.dtype != torch.long:
                raise TypeError(f"Batch[{i}] mask dtype {mask.dtype} != torch.long")
            if img.ndim != 3:
                raise ValueError(f"Batch[{i}] image ndim {img.ndim} != 3")
            if mask.ndim not in [2, 3]:
                raise ValueError(f"Batch[{i}] mask ndim {mask.ndim} not in [2, 3]")
            if img.shape != first_shape:
                raise ValueError(f"Batch[{i}] image shape {img.shape} != {first_shape}")

        images = torch.stack([item["image"] for item in batch], dim=0)
        
        masks = []
        for item in batch:
            m = item["mask"]
            if m.ndim == 3:
                if m.shape[0] != 1:
                    raise ValueError(f"Expected 1 channel mask, got {m.shape[0]}")
                m = m.squeeze(0)
            masks.append(m)
        semantic_masks = torch.stack(masks, dim=0)
        
        metadata = None
        if "metadata" in batch[0] and isinstance(batch[0]["metadata"], dict):
            expected_keys = set(batch[0]["metadata"].keys())
            for item in batch:
                if set(item.get("metadata", {}).keys()) != expected_keys:
                    raise KeyError(f"Metadata key mismatch. Expected {expected_keys}")
                    
            metadata = {}
            for key in expected_keys:
                val = batch[0]["metadata"][key]
                if torch.is_tensor(val):
                    metadata[key] = torch.stack([item["metadata"][key] for item in batch], dim=0)
                else:
                    metadata[key] = [item["metadata"][key] for item in batch]
                
        return {"image": images, "mask": semantic_masks, "metadata": metadata}