"""
Collate function for Mask2Former training.


Converts list of samples into batches with proper target format.
"""

import torch


def multitask_collate_fn(batch):
    """
    Collate function for multi-task Mask2Former training.
    
    Each sample in batch is a dict:
    {
        "image": Tensor[C, H, W],
        "mask": Tensor[H, W],
        "task": str,
        "image_id": str,
    }
    
    Returns:
    {
        "image": Tensor[B, C, H, W],
        "mask": Tensor[B, H, W],
        "task": List[str],
        "image_id": List[str],
        "targets": List[Dict]  # Mask2Former target format
    }
    """
    images = []
    masks = []
    tasks = []
    image_ids = []
    targets = []
    
    for sample in batch:
        images.append(sample["image"])
        masks.append(sample["mask"])
        tasks.append(sample["task"])
        image_ids.append(sample["image_id"])
        
        # Convert semantic mask to Mask2Former target format
        mask = sample["mask"]
        
        # Handle tensor vs numpy
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        mask = torch.from_numpy(mask).long().to("cpu")
        
        # Get unique classes (excluding ignore_index=255)
        classes = torch.unique(mask)
        classes = classes[classes != 255]
        
        gt_masks = []
        gt_classes = []
        
        for cls in classes:
            gt_masks.append(mask == cls)
            gt_classes.append(cls)
        
        if len(gt_masks) == 0:
            # Empty mask - create dummy
            gt_masks = torch.zeros(
                (0, mask.shape[0], mask.shape[1]),
                dtype=torch.bool,
            )
            gt_classes = torch.zeros((0,), dtype=torch.long)
        else:
            gt_masks = torch.stack(gt_masks)
            gt_classes = torch.stack(gt_classes)
        
        targets.append({
            "labels": gt_classes,
            "masks": gt_masks.float(),
        })
    
    return {
        "image": torch.stack(images),
        "mask": torch.stack(masks),
        "task": tasks,
        "image_id": image_ids,
        "targets": targets,
    }
