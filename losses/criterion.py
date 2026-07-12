"""
criterion.py

Mask2Former SetCriterion adapted for the TerraMind multitask
semantic segmentation pipeline.

Compared to the official implementation:
- Uses local losses.py
- Removes distributed training utilities
- Keeps auxiliary decoder supervision
- Keeps PointRend uncertainty sampling
- Keeps Hungarian matching unchanged
- ENFORCES strict tensor contracts and weight validation.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import (
    dice_loss,
    sigmoid_ce_loss,
    calculate_uncertainty,
    sample_points,
    sample_uncertain_points,
)


class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: Dict[str, float],
        eos_coef: float,
        losses: List[str],
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.eos_coef = eos_coef

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        src_logits = outputs["pred_logits"].float()
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {"loss_ce": loss_ce}

    def loss_masks(self, outputs, targets, indices, num_masks):
        src_idx = self._get_src_permutation_idx(indices)

        src_masks = outputs["pred_masks"][src_idx]
        # Select each target's matched masks (by J) per-target BEFORE concatenating,
        # so the batch dimension is never collapsed. Do NOT index the already-flattened
        # concatenation with (batch_idx, tgt_idx) -- that indexes two dimensions at
        # once and silently corrupts the mask's spatial shape.
        target_masks = torch.cat([t["masks"][J] for t, (_, J) in zip(targets, indices)], dim=0).to(src_masks)

        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            point_coords = sample_uncertain_points(
                logits=src_masks,
                uncertainty_fn=calculate_uncertainty,
                num_points=self.num_points,
                oversample_ratio=self.oversample_ratio,
                importance_sample_ratio=self.importance_sample_ratio,
            )
            point_labels = sample_points(target_masks, point_coords, align_corners=False).squeeze(1)

        point_logits = sample_points(src_masks, point_coords, align_corners=False).squeeze(1)

        return {
            "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

    def _get_src_permutation_idx(self, indices):
        indices = [(s.cuda(), t.cuda()) for s, t in indices]
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for _, tgt in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        # ---------------------------------------------------------
        # Strict Tensor Contract Validation
        # ---------------------------------------------------------
        if len(targets) != outputs["pred_logits"].shape[0]:
            raise ValueError(f"Batch size mismatch: {len(targets)} targets vs {outputs['pred_logits'].shape[0]} predictions.")
            
        for i, t in enumerate(targets):
            if t["labels"].dtype != torch.long:
                raise TypeError(f"Target {i} labels must be torch.long, got {t['labels'].dtype}")
            if t["masks"].dtype != torch.float32:
                raise TypeError(f"Target {i} masks must be torch.float32, got {t['masks'].dtype}")

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)

        num_masks = max(sum(len(t["labels"]) for t in targets), 1)

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    aux_loss = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    aux_loss = {k + f"_{i}": v for k, v in aux_loss.items()}
                    losses.update(aux_loss)

        # ---------------------------------------------------------
        # Loss Weight Validation
        # ---------------------------------------------------------
        missing_weights = set(losses.keys()) - set(self.weight_dict.keys())
        if missing_weights:
            raise ValueError(f"The following computed losses are missing from weight_dict: {missing_weights}")

        return losses