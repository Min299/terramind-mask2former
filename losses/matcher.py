"""
matcher.py

Hungarian Matcher used by Mask2Former.

Computes the optimal bipartite assignment between predicted queries
and ground-truth masks.

Matching cost:

    cost =
        cost_class +
        cost_mask +
        cost_dice

The implementation closely follows the official Mask2Former matcher,
adapted for the TerraMind multitask segmentation pipeline.
"""

from typing import Dict, List, Tuple

import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

from losses import (
    batch_dice_cost,
    batch_sigmoid_ce_cost,
    sample_points,
)


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for Mask2Former.

    For every image independently, computes the minimum-cost
    assignment between decoder queries and ground-truth masks.
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        num_points: int = 12544,
    ):
        super().__init__()

        # Replaced assert with safe ValueError for runtime robustness
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("All matching costs cannot be zero.")

        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs Hungarian matching independently
        for every image in the batch.
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []

        for b in range(bs):
            tgt_labels = targets[b]["labels"]
            
            # Short-circuit if there are no ground-truth targets in this image.
            # Prevents CUDA errors from pushing empty tensors into einsum/PointRend.
            if len(tgt_labels) == 0:
                indices.append((
                    torch.empty(0, dtype=torch.int64, device=outputs["pred_logits"].device), 
                    torch.empty(0, dtype=torch.int64, device=outputs["pred_logits"].device)
                ))
                continue

            # 1. Classification Cost
            out_prob = outputs["pred_logits"][b].softmax(-1)
            cost_class = -out_prob[:, tgt_labels]

            # 2. Mask Cost Pre-processing
            pred_masks = outputs["pred_masks"][b]
            gt_masks = targets[b]["masks"].to(pred_masks)

            pred_masks = pred_masks[:, None]
            gt_masks = gt_masks[:, None]

            # PointRend sampling to reduce memory footprint from O(H*W) to O(num_points)
            point_coords = torch.rand(1, self.num_points, 2, device=pred_masks.device)

            sampled_gt = sample_points(
                gt_masks, 
                point_coords.repeat(gt_masks.shape[0], 1, 1), 
                align_corners=False
            ).squeeze(1)
            
            sampled_pred = sample_points(
                pred_masks, 
                point_coords.repeat(pred_masks.shape[0], 1, 1), 
                align_corners=False
            ).squeeze(1)

            # 3. Pairwise Costs 
            # Explicitly disable autocast for numerical stability in the cost matrix
            device_type = "cuda" if pred_masks.is_cuda else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):
                sampled_pred = sampled_pred.float()
                sampled_gt = sampled_gt.float()

                cost_mask = batch_sigmoid_ce_cost(sampled_pred, sampled_gt)
                cost_dice = batch_dice_cost(sampled_pred, sampled_gt)

            # 4. Final Cost Matrix & Hungarian Assignment
            cost = (
                self.cost_class * cost_class
                + self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
            )

            # Move to CPU because scipy linear_sum_assignment does not support CUDA tensors
            cost = cost.reshape(num_queries, -1).cpu()
            pred_ind, tgt_ind = linear_sum_assignment(cost)

            indices.append((
                torch.as_tensor(pred_ind, dtype=torch.int64),
                torch.as_tensor(tgt_ind, dtype=torch.int64),
            ))

        return indices

    @torch.no_grad()
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: List[Dict[str, torch.Tensor]]
    ):
        """
        Returns
        -------
        List[
            (
                prediction_indices,
                target_indices
            )
        ]
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  cost_class={self.cost_class},\n"
            f"  cost_mask={self.cost_mask},\n"
            f"  cost_dice={self.cost_dice},\n"
            f"  num_points={self.num_points}\n"
            f")"
        )