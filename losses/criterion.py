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
    """
    Computes all losses for Mask2Former.

    Workflow

        Hungarian Matching
                ↓
        Classification Loss
                ↓
        BCE Mask Loss
                ↓
        Dice Loss
                ↓
        Auxiliary Decoder Losses
    """

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

        self.register_buffer(
            "empty_weight",
            empty_weight,
        )

        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    # ------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------

    def loss_labels(
        self,
        outputs,
        targets,
        indices,
        num_masks,
    ):

        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat(
            [
                t["labels"][J]
                for t, (_, J)
                in zip(targets, indices)
            ]
        )

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )

        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            self.empty_weight,
        )

        return {
            "loss_ce": loss_ce
        }

    # ------------------------------------------------------------
    # Mask Loss
    # ------------------------------------------------------------

    def loss_masks(
        self,
        outputs,
        targets,
        indices,
        num_masks,
    ):

        src_idx = self._get_src_permutation_idx(indices)

        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"][src_idx]

        #
        # Expected collate output:
        #
        # targets[i]["masks"] -> [K,H,W]
        #

        target_masks = torch.cat(
            [t["masks"] for t in targets],
            dim=0,
        )

        target_masks = target_masks.to(src_masks)

        target_masks = target_masks[tgt_idx]

        src_masks = src_masks[:, None]

        target_masks = target_masks[:, None]

        #
        # Uncertainty-guided point sampling
        #

        with torch.no_grad():

            point_coords = sample_uncertain_points(
                logits=src_masks,
                uncertainty_fn=calculate_uncertainty,
                num_points=self.num_points,
                oversample_ratio=self.oversample_ratio,
                importance_sample_ratio=self.importance_sample_ratio,
            )

            point_labels = sample_points(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = sample_points(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        return {

            "loss_mask":
                sigmoid_ce_loss(
                    point_logits,
                    point_labels,
                    num_masks,
                ),

            "loss_dice":
                dice_loss(
                    point_logits,
                    point_labels,
                    num_masks,
                ),
        }

    # ------------------------------------------------------------

    def _get_src_permutation_idx(
        self,
        indices,
    ):

        batch_idx = torch.cat(
            [
                torch.full_like(src, i)
                for i, (src, _)
                in enumerate(indices)
            ]
        )

        src_idx = torch.cat(
            [
                src
                for src, _
                in indices
            ]
        )

        return batch_idx, src_idx

    # ------------------------------------------------------------

    def _get_tgt_permutation_idx(
        self,
        indices,
    ):

        batch_idx = torch.cat(
            [
                torch.full_like(tgt, i)
                for i, (_, tgt)
                in enumerate(indices)
            ]
        )

        tgt_idx = torch.cat(
            [
                tgt
                for _, tgt
                in indices
            ]
        )

        return batch_idx, tgt_idx

    # ------------------------------------------------------------

    def get_loss(
        self,
        loss,
        outputs,
        targets,
        indices,
        num_masks,
    ):

        loss_map = {

            "labels":
                self.loss_labels,

            "masks":
                self.loss_masks,
        }

        return loss_map[loss](
            outputs,
            targets,
            indices,
            num_masks,
        )

    # ------------------------------------------------------------

    def forward(
        self,
        outputs,
        targets,
    ):

        outputs_without_aux = {

            k: v

            for k, v

            in outputs.items()

            if k != "aux_outputs"
        }

        indices = self.matcher(
            outputs_without_aux,
            targets,
        )

        #
        # No distributed training.
        #

        num_masks = max(
            sum(
                len(t["labels"])
                for t in targets
            ),
            1,
        )

        losses = {}

        for loss in self.losses:

            losses.update(

                self.get_loss(

                    loss,

                    outputs,

                    targets,

                    indices,

                    num_masks,
                )
            )

        #
        # Deep supervision
        #

        if "aux_outputs" in outputs:

            for i, aux_outputs in enumerate(
                outputs["aux_outputs"]
            ):

                indices = self.matcher(
                    aux_outputs,
                    targets,
                )

                for loss in self.losses:

                    aux_loss = self.get_loss(
                        loss,
                        aux_outputs,
                        targets,
                        indices,
                        num_masks,
                    )

                    aux_loss = {
                        k + f"_{i}": v
                        for k, v
                        in aux_loss.items()
                    }

                    losses.update(aux_loss)

        return losses

    def __repr__(self):

        body = [

            f"matcher={self.matcher}",

            f"losses={self.losses}",

            f"weight_dict={self.weight_dict}",

            f"num_classes={self.num_classes}",

            f"eos_coef={self.eos_coef}",

            f"num_points={self.num_points}",

            f"oversample_ratio={self.oversample_ratio}",

            f"importance_sample_ratio={self.importance_sample_ratio}",
        ]

        return (
            self.__class__.__name__
            + "(\n  "
            + ",\n  ".join(body)
            + "\n)"
        )