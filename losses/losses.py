"""
losses.py

Shared loss utilities for TerraMind Mask2Former.

This module centralizes all loss functions and point sampling utilities
used during Hungarian matching and criterion computation.

The mathematical implementations follow the official Mask2Former paper,
while Detectron2 is used only for PointRend's uncertainty-guided point
sampling.

References
----------
Mask2Former:
    https://github.com/facebookresearch/Mask2Former

Detectron2 PointRend:
    https://github.com/facebookresearch/detectron2
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from detectron2.projects.point_rend.point_features import (
    point_sample as d2_point_sample,
    get_uncertain_point_coords_with_randomness,
)


# ---------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------

def dice_loss(
    inputs: Tensor,
    targets: Tensor,
    num_masks: float,
) -> Tensor:
    """
    Official Mask2Former Dice Loss.

    Parameters
    ----------
    inputs
        Predicted mask logits
        Shape:
            [N, P]

    targets
        Binary ground-truth masks
        Shape:
            [N, P]

    num_masks
        Number of matched masks used for normalization.

    Returns
    -------
    Scalar Dice loss.
    """

    inputs = inputs.sigmoid()

    numerator = 2 * (inputs * targets).sum(-1)

    denominator = (
        inputs.sum(-1)
        + targets.sum(-1)
    )

    loss = 1 - (
        numerator + 1
    ) / (
        denominator + 1
    )

    return loss.sum() / num_masks


# ---------------------------------------------------------------------
# BCE Loss
# ---------------------------------------------------------------------

def sigmoid_ce_loss(
    inputs: Tensor,
    targets: Tensor,
    num_masks: float,
) -> Tensor:
    """
    Official sigmoid BCE loss used by Mask2Former.
    """

    loss = F.binary_cross_entropy_with_logits(
        inputs,
        targets,
        reduction="none",
    )

    return loss.mean(1).sum() / num_masks


# ---------------------------------------------------------------------
# Pairwise Dice Cost
# ---------------------------------------------------------------------

def batch_dice_cost(
    inputs: Tensor,
    targets: Tensor,
) -> Tensor:
    """
    Pairwise Dice cost used during Hungarian matching.

    Parameters
    ----------
    inputs
        Shape:
            [num_queries, num_points]

    targets
        Shape:
            [num_targets, num_points]

    Returns
    -------
    Tensor
        Shape:
            [num_queries, num_targets]
    """

    inputs = inputs.sigmoid()

    numerator = 2 * torch.einsum(
        "nc,mc->nm",
        inputs,
        targets,
    )

    denominator = (
        inputs.sum(-1)[:, None]
        + targets.sum(-1)[None]
    )

    return 1 - (
        numerator + 1
    ) / (
        denominator + 1
    )


# ---------------------------------------------------------------------
# Pairwise BCE Cost
# ---------------------------------------------------------------------

def batch_sigmoid_ce_cost(
    inputs: Tensor,
    targets: Tensor,
) -> Tensor:
    """
    Pairwise sigmoid BCE cost used during Hungarian matching.
    
    Optimized implementation using einsum to avoid massive memory 
    allocations from broadcasting [queries, targets, points].

    Returns
    -------
    Tensor
        Shape:
            [num_queries, num_targets]
    """

    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs,
        torch.ones_like(inputs),
        reduction="none",
    )

    neg = F.binary_cross_entropy_with_logits(
        inputs,
        torch.zeros_like(inputs),
        reduction="none",
    )

    loss = (
        torch.einsum("nc,mc->nm", pos, targets)
        +
        torch.einsum("nc,mc->nm", neg, 1 - targets)
    )

    return loss / hw


# ---------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------

def calculate_uncertainty(
    logits: Tensor,
) -> Tensor:
    """
    Official uncertainty function from Mask2Former.

    Pixels whose logits are close to zero are considered
    the most uncertain.

    Parameters
    ----------
    logits
        Shape:
            [N, 1, H, W]

    Returns
    -------
    Tensor
        Uncertainty map with identical spatial dimensions.
    """

    if logits.shape[1] != 1:
        raise ValueError(
            f"calculate_uncertainty expects logits with shape [N,1,H,W], "
            f"got {tuple(logits.shape)}."
        )

    return -torch.abs(logits)


# ---------------------------------------------------------------------
# Detectron2 Wrappers
# ---------------------------------------------------------------------

@torch.jit.ignore
def sample_points(
    features: Tensor,
    point_coords: Tensor,
    align_corners: bool = False,
) -> Tensor:
    """
    Wrapper around Detectron2 PointRend point_sample().

    Keeps Detectron2 isolated from the rest of the codebase.
    """

    return d2_point_sample(
        features,
        point_coords,
        align_corners=align_corners,
    )


@torch.jit.ignore
def sample_uncertain_points(
    logits: Tensor,
    uncertainty_fn: Callable[[Tensor], Tensor],
    num_points: int,
    oversample_ratio: float = 3.0,
    importance_sample_ratio: float = 0.75,
) -> Tensor:
    """
    Wrapper around Detectron2's uncertainty-guided point sampling.
    Uses official Mask2Former defaults for oversample and importance ratios.

    Returns normalized point coordinates in [0,1]^2.
    """

    return get_uncertain_point_coords_with_randomness(
        logits,
        uncertainty_fn,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
    )


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

__all__ = [
    "dice_loss",
    "sigmoid_ce_loss",
    "batch_dice_cost",
    "batch_sigmoid_ce_cost",
    "calculate_uncertainty",
    "sample_points",
    "sample_uncertain_points",
]