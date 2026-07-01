"""
losses/point_features.py


Minimal PointRend utilities used by Mask2Former.
"""


from __future__ import annotations


import torch
import torch.nn.functional as F




def point_sample(
    input: torch.Tensor,
    point_coords: torch.Tensor,
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Args:
        input:
            Tensor[B,C,H,W]


        point_coords:
            Tensor[B,P,2] in [0,1]


    Returns
    -------
        Tensor[B,C,P]
    """


    if point_coords.dim() == 3:
        point_coords = point_coords.unsqueeze(2)


    output = F.grid_sample(
        input,
        2.0 * point_coords - 1.0,
        mode="bilinear",
        align_corners=align_corners,
    )


    return output.squeeze(3)




@torch.no_grad()
def get_uncertain_point_coords_with_randomness(
    coarse_logits: torch.Tensor,
    uncertainty_func,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
):
    """
    Uncertainty-guided point sampling from PointRend /
    Mask2Former.


    Returns:
        Tensor[B,num_points,2]
    """


    assert oversample_ratio >= 1.0
    assert 0.0 <= importance_sample_ratio <= 1.0


    batch_size = coarse_logits.shape[0]


    num_sampled = int(num_points * oversample_ratio)


    point_coords = torch.rand(
        batch_size,
        num_sampled,
        2,
        device=coarse_logits.device,
    )


    point_logits = point_sample(
        coarse_logits,
        point_coords,
        align_corners=False,
    )


    uncertainties = uncertainty_func(point_logits)


    num_uncertain = int(
        importance_sample_ratio * num_points
    )


    num_random = num_points - num_uncertain


    idx = torch.topk(
        uncertainties[:, 0],
        k=num_uncertain,
        dim=1,
    )[1]


    shift = (
        torch.arange(
            batch_size,
            device=coarse_logits.device,
        )
        * num_sampled
    )


    idx = idx + shift[:, None]


    point_coords = point_coords.reshape(
        batch_size * num_sampled,
        2,
    )


    point_coords = point_coords[idx.reshape(-1)]


    point_coords = point_coords.reshape(
        batch_size,
        num_uncertain,
        2,
    )


    if num_random > 0:


        random_coords = torch.rand(
            batch_size,
            num_random,
            2,
            device=coarse_logits.device,
        )


        point_coords = torch.cat(
            [
                point_coords,
                random_coords,
            ],
            dim=1,
        )


    return point_coords




def calculate_uncertainty(
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Mask2Former uncertainty.


    Highest uncertainty corresponds to logits
    closest to zero.
    """


    return -torch.abs(logits)
