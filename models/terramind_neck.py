"""
TerraMind Neck

Adapts TerraMind transformer features to the multi-scale feature pyramid
expected by the original Mask2Former MSDeformAttn Pixel Decoder.

Pipeline
--------
1. Dynamically select four evenly spaced transformer layers.
2. Convert (B,N,C) -> (B,C,H,W).
3. Project every feature to a common hidden dimension.
4. Generate a spatial feature pyramid

    res5 : 1/32
    res4 : 1/16
    res3 : 1/8
    res2 : 1/4

The pixel decoder is then responsible for the original FPN fusion.
"""

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ProjectionBlock


class TerraMindNeck(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        num_feature_levels: int = 4,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels

        self.projections = nn.ModuleList(
            [
                ProjectionBlock(
                    embed_dim,
                    hidden_dim,
                )
                for _ in range(num_feature_levels)
            ]
        )

        #
        # Spatial pyramid generators
        #

        self.downsample = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU(),
        )

        self.refine_res3 = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU(),
        )

        self.refine_res2 = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU(),
        )

        self.out_channels = [
            hidden_dim,
            hidden_dim,
            hidden_dim,
            hidden_dim,
        ]
    @staticmethod
    def _reshape_tokens(tokens: torch.Tensor) -> torch.Tensor:
        B, N, C = tokens.shape
        H = int(math.sqrt(N))
        if H * H != N:
            raise ValueError(f"Neck expected square token grid, but H*H ({H*H}) != N ({N})")
        return tokens.transpose(1, 2).contiguous().reshape(B, C, H, H)

    def _select_layers(
        self,
        encoder_outputs: List[torch.Tensor],
    ) -> List[torch.Tensor]:

        num_layers = len(encoder_outputs)

        if num_layers < self.num_feature_levels:
            raise ValueError(
                f"Expected at least {self.num_feature_levels} encoder layers."
            )

        #
        # Use the deeper 75% of the network.
        #

        start = max(1, num_layers // 4)

        indices = [
            round(
                start
                + i * (num_layers - 1 - start)
                / (self.num_feature_levels - 1)
            )
            for i in range(self.num_feature_levels)
        ]

        #
        # deepest -> shallowest
        #

        selected = [
            encoder_outputs[i]
            for i in indices[::-1]
        ]

        return selected

    def forward(
        self,
        encoder_outputs: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        selected = self._select_layers(
            encoder_outputs
        )

        projected = []

        for feature, projection in zip(
            selected,
            self.projections,
        ):

            feature = self._reshape_tokens(feature)

            feature = projection(feature)

            projected.append(feature)

        #
        # projected
        #
        # 0 -> deepest
        # 1
        # 2
        # 3 -> shallowest
        #

        res4 = projected[1]

        #
        # res5
        #
        res5 = self.downsample(
            projected[0]
        )

        #
        # res3
        #
        res3 = F.interpolate(
            projected[2],
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        res3 = self.refine_res3(
            res3
        )

        #
        # res2
        #
        res2 = F.interpolate(
            projected[3],
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )

        res2 = self.refine_res2(
            res2
        )

        return {
            "res2": res2,
            "res3": res3,
            "res4": res4,
            "res5": res5,
        }