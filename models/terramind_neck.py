"""
TerraMind Neck


Converts TerraMind transformer tokens


    [B, N, C]


into image feature maps


    [B, C, H, W]


for Mask2Former.


Unlike the previous version, this neck DOES NOT build a feature pyramid.
It simply:


1. Selects transformer layers
2. Reshapes tokens to images
3. Projects channels to a common dimension


The original MSDeformAttn Pixel Decoder is then responsible for
multi-scale feature fusion exactly as in the official Mask2Former.
"""


import math
from typing import Dict, List


import torch
import torch.nn as nn


from .layers import ProjectionBlock




class TerraMindNeck(nn.Module):
    """
    TerraMind -> Mask2Former bridge.
    """


    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        feature_indices=(2, 5, 8, 11),
    ):
        super().__init__()


        self.feature_indices = feature_indices


        #
        # Every selected transformer layer is projected into the
        # same hidden dimension expected by Mask2Former.
        #


        self.projections = nn.ModuleList(
            [
                ProjectionBlock(embed_dim, hidden_dim)
                for _ in feature_indices
            ]
        )


        self.out_channels = [
            hidden_dim
            for _ in feature_indices
        ]


    def _reshape_tokens(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Converts


            [B,N,C]


        →


            [B,C,H,W]
        """


        B, N, C = x.shape


        H = int(math.sqrt(N))
        W = H


        if H * W != N:
            raise ValueError(
                f"Token count {N} is not a square."
            )


        x = (
            x.transpose(1, 2)
             .contiguous()
             .reshape(B, C, H, W)
        )


        return x


    def forward(
        self,
        encoder_outputs: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:


        features = {}


        names = [
            "res2",
            "res3",
            "res4",
            "res5",
        ]


        for name, idx, proj in zip(
            names,
            self.feature_indices,
            self.projections,
        ):


            x = encoder_outputs[idx]


            #
            # B,N,C
            #


            x = self._reshape_tokens(x)


            #
            # B,C,H,W
            #


            x = proj(x)


            features[name] = x


        return features
