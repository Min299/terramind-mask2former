from typing import List

import torch
from torch import nn

from terratorch.registry import BACKBONE_REGISTRY


class TerraMindEncoder(nn.Module):

    def __init__(
        self,
        backbone_name="terramind_v1_base",
        pretrained=True,
        modalities=("S2L1C","S2L2A","S1GRD","S1RTC","DEM","RGB","Coordinates"),
        freeze=True,
    ):
        super().__init__()

        self.encoder = BACKBONE_REGISTRY.build(
            backbone_name,
            pretrained=pretrained,
            modalities=list(modalities),
        )

        self.out_channels = self.encoder.out_channels

        if freeze:
            self.freeze()

    def freeze(self):

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder.eval()

    def forward(
        self,
        x,
    ) -> List[torch.Tensor]:

        return self.encoder(x)