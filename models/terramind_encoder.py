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

        channels = self.encoder.out_channels

        # TerraTorch may expose out_channels as a list/tuple
        if isinstance(channels, (list, tuple)):
            if len(channels) == 0:
                raise RuntimeError("Encoder out_channels is empty.")

            # TerraMind should have the same embedding dimension at every extracted layer
            if len(set(channels)) != 1:
                raise RuntimeError(
                    f"Inconsistent encoder embedding dimensions: {channels}"
                )

            channels = channels[0]

        elif not isinstance(channels, int):
            raise TypeError(
                f"Expected encoder.out_channels to be int or list[int], got {type(channels)}"
            )

        if channels <= 0:
            raise ValueError(f"Invalid encoder out_channels: {channels}")

        self.out_channels = channels

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