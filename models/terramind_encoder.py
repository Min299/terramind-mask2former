"""
Frozen TerraMind backbone wrapper.


Outputs a list of transformer features.


Input:
    [B,C,H,W]


Output:
    List[
        [B,N,C],
        [B,N,C],
        ...
    ]
"""


from typing import List


import torch
from torch import nn


from terratorch.registry import BACKBONE_REGISTRY




class TerraMindEncoder(nn.Module):


    def __init__(
        self,
        backbone_name="terramind_v1_base",
        pretrained=True,
        modalities=("S2L2A",),
        merge_method="mean",
        freeze=True,
    ):


        super().__init__()


        self.encoder = BACKBONE_REGISTRY.build(
            backbone_name,
            pretrained=pretrained,
            modalities=list(modalities),
            merge_method=merge_method,
        )


        self.out_channels = self.encoder.out_channels


        if freeze:


            self.freeze()


    def freeze(self):


        for p in self.encoder.parameters():


            p.requires_grad = False


        self.encoder.eval()


    def train(self, mode=True):


        """
        Override train().


        Frozen encoder always remains in eval mode.
        """


        if any(p.requires_grad for p in self.encoder.parameters()):


            self.encoder.train(mode)


        else:


            self.encoder.eval()


        return super().train(mode)


    def forward(
        self,
        x,
    ) -> List[torch.Tensor]:


        """
        Returns


        [
            layer1,
            layer2,
            ...
            layer12
        ]


        each of shape


            [B,N,C]
        """


        if isinstance(x, dict):


            features = self.encoder(x)


        else:


            features = self.encoder(x)


        return features
