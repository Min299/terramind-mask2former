import torch
import torch.nn as nn

class ConvNormAct(nn.Module):
    """
    Conv -> GroupNorm -> GELU


    This block is repeatedly used throughout the neck.
    """


    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        groups=32,
    ):
        super().__init__()


        if padding is None:
            padding = kernel_size // 2


        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )


    def forward(self, x):
        return self.block(x)




class ProjectionBlock(nn.Module):
    """
    1×1 → 3×3 → 1×1


    Used to project TerraMind embeddings before pyramid generation.
    """


    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()


        self.layers = nn.Sequential(
            ConvNormAct(
                in_channels,
                out_channels,
                1,
            ),
            ConvNormAct(
                out_channels,
                out_channels,
                3,
            ),
            ConvNormAct(
                out_channels,
                out_channels,
                1,
            ),
        )


    def forward(self, x):
        return self.layers(x)




class UpsampleBlock(nn.Module):
    """
    Bilinear + 3×3 refinement
    """


    def __init__(
        self,
        channels,
    ):
        super().__init__()


        self.conv = ConvNormAct(
            channels,
            channels,
            3,
        )


    def forward(self, x):


        x = nn.functional.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )


        return self.conv(x)




class DownsampleBlock(nn.Module):
    """
    3×3 stride-2
    """


    def __init__(self, channels):


        super().__init__()


        self.block = ConvNormAct(
            channels,
            channels,
            kernel_size=3,
            stride=2,
        )


    def forward(self, x):


        return self.block(x)
