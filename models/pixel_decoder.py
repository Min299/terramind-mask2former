"""
MSDeformAttn Pixel Decoder.

This decoder takes multi-scale features from the TerraMind neck and produces
the mask_features and multi-scale features needed by the transformer decoder.
"""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ms_deform_attn import MSDeformAttnTransformerEncoderOnly
from .position_encoding import PositionEmbeddingSine
from .layers import ConvNormAct


class MSDeformAttnPixelDecoder(nn.Module):
    """
    MSDeformAttn-based Pixel Decoder.
    
    Takes multi-scale feature maps and produces:
    - mask_features: Final high-resolution features for mask prediction
    - multi_scale_features: Features at multiple scales for transformer decoder
    """

    def __init__(
        self,
        in_channels: List[int],
        transformer_dropout: float = 0.0,
        transformer_nheads: int = 8,
        transformer_dim_feedforward: int = 1024,
        transformer_enc_layers: int = 6,
        conv_dim: int = 256,
        mask_dim: int = 256,
        num_feature_levels: int = 4,
        common_stride: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.conv_dim = conv_dim
        self.mask_dim = mask_dim
        self.num_feature_levels = num_feature_levels
        self.common_stride = common_stride

        # Input projections for each feature level
        # From low resolution to high resolution
        input_proj_list = []
        for in_ch in in_channels[::-1]:
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_ch, conv_dim, kernel_size=1),
                nn.GroupNorm(32, conv_dim),
            ))
        self.input_proj = nn.ModuleList(input_proj_list)

        # Initialize projections
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # MSDeformAttn Transformer Encoder
        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=num_feature_levels,
        )

        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # Mask feature projection
        self.mask_features = nn.Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # FPN for multi-scale features
        self.maskformer_num_feature_levels = 3
        self._build_fpn()

    def _build_fpn(self):
        """Build FPN for multi-scale features."""
        in_strides = [4, 8, 16, 32]  # Assuming standard res2, res3, res4, res5
        
        # Number of FPN levels needed
        min_stride = min(in_strides[:len(self.in_channels)])
        self.num_fpn_levels = max(0, int(np.log2(min_stride) - np.log2(self.common_stride)))

        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for idx, in_ch in enumerate(self.in_channels[:self.num_fpn_levels]):
            lateral_conv = nn.Conv2d(in_ch, self.conv_dim, kernel_size=1, bias=True)
            output_conv = ConvNormAct(self.conv_dim, self.conv_dim, 3)
            
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

    def forward(self, features: Dict[str, torch.Tensor]) -> tuple:
        """
        Forward pass.
        
        Args:
            features: Dict of feature maps from TerraMind neck
                      Keys: 'res2', 'res3', 'res4', 'res5'
                      Values: [B, C, H, W]
        
        Returns:
            mask_features: Final mask features [B, mask_dim, H, W]
            multi_scale_features: List of [B, C, H, W] at multiple scales
        """
        srcs = []
        pos = []
        
        # Get features in order (low to high resolution)
        feature_names = ['res5', 'res4', 'res3', 'res2'][:self.num_feature_levels]
        
        for idx, name in enumerate(feature_names):
            if name in features:
                x = features[name].float()
                src = self.input_proj[idx](x)
                srcs.append(src)
                pos.append(self.pe_layer(x))

        if len(srcs) == 0:
            raise ValueError("No features provided to pixel decoder")

        # Run MSDeformAttn encoder
        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]

        # Split output back to feature maps
        split_sizes = []
        for i in range(len(spatial_shapes)):
            if i < len(spatial_shapes) - 1:
                split_sizes.append(level_start_index[i + 1].item() - level_start_index[i].item())
            else:
                split_sizes.append(y.shape[1] - level_start_index[i].item())
        
        y = torch.split(y, split_sizes, dim=1)
        out = []
        for i, z in enumerate(y):
            h, w = spatial_shapes[i]
            out.append(z.transpose(1, 2).view(bs, -1, h, w))

        # Apply FPN on lower-resolution features
        for idx, lateral_conv in enumerate(self.lateral_convs):
            if idx < len(out):
                x = features.get(feature_names[idx], out[idx])
                cur_fpn = lateral_conv(x)
                y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
                y = self.output_convs[idx](y)
                out.append(y)

        # Collect multi-scale features for transformer decoder
        multi_scale_features = out[:self.maskformer_num_feature_levels]
        
        # Final mask features from highest resolution
        mask_features = self.mask_features(out[-1])

        return mask_features, multi_scale_features
