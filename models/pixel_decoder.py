"""
MSDeformAttn Pixel Decoder matched to feature schemas returned by the TerraMind backbone.
Optimized for zero-overhead routing, exact Mask2Former architectural fidelity, 
and consistent GELU activation profiles.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ms_deform_attn import MSDeformAttnTransformerEncoderOnly
from .position_encoding import PositionEmbeddingSine


def _get_norm_groups(conv_dim: int) -> int:
    """Dynamically determine GroupNorm groups to avoid crashes on non-standard channel sizes."""
    groups = min(32, conv_dim)
    while conv_dim % groups != 0:
        groups -= 1
    return groups


class MSDeformAttnPixelDecoder(nn.Module):
    def __init__(
        self,
        transformer_dropout: float = 0.0,
        transformer_nheads: int = 8,
        transformer_dim_feedforward: int = 1024,
        transformer_enc_layers: int = 6,
        conv_dim: int = 256,
        mask_dim: int = 256,
    ):
        super().__init__()
        
        self.conv_dim = conv_dim
        self.mask_dim = mask_dim
        
        # Explicit feature level constants
        self.transformer_levels = ("res5", "res4", "res3")
        self.fpn_level = "res2"
        self.transformer_num_feature_levels = len(self.transformer_levels)
        self.maskformer_num_feature_levels = 3  # Kept for external reference/consistency

        # 1. Transformer strictly limited to 3 semantic levels, using GELU
        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
            activation="gelu",  # Ensures consistency with TerraMind & FPN Neck
        )

        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # 2. Single-step FPN for res2 ONLY
        self.lateral_conv = nn.Conv2d(conv_dim, conv_dim, kernel_size=1)
        self.output_conv = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(_get_norm_groups(conv_dim), conv_dim),
            nn.GELU() 
        )

        # 3. Final Mask Features projection
        self.mask_features = nn.Conv2d(conv_dim, mask_dim, kernel_size=1, stride=1, padding=0)

        self._init_weights()

    def _init_weights(self):
        """Apply Xavier/MSRA initialization matching the official Mask2Former implementation."""
        for module in [self.lateral_conv, self.output_conv[0], self.mask_features]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        srcs, pos = [], []
        for name in self.transformer_levels:
            if name not in features:
                raise KeyError(f"PixelDecoder expected feature '{name}' from backbone.")
            
            x = features[name]
            # A3: Channel validation
            if x.shape[1] != self.conv_dim:
                raise ValueError(f"PixelDecoder expected {self.conv_dim} channels, got {x.shape[1]} from '{name}'.")
            
            srcs.append(x)
            pos.append(self.pe_layer(x))
        # ... rest of forward        

        # Run multi-scale deformable attention encoder
        y, spatial_shapes, _ = self.transformer(srcs, pos)
        bs = y.shape[0]

        # Split the flattened output back into separate levels
        split_sizes = [spatial_shapes[i][0] * spatial_shapes[i][1] for i in range(len(spatial_shapes))]
        y = torch.split(y, split_sizes, dim=1)
        
        # Reshape transformer tokens back to 2D spatial grids
        transformer_features = []
        for i, z in enumerate(y):
            h, w = spatial_shapes[i]
            # .reshape() replaces .view() to safely handle non-contiguous memory from .transpose()
            transformer_features.append(z.transpose(1, 2).reshape(bs, self.conv_dim, h, w))
            
        assert len(transformer_features) == 3, "Transformer must output exactly 3 levels."
        
        # transformer_features:
        # [0] res5'
        # [1] res4'
        # [2] res3'

        # ----------------------------------------------------------------------
        # PIPELINE 2: FPN (Boundary Mask Feature)
        # ----------------------------------------------------------------------
        if self.fpn_level not in features:
            raise ValueError(f"Expected feature '{self.fpn_level}' from backbone neck for FPN.")
            
        raw_res2 = features[self.fpn_level]     # No .float() cast
        assert raw_res2.shape[1] == self.conv_dim, f"FPN input must have {self.conv_dim} channels."
        
        # 1x1 lateral projection to align the raw backbone feature
        lateral = self.lateral_conv(raw_res2)
        
        # Upsample the contextualized res3' from the transformer
        upsampled_res3 = F.interpolate(
            transformer_features[-1], 
            size=lateral.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        # Fuse and smooth
        fpn_res2 = self.output_conv(lateral + upsampled_res3)
        
        # Generate final highest-resolution mask feature
        mask_features = self.mask_features(fpn_res2)

        # ----------------------------------------------------------------------
        # FINAL OUTPUTS
        # ----------------------------------------------------------------------
        return (
            mask_features,
            transformer_features,
        )