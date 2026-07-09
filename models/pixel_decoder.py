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
            # 1. Validate required features exist (allows extra unused features for extensibility)
            expected_keys = set(self.transformer_levels) | {self.fpn_level}
            missing_keys = expected_keys - set(features.keys())
            if missing_keys:
                raise KeyError(f"PixelDecoder missing required features from backbone: {missing_keys}")

            # 2. Check spatial ordering (Resolutions must strictly decrease: res2 > res3 > res4 > res5)
            spatial_shapes = [features[k].shape[-2:] for k in [self.fpn_level] + list(self.transformer_levels)]
            for i in range(len(spatial_shapes) - 1):
                h1, w1 = spatial_shapes[i]
                h2, w2 = spatial_shapes[i+1]
                if h1 <= h2 or w1 <= w2:
                    raise RuntimeError(
                        f"Spatial ordering invalid. Level {i} ({h1}x{w1}) "
                        f"not strictly larger than {i+1} ({h2}x{w2})"
                    )

            # ----------------------------------------------------------------------
            # PIPELINE 1: Transformer (Semantic Features)
            # ----------------------------------------------------------------------
            srcs, pos = [], []
            for name in self.transformer_levels:
                x = features[name]  # No .float() cast; preserves AMP
                
                # Channel validation
                if x.shape[1] != self.conv_dim:
                    raise ValueError(f"PixelDecoder expected {self.conv_dim} channels, got {x.shape[1]} from '{name}'.")
                
                srcs.append(x)
                pos.append(self.pe_layer(x))

            # Run multi-scale deformable attention encoder
            y, spatial_shapes_transformer, _ = self.transformer(srcs, pos)
            bs = y.shape[0]

            # Split the flattened output back into separate levels
            split_sizes = [h * w for h, w in spatial_shapes_transformer]
            y = torch.split(y, split_sizes, dim=1)
            
            # Reshape transformer tokens back to 2D spatial grids
            transformer_features = []
            for i, z in enumerate(y):
                h, w = spatial_shapes_transformer[i]
                # .reshape() replaces .view() to safely handle non-contiguous memory from .transpose()
                transformer_features.append(z.transpose(1, 2).reshape(bs, self.conv_dim, h, w))
                
            if len(transformer_features) != self.transformer_num_feature_levels:
                raise ValueError(f"Transformer must output exactly {self.transformer_num_feature_levels} levels.")
            
            # transformer_features:
            # [0] res5'
            # [1] res4'
            # [2] res3'

            # ----------------------------------------------------------------------
            # PIPELINE 2: FPN (Boundary Mask Feature)
            # ----------------------------------------------------------------------
            raw_res2 = features[self.fpn_level]
            
            if raw_res2.shape[1] != self.conv_dim:
                raise ValueError(f"FPN input '{self.fpn_level}' must have {self.conv_dim} channels, got {raw_res2.shape[1]}")
            
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
            
            # Output dim validation
            if mask_features.shape[1] != self.mask_dim:
                raise ValueError(f"Mask features channels {mask_features.shape[1]} != expected mask_dim {self.mask_dim}")

            # ----------------------------------------------------------------------
            # FINAL OUTPUTS
            # ----------------------------------------------------------------------
            return mask_features, transformer_features