"""
Global configuration for TerraMind-Mask2Former.


This file intentionally replaces Detectron2's cfg system with a simple
Python dataclass so the implementation remains lightweight.
"""


from dataclasses import dataclass, field
from typing import List




@dataclass
class ModelConfig:
    # ------------------------------------------------------------------
    # Backbone
    # ------------------------------------------------------------------


    backbone_name: str = "terramind_v1_base"


    pretrained: bool = True


    freeze_backbone: bool = True


    modalities: List[str] = field(
        default_factory=lambda: ["S2L2A"]
    )


    merge_method: str = "mean"


    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------


    image_size: int = 224


    patch_size: int = 16


    num_input_channels: int = 12


    # ------------------------------------------------------------------
    # Transformer
    # ------------------------------------------------------------------


    hidden_dim: int = 256


    num_queries: int = 100


    nheads: int = 8


    dim_feedforward: int = 2048


    dec_layers: int = 9


    dropout: float = 0.0


    pre_norm: bool = False


    # ------------------------------------------------------------------
    # Pixel Decoder
    # ------------------------------------------------------------------


    mask_dim: int = 256


    conv_dim: int = 256


    transformer_in_features = [
        "res3",
        "res4",
        "res5"
    ]


    common_stride: int = 4


    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------


    no_object_weight: float = 0.1


    dice_weight: float = 5.0


    mask_weight: float = 5.0


    class_weight: float = 2.0


    num_points: int = 12544


    oversample_ratio: float = 3.0


    importance_sample_ratio: float = 0.75


    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------


    lr: float = 1e-4


    weight_decay: float = 0.05


    batch_size: int = 4


    epochs: int = 100


    workers: int = 4


    device: str = "cuda"


    amp: bool = True


    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------


    ignore_index: int = 255
