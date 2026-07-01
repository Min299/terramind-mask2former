"""
MultiScaleMaskedTransformerDecoder for TerraMind-Mask2Former.

This is a clean PyTorch implementation of the Mask2Former transformer decoder,
adapted from the original Detectron2 implementation.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .position_encoding import PositionEmbeddingSine


class SelfAttentionLayer(nn.Module):
    """Self-attention layer for decoder."""

    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for decoder."""

    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):
    """Feed-forward network layer."""

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class MLP(nn.Module):
    """Simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class MultiScaleMaskedTransformerDecoder(nn.Module):
    """
    Multi-Scale Masked Transformer Decoder.
    
    Takes multi-scale pixel features and produces class predictions and mask logits.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_queries: int = 100,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dec_layers: int = 9,
        pre_norm: bool = False,
        mask_dim: int = 256,
        enforce_input_project: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.nheads = nheads
        self.dec_layers = dec_layers
        self.mask_dim = mask_dim

        # Position encoding for features
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # Transformer layers
        self.num_heads = nheads
        self.num_layers = dec_layers
        
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm)
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm)
            )
            self.transformer_ffn_layers.append(
                FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm)
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # Learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # Learnable query positional encoding
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Level embeddings for multi-scale features
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        # Input projections
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
            else:
                self.input_proj.append(nn.Sequential())

        # Output heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object class
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, x: List[torch.Tensor], mask_features: torch.Tensor):
        """
        Forward pass.
        
        Args:
            x: List of multi-scale features [B, C, H, W]
            mask_features: Final mask features [B, mask_dim, H, W]
        
        Returns:
            Dictionary with:
            - pred_logits: Class predictions [B, num_queries, num_classes + 1]
            - pred_masks: Mask logits [B, num_queries, H, W]
            - aux_outputs: List of auxiliary outputs
        """
        assert len(x) == self.num_feature_levels
        
        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            
            # Convert to HWxNxC format
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # Initialize queries
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # Initial prediction
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        # Decoder layers
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            
            # Cross-attention
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed
            )

            # Self-attention
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](output)

            # Prediction heads
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(predictions_class[:-1], predictions_mask[:-1])
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        """Compute class predictions and mask logits."""
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # Create attention mask for masking
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        """Generate auxiliary outputs for deep supervision."""
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class, outputs_seg_masks)
        ]
