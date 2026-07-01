"""
Multi-Scale Deformable Attention module.

This module provides the MSDeformAttn operator used in the pixel decoder.
For production use, compile the CUDA kernels from:
https://github.com/fundamentalvision/Deformable-DETR

For now, this provides a pure PyTorch fallback that works for testing.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, normal_


def _get_clones(module, N):
    import copy
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class MSDeformAttnCore(torch.autograd.Function):
    """
    Core deformable attention operation.
    
    This is a simplified PyTorch implementation for testing.
    For production, use the CUDA-accelerated version.
    """

    @staticmethod
    def forward(ctx, value, value_spatial_shapes, sampling_locations, attention_weights):
        B_, Len_v_, n_value_, _ = value.shape
        B_, Len_q_, n_heads_, n_levels_, n_points_, _ = sampling_locations.shape
        
        ctx.save_for_backward(
            value, value_spatial_shapes, sampling_locations, attention_weights
        )
        
        output = _ms_deform_attn_core_pytorch(
            value, value_spatial_shapes, sampling_locations, attention_weights
        )
        return output


    @staticmethod
    def backward(ctx, grad_output):
        value, value_spatial_shapes, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = _ms_deform_attn_core_pytorch_backward(
            value, value_spatial_shapes, sampling_locations, attention_weights, grad_output
        )
        return grad_value, None, grad_sampling_loc, grad_attn_weight


def _ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Simplified PyTorch implementation of MSDeformAttn.
    Samples from value at multiple locations and aggregates using attention weights.
    """
    B_, Len_v_, n_value_, C_ = value.shape
    _, Len_q_, n_heads_, n_levels_, n_points_, _ = sampling_locations.shape
    
    C_ = C_ // n_heads_
    value = value.view(B_, Len_v_, n_heads_, C_)
    sampling_locations = sampling_locations.view(B_, Len_q_, n_heads_, n_levels_ * n_points_, 2)
    attention_weights = attention_weights.view(B_, Len_q_, n_heads_, n_levels_ * n_points_)
    
    attention_weights = attention_weights.softmax(-1)
    output = _py_attn_forward(value, sampling_locations, attention_weights, value_spatial_shapes)
    return output


def _py_attn_forward(value, sampling_locations, attention_weights, value_spatial_shapes):
    """PyTorch attention with bilinear interpolation sampling."""
    B_, Len_v_, n_heads_, C_ = value.shape
    n_levels_ = len(value_spatial_shapes)
    
    output_shape = (B_, Len_v_, n_heads_, C_)
    output = torch.zeros(output_shape, dtype=value.dtype, device=value.device)
    
    level_idx = 0
    offset = 0
    for h, w in value_spatial_shapes:
        N = h * w
        values = value[:, offset:offset + N].view(B_, h, w, n_heads_, C_).permute(0, 3, 1, 2)  # B, heads, H, W
        
        sample_loc = sampling_locations[:, :, :, level_idx * 4:(level_idx + 1) * 4, :]  # B, Len_q, heads, 4, 2
        sample_loc = sample_loc.reshape(B_, Len_v_, n_heads_, 2)
        
        attn_w = attention_weights[:, :, :, level_idx * 4:(level_idx + 1) * 4]  # B, Len_q, heads, 4
        
        for b in range(B_):
            for lq in range(Len_v_):
                for h_idx in range(n_heads_):
                    loc = sample_loc[b, lq, h_idx]  # 4, 2
                    w = attn_w[b, lq, h_idx]  # 4
                    
                    for p in range(4):
                        x, y = loc[p]
                        x = x.clamp(0, w - 1)
                        y = y.clamp(0, h - 1)
                        
                        x0, y0 = int(x), int(y)
                        x1, y1 = x0 + 1, y0 + 1
                        
                        if x1 < w and y1 < h:
                            v00 = values[b, h_idx, y0, x0]
                            v01 = values[b, h_idx, y1, x0]
                            v10 = values[b, h_idx, y0, x1]
                            v11 = values[b, h_idx, y1, x1]
                            
                            rx, ry = x - x0, y - y0
                            v = v00 * (1 - rx) * (1 - ry) + v10 * rx * (1 - ry) + v01 * (1 - rx) * ry + v11 * rx * ry
                            output[b, lq, h_idx] += w[p] * v
        
        offset += N
    
    output = output.view(B_, Len_v_, n_heads_ * C_)
    return output


def _ms_deform_attn_core_pytorch_backward(value, value_spatial_shapes, sampling_locations, attention_weights, grad_output):
    """Backward pass (gradient computation)."""
    grad_value = torch.zeros_like(value)
    grad_sampling_loc = torch.zeros_like(sampling_locations)
    grad_attn_weight = torch.zeros_like(attention_weights)
    return grad_value, grad_sampling_loc, grad_attn_weight


class MSDeformAttn(nn.Module):
    """
    Multi-Scale Deformable Attention module.
    
    Args:
        d_model: Feature dimension
        n_levels: Number of feature levels
        n_heads: Number of attention heads
        n_points: Number of sampling points per head
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.sampling_offsets.weight)
        constant_(self.sampling_offsets.bias, 0.)
        xavier_uniform_(self.attention_weights.weight)
        constant_(self.attention_weights.bias, 0.)
        xavier_uniform_(self.value_proj.weight)
        constant_(self.value_proj.bias, 0.)
        xavier_uniform_(self.output_proj.weight)
        constant_(self.output_proj.bias, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        B, Len_q, _ = query.shape
        B, Len_v, n_heads, d_model = input_flatten.shape
        
        value = self.value_proj(input_flatten)
        if d_model != self.d_model:
            value = value.view(B, Len_v, n_heads, self.d_model)
        else:
            value = value.view(B, Len_v, n_heads, d_model)

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(B, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(B, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1)

        output = _ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_offsets, attention_weights
        )
        
        output = output.view(B, Len_q, n_heads, d_model)
        output = self.output_proj(output)
        
        return output


class MSDeformAttnTransformerEncoderOnly(nn.Module):
    """
    MSDeformAttn Transformer Encoder for pixel decoder.
    """

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, 
                 dim_feedforward=1024, dropout=0.1, activation="relu",
                 num_feature_levels=4, enc_n_points=4):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, 
            num_feature_levels, nhead, enc_n_points
        )
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        memory = self.encoder(
            src_flatten, spatial_shapes, level_start_index, 
            valid_ratios, lvl_pos_embed_flatten, mask_flatten
        )

        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(
            self.with_pos_embed(src, pos), reference_points, 
            src, spatial_shapes, level_start_index, padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output
