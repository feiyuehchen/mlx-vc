"""Diffusion Transformer (DiT) for Seed-VC, ported to MLX.

Architecture: Transformer with RoPE, AdaLN, UViT skip connections,
optional cross-attention. Based on Seed-VC's diffusion_transformer.py.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class DiTConfig:
    hidden_dim: int = 512
    num_heads: int = 8
    depth: int = 13
    head_dim: int = 64
    intermediate_size: Optional[int] = None
    block_size: int = 8192
    in_channels: int = 80
    content_dim: int = 512
    style_condition: bool = True
    uvit_skip_connection: bool = True
    has_cross_attention: bool = False
    context_dim: int = 0
    norm_eps: float = 1e-5
    rope_base: float = 10000.0
    time_as_token: bool = False

    def __post_init__(self):
        if self.intermediate_size is None:
            hidden = 4 * self.hidden_dim
            n_hidden = int(2 * hidden / 3)
            # Round up to multiple of 256
            self.intermediate_size = n_hidden + (256 - n_hidden % 256) % 256
        if self.head_dim == 0:
            self.head_dim = self.hidden_dim // self.num_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x * rms * self.weight


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization: modulates LayerNorm output with style."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = RMSNorm(dim, eps=eps)
        self.project_layer = nn.Linear(dim, 2 * dim)
        self.dim = dim

    def __call__(self, x: mx.array, embedding: Optional[mx.array] = None) -> mx.array:
        if embedding is None:
            return self.norm(x)
        proj = self.project_layer(embedding)
        weight = proj[..., : self.dim]
        bias = proj[..., self.dim :]
        return weight * self.norm(x) + bias


def precompute_freqs_cis(
    seq_len: int, head_dim: int, rope_base: float = 10000.0
) -> mx.array:
    """Precompute RoPE frequency pairs as complex exponentials."""
    freqs = 1.0 / (
        rope_base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
    )
    t = mx.arange(seq_len, dtype=mx.float32)
    freqs = mx.outer(t, freqs)  # [seq_len, head_dim/2]
    # Store as [cos, sin] pairs: [seq_len, head_dim/2, 2]
    cos_f = mx.cos(freqs)
    sin_f = mx.sin(freqs)
    return mx.stack([cos_f, sin_f], axis=-1)


def apply_rotary_emb(x: mx.array, freqs_cis: mx.array) -> mx.array:
    """Apply rotary positional embedding.

    Args:
        x: [B, seq_len, n_heads, head_dim]
        freqs_cis: [seq_len, head_dim/2, 2]
    """
    # Split into pairs
    x_r = x[..., 0::2]  # [B, T, H, head_dim/2]
    x_i = x[..., 1::2]

    cos = freqs_cis[:, :, 0]  # [T, head_dim/2]
    sin = freqs_cis[:, :, 1]

    # Broadcast: add batch and head dims
    cos = cos[None, :, None, :]  # [1, T, 1, head_dim/2]
    sin = sin[None, :, None, :]

    out_r = x_r * cos - x_i * sin
    out_i = x_r * sin + x_i * cos

    # Interleave back
    B, T, H, half = out_r.shape
    out = mx.zeros((B, T, H, half * 2), dtype=x.dtype)
    out = out.at[..., 0::2].add(out_r)
    out = out.at[..., 1::2].add(out_i)
    return out


class Attention(nn.Module):
    def __init__(self, config: DiTConfig, is_cross_attention: bool = False):
        super().__init__()
        self.n_head = config.num_heads
        self.head_dim = config.head_dim
        self.n_local_heads = config.num_heads
        self.is_cross_attention = is_cross_attention

        if is_cross_attention:
            self.wq = nn.Linear(
                config.hidden_dim, config.num_heads * config.head_dim, bias=False
            )
            self.wkv = nn.Linear(
                config.context_dim,
                2 * config.num_heads * config.head_dim,
                bias=False,
            )
        else:
            total_dim = (config.num_heads + 2 * config.num_heads) * config.head_dim
            self.wqkv = nn.Linear(config.hidden_dim, total_dim, bias=False)

        self.wo = nn.Linear(
            config.num_heads * config.head_dim, config.hidden_dim, bias=False
        )
        self.scale = math.sqrt(self.head_dim)

    def __call__(
        self,
        x: mx.array,
        freqs_cis: mx.array,
        mask: Optional[mx.array] = None,
        context: Optional[mx.array] = None,
        context_freqs_cis: Optional[mx.array] = None,
    ) -> mx.array:
        B, T, _ = x.shape
        kv_size = self.n_local_heads * self.head_dim

        if context is None:
            qkv = self.wqkv(x)
            q = qkv[..., :kv_size]
            k = qkv[..., kv_size : 2 * kv_size]
            v = qkv[..., 2 * kv_size :]
            ctx_len = T
        else:
            q = self.wq(x)
            kv = self.wkv(context)
            k = kv[..., :kv_size]
            v = kv[..., kv_size:]
            ctx_len = context.shape[1]

        q = q.reshape(B, T, self.n_head, self.head_dim)
        k = k.reshape(B, ctx_len, self.n_local_heads, self.head_dim)
        v = v.reshape(B, ctx_len, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k_freqs = context_freqs_cis if context_freqs_cis is not None else freqs_cis
        k = apply_rotary_emb(k, k_freqs)

        # [B, H, T, D]
        q = mx.transpose(q, [0, 2, 1, 3])
        k = mx.transpose(k, [0, 2, 1, 3])
        v = mx.transpose(v, [0, 2, 1, 3])

        scores = (q @ mx.transpose(k, [0, 1, 3, 2])) / self.scale
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        y = weights @ v

        y = mx.transpose(y, [0, 2, 1, 3]).reshape(B, T, -1)
        return self.wo(y)


class FeedForward(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = AdaptiveLayerNorm(config.hidden_dim, config.norm_eps)
        self.ffn_norm = AdaptiveLayerNorm(config.hidden_dim, config.norm_eps)
        self.time_as_token = config.time_as_token

        self.has_cross_attention = config.has_cross_attention
        if config.has_cross_attention:
            self.cross_attention = Attention(config, is_cross_attention=True)
            self.cross_attention_norm = AdaptiveLayerNorm(
                config.hidden_dim, config.norm_eps
            )

        self.uvit_skip_connection = config.uvit_skip_connection
        if config.uvit_skip_connection:
            self.skip_in_linear = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

    def __call__(
        self,
        x: mx.array,
        c: Optional[mx.array],
        freqs_cis: mx.array,
        mask: Optional[mx.array] = None,
        context: Optional[mx.array] = None,
        context_freqs_cis: Optional[mx.array] = None,
        skip_in_x: Optional[mx.array] = None,
    ) -> mx.array:
        cond = None if self.time_as_token else c

        if self.uvit_skip_connection and skip_in_x is not None:
            x = self.skip_in_linear(mx.concatenate([x, skip_in_x], axis=-1))

        h = x + self.attention(self.attention_norm(x, cond), freqs_cis, mask)

        if self.has_cross_attention and context is not None:
            h = h + self.cross_attention(
                self.cross_attention_norm(h, cond),
                freqs_cis,
                None,
                context,
                context_freqs_cis,
            )

        out = h + self.feed_forward(self.ffn_norm(h, cond))
        return out


class Transformer(nn.Module):
    """DiT Transformer backbone with UViT skip connections."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        self.layers = [TransformerBlock(config) for _ in range(config.depth)]
        self.norm = AdaptiveLayerNorm(config.hidden_dim, config.norm_eps)

        # Precompute RoPE
        self.freqs_cis = precompute_freqs_cis(
            config.block_size, config.head_dim, config.rope_base
        )

        # UViT skip connection indices
        if config.uvit_skip_connection:
            self.layers_emit_skip = [
                i for i in range(config.depth) if i < config.depth // 2
            ]
            self.layers_receive_skip = [
                i for i in range(config.depth) if i > config.depth // 2
            ]
        else:
            self.layers_emit_skip = []
            self.layers_receive_skip = []

    def __call__(
        self,
        x: mx.array,
        c: mx.array,
        input_pos: mx.array,
        mask: Optional[mx.array] = None,
        context: Optional[mx.array] = None,
        context_input_pos: Optional[mx.array] = None,
    ) -> mx.array:
        freqs_cis = self.freqs_cis[input_pos]
        context_freqs_cis = (
            self.freqs_cis[context_input_pos] if context_input_pos is not None else None
        )

        skip_list = []
        for i, layer in enumerate(self.layers):
            skip_in = (
                skip_list.pop(-1)
                if (
                    self.config.uvit_skip_connection
                    and i in self.layers_receive_skip
                    and skip_list
                )
                else None
            )

            x = layer(x, c, freqs_cis, mask, context, context_freqs_cis, skip_in)

            if self.config.uvit_skip_connection and i in self.layers_emit_skip:
                skip_list.append(x)

        x = self.norm(x, c)
        return x
