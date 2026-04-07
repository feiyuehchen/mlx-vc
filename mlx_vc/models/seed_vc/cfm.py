"""Conditional Flow Matching for Seed-VC, ported to MLX.

Euler ODE solver with classifier-free guidance for mel spectrogram generation.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_vc.models.seed_vc.dit import DiTConfig, Transformer


class DiT(nn.Module):
    """Full DiT estimator: input projection + Transformer + output projection."""

    def __init__(self, config):
        super().__init__()
        hidden_dim = config.hidden_dim
        in_channels = config.in_channels
        content_dim = config.content_dim
        style_condition = config.style_condition

        # Time embedding
        self.time_embed = TimestepEmbedding(hidden_dim)

        # Input projections
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.prompt_proj = nn.Linear(in_channels, hidden_dim)
        self.cond_proj = nn.Linear(content_dim, hidden_dim)

        # Style projection
        if style_condition:
            self.style_proj = nn.Linear(192, hidden_dim)  # CAMPPlus output = 192
        self.style_condition = style_condition

        # Transformer backbone
        dit_config = DiTConfig(
            hidden_dim=hidden_dim,
            num_heads=config.num_heads,
            depth=config.depth,
            head_dim=config.head_dim if hasattr(config, "head_dim") else hidden_dim // config.num_heads,
            block_size=config.block_size,
            in_channels=in_channels,
            uvit_skip_connection=config.uvit_skip_connection,
            norm_eps=getattr(config, "norm_eps", 1e-5),
            time_as_token=config.time_as_token,
        )
        self.transformer = Transformer(dit_config)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, in_channels)

    def __call__(
        self,
        x: mx.array,
        prompt: mx.array,
        x_lens: mx.array,
        t: mx.array,
        style: mx.array,
        mu: mx.array,
        prompt_lens: Optional[mx.array] = None,
    ) -> mx.array:
        """Estimate velocity field for flow matching.

        Args:
            x: Noisy mel [B, C, T] (channels first)
            prompt: Reference mel prompt [B, C, T]
            x_lens: Lengths [B]
            t: Timestep [B]
            style: Speaker embedding [B, 192]
            mu: Content features [B, T, content_dim]
        """
        B = x.shape[0]
        T = x.shape[2]

        # Time conditioning
        t_emb = self.time_embed(t)  # [B, hidden_dim]
        if self.style_condition:
            t_emb = t_emb + self.style_proj(style)

        # Transpose to [B, T, C] for transformer
        x_t = mx.transpose(x, [0, 2, 1])  # [B, T, in_channels]
        prompt_t = mx.transpose(prompt, [0, 2, 1])

        # Project inputs
        h = self.input_proj(x_t) + self.prompt_proj(prompt_t) + self.cond_proj(mu)

        # Position indices
        input_pos = mx.arange(T)

        # Run transformer
        h = self.transformer(h, t_emb, input_pos)

        # Project back to mel channels
        out = self.output_proj(h)  # [B, T, in_channels]
        out = mx.transpose(out, [0, 2, 1])  # [B, in_channels, T]
        return out


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding → MLP."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def __call__(self, t: mx.array) -> mx.array:
        half_dim = self.dim // 2
        freqs = mx.exp(
            -mx.log(mx.array(self.max_period, dtype=mx.float32))
            * mx.arange(half_dim, dtype=mx.float32)
            / half_dim
        )
        args = t[:, None].astype(mx.float32) * freqs[None, :]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        return self.mlp(emb)


class CFM(nn.Module):
    """Conditional Flow Matching with Euler ODE solver."""

    def __init__(self, config):
        super().__init__()
        self.sigma_min = 1e-6
        self.in_channels = config.in_channels
        self.estimator = DiT(config)
        self.zero_prompt_speech_token = getattr(
            config, "zero_prompt_speech_token", False
        )

    def inference(
        self,
        mu: mx.array,
        x_lens: mx.array,
        prompt: mx.array,
        style: mx.array,
        f0: Optional[mx.array],
        n_timesteps: int,
        temperature: float = 1.0,
        inference_cfg_rate: float = 0.5,
    ) -> mx.array:
        """Generate mel spectrogram via flow matching.

        Args:
            mu: Content condition [B, content_dim, T]
            x_lens: Output lengths [B]
            prompt: Reference mel [B, C, T_ref]
            style: Speaker embedding [B, 192]
            f0: Not used in non-f0 mode
            n_timesteps: Number of Euler steps
            temperature: Noise temperature
            inference_cfg_rate: Classifier-free guidance rate

        Returns:
            Generated mel [B, C, T]
        """
        B = mu.shape[0]
        T = mu.shape[1]

        # Start from noise
        z = mx.random.normal((B, self.in_channels, T)) * temperature
        t_span = mx.linspace(0, 1, n_timesteps + 1)

        return self._solve_euler(
            z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate
        )

    def _solve_euler(
        self,
        x: mx.array,
        x_lens: mx.array,
        prompt: mx.array,
        mu: mx.array,
        style: mx.array,
        f0: Optional[mx.array],
        t_span: mx.array,
        inference_cfg_rate: float = 0.5,
    ) -> mx.array:
        """Fixed Euler solver for the ODE."""
        prompt_len = prompt.shape[-1]
        prompt_x = mx.zeros_like(x)
        prompt_x = prompt_x.at[:, :, :prompt_len].add(prompt[:, :, :prompt_len])
        x = x.at[:, :, :prompt_len].add(-x[:, :, :prompt_len])  # zero out prompt region

        if self.zero_prompt_speech_token:
            mu = mu.at[:, :prompt_len, :].add(-mu[:, :prompt_len, :])

        t = t_span[0]
        for step in range(1, len(t_span)):
            dt = t_span[step] - t_span[step - 1]

            if inference_cfg_rate > 0:
                # Batched CFG: stack conditioned and unconditioned
                stacked_x = mx.concatenate([x, x], axis=0)
                stacked_prompt = mx.concatenate(
                    [prompt_x, mx.zeros_like(prompt_x)], axis=0
                )
                stacked_style = mx.concatenate(
                    [style, mx.zeros_like(style)], axis=0
                )
                stacked_mu = mx.concatenate([mu, mx.zeros_like(mu)], axis=0)
                stacked_t = mx.concatenate(
                    [mx.array([t]), mx.array([t])], axis=0
                )

                dphi_dt = self.estimator(
                    stacked_x, stacked_prompt, x_lens, stacked_t, stacked_style, stacked_mu
                )

                cond, uncond = mx.split(dphi_dt, 2, axis=0)
                dphi_dt = (1.0 + inference_cfg_rate) * cond - inference_cfg_rate * uncond
            else:
                dphi_dt = self.estimator(
                    x, prompt_x, x_lens, mx.array([t]), style, mu
                )

            x = x + dt * dphi_dt
            t = t + dt
            # Zero out prompt region
            x = x.at[:, :, :prompt_len].add(-x[:, :, :prompt_len])

        return x
