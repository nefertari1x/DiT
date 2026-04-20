# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class AttentionQKNorm(nn.Module):
    """
    Self-attention with RMSNorm applied to Q and K per-head before the dot product.
    Drop-in replacement for timm's Attention when qk_norm is enabled.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)            # each: (B, h, N, d)
        q = self.q_norm(q)
        k = self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v)   # (B, h, N, d)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, output_norm=False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        # Output LayerNorm to anchor mean=0/var=1 and prevent long-horizon DC drift.
        self.output_norm = nn.LayerNorm(hidden_size) if output_norm else nn.Identity()

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return self.output_norm(t_emb)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


def leaf_2d_sincos(centers, embed_dim, orig_image_size=256, token_grid_size=16):
    """
    2D sin-cos positional encoding for quadtree leaves, matched to DiT's image-token
    pos_embed (same omega, same h+w concatenation order) so cross-attention can
    reason geometrically.

    centers: (B, N, 2) float tensor, (cx, cy) in [0, orig_image_size]
    Returns: (B, N, embed_dim)
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
    scale = orig_image_size / token_grid_size
    pos = centers / scale                                           # (B, N, 2)
    quarter = embed_dim // 4
    omega = torch.arange(quarter, dtype=pos.dtype, device=pos.device) / quarter
    omega = 1.0 / (10000 ** omega)                                   # (quarter,)
    # h (y) first, then w (x) — matches get_2d_sincos_pos_embed_from_grid ordering
    y = pos[..., 1:2] * omega                                        # (B, N, quarter)
    x = pos[..., 0:1] * omega
    y_emb = torch.cat([y.sin(), y.cos()], dim=-1)                    # (B, N, embed_dim/2)
    x_emb = torch.cat([x.sin(), x.cos()], dim=-1)
    return torch.cat([y_emb, x_emb], dim=-1)                         # (B, N, embed_dim)


class CrossAttentionQKNorm(nn.Module):
    """
    Cross-attention with RMSNorm on Q (K is pre-normalized upstream and shared
    across all DiT blocks).  Only the Q projection is per-block.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, k, v):
        """
        x: (B, T, D) query tokens.
        k, v: (B, h, N_kv, head_dim) precomputed once in DiT.forward.
        """
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)                                            # (B, h, T, d)
        out = F.scaled_dot_product_attention(q, k, v)                 # (B, h, T, d)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    When use_cross=True, includes a cross-attention sub-block with its own
    adaLN modulation (9-way chunk: self-attn / cross-attn / mlp).
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, adaln_clamp=0,
                 qk_norm=False, use_cross=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if qk_norm:
            self.attn = AttentionQKNorm(hidden_size, num_heads=num_heads, qkv_bias=True)
        else:
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaln_clamp = adaln_clamp

        self.use_cross = use_cross
        if use_cross:
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.cross_attn = CrossAttentionQKNorm(hidden_size, num_heads=num_heads, qkv_bias=True)
            num_adaln_outs = 9
        else:
            num_adaln_outs = 6
        self.num_adaln_outs = num_adaln_outs
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, num_adaln_outs * hidden_size, bias=True)
        )

    def forward(self, x, c, depth_k=None, depth_v=None):
        raw = self.adaLN_modulation(c)
        if self.adaln_clamp > 0:
            C = self.adaln_clamp
            raw = C * torch.tanh(raw / C)
        self._adaln_post_clamp = raw  # 供 GradientMonitor 读取 clamp 后的值
        if self.use_cross:
            (shift_msa, scale_msa, gate_msa,
             shift_ca,  scale_ca,  gate_ca,
             shift_mlp, scale_mlp, gate_mlp) = raw.chunk(9, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = raw.chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.use_cross:
            x = x + gate_ca.unsqueeze(1) * self.cross_attn(
                modulate(self.norm_cross(x), shift_ca, scale_ca), depth_k, depth_v
            )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, adaln_clamp=0):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.adaln_clamp = adaln_clamp

    def forward(self, x, c):
        raw = self.adaLN_modulation(c)
        if self.adaln_clamp > 0:
            C = self.adaln_clamp
            raw = C * torch.tanh(raw / C)
        self._adaln_post_clamp = raw  # 供 GradientMonitor 读取 clamp 后的值
        shift, scale = raw.chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        num_depth_levels=0,
        orig_image_size=256,
        adaln_clamp=0,
        qk_norm=False,
        t_embed_norm=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.orig_image_size = orig_image_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size, output_norm=t_embed_norm)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.token_grid_size = int(num_patches ** 0.5)
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.use_depth = num_depth_levels > 0
        if self.use_depth:
            # Shared across all DiT blocks — computed once per forward.
            self.depth_value_embed = nn.Embedding(num_depth_levels, hidden_size)
            self.depth_k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
            self.depth_v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
            self.depth_k_norm = RMSNorm(hidden_size // num_heads)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, adaln_clamp=adaln_clamp,
                     qk_norm=qk_norm, use_cross=self.use_depth)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, adaln_clamp=adaln_clamp)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize depth embedding table and K/V projections (if used):
        if self.use_depth:
            nn.init.normal_(self.depth_value_embed.weight, std=0.02)
            nn.init.xavier_uniform_(self.depth_k_proj.weight)
            nn.init.constant_(self.depth_k_proj.bias, 0)
            nn.init.xavier_uniform_(self.depth_v_proj.weight)
            nn.init.constant_(self.depth_v_proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def _build_depth_kv(self, depth_seq, depth_centers):
        """
        Build shared K, V for depth cross-attention.  Called once per forward.
        depth_seq:      (N, N_leaves) int tensor of depth values per leaf (z-order)
        depth_centers:  (N, N_leaves, 2) float tensor of (cx, cy) in original-image coords
        Returns depth_k, depth_v: each (N, num_heads, N_leaves, head_dim)
        """
        N, L = depth_seq.shape
        head_dim = self.hidden_size // self.num_heads
        tokens = self.depth_value_embed(depth_seq)                        # (N, L, D)
        tokens = tokens + leaf_2d_sincos(
            depth_centers, self.hidden_size,
            orig_image_size=self.orig_image_size,
            token_grid_size=self.token_grid_size,
        ).to(tokens.dtype)
        k = self.depth_k_proj(tokens).reshape(N, L, self.num_heads, head_dim).transpose(1, 2)
        v = self.depth_v_proj(tokens).reshape(N, L, self.num_heads, head_dim).transpose(1, 2)
        k = self.depth_k_norm(k)                                          # (N, h, L, d)
        return k, v

    def forward(self, x, t, y, depth_seq=None, depth_centers=None):
        """
        x: (N, C, H, W) latent inputs.
        t: (N,) diffusion timesteps.
        y: (N,) class labels.
        depth_seq:     (N, N_leaves) int tensor, required when use_depth=True.
        depth_centers: (N, N_leaves, 2) float tensor, required when use_depth=True.
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        if self.use_depth:
            assert depth_seq is not None and depth_centers is not None, \
                "depth_seq and depth_centers required when use_depth=True"
            depth_k, depth_v = self._build_depth_kv(depth_seq, depth_centers)
        else:
            depth_k = depth_v = None
        for block in self.blocks:
            x = block(x, c, depth_k, depth_v)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, depth_seq=None, depth_centers=None):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        if depth_seq is not None:
            ds_half = depth_seq[: len(depth_seq) // 2]
            depth_seq = torch.cat([ds_half, ds_half], dim=0)
        if depth_centers is not None:
            dc_half = depth_centers[: len(depth_centers) // 2]
            depth_centers = torch.cat([dc_half, dc_half], dim=0)
        model_out = self.forward(combined, t, y,
                                 depth_seq=depth_seq, depth_centers=depth_centers)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
