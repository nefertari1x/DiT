"""
QDTAutoregressive: small causal transformer for quadtree structure generation.

Predicts a DFS decision sequence (split/leaf) conditioned on class label.
Each input token embeds: action + depth + remaining_budget + class.
Output: 2-class logits (SPLIT=0, LEAF=1) for the next decision.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from qdt_gen.data import SPLIT, LEAF, NUM_LEAVES


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using F.scaled_dot_product_attention."""

    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)                      # each (B, T, num_heads, head_dim)
        q = q.transpose(1, 2)                         # (B, num_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # is_causal=True → FlashAttention / memory-efficient kernel, no explicit mask
        x = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, T, C)       # (B, T, C)
        return self.out_proj(x)


class DecoderBlock(nn.Module):
    """Pre-norm transformer decoder block (self-attention only, no cross-attention)."""

    def __init__(self, hidden_size, num_heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, num_heads, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ff_mult),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * ff_mult, hidden_size),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.drop2(self.mlp(self.norm2(x)))
        return x


class QDTAutoregressive(nn.Module):
    """
    Autoregressive transformer decoder for QDT decision sequences.

    Architecture:
        - Input: (action, depth, remaining, class_label) per token
        - Learned embeddings for action (3: SPLIT, LEAF, BOS) and depth (max_depth+1)
        - Linear projection for remaining_leaves (scalar → hidden)
        - Class embedding (num_classes → hidden)
        - Decoder-only causal transformer (FlashAttention via SDPA)
        - Linear head → 2 logits (SPLIT, LEAF)

    Training (teacher forcing):
        Input:  [BOS, decision_0, ..., decision_{T-2}]
        Target: [action_0, action_1, ..., action_{T-1}]
    """

    def __init__(
        self,
        num_classes=1000,
        max_depth=7,
        hidden_size=384,
        num_layers=6,
        num_heads=6,
        ff_mult=4,
        max_seq_len=5462,  # 5461 decisions + 1 BOS
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.num_actions = 2  # SPLIT, LEAF

        BOS_ID = 2  # action ids: 0=SPLIT, 1=LEAF, 2=BOS
        self.bos_id = BOS_ID

        # Token embeddings
        self.action_embed = nn.Embedding(3, hidden_size)  # SPLIT, LEAF, BOS
        self.depth_embed = nn.Embedding(max_depth + 1, hidden_size)
        self.budget_proj = nn.Linear(1, hidden_size, bias=False)
        self.class_embed = nn.Embedding(num_classes, hidden_size)

        # Learnable positional encoding
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)

        # Input layernorm
        self.input_norm = nn.LayerNorm(hidden_size)

        # Decoder-only transformer blocks (no cross-attention)
        self.blocks = nn.ModuleList([
            DecoderBlock(hidden_size, num_heads, ff_mult, dropout)
            for _ in range(num_layers)
        ])

        # Output head
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_head = nn.Linear(hidden_size, self.num_actions)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _embed_tokens(self, actions, depths, budgets, class_labels):
        """
        Embed a sequence of decision tokens.

        Args:
            actions:      (B, T) int — action ids (0=SPLIT, 1=LEAF, 2=BOS)
            depths:       (B, T) int — node depths (0 for BOS)
            budgets:      (B, T) float — remaining leaves normalized to [0, 1]
            class_labels: (B,) int — class labels

        Returns:
            (B, T, hidden_size)
        """
        T = actions.shape[1]
        h = (
            self.action_embed(actions)
            + self.depth_embed(depths)
            + self.budget_proj(budgets.unsqueeze(-1))
            + self.class_embed(class_labels).unsqueeze(1).expand(-1, T, -1)
            + self.pos_embed(torch.arange(T, device=actions.device)).unsqueeze(0)
        )
        return self.input_norm(h)

    def forward(self, decisions, class_labels):
        """
        Teacher-forced forward pass.

        Args:
            decisions:    (B, T, 3) int32 — full decision sequences
                          each row: [action, depth, remaining_leaves]
            class_labels: (B,) int — class labels

        Returns:
            logits: (B, T, 2) — predicted action logits for each position
        """
        B, T, _ = decisions.shape
        device = decisions.device

        # Prepend BOS: shift decisions right by 1
        actions_in = torch.cat([
            torch.full((B, 1), self.bos_id, dtype=torch.long, device=device),
            decisions[:, :-1, 0].long(),
        ], dim=1)  # (B, T)

        depths_in = torch.cat([
            torch.zeros((B, 1), dtype=torch.long, device=device),
            decisions[:, :-1, 1].long(),
        ], dim=1)  # (B, T)

        budgets_raw = torch.cat([
            torch.full((B, 1), NUM_LEAVES, dtype=torch.float32, device=device),
            decisions[:, :-1, 2].float(),
        ], dim=1)  # (B, T)
        budgets_in = budgets_raw / NUM_LEAVES  # normalize to [0, 1]

        # Embed
        h = self._embed_tokens(actions_in, depths_in, budgets_in, class_labels)

        # Decoder-only transformer with causal SDPA (FlashAttention)
        for block in self.blocks:
            h = block(h)

        # Predict actions
        logits = self.output_head(self.output_norm(h))  # (B, T, 2)
        return logits

    def compute_loss(self, decisions, class_labels):
        """
        Compute cross-entropy loss for teacher-forced training.

        Args:
            decisions:    (B, T, 3) int32
            class_labels: (B,) int

        Returns:
            loss: scalar tensor
            logits: (B, T, 2) — reuse for accuracy, avoid double forward
        """
        logits = self.forward(decisions, class_labels)  # (B, T, 2)
        targets = decisions[:, :, 0].long()  # (B, T) — action at each position
        loss = F.cross_entropy(
            logits.reshape(-1, self.num_actions),
            targets.reshape(-1),
        )
        return loss, logits


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# Model registry — easy to add size variants
QDT_models = {
    "QDT-S": dict(hidden_size=256, num_layers=4, num_heads=4),   # ~5M
    "QDT-B": dict(hidden_size=384, num_layers=6, num_heads=6),   # ~13M
    "QDT-L": dict(hidden_size=512, num_layers=8, num_heads=8),   # ~23M
}


def build_qdt_model(name="QDT-B", **kwargs):
    """Build a QDTAutoregressive model by name."""
    config = QDT_models[name].copy()
    config.update(kwargs)
    return QDTAutoregressive(**config)
