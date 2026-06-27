"""Self-attention feature extractor for the tokenized PPO2 observation.

This module is only imported in the ML path (torch + stable_baselines3 required).
Do NOT import this from encode.py or any path that must stay import-safe.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from locma.envs.encode import N_TACTICAL, NUM_CARDS, TOKEN_FEATS


class TokenSetExtractor(BaseFeaturesExtractor):
    """Set-based self-attention feature extractor for tokenized card observations.

    Architecture (forward pass for batch size B):

    1. card_ids (B,20) → Embedding(161, id_dim) → (B,20,id_dim)
    2. cat([tokens, id_embed], dim=-1) → Linear(TOKEN_FEATS+id_dim, d_model) → (B,20,d_model)
    3. Prepend learned CLS → (B,21,d_model); build key_padding_mask from token_mask
    4. TransformerEncoder (n_layers, batch_first) with src_key_padding_mask
    5. Pool: CLS slot z[:,0] (pool="cls") or learned-query MHA over token outputs (pool="attn")
    6. scalar_mlp(scalars) → (B,d_model)
    7. head(cat([pool_out, scalar_out])) → (B,features_dim)

    No positional encoding is added, making the encoder permutation-equivariant
    (set-based): the output depends only on the *set* of real tokens, not their order.
    """

    def __init__(
        self,
        observation_space,
        *,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        id_dim: int = 16,
        ff_mult: int = 2,
        dropout: float = 0.1,
        features_dim: int = 128,
        pool: str = "cls",
    ) -> None:
        super().__init__(observation_space, features_dim)

        if pool not in ("cls", "attn"):
            raise ValueError(f"pool must be 'cls' or 'attn', got {pool!r}")

        self.pool = pool

        # Card-id embedding: index 0 is PAD (padding_idx → zero, no gradient).
        self.id_embed = nn.Embedding(NUM_CARDS + 1, id_dim, padding_idx=0)

        # Project token features + id embedding to d_model.
        self.proj = nn.Linear(TOKEN_FEATS + id_dim, d_model)

        # Learned CLS token prepended before the transformer.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer encoder (batch_first=True throughout).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Scalar MLP: project N_TACTICAL scalars → d_model.
        self.scalar_mlp = nn.Sequential(
            nn.Linear(N_TACTICAL, d_model),
            nn.ReLU(),
        )

        # Approach-C fallback: single learned-query MHA pooling over token outputs.
        if pool == "attn":
            self.attn_query = nn.Parameter(torch.zeros(1, 1, d_model))
            self.attn_pool = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=0.0,  # pooling head: no dropout
                batch_first=True,
            )

        # Final head: fuse CLS/pool output with scalar branch.
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        # 1. Embed card IDs (cast float32 → long; SB3 batches Box obs as float32).
        ids = obs["card_ids"].long()  # (B, 20)
        id_embed = self.id_embed(ids)  # (B, 20, id_dim)

        # 2. Project concatenated numeric + id features to d_model.
        x = self.proj(torch.cat([obs["tokens"], id_embed], dim=-1))  # (B, 20, d_model)

        # 3. Prepend learned CLS token.
        B = x.size(0)
        cls = self.cls_token.expand(B, 1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (B, 21, d_model)

        # 4. Build src_key_padding_mask for TransformerEncoder.
        #    Convention: True = IGNORE this position.
        #    token_mask is 1.0 for real cards, 0.0 for pads → invert for kpm.
        kpm_tokens = obs["token_mask"] == 0  # (B, 20) True for pads
        cls_false = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        kpm = torch.cat([cls_false, kpm_tokens], dim=1)  # (B, 21)

        # 5. Run transformer.
        z = self.transformer(x, src_key_padding_mask=kpm)  # (B, 21, d_model)

        # 6. Pool.
        if self.pool == "cls":
            pool_out = z[:, 0]  # (B, d_model)
        else:  # pool == "attn"
            # Learned-query MHA over the 20 token outputs (not the CLS slot).
            token_z = z[:, 1:]  # (B, 20, d_model)
            q = self.attn_query.expand(B, 1, -1)  # (B, 1, d_model)
            attn_out, _ = self.attn_pool(
                q,
                token_z,
                token_z,
                key_padding_mask=kpm_tokens,  # (B, 20)
            )
            pool_out = attn_out.squeeze(1)  # (B, d_model)

        # 7. Scalar branch.
        s = self.scalar_mlp(obs["scalars"])  # (B, d_model)

        # 8. Head: fuse and project to features_dim.
        return self.head(torch.cat([pool_out, s], dim=-1))  # (B, features_dim)
