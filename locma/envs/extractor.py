"""Self-attention feature extractor for the tokenized PPO2 observation.

This module is only imported in the ML path (torch + stable_baselines3 required).
Do NOT import this from encode.py or any path that must stay import-safe.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from locma.envs.encode import MAX_TOKENS, NUM_CARDS, TOKEN_FEATS


class TokenSetExtractor(BaseFeaturesExtractor):
    """Slot-addressable self-attention feature extractor for tokenized card observations.

    Architecture (forward pass for batch size B):

    1. card_ids (B,20) → Embedding(161, id_dim) → (B,20,id_dim)
    2. cat([tokens, id_embed], dim=-1) → Linear(TOKEN_FEATS+id_dim, d_model) →
       LayerNorm(d_model) → (B,20,d_model)
    3. Add learned per-slot positional embedding (1,20,d_model) — breaks pure
       set-invariance intentionally (see "Why slot-addressable" below).
    4. Build key_padding_mask from token_mask (pad=True); all-pad guard unmasks
       fully-padded rows so attention is defined.
    5. TransformerEncoder (n_layers, batch_first=True) with src_key_padding_mask
       provides cross-card relational mixing while preserving slot identity.
    6. Flatten per-slot outputs: z.reshape(B, MAX_TOKENS * d_model) — slot s occupies
       the fixed offset range [s*d_model : (s+1)*d_model], making slot-content
       associations directly addressable by the downstream policy head.
    7. scalar_mlp(scalars): LayerNorm(n_scalar) → Linear(n_scalar, d_model) → ReLU
       normalizes raw scalar magnitudes (health≈30, turn≈50, board totals≈60).
       n_scalar is read from the obs space so any variant (v0=13, v1=18) is drop-in.
    8. head(cat([flat, s], dim=-1)) → (B, features_dim)

    Why slot-addressable (NOT permutation-invariant):
    The 155-action space indexes actions by slot position: Summon→1+s, Use→9+s*13+tc,
    Attack→113+a*7+tc, where s/a are the card's slot in hand/board. A permutation-
    invariant pooled feature (e.g. CLS pooling) produces identical logits when two cards
    swap slots, so the policy CANNOT learn slot-content-specific actions. The per-slot
    positional embedding + flat reshape preserves slot identity in the features, enabling
    the policy head to associate each slot's d_model features with the corresponding
    action indices. The transformer still provides cross-card relational mixing; only the
    permutation-invariant POOLING is replaced.
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
        # 0.1 is empirically validated at the tuned recipe (lr=1e-4 + target_kl):
        # removing it regressed -0.028 paired avg-hard3 (worklog 2026-07-02
        # N-battery). Known PPO caveat: SB3 collects rollouts in eval mode but
        # trains in train mode, so dropout>0 adds noise to the importance ratio
        # (inflates approx_kl) -- here the regularization benefit measurably
        # dominates that cost. Change only with a paired ceiling-eval.
        dropout: float = 0.1,
        features_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim)

        # Card-id embedding: index 0 is PAD (padding_idx → zero, no gradient).
        self.id_embed = nn.Embedding(NUM_CARDS + 1, id_dim, padding_idx=0)

        # Project token features + id embedding to d_model.
        self.proj = nn.Linear(TOKEN_FEATS + id_dim, d_model)

        # Normalize projected token vectors before adding positional embedding.
        # Per-token LayerNorm tames raw magnitudes (cost/attack/defense O(1–12)).
        self.token_ln = nn.LayerNorm(d_model)

        # Learned per-slot positional embedding — one d_model vector per slot.
        # Initialized to zeros; trained to encode slot identity (hand vs my-board
        # vs op-board, and position within each zone).
        self.pos_embed = nn.Parameter(torch.zeros(1, MAX_TOKENS, d_model))

        # Transformer encoder (batch_first=True throughout).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
        )
        # enable_nested_tensor=False: the default (True) + a src_key_padding_mask
        # triggers PyTorch's nested-tensor fast path, which emits a prototype-API
        # warning and is a documented footgun (can alter outputs across versions).
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # Scalar MLP: normalize then project the scalar vector → d_model.
        # The scalar dim is read from the obs space (v0=13, v1=18, or any future
        # variant), so this extractor is variant-agnostic. LayerNorm on the scalar
        # vector tames raw magnitudes (health≈30, turn≈50, board totals≈60) before
        # the linear projection.
        n_scalar = int(observation_space["scalars"].shape[0])
        self.scalar_mlp = nn.Sequential(
            nn.LayerNorm(n_scalar),
            nn.Linear(n_scalar, d_model),
            nn.ReLU(),
        )

        # Final head: fuse flattened per-slot outputs with scalar branch.
        # MAX_TOKENS * d_model from the flattened transformer outputs (20 slots,
        # each d_model wide) + d_model from the scalar branch.
        self.head = nn.Sequential(
            nn.Linear(MAX_TOKENS * d_model + d_model, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        # 1. Embed card IDs (cast float32 → long; SB3 batches Box obs as float32).
        ids = obs["card_ids"].long()  # (B, 20)
        id_embed = self.id_embed(ids)  # (B, 20, id_dim)

        # 2. Project concatenated numeric + id features, then normalize per-token.
        x = self.token_ln(
            self.proj(torch.cat([obs["tokens"], id_embed], dim=-1))
        )  # (B, 20, d_model)

        # 3. Add per-slot positional embedding (breaks pure set-invariance — intended).
        x = x + self.pos_embed  # (B, 20, d_model)

        # 4. Build src_key_padding_mask for TransformerEncoder.
        #    Convention: True = IGNORE this position.
        #    token_mask is 1.0 for real cards, 0.0 for pads → invert for kpm.
        kpm = obs["token_mask"] == 0  # (B, 20) True for pads

        # All-pad guard: a row where every position is masked makes attention
        # undefined (query attends to zero keys → NaN). For any fully-padded
        # row, unmask all positions so attention is well-defined; those slots'
        # logits are masked at the action level anyway.
        all_pad = kpm.all(dim=1, keepdim=True)  # (B, 1) True if entire row is pad
        kpm = kpm & ~all_pad  # unmask all positions for fully-padded rows

        # 5. Run transformer (cross-card relational mixing, slot order preserved).
        z = self.transformer(x, src_key_padding_mask=kpm)  # (B, 20, d_model)

        # 6. Flatten per-slot outputs: slot s at fixed offset s*d_model.
        flat = z.reshape(z.size(0), -1)  # (B, 20 * d_model)

        # 7. Scalar branch.
        s = self.scalar_mlp(obs["scalars"])  # (B, d_model)

        # 8. Head: fuse and project to features_dim.
        return self.head(torch.cat([flat, s], dim=-1))  # (B, features_dim)
