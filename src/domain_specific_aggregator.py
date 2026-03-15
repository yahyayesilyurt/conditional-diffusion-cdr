import torch
import torch.nn as nn


class DomainSpecificAggregator(nn.Module):
    """
    Pre-Norm Transformer block — Self-Attention + FFN.
    Summarizes a variable-length item sequence into a single user embedding.

    Changes from original:
      - Post-Norm → Pre-Norm  (reduced gradient vanishing risk)
      - Double Dropout in FFN → single Dropout  (prevented excessive regularization)
      - float() cast → masked_fill  (mixed precision compatibility)
      - super(ClassName, self) → super()  (modern Python style)
    """

    def __init__(self, embed_dim=256, num_heads=8, ffn_dim=1024, dropout=0.1):
        super().__init__()

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
        )

        # Self-Attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # Feed-Forward Network
        # ffn_dim is provided externally (E2EWrapper: embed_dim * 4)
        # → 256*4=1024, 128*4=512, 64*4=256 
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, item_seq_embs, key_padding_mask=None):
        """
        item_seq_embs   : (B, S, D)
        key_padding_mask: (B, S) — True where padding

        Returns:
            h_u_d : (B, D)
        """
        # --- Pre-Norm Self-Attention + Residual ---
        normed = self.norm1(item_seq_embs)
        attn_output, _ = self.self_attention(
            query=normed,
            key=normed,
            value=normed,
            key_padding_mask=key_padding_mask
        )
        x = item_seq_embs + self.dropout(attn_output)

        # --- Pre-Norm FFN + Residual ---
        x = x + self.dropout(self.ffn(self.norm2(x)))

        # --- Masked Mean Pooling ---
        # mean instead of sum: representation independent of varying history lengths
        if key_padding_mask is not None:
            valid_mask   = (~key_padding_mask).unsqueeze(-1)            # (B, S, 1) bool
            x            = x.masked_fill(~valid_mask, 0.0)
            valid_counts = valid_mask.sum(dim=1).clamp(min=1).float()   # (B, 1)
            h_u_d        = x.sum(dim=1) / valid_counts                  # (B, D)
        else:
            h_u_d = x.mean(dim=1)

        return h_u_d