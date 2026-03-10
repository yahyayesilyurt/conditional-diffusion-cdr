import torch
import torch.nn as nn


class DomainSpecificAggregator(nn.Module):
    """
    Self-attention-based aggregator

    Changes from original:
    - sum → normalized mean (representation independent of sequence length)
    - Added FFN (Feed-Forward Network) following Transformer standard
    - num_heads=4 (embed_dim must be divisible by num_heads for richer attention heads)
    - Added LayerNorm (for training stability)
    """

    def __init__(self, embed_dim=128, num_heads=4, ffn_dim=512, dropout=0.1):
        super(DomainSpecificAggregator, self).__init__()

        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."

        # Self-Attention mechanism
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # Feed-Forward Network (second half of the Transformer block)
        # Increases representational capacity.
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, item_seq_embs, key_padding_mask=None):
        """
        item_seq_embs : (batch_size, seq_len, embed_dim)
        key_padding_mask : (batch_size, seq_len) — True where padding

        Returns:
            h_u_d : (batch_size, embed_dim)
        """
        # --- Self-Attention + Residual ---
        attn_output, _ = self.self_attention(
            query=item_seq_embs,
            key=item_seq_embs,
            value=item_seq_embs,
            key_padding_mask=key_padding_mask
        )
        # Residual connection + LayerNorm
        x = self.norm1(item_seq_embs + attn_output)

        # --- FFN + Residual ---
        x = self.norm2(x + self.ffn(x))

        # --- Summarize with normalized mean ---
        # sum creates large magnitude differences between users with different
        # history lengths (10 movies watched vs. 1 movie watched).
        # We take the mean over valid (non-padding) tokens.
        if key_padding_mask is not None:
            valid_mask = (~key_padding_mask).unsqueeze(-1).float()  # (B, S, 1)
            x = x * valid_mask
            valid_counts = valid_mask.sum(dim=1).clamp(min=1)       # (B, 1)
            h_u_d = x.sum(dim=1) / valid_counts                     # (B, embed_dim)
        else:
            h_u_d = x.mean(dim=1)

        return h_u_d