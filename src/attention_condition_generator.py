import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionConditionGenerator(nn.Module):
    """
    Cross-domain condition generator using attention.

    Input vectors:
      h_u_cross  : Cross-domain signal from the GNN user embedding.
                   The user's representation after receiving messages from both
                   the book and movie graphs.
                   In e2e_wrapper: user_proj(user_embedding[user_id])
      h_u_target : Domain-specific signal from the target domain aggregator.
                   Self-attention output over the user's movie history.
                   In e2e_wrapper: movie_aggregator(movie_seq_embs)

    Combines these two signals with cross-attention to produce
    the c_ud condition vector for diffusion.
    """

    def __init__(self, embed_dim=64, num_heads=4, ffn_dim=256, dropout=0.1):
        super(AttentionConditionGenerator, self).__init__()

        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."

        # Learnable domain indicators — v_A, v_B
        # Signal for "which domain am I recommending for?"
        # Same for all users — query is user-independent.
        self.domain_indicator_Book  = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.domain_indicator_Movie = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        # Cross-Attention:
        # Query   = domain_indicator          — "what am I looking for?"
        # Key/Val = [h_u_cross; h_u_target]   — "what can I attend to?"
        # Attention weights learn how much to trust the cross-domain
        # and target-domain signals respectively.
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, h_u_cross, h_u_target, target_domain='Movie'):
        """
        h_u_cross  : (B, D) — GNN user embedding (cross-domain signal)
        h_u_target : (B, D) — movie history aggregator output (domain-specific)
        target_domain : 'Movie' or 'Book'

        Returns:
            c_ud : (B, D) — condition vector (normalization is done in e2e_wrapper)
        """
        batch_size = h_u_cross.size(0)

        # 1. KV: stack the two signals in sequence → (B, 2, D)
        #    [0] = cross-domain signal (GNN user)
        #    [1] = target-domain signal (movie history)
        kv = torch.stack([h_u_cross, h_u_target], dim=1)

        # 2. Query: target domain indicator → (B, 1, D)
        if target_domain == 'Movie':
            query = self.domain_indicator_Movie.expand(batch_size, 1, -1)
        else:
            query = self.domain_indicator_Book.expand(batch_size, 1, -1)

        # 3. Cross-Attention
        attn_out, _ = self.cross_attention(query, kv, kv)  # (B, 1, D)

        # 4. Residual + LayerNorm
        c_ud = self.norm1(query.clone() + attn_out)        # (B, 1, D)

        # 5. FFN + Residual
        c_ud = self.norm2(c_ud + self.ffn(c_ud))           # (B, 1, D)

        return c_ud.squeeze(1)                              # (B, D)