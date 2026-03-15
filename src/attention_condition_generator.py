import torch
import torch.nn as nn


class AttentionConditionGenerator(nn.Module):
    """
    GNN-Anchored Cross-Domain Condition Generator.

    Triple-Stream design:
      Query   = h_u_cross                        (GNN identity — "Who am I?")
      Key/Val = [h_u_source || h_u_target]        (Two intent sources)

    Differences from the old design:
      - domain_indicator (static, user-independent Query) removed
      - h_u_cross taken as Query / GNN-Anchor (user-specific, dynamic)
      - h_u_source (book/movie history) added as 3rd stream to K/V
      - target_domain parameter removed (was unused)
      - Post-Norm → Pre-Norm (reduced gradient vanishing risk)
      - Double Dropout in FFN → single Dropout
      - query.clone() removed (unnecessary memory copy)
      - super(ClassName, self) → super() (modern Python style)

    Input vectors:
      h_u_cross  : (B, D) — GNN user embedding (Query / Anchor)
      h_u_source : (B, D) — source domain aggregator output (Key/Value)
                            Amazon: book history | Douban: movie history
      h_u_target : (B, D) — target domain aggregator output (Key/Value)
                            Amazon: movie history | Douban: music history

    The model's internal question (separate for each user and step):
      "For the user at this position in the GNN space, should the source
       domain or the target domain history be more decisive?"

    Returns:
      c_ud : (B, D) — diffusion condition vector (L2 norm applied in E2EWrapper)
    """

    def __init__(self, embed_dim=256, num_heads=8, ffn_dim=1024, dropout=0.1):
        super().__init__()

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
        )

        # Cross-Attention:
        # Query   = h_u_cross                  (GNN identity)
        # Key/Val = [h_u_source; h_u_target]   (two intent sources)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # Feed-Forward Network
        # ffn_dim is provided externally (E2EWrapper: embed_dim * 4)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, h_u_cross, h_u_source, h_u_target):
        """
        h_u_cross  : (B, D) — GNN user embedding
        h_u_source : (B, D) — source domain intent (book/movie history)
        h_u_target : (B, D) — target domain intent (movie/music history)

        Returns:
            c_ud : (B, D)
        """
        # 1. K/V: stack source and target intents side by side → (B, 2, D)
        kv = torch.stack([h_u_source, h_u_target], dim=1)

        # 2. Query: GNN identity → (B, 1, D)
        query = h_u_cross.unsqueeze(1)

        # 3. Pre-Norm Cross-Attention + Residual
        attn_out, _ = self.cross_attention(
            query=self.norm1(query),
            key  =self.norm1(kv),
            value=self.norm1(kv)
        )
        x = query + self.dropout(attn_out)              # (B, 1, D)

        # 4. Pre-Norm FFN + Residual
        x = x + self.dropout(self.ffn(self.norm2(x)))  # (B, 1, D)

        return x.squeeze(1)                             # (B, D)