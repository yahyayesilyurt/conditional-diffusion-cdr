import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionConditionGenerator(nn.Module):
    """
    Makaledeki Denklem (3) ve (4)'ü uygulayan çapraz alan koşul üreticisi.

    Giriş vektörleri:
      h_u_cross  : GNN user embedding'inden gelen cross-domain sinyal.
                   Kullanıcının hem kitap hem film graph'ından mesaj almış temsili.
                   e2e_wrapper'da: user_proj(user_embedding[user_id])
      h_u_target : Hedef domain aggregator'ından gelen domain-specific sinyal.
                   Kullanıcının film geçmişinin self-attention çıktısı.
                   e2e_wrapper'da: movie_aggregator(movie_seq_embs)

    Bu iki sinyali cross-attention ile birleştirip diffusion'a verilecek
    c_ud koşul vektörünü üretir.
    """

    def __init__(self, embed_dim=64, num_heads=4, ffn_dim=256, dropout=0.1):
        super(AttentionConditionGenerator, self).__init__()

        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) num_heads ({num_heads})'e tam bölünebilmeli."

        # Öğrenilebilir domain göstergeleri — makaledeki v_A, v_B
        # "Hangi domain için öneri üretiyorum?" sinyali.
        # Tüm kullanıcılar için aynı — query kullanıcıdan bağımsız.
        self.domain_indicator_Book  = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.domain_indicator_Movie = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        # Cross-Attention:
        # Query   = domain_indicator          — "ne arıyorum?"
        # Key/Val = [h_u_cross; h_u_target]   — "neye bakabilirim?"
        # Attention ağırlıkları: cross-domain ve target-domain sinyallerine
        # ne kadar güvenileceğini öğrenir.
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
        h_u_cross  : (B, D) — GNN user embedding (cross-domain sinyal)
        h_u_target : (B, D) — film geçmişi aggregator çıktısı (domain-specific)
        target_domain : 'Movie' veya 'Book'

        Döndürür:
            c_ud : (B, D) — koşul vektörü (normalize e2e_wrapper'da yapılır)
        """
        batch_size = h_u_cross.size(0)

        # 1. KV: iki sinyali sıralı diz → (B, 2, D)
        #    [0] = cross-domain sinyal (GNN user)
        #    [1] = target-domain sinyal (film geçmişi)
        kv = torch.stack([h_u_cross, h_u_target], dim=1)

        # 2. Query: hedef domain indicator → (B, 1, D)
        if target_domain == 'Movie':
            query = self.domain_indicator_Movie.expand(batch_size, 1, -1)
        else:
            query = self.domain_indicator_Book.expand(batch_size, 1, -1)

        # 3. Cross-Attention — Denklem 3 & 4
        attn_out, _ = self.cross_attention(query, kv, kv)  # (B, 1, D)

        # 4. Residual + LayerNorm
        c_ud = self.norm1(query.clone() + attn_out)        # (B, 1, D)

        # 5. FFN + Residual
        c_ud = self.norm2(c_ud + self.ffn(c_ud))           # (B, 1, D)

        return c_ud.squeeze(1)                              # (B, D)