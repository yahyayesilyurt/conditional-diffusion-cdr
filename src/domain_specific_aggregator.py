import torch
import torch.nn as nn


class DomainSpecificAggregator(nn.Module):
    """
    Makaledeki Denklem (1) ve (2)'yi uygulayan öz-dikkat tabanlı toplayıcı.

    Değişiklikler (orijinale göre):
    - sum → normalize edilmiş mean (sequence uzunluğundan bağımsız temsil)
    - Transformer standardına uygun FFN (Feed-Forward Network) eklendi
    - num_heads=4 (daha zengin dikkat başlıkları için embed_dim 4'e bölünebilmeli)
    - LayerNorm eklendi (eğitim kararlılığı için)
    """

    def __init__(self, embed_dim=64, num_heads=4, ffn_dim=256, dropout=0.1):
        super(DomainSpecificAggregator, self).__init__()

        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) num_heads ({num_heads})'e tam bölünebilmeli."

        # Öz-dikkat (Self-Attention) mekanizması
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # Feed-Forward Network (Transformer bloğunun ikinci yarısı)
        # Orijinal kodda yoktu — temsil kapasitesini artırır.
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
        key_padding_mask : (batch_size, seq_len) — padding yerleri True

        Döndürür:
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

        # --- Normalize edilmiş ortalama (mean) ile özetle ---
        # Düzeltme: orijinal kodda sum kullanılıyordu.
        # sum, farklı geçmiş uzunluklarına sahip kullanıcılar arasında büyük 
        # ölçek farkları yaratıyor (10 film izleyen vs 1 film izleyen).
        # Geçerli (padding olmayan) token'ların ortalamasını alıyoruz.
        if key_padding_mask is not None:
            valid_mask = (~key_padding_mask).unsqueeze(-1).float()  # (B, S, 1)
            x = x * valid_mask
            valid_counts = valid_mask.sum(dim=1).clamp(min=1)       # (B, 1)
            h_u_d = x.sum(dim=1) / valid_counts                     # (B, embed_dim)
        else:
            h_u_d = x.mean(dim=1)

        return h_u_d