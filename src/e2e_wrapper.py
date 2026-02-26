import torch
import torch.nn as nn
import torch.nn.functional as F

from src.domain_specific_aggregator import DomainSpecificAggregator
from src.attention_condition_generator import AttentionConditionGenerator


class E2EWrapper(nn.Module):
    """
    Uçtan uca sarıcı.

    Pipeline:
        user_id → user_embedding[id+1] → user_proj → h_u_cross   (cross-domain)
        movie_seq → movie_embedding → movie_proj → aggregator → h_u_target (domain-specific)
        [h_u_cross, h_u_target] → condition_generator → c_ud → L2 normalize
    
    Neden id+1?
        padded_user_embs[0]  = sıfır vektör (PAD)
        padded_user_embs[1]  = user_mapping'deki 0. kullanıcı
        padded_user_embs[i+1]= user_mapping'deki i. kullanıcı
        
        Dataset'te movie_id zaten +1 ile kaydediliyor (movie_mapping[i]+1).
        User için aynı offset'i forward() içinde uyguluyoruz.
    """

    def __init__(self, padded_user_embs, padded_movie_embs,
                 embed_dim=64, num_heads=4):
        super(E2EWrapper, self).__init__()

        # GNN Embedding'leri — dondurulmuş
        self.user_embedding = nn.Embedding.from_pretrained(
            padded_user_embs, freeze=True, padding_idx=0
        )
        self.movie_embedding = nn.Embedding.from_pretrained(
            padded_movie_embs, freeze=True, padding_idx=0
        )

        # user_proj: GNN user uzayı → diffusion uzayı (cross-domain köprü)
        self.user_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # movie_proj: GNN movie embedding → aggregator girişi
        self.movie_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),  # user_proj ile tutarlı hale getirildi
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        self.movie_aggregator = DomainSpecificAggregator(
            embed_dim=embed_dim, num_heads=num_heads
        )

        self.condition_generator = AttentionConditionGenerator(
            embed_dim=embed_dim, num_heads=num_heads
        )

    def forward(self, user_ids, movie_seq_ids, movie_mask,
                target_domain='Movie'):
        """
        user_ids      : (B,)    — user_mapping'deki 0-indexed ID'ler
        movie_seq_ids : (B, S)  — +1 offset'li movie ID'leri (0=PAD)
        movie_mask    : (B, S)  — True = padding pozisyonu

        Döndürür:
            c_ud : (B, D) — L2 normalize edilmiş koşul vektörü
        """
        # 1. Cross-domain sinyal
        # +1: user_mapping 0-indexed, padded_user_embs[0]=PAD olduğu için
        # user_mapping'deki i. kullanıcı → padded_user_embs[i+1]
        raw_user_emb = self.user_embedding(user_ids + 1)  # (B, D)
        h_u_cross    = self.user_proj(raw_user_emb)       # (B, D)

        # 2. Domain-specific sinyal
        # movie_seq_ids zaten +1 offset'li (dataset'te movie_mapping[i]+1)
        movie_raw      = self.movie_embedding(movie_seq_ids)   # (B, S, D)
        movie_seq_embs = self.movie_proj(movie_raw)            # (B, S, D)
        h_u_target     = self.movie_aggregator(
            movie_seq_embs, key_padding_mask=movie_mask
        )                                                      # (B, D)

        # 3. Koşul vektörü
        c_ud = self.condition_generator(
            h_u_cross, h_u_target, target_domain=target_domain
        )                                                      # (B, D)

        # 4. L2 Normalize
        c_ud = F.normalize(c_ud, p=2, dim=1)

        return c_ud