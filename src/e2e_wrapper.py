import torch
import torch.nn as nn
import torch.nn.functional as F

from src.domain_specific_aggregator import DomainSpecificAggregator
from src.attention_condition_generator import AttentionConditionGenerator


class E2EWrapper(nn.Module):
    """
    End-to-end wrapper.

    Pipeline:
        user_id → user_embedding[id+1] → user_proj → h_u_cross   (cross-domain)
        movie_seq → movie_embedding → movie_proj → aggregator → h_u_target (domain-specific)
        [h_u_cross, h_u_target] → condition_generator → c_ud → L2 normalize
    
    Why id+1?
        padded_user_embs[0]  = zero vector (PAD)
        padded_user_embs[1]  = first user in user_mapping (index 0)
        padded_user_embs[i+1]= i-th user in user_mapping
        
        movie_id is already stored with +1 offset in the dataset (movie_mapping[i]+1).
        The same offset is applied in forward() for users.
    """

    def __init__(self, padded_user_embs, padded_movie_embs,
                 embed_dim=128, num_heads=4):
        super(E2EWrapper, self).__init__()

        # GNN Embeddings — frozen
        self.user_embedding = nn.Embedding.from_pretrained(
            padded_user_embs, freeze=True, padding_idx=0
        )
        self.movie_embedding = nn.Embedding.from_pretrained(
            padded_movie_embs, freeze=True, padding_idx=0
        )

        # user_proj: GNN user space → diffusion space (cross-domain bridge)
        self.user_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # movie_proj: GNN movie embedding → aggregator input
        self.movie_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),  # aligned with user_proj
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        self.movie_aggregator = DomainSpecificAggregator(
            embed_dim=embed_dim, num_heads=num_heads, ffn_dim=embed_dim * 4
        )

        self.condition_generator = AttentionConditionGenerator(
            embed_dim=embed_dim, num_heads=num_heads, ffn_dim=embed_dim * 4
        )

    def forward(self, user_ids, movie_seq_ids, movie_mask,
                target_domain='Movie'):
        """
        user_ids      : (B,)    — 0-indexed IDs from user_mapping
        movie_seq_ids : (B, S)  — movie IDs with +1 offset (0=PAD)
        movie_mask    : (B, S)  — True = padding position

        Returns:
            c_ud : (B, D) — L2-normalized condition vector
        """
        # 1. Cross-domain signal
        # +1: user_mapping is 0-indexed; padded_user_embs[0]=PAD
        # i-th user in user_mapping → padded_user_embs[i+1]
        raw_user_emb = self.user_embedding(user_ids + 1)  # (B, D)
        h_u_cross    = self.user_proj(raw_user_emb)       # (B, D)

        # 2. Domain-specific signal
        # movie_seq_ids already carry a +1 offset (movie_mapping[i]+1 in the dataset)
        movie_raw      = self.movie_embedding(movie_seq_ids)   # (B, S, D)
        movie_seq_embs = self.movie_proj(movie_raw)            # (B, S, D)
        h_u_target     = self.movie_aggregator(
            movie_seq_embs, key_padding_mask=movie_mask
        )                                                      # (B, D)

        # 3. Condition vector
        c_ud = self.condition_generator(
            h_u_cross, h_u_target, target_domain=target_domain
        )                                                      # (B, D)

        # 4. L2 Normalize
        c_ud = F.normalize(c_ud, p=2, dim=1)

        return c_ud