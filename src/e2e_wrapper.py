import torch
import torch.nn as nn
import torch.nn.functional as F

from src.domain_specific_aggregator import DomainSpecificAggregator
from src.attention_condition_generator import AttentionConditionGenerator


class E2EWrapper(nn.Module):
    """
    End-to-end wrapper for Triple-Stream Architecture

    Domain mapping:
      Source domain : Book  (user's reading history → h_u_source)
      Target domain : Movie (predict next movie    → h_u_target)

    Triple-Stream:
      Stream 1 — GNN Anchor    : h_u_cross  (GNN user embedding)
      Stream 2 — Source Intent : h_u_source (book sequence aggregation)
      Stream 3 — Target Intent : h_u_target (movie sequence aggregation)

    Ablation flag:
      use_book_stream=False → h_u_source = h_u_cross (book stream disabled)
      Experimentally, book stream showed a negative effect on Amazon Books→Movies.
      Can be easily toggled from config.

    Embedding strategy:
      All embeddings freeze=True — GNN topological structure must be preserved.

    Index conventions (must match CrossDomainDataset output):
      user_ids      : 0-indexed → +1 offset applied internally
      movie_seq_ids : 1-indexed → 0 = padding token
      book_seq_ids  : 1-indexed → 0 = padding token

    embed_dim independence:
      All intermediate dimensions are proportional to embed_dim (embed_dim * 2, embed_dim * 4).
      Changing embed_dim in config.yaml is sufficient — no code changes needed.
    """

    def __init__(
        self,
        padded_user_embs,
        padded_book_embs,
        padded_movie_embs,
        embed_dim=256,
        num_heads=8,
        dropout=0.1,
        use_book_stream=True
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
        )

        self.use_book_stream = use_book_stream

        # --- GNN Embeddings (frozen) ---
        self.user_embedding = nn.Embedding.from_pretrained(
            padded_user_embs, freeze=True, padding_idx=0
        )
        self.movie_embedding = nn.Embedding.from_pretrained(
            padded_movie_embs, freeze=True, padding_idx=0
        )
        self.book_embedding = nn.Embedding.from_pretrained(
            padded_book_embs, freeze=True, padding_idx=0
        )

        # --- Projection layers (shared architecture, separate weights) ---
        # LayerNorm first: normalizes magnitude differences in raw GNN vectors
        self.user_proj  = self._make_proj(embed_dim, dropout)
        self.movie_proj = self._make_proj(embed_dim, dropout)

        if self.use_book_stream:
            self.book_proj = self._make_proj(embed_dim, dropout)
            self.book_aggregator = DomainSpecificAggregator(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim  =embed_dim * 4,
                dropout  =dropout
            )

        # --- Domain-Specific Aggregators ---
        self.movie_aggregator = DomainSpecificAggregator(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim  =embed_dim * 4,
            dropout  =dropout
        )

        # --- GNN-Anchored Condition Generator ---
        self.condition_generator = AttentionConditionGenerator(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim  =embed_dim * 4,
            dropout  =dropout
        )

    @staticmethod
    def _make_proj(embed_dim: int, dropout: float) -> nn.Sequential:
        """
        Projection block: normalize → expand → compress.

        LayerNorm(D) → Linear(D→2D) → GELU → Dropout → Linear(2D→D)

        embed_dim=256 : D=256, 2D=512
        embed_dim=128 : D=128, 2D=256
        embed_dim=64  : D=64,  2D=128
        """
        return nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(
        self,
        user_ids,
        movie_seq_ids,
        movie_mask,
        book_seq_ids=None,
        book_mask=None
    ):
        """
        Args:
            user_ids      : (B,)   — 0-indexed user IDs
            movie_seq_ids : (B, S) — 1-indexed movie history, 0=padding
            movie_mask    : (B, S) — True where padding
            book_seq_ids  : (B, S) — 1-indexed book history, 0=padding
                            (not used when use_book_stream=False)
            book_mask     : (B, S) — True where padding
                            (not used when use_book_stream=False)

        Returns:
            c_ud : (B, D) — L2-normalized diffusion condition vector
        """

        # ── Stream 1: GNN Anchor ─────────────────────────────────────────
        # user_ids are 0-indexed → map to 1-indexed embedding table with +1
        h_u_cross = self.user_proj(
            self.user_embedding(user_ids + 1)
        )                                                          # (B, D)

        # ── All-padding detection (before passing to aggregator) ─────────────────────
        # MultiheadAttention produces NaN when all tokens are masked.
        # Safe mask: treat at least 1 token as valid.
        # No semantic harm since the result will be overwritten by fallback.
        movie_empty     = movie_mask.all(dim=1)                   # (B,) bool
        safe_movie_mask = movie_mask.clone()
        safe_movie_mask[movie_empty, 0] = False

        # ── Stream 3: Short-Term Target Intent (Movie) ───────────────────
        h_u_target = self.movie_aggregator(
            self.movie_proj(self.movie_embedding(movie_seq_ids)),
            key_padding_mask=safe_movie_mask
        )                                                          # (B, D)
        h_u_target = torch.where(
            movie_empty.unsqueeze(1), h_u_cross, h_u_target
        )

        # ── Stream 2: Long-Term Source Intent (Book) ─────────────────────
        if self.use_book_stream:
            assert book_seq_ids is not None and book_mask is not None, (
                "book_seq_ids and book_mask required when use_book_stream=True"
            )
            book_empty     = book_mask.all(dim=1)
            safe_book_mask = book_mask.clone()
            safe_book_mask[book_empty, 0] = False

            h_u_source = self.book_aggregator(
                self.book_proj(self.book_embedding(book_seq_ids)),
                key_padding_mask=safe_book_mask
            )                                                      # (B, D)
            h_u_source = torch.where(
                book_empty.unsqueeze(1), h_u_cross, h_u_source
            )
        else:
            # Ablation: no book stream → GNN anchor acts as its own source
            h_u_source = h_u_cross                                 # (B, D)

        # ── Triple-Stream Fusion ─────────────────────────────────────────
        c_ud = self.condition_generator(
            h_u_cross =h_u_cross,
            h_u_source=h_u_source,
            h_u_target=h_u_target
        )                                                          # (B, D)

        # ── L2 Normalize → Diffusion model input ─────────────────────────
        return F.normalize(c_ud, p=2, dim=1)                      # (B, D)