import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalTimeEmbeddings(nn.Module):
    """
    Standard sinusoidal time encoding (Transformer-style).
    Converts scalar timestep t into a D-dimensional vector.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        time : (B,) long tensor of timestep indices
        Returns: (B, dim)
        """
        device   = time.device
        half_dim = self.dim // 2
        # max(half_dim - 1, 1): prevents division by zero when dim<=2
        scale      = math.log(10000) / max(half_dim - 1, 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -scale)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class DenoisingMLP(nn.Module):
    """
    Noise predictor: ε_θ(x_t, c_ud, t) → ε̂

    Architecture: [x_t || c_ud || t_emb] → MLP → ε̂

    Intermediate layer sizes are proportional to item_dim — not hardcoded:
      input_dim = item_dim + cond_dim + time_dim
      hidden1   = item_dim * 4
      hidden2   = item_dim * 2

      embed_dim=256: 768 → 1024 → 512 → 512 → 256
      embed_dim=128: 384 →  512 → 256 → 256 → 128
      embed_dim=64:  192 →  256 → 128 → 128 →  64
    """

    def __init__(self, item_dim=256, cond_dim=256, time_dim=None, dropout=0.1):
        super().__init__()

        # if time_dim is not specified, default to item_dim
        time_dim = time_dim or item_dim

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),             
            nn.Linear(time_dim * 2, time_dim)
        )

        input_dim = item_dim + cond_dim + time_dim
        hidden1   = item_dim * 4
        hidden2   = item_dim * 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),

            nn.Linear(hidden2, item_dim)
        )

        self.item_dim = item_dim

    def forward(self, x_noisy, condition, time):
        """
        x_noisy   : (B, item_dim)
        condition : (B, cond_dim)
        time      : (B,) long

        Returns: (B, item_dim) — predicted noise ε̂
        """
        t_emb   = self.time_mlp(time)
        x_input = torch.cat([x_noisy, condition, t_emb], dim=-1)
        return self.net(x_input)


class ConditionalDiffusion(nn.Module):
    """
    Conditional Denoising Diffusion Probabilistic Model (DDPM)
    with Classifier-Free Guidance (CFG).

    embed_dim independent: when item_dim changes, all intermediate sizes adapt automatically.

    Training : forward(target_item_emb, condition) → MSE loss
    Inference: sample(condition, target_domain_embs) → top-k item indices
    """

    def __init__(
        self,
        steps=200,
        item_dim=256,
        cond_dim=256,
        dropout=0.1,
        p_uncond=0.1
    ):
        super().__init__()
        self.steps    = steps
        self.p_uncond = p_uncond
        self.item_dim = item_dim

        self.model = DenoisingMLP(
            item_dim=item_dim,
            cond_dim=cond_dim,
            time_dim=item_dim,   # time_dim is always equal to item_dim
            dropout =dropout
        )

        # ── Noise schedule (linear beta) ─────────────────────────────────
        betas               = torch.linspace(1e-4, 0.02, steps)
        alphas              = 1.0 - betas
        alphas_cumprod      = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.register_buffer('betas',                         betas)
        self.register_buffer('alphas',                        alphas)
        self.register_buffer('alphas_cumprod',                alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev',           alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod',           alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1.0 - alphas_cumprod).sqrt())
        self.register_buffer(
            'posterior_variance',
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def forward(self, target_item_emb, condition):
        """
        Args:
            target_item_emb : (B, D) — raw target item embedding
            condition       : (B, D) — L2-normalized c_ud from E2EWrapper

        Returns:
            loss : scalar MSE loss
        """
        # Normalize: ensures consistency with the space used for cosine similarity at inference
        target_item_emb = F.normalize(target_item_emb, p=2, dim=1)

        batch_size = target_item_emb.shape[0]
        device     = target_item_emb.device

        t     = torch.randint(0, self.steps, (batch_size,), device=device).long()
        noise = torch.randn_like(target_item_emb)

        # Forward diffusion: q(x_t | x_0)
        sqrt_ab        = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_1ab       = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        noisy_item_emb = sqrt_ab * target_item_emb + sqrt_1ab * noise

        # Classifier-Free Guidance: randomly zero out the condition during training
        if self.p_uncond > 0:
            uncond_mask = torch.rand(batch_size, device=device) < self.p_uncond
            condition   = torch.where(
                uncond_mask.unsqueeze(-1),
                torch.zeros_like(condition),
                condition
            )

        predicted_noise = self.model(noisy_item_emb, condition, t)
        return F.mse_loss(predicted_noise, noise)

    # ------------------------------------------------------------------
    # INFERENCE (shared reverse diffusion core)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _reverse_diffusion(self, x, condition, w):
        """
        Shared DDPM reverse loop with Classifier-Free Guidance.
        generate() and sample() call this method — no code duplication.

        Args:
            x         : (B, D) — initial Gaussian noise
            condition : (B, D) — L2-normalized c_ud
            w         : CFG weight

        Returns:
            x : (B, D) — denoised vector x_0
        """
        batch_size = condition.shape[0]
        device     = condition.device
        uncond     = torch.zeros_like(condition)

        for t in reversed(range(self.steps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            eps_cond   = self.model(x, condition, t_tensor)
            eps_uncond = self.model(x, uncond,    t_tensor)
            eps_cfg    = (1 + w) * eps_cond - w * eps_uncond

            sqrt_ab  = self.sqrt_alphas_cumprod[t]
            sqrt_1ab = self.sqrt_one_minus_alphas_cumprod[t]
            pred_x0  = (x - sqrt_1ab * eps_cfg) / sqrt_ab.clamp(min=1e-6)

            if t > 0:
                coef1 = (
                    self.alphas_cumprod_prev[t].sqrt() * self.betas[t]
                    / (1.0 - self.alphas_cumprod[t])
                )
                coef2 = (
                    self.alphas[t].sqrt() * (1.0 - self.alphas_cumprod_prev[t])
                    / (1.0 - self.alphas_cumprod[t])
                )
                x = (coef1 * pred_x0 + coef2 * x
                     + self.posterior_variance[t].sqrt() * torch.randn_like(x))
            else:
                x = pred_x0

        return x

    @torch.no_grad()
    def generate(self, condition, w=2.0):
        """
        Generates an ideal item vector via reverse diffusion.
        Used for embedding-space analysis / visualization.

        Args:
            condition : (B, D) — L2-normalized c_ud
            w         : CFG weight

        Returns:
            x : (B, D) — generated raw (un-normalized) vector
        """
        x = torch.randn(condition.shape[0], self.item_dim, device=condition.device)
        return self._reverse_diffusion(x, condition, w)

    @torch.no_grad()
    def sample(self, condition, target_domain_embs, watched_ids=None, w=2.0, k=10):
        """
        Generates top-k recommendations via reverse diffusion + cosine similarity.

        Args:
            condition          : (B, D) — L2-normalized c_ud
            target_domain_embs : (N, D) — all target embeddings (index 0 = PAD)
            watched_ids        : (B, S) optional — suppress already-watched items
            w                  : CFG weight
            k                  : number of recommendations

        Returns:
            top_k_indices : (B, k) — recommended item indices (1-indexed)
        """
        x = torch.randn(condition.shape[0], self.item_dim, device=condition.device)
        x = self._reverse_diffusion(x, condition, w)

        # Catalog matching via cosine similarity
        x_norm      = F.normalize(x,                  p=2, dim=1)  # (B, D)
        target_norm = F.normalize(target_domain_embs, p=2, dim=1)  # (N, D)
        sim         = torch.matmul(x_norm, target_norm.T)           # (B, N)

        # Suppress PAD token (index 0)
        sim[:, 0] = -1e9

        # Suppress already-interacted items
        if watched_ids is not None:
            sim.scatter_(1, watched_ids, -1e9)

        _, top_k_indices = torch.topk(sim, k=k, dim=1)
        return top_k_indices                                         # (B, k)