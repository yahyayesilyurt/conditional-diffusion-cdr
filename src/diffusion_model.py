import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalTimeEmbeddings(nn.Module):
    """Standard Transformer sinusoidal time encoding that converts a timestep (t) into a vector the model can consume."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device   = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class DenoisingMLP(nn.Module):
    """MLP that predicts the noise (epsilon)."""

    def __init__(self, item_dim=128, cond_dim=128, time_dim=128):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim)
        )

        input_dim = item_dim + cond_dim + time_dim  # 128+128+128 = 384

        # (128 Sizes Available): 384 -> 512 -> 256 -> 256 -> 128
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),

            nn.Linear(256, item_dim) # Out: 128
        )

        self.item_dim = item_dim

    def forward(self, x_noisy, condition, time):
        t_emb   = self.time_mlp(time)
        x_input = torch.cat([x_noisy, condition, t_emb], dim=-1)
        return self.net(x_input)


class ConditionalDiffusion(nn.Module):

    def __init__(self, steps=100, item_dim=128, cond_dim=128, p_uncond=0.1):
        super().__init__()
        self.steps    = steps
        self.p_uncond = p_uncond
        self.item_dim = item_dim

        self.model = DenoisingMLP(item_dim=item_dim, cond_dim=cond_dim)

        betas               = torch.linspace(1e-4, 0.02, steps)
        alphas              = 1. - betas
        alphas_cumprod      = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.register_buffer('betas',                         betas)
        self.register_buffer('alphas',                        alphas)
        self.register_buffer('alphas_cumprod',                alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev',           alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod',           torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer(
            'posterior_variance',
            betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        )

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def forward(self, target_item_emb, condition):
        """
        target_item_emb : (B, D) — L2-normalized target movie embedding
        condition       : (B, D) — L2-normalized condition vector (c_ud)
        """
        batch_size = target_item_emb.shape[0]
        device     = target_item_emb.device

        t     = torch.randint(0, self.steps, (batch_size,), device=device).long()
        noise = torch.randn_like(target_item_emb)

        sqrt_ab  = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_1ab = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        noisy_item_emb = sqrt_ab * target_item_emb + sqrt_1ab * noise

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
    # INFERENCE
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, condition, w=2.0):
        """
        Generates an ideal movie vector via reverse diffusion.

        condition : (B, D) — L2-normalized condition vector
        w         : CFG weight
        Returns   : (B, D) — generated raw vector (not normalized)
        """
        batch_size = condition.shape[0]
        device     = condition.device

        x      = torch.randn(batch_size, self.item_dim, device=device)
        uncond = torch.zeros_like(condition)

        for t in reversed(range(self.steps)):
            t_tensor   = torch.full((batch_size,), t, device=device, dtype=torch.long)
            eps_cond   = self.model(x, condition, t_tensor)
            eps_uncond = self.model(x, uncond,    t_tensor)
            eps_cfg    = (1 + w) * eps_cond - w * eps_uncond

            sqrt_ab  = self.sqrt_alphas_cumprod[t]
            sqrt_1ab = self.sqrt_one_minus_alphas_cumprod[t]
            pred_x0  = (x - sqrt_1ab * eps_cfg) / sqrt_ab.clamp(min=1e-6)

            if t > 0:
                coef1 = (self.alphas_cumprod_prev[t].sqrt() * self.betas[t]) \
                        / (1. - self.alphas_cumprod[t])
                coef2 = (self.alphas[t].sqrt() * (1. - self.alphas_cumprod_prev[t])) \
                        / (1. - self.alphas_cumprod[t])
                x = coef1 * pred_x0 + coef2 * x \
                    + self.posterior_variance[t].sqrt() * torch.randn_like(x)
            else:
                x = pred_x0

        return x

    @torch.no_grad()
    def sample(self, condition, target_domain_embs, w=2.0, k=10):
        """
        condition          : (B, D) — L2-normalized condition vector
        target_domain_embs : (N, D) — embeddings for all movies (including PAD)
        w                  : CFG weight
        k                  : number of movies to recommend
        """
        batch_size = condition.shape[0]
        device     = condition.device

        x      = torch.randn(batch_size, self.item_dim, device=device)
        uncond = torch.zeros_like(condition)

        for t in reversed(range(self.steps)):
            t_tensor   = torch.full((batch_size,), t, device=device, dtype=torch.long)
            eps_cond   = self.model(x, condition, t_tensor)
            eps_uncond = self.model(x, uncond,    t_tensor)
            eps_cfg    = (1 + w) * eps_cond - w * eps_uncond

            sqrt_ab  = self.sqrt_alphas_cumprod[t]
            sqrt_1ab = self.sqrt_one_minus_alphas_cumprod[t]
            pred_x0  = (x - sqrt_1ab * eps_cfg) / sqrt_ab.clamp(min=1e-6)

            if t > 0:
                coef1 = (self.alphas_cumprod_prev[t].sqrt() * self.betas[t]) \
                        / (1. - self.alphas_cumprod[t])
                coef2 = (self.alphas[t].sqrt() * (1. - self.alphas_cumprod_prev[t])) \
                        / (1. - self.alphas_cumprod[t])
                x = coef1 * pred_x0 + coef2 * x \
                    + self.posterior_variance[t].sqrt() * torch.randn_like(x)
            else:
                x = pred_x0

        x_norm      = F.normalize(x,                  p=2, dim=1)
        target_norm = F.normalize(target_domain_embs, p=2, dim=1)

        sim          = torch.matmul(x_norm, target_norm.T)
        sim[:, 0]    = -1e9  # suppress PAD index

        _, top_k_indices = torch.topk(sim, k=k, dim=1)
        return top_k_indices