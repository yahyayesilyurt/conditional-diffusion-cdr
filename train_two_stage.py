import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.config_loader import load_config
from src.dataset import load_and_pad_embeddings, CrossDomainDataset
from src.e2e_wrapper import E2EWrapper
from src.diffusion_model import ConditionalDiffusion
from src.metrics import calculate_metrics


# ---------------------------------------------------------------------------
# Two-Stage Retrieval
# ---------------------------------------------------------------------------

@torch.no_grad()
def two_stage_retrieve(
    user_ids:         torch.Tensor,        # (B,)
    c_ud:             torch.Tensor,        # (B, D)
    diffusion_model:  ConditionalDiffusion,
    user_emb_matrix:  torch.Tensor,        # (N_users+1, D) — padded
    movie_emb_matrix: torch.Tensor,        # (N_movies+1, D) — padded, L2 normalize
    recall_k:         int   = 1000,
    top_k:            int   = 10,
    cfg_w:            float = 2.0,
) -> torch.Tensor:
    """
    Stage 1 — GNN Recall (recall_k):
        cosine similarity between user embedding and all movies.
        PAD (index 0) is suppressed.

    Stage 2 — Diffusion Rerank (recall_k → top_k):
        ideal vector is generated with diffusion_model.generate().
        KNN is performed over only recall_k candidates.

    Returns: (B, top_k) — movie_emb_matrix indices (1-indexed, PAD excluded)
    """
    device = user_ids.device

    # -----------------------------------------------------------------------
    # Stage 1: Fast recall with GNN user embedding
    # -----------------------------------------------------------------------
    # user_ids+1 is applied in e2e_wrapper — same offset used here
    real_user_idx = (user_ids + 1).clamp(0, user_emb_matrix.shape[0] - 1)
    user_vecs     = F.normalize(user_emb_matrix[real_user_idx], p=2, dim=1)  # (B, D)

    stage1_sim       = torch.matmul(user_vecs, movie_emb_matrix.T)  # (B, N+1)
    stage1_sim[:, 0] = -1e9  # suppress PAD

    _, recall_indices = torch.topk(
        stage1_sim, k=min(recall_k, stage1_sim.shape[1] - 1), dim=1
    )  # (B, recall_k)

    # -----------------------------------------------------------------------
    # Stage 2: Rerank with Diffusion
    # -----------------------------------------------------------------------
    candidate_embs = movie_emb_matrix[recall_indices]  # (B, recall_k, D)

    generated = diffusion_model.generate(c_ud, w=cfg_w)          # (B, D)
    gen_norm  = F.normalize(generated, p=2, dim=1).unsqueeze(1)  # (B, 1, D)

    stage2_sim          = (gen_norm * candidate_embs).sum(dim=-1)  # (B, recall_k)
    _, local_top_k      = torch.topk(stage2_sim, k=min(top_k, recall_k), dim=1)
    top_k_global        = recall_indices.gather(1, local_top_k)    # (B, top_k)

    return top_k_global


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    cfg      = load_config()
    paths    = cfg['paths']
    tr       = cfg['training']
    dl       = cfg['dataloader']
    mdl      = cfg['model']
    ts       = cfg['two_stage']

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    # Hyperparameters
    batch_size   = tr['batch_size']
    lr           = tr['learning_rate']
    weight_decay = tr['weight_decay']
    num_epochs   = tr['num_epochs']
    embed_dim    = mdl['embed_dim']
    num_heads    = mdl['num_heads']
    recall_k     = ts['recall_k']
    cfg_w        = ts['cfg_w']

    # Data
    print("Loading embeddings...")
    padded_user_embs, _, padded_movie_embs = load_and_pad_embeddings(
        paths['embeddings']
    )

    train_dataset = CrossDomainDataset(
        book_inter_path   =paths['inters']['book_train'],
        movie_inter_path  =paths['inters']['movie_train'],
        book_mapping_path =paths['mappings']['book'],
        movie_mapping_path=paths['mappings']['movie'],
        user_mapping_path =paths['mappings']['user'],
        max_seq_len=tr['max_seq_len'],
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=dl['train_num_workers'], pin_memory=dl['train_pin_memory']
    )

    valid_dataset = CrossDomainDataset(
        book_inter_path        =paths['inters']['book_train'],
        movie_inter_path       =paths['inters']['movie_valid'],
        train_movie_inter_path =paths['inters']['movie_train'],
        book_mapping_path      =paths['mappings']['book'],
        movie_mapping_path     =paths['mappings']['movie'],
        user_mapping_path      =paths['mappings']['user'],
        max_seq_len=tr['max_seq_len'],
        mode='valid',
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=dl['valid_batch_size'], shuffle=False,
        num_workers=dl['valid_num_workers'], pin_memory=dl['valid_pin_memory']
    )

    # Models
    e2e_model = E2EWrapper(
        padded_user_embs =padded_user_embs,
        padded_movie_embs=padded_movie_embs,
        embed_dim=embed_dim,
        num_heads=num_heads,
    ).to(device)

    diffusion_model = ConditionalDiffusion(
        steps=mdl['diffusion']['steps'],
        item_dim=embed_dim,
        cond_dim=embed_dim,
        p_uncond=mdl['diffusion']['p_uncond']
    ).to(device)

    all_params = list(e2e_model.parameters()) + list(diffusion_model.parameters())
    optimizer  = torch.optim.Adam(all_params, lr=lr, weight_decay=weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=mdl['scheduler']['eta_min']
    )

    best_hr    = 0.0
    best_epoch = 0

    print("Training started...")
    for epoch in range(num_epochs):
        e2e_model.train()
        diffusion_model.train()
        epoch_loss = 0.0

        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in bar:
            user_ids         = batch['user_id'].to(device)
            movie_seq        = batch['movie_seq'].to(device)
            movie_mask       = batch['movie_mask'].to(device)
            target_movie_ids = batch['target_movie_id'].to(device)

            optimizer.zero_grad()

            c_ud       = e2e_model(user_ids, movie_seq, movie_mask)
            raw_target = e2e_model.movie_embedding(target_movie_ids)
            target_emb = F.normalize(raw_target, p=2, dim=1)

            loss = diffusion_model(target_emb, c_ud)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, tr['grad_clip_norm'])
            optimizer.step()

            epoch_loss += loss.item()
            bar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss={avg_loss:.4f} "
              f"LR={optimizer.param_groups[0]['lr']:.6f}")

        # -------------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------------
        if (epoch + 1) % tr['validation_freq'] == 0:
            e2e_model.eval()
            diffusion_model.eval()

            # Normalize all movie embeddings — kept fixed throughout validation
            all_movie_embs_norm = F.normalize(
                e2e_model.movie_embedding.weight, p=2, dim=1
            )  # (N_movies+1, D)

            all_hits  = []
            all_ndcgs = []

            with torch.no_grad():
                for batch in tqdm(valid_loader, desc="Validation"):
                    user_ids         = batch['user_id'].to(device)
                    movie_seq        = batch['movie_seq'].to(device)
                    movie_mask       = batch['movie_mask'].to(device)
                    target_movie_ids = batch['target_movie_id'].to(device)

                    c_ud = e2e_model(user_ids, movie_seq, movie_mask)

                    top_k_indices = two_stage_retrieve(
                        user_ids         =user_ids,
                        c_ud             =c_ud,
                        diffusion_model  =diffusion_model,
                        user_emb_matrix  =e2e_model.user_embedding.weight,
                        movie_emb_matrix =all_movie_embs_norm,
                        recall_k         =recall_k,
                        top_k            =ts['top_k'],
                        cfg_w            =cfg_w,
                    )

                    batch_hr, batch_ndcg = calculate_metrics(
                        top_k_indices, target_movie_ids, k=10
                    )
                    all_hits.append(batch_hr)
                    all_ndcgs.append(batch_ndcg)

            mean_hr   = np.mean(all_hits)
            mean_ndcg = np.mean(all_ndcgs)
            print(f"[Epoch {epoch+1}] HR@10={mean_hr:.4f} | NDCG@10={mean_ndcg:.4f}")

            if mean_hr > best_hr:
                best_hr    = mean_hr
                best_epoch = epoch + 1
                torch.save({
                    'epoch'               : best_epoch,
                    'e2e_state_dict'      : e2e_model.state_dict(),
                    'diffusion_state_dict': diffusion_model.state_dict(),
                    'hr'                  : best_hr,
                    'ndcg'               : mean_ndcg,
                }, paths['checkpoints']['best_model_two_stage'])
                print(f"✓ Saved (Epoch {best_epoch}, HR@10={best_hr:.4f})")

    print(f"\nTraining complete. Best HR@10={best_hr:.4f} (Epoch {best_epoch})")


if __name__ == "__main__":
    train()