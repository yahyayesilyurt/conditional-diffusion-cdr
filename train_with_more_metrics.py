import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from src.config_loader import load_config
from src.dataset import load_and_pad_embeddings, CrossDomainDataset
from src.e2e_wrapper import E2EWrapper
from src.diffusion_model import ConditionalDiffusion
from src.more_metrics import calculate_metrics


def train():
    # -------------------------------------------------------------------------
    # 0. CONFIGURATION
    # -------------------------------------------------------------------------
    cfg        = load_config()
    paths      = cfg['paths']
    tr         = cfg['training']
    dl         = cfg['dataloader']
    mdl        = cfg['model']
    val        = cfg['validation']

    # -------------------------------------------------------------------------
    # 1. HYPERPARAMETERS
    # -------------------------------------------------------------------------
    device = (
        torch.device("cuda")  if torch.cuda.is_available()  else
        torch.device("mps")   if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Using device: {device}")

    batch_size    = tr['batch_size']
    learning_rate = tr['learning_rate']
    weight_decay  = tr['weight_decay']
    num_epochs    = tr['num_epochs']
    embed_dim     = mdl['embed_dim']
    num_heads     = mdl['num_heads']

    # -------------------------------------------------------------------------
    # 2. DATA LOADING
    # -------------------------------------------------------------------------
    print("Loading GNN embeddings...")
    # padded_book_embs is loaded but E2EWrapper no longer uses it.
    # Cross-domain signal comes from user_emb instead of book_emb.
    # load_and_pad_embeddings() returns three values; the second is ignored with _.
    padded_user_embs, _, padded_movie_embs = load_and_pad_embeddings(
        paths['embeddings']
    )

    print("Loading training data...")
    train_dataset = CrossDomainDataset(
        book_inter_path   =paths['inters']['book_train'],
        movie_inter_path  =paths['inters']['movie_train'],
        book_mapping_path =paths['mappings']['book'],
        movie_mapping_path=paths['mappings']['movie'],
        user_mapping_path =paths['mappings']['user'],
        max_seq_len=tr['max_seq_len']
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=dl['train_num_workers'], pin_memory=dl['train_pin_memory']
    )

    print("Loading validation data...")
    valid_dataset = CrossDomainDataset(
        book_inter_path        =paths['inters']['book_train'],
        movie_inter_path       =paths['inters']['movie_valid'],
        train_movie_inter_path =paths['inters']['movie_train'],
        book_mapping_path      =paths['mappings']['book'],
        movie_mapping_path     =paths['mappings']['movie'],
        user_mapping_path      =paths['mappings']['user'],
        max_seq_len=tr['max_seq_len'],
        mode='valid'
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=dl['valid_batch_size'], shuffle=False,
        num_workers=dl['valid_num_workers'], pin_memory=dl['valid_pin_memory']
    )

    # -------------------------------------------------------------------------
    # 3. MODEL INITIALIZATION
    # -------------------------------------------------------------------------
    e2e_model = E2EWrapper(
        padded_user_embs  =padded_user_embs,
        padded_movie_embs =padded_movie_embs,
        embed_dim         =embed_dim,
        num_heads         =num_heads,
    ).to(device)

    diffusion_model = ConditionalDiffusion(
        steps=mdl['diffusion']['steps'],
        item_dim=embed_dim,
        cond_dim=embed_dim,
        p_uncond=mdl['diffusion']['p_uncond']
    ).to(device)

    all_parameters = (
        list(e2e_model.parameters()) +
        list(diffusion_model.parameters())
    )
    optimizer = optim.Adam(all_parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=mdl['scheduler']['eta_min']
    )

    best_hr    = 0.0
    best_epoch = 0

    # -------------------------------------------------------------------------
    # 4. TRAINING LOOP
    # -------------------------------------------------------------------------
    print("Training started...")
    for epoch in range(num_epochs):
        e2e_model.train()
        diffusion_model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # book_seq comes from the dataset but is not used in this architecture.
            # Cross-domain signal is provided via user_emb inside e2e_wrapper.
            user_ids         = batch['user_id'].to(device)         # (B,)
            movie_seq        = batch['movie_seq'].to(device)       # (B, 10)
            movie_mask       = batch['movie_mask'].to(device)      # (B, 10)
            target_movie_ids = batch['target_movie_id'].to(device) # (B,)

            optimizer.zero_grad()

            # c_ud: user_emb (cross-domain) + movie_seq (domain-specific) → condition vector
            # user_ids+1 offset is applied inside e2e_wrapper (PAD=0 convention)
            c_ud = e2e_model(
                user_ids=user_ids,
                movie_seq_ids=movie_seq,
                movie_mask=movie_mask,
                target_domain='Movie'
            )

            # Target movie embedding — target_movie_ids are already 1-indexed (+1 applied in dataset)
            raw_target_embs  = e2e_model.movie_embedding(target_movie_ids)
            target_item_embs = F.normalize(raw_target_embs, p=2, dim=1)

            loss = diffusion_model(target_item_embs, c_ud)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=tr['grad_clip_norm'])
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss   = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        # -------------------------------------------------------------------------
        # 5. VALIDATION
        # -------------------------------------------------------------------------
        if (epoch + 1) % tr['validation_freq'] == 0:
            print(f"--- Epoch {epoch+1} Validation ---")
            e2e_model.eval()
            diffusion_model.eval()

            all_hits  = []
            all_ndcgs = []

            with torch.no_grad():
                # Normalize all movie embeddings — kept fixed throughout validation
                all_movie_embs = F.normalize(
                    e2e_model.movie_embedding.weight, p=2, dim=1
                )  # (N_movies+1, D)

                for batch in tqdm(valid_loader, desc="Validation"):
                    user_ids         = batch['user_id'].to(device)
                    movie_seq        = batch['movie_seq'].to(device)
                    movie_mask       = batch['movie_mask'].to(device)
                    target_movie_ids = batch['target_movie_id'].to(device)

                    c_ud = e2e_model(
                        user_ids=user_ids,
                        movie_seq_ids=movie_seq,
                        movie_mask=movie_mask,
                        target_domain='Movie'
                    )

                    # sample() — full KNN over the data pool (single-stage)
                    # sim[:, 0] is suppressed with -1e9 to mask PAD (inside diffusion_model)
                    top_k_indices = diffusion_model.sample(
                        condition=c_ud,
                        target_domain_embs=all_movie_embs,
                        w=val['cfg_w'], k=max(val['k_list'])
                    )

                    batch_metrics = calculate_metrics(top_k_indices, target_movie_ids, k_list=val['k_list'])
                    all_hits.append(batch_metrics)

            print(f"--- Epoch {epoch+1} Validation Results ---")
            for k in val['k_list']:
                mean_hr = np.mean([b[k][0] for b in all_hits])
                mean_ndcg = np.mean([b[k][1] for b in all_hits])
                print(f"HR@{k:<3}: {mean_hr:.4f} | NDCG@{k:<3}: {mean_ndcg:.4f}")
            print("-" * 50)

            mean_hr_10 = np.mean([b[val['k_list'][0]][0] for b in all_hits])
            mean_ndcg_10 = np.mean([b[val['k_list'][0]][1] for b in all_hits])

            if mean_hr_10 > best_hr:
                best_hr    = mean_hr_10
                best_epoch = epoch + 1
                torch.save({
                    'epoch'               : best_epoch,
                    'e2e_state_dict'      : e2e_model.state_dict(),
                    'diffusion_state_dict': diffusion_model.state_dict(),
                    'hr'                  : best_hr,
                    'ndcg'               : mean_ndcg_10,
                }, paths['checkpoints']['best_model'])
                print(f"✓ Best model saved (Epoch {best_epoch}, HR@{val['k_list'][0]}={best_hr:.4f})")

    print(f"\nTraining complete. Best HR@10={best_hr:.4f} (Epoch {best_epoch})")


if __name__ == "__main__":
    train()