import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from src.dataset import load_and_pad_embeddings, CrossDomainDataset
from src.e2e_wrapper import E2EWrapper
from src.diffusion_model import ConditionalDiffusion
from src.more_metrics import calculate_metrics


def train():
    # -------------------------------------------------------------------------
    # 1. HİPERPARAMETRELER
    # -------------------------------------------------------------------------
    device = (
        torch.device("cuda")  if torch.cuda.is_available()  else
        torch.device("mps")   if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Kullanılan cihaz: {device}")

    batch_size    = 2048
    learning_rate = 1e-3
    weight_decay  = 1e-6
    num_epochs    = 50
    embed_dim     = 32
    num_heads     = 4

    # -------------------------------------------------------------------------
    # 2. VERİ YÜKLEME
    # -------------------------------------------------------------------------
    print("GNN Embedding'leri yükleniyor...")
    # padded_book_embs yükleniyor ama E2EWrapper artık kullanmıyor.
    # Cross-domain sinyal book_emb yerine user_emb'den geliyor.
    # load_and_pad_embeddings() üç değer döndürdüğü için _ ile görmezden geliyoruz.
    padded_user_embs, _, padded_movie_embs = load_and_pad_embeddings(
        "assets/27_gat_embeddings.pt"
    )

    print("Eğitim verisi yükleniyor...")
    train_dataset = CrossDomainDataset(
        book_inter_path   ="inters/AmazonBooks.train.inter",
        movie_inter_path  ="inters/AmazonMovies.train.inter",
        book_mapping_path ="mappings/book_mapping.json",
        movie_mapping_path="mappings/movie_mapping.json",
        user_mapping_path ="mappings/user_mapping.json",
        max_seq_len=10
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )

    print("Doğrulama verisi yükleniyor...")
    valid_dataset = CrossDomainDataset(
        book_inter_path        ="inters/AmazonBooks.train.inter",
        movie_inter_path       ="inters/AmazonMovies.valid.inter",
        train_movie_inter_path ="inters/AmazonMovies.train.inter",
        book_mapping_path      ="mappings/book_mapping.json",
        movie_mapping_path     ="mappings/movie_mapping.json",
        user_mapping_path      ="mappings/user_mapping.json",
        max_seq_len=10,
        mode='valid'
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=512, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # -------------------------------------------------------------------------
    # 3. MODELLERİ BAŞLATMA
    # -------------------------------------------------------------------------
    e2e_model = E2EWrapper(
        padded_user_embs  =padded_user_embs,
        padded_movie_embs =padded_movie_embs,
        embed_dim         =embed_dim,
        num_heads         =num_heads,
    ).to(device)

    diffusion_model = ConditionalDiffusion(
        steps=100, item_dim=embed_dim, cond_dim=embed_dim, p_uncond=0.1
    ).to(device)

    all_parameters = (
        list(e2e_model.parameters()) +
        list(diffusion_model.parameters())
    )
    optimizer = optim.Adam(all_parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5
    )

    best_hr    = 0.0
    best_epoch = 0

    # -------------------------------------------------------------------------
    # 4. EĞİTİM DÖNGÜSÜ
    # -------------------------------------------------------------------------
    print("Eğitim başlıyor...")
    for epoch in range(num_epochs):
        e2e_model.train()
        diffusion_model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # book_seq dataset'ten geliyor ama bu mimaride kullanılmıyor.
            # Cross-domain sinyal user_emb üzerinden e2e_wrapper içinde sağlanıyor.
            user_ids         = batch['user_id'].to(device)         # (B,)
            movie_seq        = batch['movie_seq'].to(device)       # (B, 10)
            movie_mask       = batch['movie_mask'].to(device)      # (B, 10)
            target_movie_ids = batch['target_movie_id'].to(device) # (B,)

            optimizer.zero_grad()

            # c_ud: user_emb (cross-domain) + movie_seq (domain-specific) → koşul vektörü
            # e2e_wrapper içinde user_ids+1 offset uygulanıyor (PAD=0 için)
            c_ud = e2e_model(
                user_ids=user_ids,
                movie_seq_ids=movie_seq,
                movie_mask=movie_mask,
                target_domain='Movie'
            )

            # Hedef film embedding'i — target_movie_ids zaten 1-indexed (dataset'te +1)
            raw_target_embs  = e2e_model.movie_embedding(target_movie_ids)
            target_item_embs = F.normalize(raw_target_embs, p=2, dim=1)

            loss = diffusion_model(target_item_embs, c_ud)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss   = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        # -------------------------------------------------------------------------
        # 5. DOĞRULAMA
        # -------------------------------------------------------------------------
        if (epoch + 1) % 5 == 0:
            print(f"--- Epoch {epoch+1} Doğrulama ---")
            e2e_model.eval()
            diffusion_model.eval()

            all_hits  = []
            all_ndcgs = []

            with torch.no_grad():
                # Tüm film embedding'lerini normalize et — validation boyunca sabit
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

                    # sample() — 153K pool üzerinde tam KNN (single-stage)
                    # sim[:, 0] = -1e9 ile PAD bastırılıyor (diffusion_model içinde)
                    top_k_indices = diffusion_model.sample(
                        condition=c_ud,
                        target_domain_embs=all_movie_embs,
                        w=2.0, k=500
                    )

                    batch_metrics = calculate_metrics(top_k_indices, target_movie_ids, k_list=[10, 100, 500])
                    all_hits.append(batch_metrics)

            print(f"--- Epoch {epoch+1} Validasyon Sonuçları ---")
            for k in [10, 100, 500]:
                mean_hr = np.mean([b[k][0] for b in all_hits])
                mean_ndcg = np.mean([b[k][1] for b in all_hits])
                print(f"HR@{k:<3}: {mean_hr:.4f} | NDCG@{k:<3}: {mean_ndcg:.4f}")
            print("-" * 50)

            mean_hr_10 = np.mean([b[10][0] for b in all_hits])
            mean_ndcg_10 = np.mean([b[10][1] for b in all_hits])

            if mean_hr_10 > best_hr:
                best_hr    = mean_hr_10
                best_epoch = epoch + 1
                torch.save({
                    'epoch'               : best_epoch,
                    'e2e_state_dict'      : e2e_model.state_dict(),
                    'diffusion_state_dict': diffusion_model.state_dict(),
                    'hr'                  : best_hr,
                    'ndcg'               : mean_ndcg_10,
                }, "checkpoints/best_model.pt")
                print(f"✓ En iyi model kaydedildi (Epoch {best_epoch}, HR@10={best_hr:.4f})")

    print(f"\nEğitim tamamlandı. En iyi HR@10={best_hr:.4f} (Epoch {best_epoch})")


if __name__ == "__main__":
    train()