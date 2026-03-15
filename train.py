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
from src.metrics import calculate_metrics


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, warmup_steps, total_steps, eta_min):
    """
    Linear warm-up for `warmup_steps` steps, then Cosine Annealing decay.
    scheduler.step() is called after each batch (step-based).
    """
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps - warmup_steps, 1),
        eta_min=eta_min
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps]
    )


def train():

    # ── 0. Config ────────────────────────────────────────────────────────
    cfg   = load_config()
    paths = cfg['paths']
    tr    = cfg['training']
    dl    = cfg['dataloader']
    mdl   = cfg['model']
    val   = cfg['validation']

    use_book_stream = mdl['use_book_stream']

    # ── 1. Seed + Device ─────────────────────────────────────────────────
    set_seed(tr['seed'])

    device = (
        torch.device("cuda") if torch.cuda.is_available()        else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Using device: {device}")
    print(
        f"Mode: {'Triple-Stream (GNN + Book + Movie)' if use_book_stream else 'Ablation (GNN + Movie only)'}"
    )

    batch_size    = tr['batch_size']
    learning_rate = tr['learning_rate']
    weight_decay  = tr['weight_decay']
    num_epochs    = tr['num_epochs']
    embed_dim     = mdl['embed_dim']
    num_heads     = mdl['num_heads']
    dropout       = mdl['dropout']
    patience      = tr['early_stop_patience']

    # ── 2. Data ──────────────────────────────────────────────────────────
    print("Loading GNN embeddings...")
    padded_user_embs, padded_book_embs, padded_movie_embs = load_and_pad_embeddings(
        paths['embeddings']
    )

    print("Loading training data...")
    train_dataset = CrossDomainDataset(
        book_inter_path   =paths['inters']['book_train'],
        movie_inter_path  =paths['inters']['movie_train'],
        book_mapping_path =paths['mappings']['book'],
        movie_mapping_path=paths['mappings']['movie'],
        user_mapping_path =paths['mappings']['user'],
        max_seq_len       =tr['max_seq_len'],
        mode              ='train'
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size =batch_size,
        shuffle    =True,
        num_workers=dl['train_num_workers'],
        pin_memory =dl['train_pin_memory']
    )

    print("Loading validation data...")
    valid_dataset = CrossDomainDataset(
        # book_train intentionally used for valid/test:
        # book history is cross-domain context signal, not a separate eval set.
        book_inter_path        =paths['inters']['book_train'],
        movie_inter_path       =paths['inters']['movie_valid'],
        train_movie_inter_path =paths['inters']['movie_train'],
        book_mapping_path      =paths['mappings']['book'],
        movie_mapping_path     =paths['mappings']['movie'],
        user_mapping_path      =paths['mappings']['user'],
        max_seq_len            =tr['max_seq_len'],
        mode                   ='valid'
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size =dl['valid_batch_size'],
        shuffle    =False,
        num_workers=dl['valid_num_workers'],
        pin_memory =dl['valid_pin_memory']
    )

    # ── 3. Models ────────────────────────────────────────────────────────
    e2e_model = E2EWrapper(
        padded_user_embs  =padded_user_embs,
        padded_book_embs  =padded_book_embs,
        padded_movie_embs =padded_movie_embs,
        embed_dim         =embed_dim,
        num_heads         =num_heads,
        dropout           =dropout,
        use_book_stream   =use_book_stream
    ).to(device)

    diffusion_model = ConditionalDiffusion(
        steps   =mdl['diffusion']['steps'],
        item_dim=embed_dim,
        cond_dim=embed_dim,
        dropout =dropout,
        p_uncond=mdl['diffusion']['p_uncond']
    ).to(device)

    all_trainable = [
        p for p in
        list(e2e_model.parameters()) + list(diffusion_model.parameters())
        if p.requires_grad
    ]

    # AdamW: weight decay is correctly applied excluding bias/norm parameters
    optimizer = optim.AdamW(
        all_trainable,
        lr          =learning_rate,
        weight_decay=weight_decay
    )

    # Step-based: linear warm-up → cosine annealing
    total_steps  = num_epochs * len(train_loader)
    warmup_steps = mdl['scheduler']['warmup_steps']
    scheduler    = build_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps =total_steps,
        eta_min     =mdl['scheduler']['eta_min']
    )

    print(
        f"Total steps: {total_steps} | "
        f"Warm-up steps: {warmup_steps} | "
        f"Steps/epoch: {len(train_loader)}"
    )

    best_hr           = 0.0
    best_epoch        = 0
    epochs_no_improve = 0

    # ── 4. Training Loop ─────────────────────────────────────────────────
    print("Training started...")
    for epoch in range(num_epochs):
        e2e_model.train()
        diffusion_model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            user_ids         = batch['user_id'].to(device)
            movie_seq        = batch['movie_seq'].to(device)
            movie_mask       = batch['movie_mask'].to(device)
            target_movie_ids = batch['target_movie_id'].to(device)

            optimizer.zero_grad()

            if use_book_stream:
                book_seq  = batch['book_seq'].to(device)
                book_mask = batch['book_mask'].to(device)
                c_ud = e2e_model(
                    user_ids     =user_ids,
                    movie_seq_ids=movie_seq,
                    movie_mask   =movie_mask,
                    book_seq_ids =book_seq,
                    book_mask    =book_mask
                )
            else:
                c_ud = e2e_model(
                    user_ids     =user_ids,
                    movie_seq_ids=movie_seq,
                    movie_mask   =movie_mask
                )

            # Raw target embedding — diffusion_model.forward() normalizes internally
            raw_target_embs = e2e_model.movie_embedding(target_movie_ids)
            loss = diffusion_model(raw_target_embs, c_ud)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=tr['grad_clip_norm'])
            optimizer.step()
            scheduler.step()   # step-based update

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss   = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e}"
        )

        # ── 5. Validation ────────────────────────────────────────────────
        if (epoch + 1) % tr['validation_freq'] == 0:
            print(f"--- Epoch {epoch+1} Validation ---")
            e2e_model.eval()
            diffusion_model.eval()

            all_hrs   = []
            all_ndcgs = []

            with torch.no_grad():
                # Normalize all movie embeddings once
                all_movie_embs = F.normalize(
                    e2e_model.movie_embedding.weight, p=2, dim=1
                )                                                   # (N+1, D)

                for batch in tqdm(valid_loader, desc="Validation"):
                    user_ids         = batch['user_id'].to(device)
                    movie_seq        = batch['movie_seq'].to(device)
                    movie_mask       = batch['movie_mask'].to(device)
                    target_movie_ids = batch['target_movie_id'].to(device)

                    if use_book_stream:
                        book_seq  = batch['book_seq'].to(device)
                        book_mask = batch['book_mask'].to(device)
                        c_ud = e2e_model(
                            user_ids     =user_ids,
                            movie_seq_ids=movie_seq,
                            movie_mask   =movie_mask,
                            book_seq_ids =book_seq,
                            book_mask    =book_mask
                        )
                    else:
                        c_ud = e2e_model(
                            user_ids     =user_ids,
                            movie_seq_ids=movie_seq,
                            movie_mask   =movie_mask
                        )

                    top_k_indices = diffusion_model.sample(
                        condition         =c_ud,
                        target_domain_embs=all_movie_embs,
                        watched_ids       =movie_seq,   # suppress already-watched items
                        w                 =val['cfg_w'],
                        k                 =val['top_k']
                    )

                    batch_hr, batch_ndcg = calculate_metrics(
                        top_k_indices, target_movie_ids, k=val['top_k']
                    )
                    all_hrs.append(batch_hr)
                    all_ndcgs.append(batch_ndcg)

            mean_hr   = float(np.mean(all_hrs))
            mean_ndcg = float(np.mean(all_ndcgs))
            print(f"Validation HR@{val['top_k']}:   {mean_hr:.4f}")
            print(f"Validation NDCG@{val['top_k']}: {mean_ndcg:.4f}")

            # ── Early Stopping ───────────────────────────────────────────
            if mean_hr > best_hr:
                best_hr           = mean_hr
                best_epoch        = epoch + 1
                epochs_no_improve = 0
                torch.save({
                    'epoch'               : best_epoch,
                    'e2e_state_dict'      : e2e_model.state_dict(),
                    'diffusion_state_dict': diffusion_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'hr'                  : best_hr,
                    'ndcg'               : mean_ndcg,
                    'use_book_stream'     : use_book_stream,
                }, paths['checkpoints']['best_model'])
                print(
                    f"✓ Best model saved "
                    f"(Epoch {best_epoch}, HR@{val['top_k']}={best_hr:.4f})"
                )
            else:
                epochs_no_improve += 1
                print(
                    f"  No improvement: {epochs_no_improve}/{patience} "
                    f"(best HR@{val['top_k']}={best_hr:.4f} @ Epoch {best_epoch})"
                )

            print("-" * 50)

            if epochs_no_improve >= patience:
                print(
                    f"\nEarly stopping triggered — "
                    f"{patience} validations without improvement."
                )
                break

    print(
        f"\nTraining complete. "
        f"Best HR@{val['top_k']}={best_hr:.4f} (Epoch {best_epoch})"
    )


if __name__ == "__main__":
    train()