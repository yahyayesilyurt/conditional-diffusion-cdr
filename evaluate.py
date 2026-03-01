import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from tqdm import tqdm
from datetime import datetime

from src.config_loader import load_config
from src.dataset import load_and_pad_embeddings, CrossDomainDataset
from src.e2e_wrapper import E2EWrapper
from src.diffusion_model import ConditionalDiffusion
from src.metrics import calculate_metrics


def evaluate():
    # -------------------------------------------------------------------------
    # 0. CONFIGURATION
    # -------------------------------------------------------------------------
    cfg   = load_config()
    paths = cfg['paths']
    tr    = cfg['training']
    dl    = cfg['dataloader']
    mdl   = cfg['model']
    val   = cfg['validation']

    device = (
        torch.device("cuda")  if torch.cuda.is_available() else
        torch.device("mps")   if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Using device: {device}")

    embed_dim = mdl['embed_dim']
    num_heads = mdl['num_heads']

    # -------------------------------------------------------------------------
    # 1. CHECKPOINT LOADING
    # -------------------------------------------------------------------------
    checkpoint_path = paths['checkpoints']['best_model']
    assert os.path.exists(checkpoint_path), \
        f"Checkpoint not found: {checkpoint_path}"

    print(f"Checkpoint loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"  Saved epoch : {checkpoint['epoch']}")
    print(f"  Validation HR@10 : {checkpoint['hr']:.4f}")
    print(f"  Validation NDCG@10: {checkpoint['ndcg']:.4f}")

    # -------------------------------------------------------------------------
    # 2. DATA LOADING
    # -------------------------------------------------------------------------
    print("\nGNN embeddings loading...")
    padded_user_embs, _, padded_movie_embs = load_and_pad_embeddings(
        paths['embeddings']
    )

    print("Test data loading...")
    test_dataset = CrossDomainDataset(
        book_inter_path        =paths['inters']['book_train'],
        movie_inter_path       =paths['inters']['movie_test'],   # ← test.inter
        train_movie_inter_path =paths['inters']['movie_train'],  # train for history
        book_mapping_path      =paths['mappings']['book'],
        movie_mapping_path     =paths['mappings']['movie'],
        user_mapping_path      =paths['mappings']['user'],
        max_seq_len=tr['max_seq_len'],
        mode='test'
    )
    test_loader = DataLoader(
        test_dataset, batch_size=dl['valid_batch_size'], shuffle=False,
        num_workers=dl['valid_num_workers'], pin_memory=dl['valid_pin_memory']
    )

    # -------------------------------------------------------------------------
    # 3. MODEL LOADING
    # -------------------------------------------------------------------------
    e2e_model = E2EWrapper(
        padded_user_embs  =padded_user_embs,
        padded_movie_embs =padded_movie_embs,
        embed_dim         =embed_dim,
        num_heads         =num_heads,
    ).to(device)

    diffusion_model = ConditionalDiffusion(
        steps    =mdl['diffusion']['steps'],
        item_dim =embed_dim,
        cond_dim =embed_dim,
        p_uncond =mdl['diffusion']['p_uncond']
    ).to(device)

    e2e_model.load_state_dict(checkpoint['e2e_state_dict'])
    diffusion_model.load_state_dict(checkpoint['diffusion_state_dict'])

    e2e_model.eval()
    diffusion_model.eval()
    print("Model weights loaded.")

    # -------------------------------------------------------------------------
    # 4. TEST EVALUATION
    # -------------------------------------------------------------------------
    print("\nTest evaluation starting...")
    all_hits  = []
    all_ndcgs = []

    with torch.no_grad():
        all_movie_embs = F.normalize(
            e2e_model.movie_embedding.weight, p=2, dim=1
        )  # (N_movies+1, D)

        for batch in tqdm(test_loader, desc="Test"):
            user_ids         = batch['user_id'].to(device)
            movie_seq        = batch['movie_seq'].to(device)
            movie_mask       = batch['movie_mask'].to(device)
            target_movie_ids = batch['target_movie_id'].to(device)

            c_ud = e2e_model(
                user_ids      =user_ids,
                movie_seq_ids =movie_seq,
                movie_mask    =movie_mask,
                target_domain ='Movie'
            )

            top_k_indices = diffusion_model.sample(
                condition          =c_ud,
                target_domain_embs =all_movie_embs,
                w=val['cfg_w'], k=val['top_k']
            )

            batch_hr, batch_ndcg = calculate_metrics(
                top_k_indices, target_movie_ids, k=10
            )
            all_hits.append(batch_hr)
            all_ndcgs.append(batch_ndcg)

    test_hr   = float(np.mean(all_hits))
    test_ndcg = float(np.mean(all_ndcgs))

    # -------------------------------------------------------------------------
    # 5. RESULTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    print("=" * 50)
    print(f"HR@10:   {test_hr:.4f}")
    print(f"NDCG@10: {test_ndcg:.4f}")
    print("=" * 50)
    print(f"\nComparison:")
    print(f"  Validation HR@10:   {checkpoint['hr']:.4f}")
    print(f"  Test       HR@10:   {test_hr:.4f}")

    # Save results to JSON
    os.makedirs("results", exist_ok=True)
    results = {
        "timestamp"        : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint"       : checkpoint_path,
        "best_val_epoch"   : checkpoint['epoch'],
        "validation_hr10"  : checkpoint['hr'],
        "validation_ndcg10": checkpoint['ndcg'],
        "test_hr10"        : test_hr,
        "test_ndcg10"      : test_ndcg,
        "embed_dim"        : embed_dim,
        "diffusion_steps"  : mdl['diffusion']['steps'],
        "cfg_w"            : val['cfg_w'],
    }
    output_path = "results/test_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    evaluate()