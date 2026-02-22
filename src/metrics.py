import numpy as np

def calculate_metrics(top_k_indices, target_item_ids, k=10):
    """
    top_k_indices: Modelin önerdiği film indeksleri matrisi (batch_size, k) (PyTorch Tensor)
    target_item_ids: Kullanıcının test setinde GERÇEKTEN izlediği film indeksleri (batch_size,)
    """
    # Tensörleri numpy dizilerine çevir
    top_k_indices = top_k_indices.cpu().numpy()
    target_item_ids = target_item_ids.cpu().numpy()
    
    hits = []
    ndcgs = []
    
    for i in range(len(target_item_ids)):
        target = target_item_ids[i]
        predictions = top_k_indices[i]
        
        # Hedef film öneriler arasında var mı?
        if target in predictions:
            # HR@K
            hits.append(1)
            
            # NDCG@K
            # Hedefin kaçıncı sırada olduğunu bul (0-indeksli olduğu için +1 ekliyoruz)
            rank = np.where(predictions == target)[0][0] + 1
            # NDCG Formülü: 1 / log2(rank + 1)
            ndcg_score = 1.0 / np.log2(rank + 1)
            ndcgs.append(ndcg_score)
        else:
            hits.append(0)
            ndcgs.append(0)
            
    # Batch için ortalama değerleri döndür
    return np.mean(hits), np.mean(ndcgs)