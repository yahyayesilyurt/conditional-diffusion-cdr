import numpy as np

def calculate_metrics(top_k_indices, target_item_ids, k_list=[10, 100, 500]):
    """
    top_k_indices: Modelin önerdiği film indeksleri matrisi (batch_size, max(k_list))
    target_item_ids: Kullanıcının gerçek hedefi (batch_size,)
    k_list: Hesaplanacak farklı K değerleri listesi
    """
    top_k_indices = top_k_indices.cpu().numpy()
    target_item_ids = target_item_ids.cpu().numpy()
    
    # Her K değeri için hits ve ndcgs listelerini tutacak sözlük
    metrics = {k: {'hits': [], 'ndcgs': []} for k in k_list}
    
    for i in range(len(target_item_ids)):
        target = target_item_ids[i]
        predictions = top_k_indices[i] 
        
        # Hedef filmin indeksini bul (eğer varsa)
        rank_arr = np.where(predictions == target)[0]
        
        for k in k_list:
            if len(rank_arr) > 0 and rank_arr[0] < k:
                metrics[k]['hits'].append(1)
                rank = rank_arr[0] + 1
                metrics[k]['ndcgs'].append(1.0 / np.log2(rank + 1))
            else:
                metrics[k]['hits'].append(0)
                metrics[k]['ndcgs'].append(0)
                
    # Her K değeri için ortalamaları (mean) sözlük olarak döndür
    return {k: (np.mean(metrics[k]['hits']), np.mean(metrics[k]['ndcgs'])) for k in k_list}