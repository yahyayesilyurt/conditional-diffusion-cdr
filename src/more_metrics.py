import numpy as np

def calculate_metrics(top_k_indices, target_item_ids, k_list=[10, 100, 500]):
    """
    top_k_indices: Matrix of movie indices recommended by the model (batch_size, max(k_list))
    target_item_ids: Ground-truth target for each user (batch_size,)
    k_list: List of K values for which metrics are computed
    """
    top_k_indices = top_k_indices.cpu().numpy()
    target_item_ids = target_item_ids.cpu().numpy()
    
    # Dictionary to hold hits and ndcgs lists for each K value
    metrics = {k: {'hits': [], 'ndcgs': []} for k in k_list}
    
    for i in range(len(target_item_ids)):
        target = target_item_ids[i]
        predictions = top_k_indices[i] 
        
        # Find the rank of the target movie (if present)
        rank_arr = np.where(predictions == target)[0]
        
        for k in k_list:
            if len(rank_arr) > 0 and rank_arr[0] < k:
                metrics[k]['hits'].append(1)
                rank = rank_arr[0] + 1
                metrics[k]['ndcgs'].append(1.0 / np.log2(rank + 1))
            else:
                metrics[k]['hits'].append(0)
                metrics[k]['ndcgs'].append(0)
                
    # Return per-K averages as a dictionary
    return {k: (np.mean(metrics[k]['hits']), np.mean(metrics[k]['ndcgs'])) for k in k_list}