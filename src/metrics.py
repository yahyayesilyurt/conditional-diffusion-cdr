import numpy as np

def calculate_metrics(top_k_indices, target_item_ids, k=10):
    """
    top_k_indices: Matrix of movie indices recommended by the model (batch_size, k) (PyTorch Tensor)
    target_item_ids: Movie indices actually watched by the user in the test set (batch_size,)
    """
    # Convert tensors to numpy arrays
    top_k_indices = top_k_indices.cpu().numpy()
    target_item_ids = target_item_ids.cpu().numpy()
    
    hits = []
    ndcgs = []
    
    for i in range(len(target_item_ids)):
        target = target_item_ids[i]
        predictions = top_k_indices[i]
        
        # Is the target movie among the recommendations?
        if target in predictions:
            # HR@K
            hits.append(1)
            
            # NDCG@K
            # Find the rank of the target (add 1 since rank is 0-indexed)
            rank = np.where(predictions == target)[0][0] + 1
            # NDCG formula: 1 / log2(rank + 1)
            ndcg_score = 1.0 / np.log2(rank + 1)
            ndcgs.append(ndcg_score)
        else:
            hits.append(0)
            ndcgs.append(0)
            
    # Return average values for the batch
    return np.mean(hits), np.mean(ndcgs)