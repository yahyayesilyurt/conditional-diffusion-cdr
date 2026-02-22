"""
Inspect and analyze exported embedding files.
Usage: python inspect_embeddings.py <embedding_file.pt>
"""
import argparse
import torch
from pathlib import Path


def inspect_embeddings(embedding_path: str):
    """
    Inspect the structure and content of an embedding file.
    
    Args:
        embedding_path: Path to the embedding .pt file
    """
    emb_path = Path(embedding_path)
    if not emb_path.exists():
        print(f"Error: File not found: {embedding_path}")
        return
    
    print(f"Loading embeddings from: {emb_path}")
    emb = torch.load(emb_path, weights_only=False)
    
    print("\n" + "="*80)
    print("EMBEDDING FILE STRUCTURE")
    print("="*80)
    
    print(f"\nFile Type: {type(emb)}")
    print(f"\nTop-level Keys: {list(emb.keys())}")
    
    # Metadata
    print("\n" + "-"*80)
    print("METADATA")
    print("-"*80)
    for k in ['model', 'ckpt_path', 'hidden']:
        if k in emb:
            print(f"  {k}: {emb[k]}")
    
    # Num nodes
    print("\n" + "-"*80)
    print("NODE COUNTS")
    print("-"*80)
    if 'num_nodes' in emb:
        for node_type, count in emb['num_nodes'].items():
            print(f"  {node_type}: {count:,}")
    
    # Embeddings dimensions
    print("\n" + "-"*80)
    print("EMBEDDING DIMENSIONS")
    print("-"*80)
    if 'embeddings' in emb:
        for node_type, tensor in emb['embeddings'].items():
            print(f"  {node_type}: shape={tensor.shape}, dtype={tensor.dtype}")
    
    # Additional metadata
    if 'metadata' in emb:
        print("\n" + "-"*80)
        print("ADDITIONAL METADATA")
        print("-"*80)
        for k, v in emb['metadata'].items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:,}")
            else:
                print(f"  {k}: {v}")
    
    # Sample embeddings
    print("\n" + "="*80)
    print("SAMPLE EMBEDDINGS")
    print("="*80)
    if 'embeddings' in emb:
        print("\nUser 0 embedding (first 10 dimensions):")
        print(f"  {emb['embeddings']['user'][0, :10]}")
        print(f"  Shape: {emb['embeddings']['user'][0].shape}")
        
        print("\nMovie 100 embedding (first 10 dimensions):")
        print(f"  {emb['embeddings']['movie'][100, :10]}")
        
        print("\nBook 500 embedding (first 10 dimensions):")
        print(f"  {emb['embeddings']['book'][500, :10]}")
    
    # Statistics
    print("\n" + "="*80)
    print("EMBEDDING STATISTICS")
    print("="*80)
    if 'embeddings' in emb:
        for node_type in ['user', 'movie', 'book']:
            if node_type in emb['embeddings']:
                tensor = emb['embeddings'][node_type]
                print(f"\n{node_type.capitalize()}:")
                print(f"  Mean: {tensor.mean():.6f}")
                print(f"  Std:  {tensor.std():.6f}")
                print(f"  Min:  {tensor.min():.6f}")
                print(f"  Max:  {tensor.max():.6f}")
                
                # Check for non-zero elements
                non_zero = (tensor != 0).sum().item()
                total = tensor.numel()
                print(f"  Non-zero elements: {non_zero:,} / {total:,} ({100*non_zero/total:.2f}%)")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Inspect exported embedding files")
    parser.add_argument(
        "embedding_file",
        type=str,
        nargs="?",
        default="./assets/gat_embeddings.pt",
        help="Path to embedding .pt file (default: ./assets/gat_embeddings.pt)"
    )
    args = parser.parse_args()
    
    inspect_embeddings(args.embedding_file)


if __name__ == "__main__":
    main()
