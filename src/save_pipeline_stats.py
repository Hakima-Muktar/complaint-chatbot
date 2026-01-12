"""
Save Pipeline Statistics

Aggregates statistics from Sampling, Chunking, and Embedding steps 
and saves them to a CSV file for documentation.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def save_pipeline_stats():
    base_dir = Path("data/processed")
    output_dir = Path("report/stats")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = []
    
    # 1. SAMPLING STATS
    sample_file = base_dir / "sample_complaints.csv"
    if sample_file.exists():
        df_sample = pd.read_csv(sample_file)
        stats.append({
            "Stage": "Sampling",
            "Metric": "Total Samples",
            "Value": len(df_sample)
        })
        product_counts = df_sample['Product'].value_counts().to_dict()
        for prod, count in product_counts.items():
            stats.append({
                "Stage": "Sampling",
                "Metric": f"Count ({prod})",
                "Value": count
            })
    
    # 2. CHUNKING STATS
    chunks_file = base_dir / "sample_chunks.csv"
    if chunks_file.exists():
        df_chunks = pd.read_csv(chunks_file)
        stats.append({
            "Stage": "Chunking",
            "Metric": "Total Chunks",
            "Value": len(df_chunks)
        })
        stats.append({
            "Stage": "Chunking",
            "Metric": "Avg Chunk Length (chars)",
            "Value": round(df_chunks['text'].str.len().mean(), 2)
        })
        stats.append({
            "Stage": "Chunking",
            "Metric": "Avg Chunks per Complaint",
            "Value": round(len(df_chunks) / len(df_sample) if len(df_sample) > 0 else 0, 2)
        })
        
    # 3. EMBEDDING STATS
    emb_file = base_dir / "sample_embeddings.npy"
    if emb_file.exists():
        embeddings = np.load(emb_file)
        stats.append({
            "Stage": "Embedding",
            "Metric": "Total Embeddings",
            "Value": embeddings.shape[0]
        })
        stats.append({
            "Stage": "Embedding",
            "Metric": "Embedding Dimension",
            "Value": embeddings.shape[1]
        })
        
    # 4. VECTOR STORE STATS
    vector_store_path = Path("vector_store/faiss_index/faiss.index")
    if vector_store_path.exists():
         stats.append({
            "Stage": "Vector Store",
            "Metric": "Index Created",
            "Value": "Yes"
        })
    
    # Save to CSV
    df_stats = pd.DataFrame(stats)
    output_file = output_dir / "pipeline_metrics.csv"
    df_stats.to_csv(output_file, index=False)
    
    print(f"Pipeline statistics saved to {output_file}")
    print(df_stats)

if __name__ == "__main__":
    save_pipeline_stats()