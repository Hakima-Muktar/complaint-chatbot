"""
Embedding Generation for Text Chunks

This script generates embeddings for text chunks using sentence-transformers.
Uses the all-MiniLM-L6-v2 model (384 dimensions) to match the pre-built embeddings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
import time


def generate_embeddings(
    chunks_file: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32
) -> tuple:
    """
    Generate embeddings for text chunks.
    
    Args:
        chunks_file: Path to the CSV file containing chunks
        model_name: Name of the sentence-transformers model
        batch_size: Batch size for encoding
    
    Returns:
        Tuple of (embeddings array, chunks DataFrame)
    """
    print(f"Loading chunks from {chunks_file}...")
    chunks_df = pd.read_csv(chunks_file)
    
    print(f"Total chunks to embed: {len(chunks_df)}")
    
    # Load the embedding model
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Extract text chunks
    texts = chunks_df['text'].tolist()
    
    # Generate embeddings
    print(f"\nGenerating embeddings with batch_size={batch_size}...")
    start_time = time.time()
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\nEmbedding generation complete!")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Embeddings per second: {len(texts) / elapsed_time:.2f}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    return embeddings, chunks_df


def save_embeddings(
    embeddings: np.ndarray,
    chunks_df: pd.DataFrame,
    output_file: str
) -> None:
    """
    Save embeddings and metadata to a file.
    
    Args:
        embeddings: Numpy array of embeddings
        chunks_df: DataFrame containing chunk metadata
        output_file: Path to save the embeddings
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy file
    np.save(output_file, embeddings)
    
    print(f"\nEmbeddings saved to {output_file}")
    print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Also save metadata separately
    metadata_file = output_file.replace('.npy', '_metadata.csv')
    chunks_df.to_csv(metadata_file, index=False)
    print(f"Metadata saved to {metadata_file}")


def analyze_embeddings(embeddings: np.ndarray) -> None:
    """
    Analyze the generated embeddings.
    
    Args:
        embeddings: Numpy array of embeddings
    """
    print("\n" + "="*80)
    print("EMBEDDING ANALYSIS")
    print("="*80)
    
    print(f"\nEmbedding statistics:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dimension: {embeddings.shape[1]}")
    print(f"  Mean: {np.mean(embeddings):.6f}")
    print(f"  Std: {np.std(embeddings):.6f}")
    print(f"  Min: {np.min(embeddings):.6f}")
    print(f"  Max: {np.max(embeddings):.6f}")
    
    # Calculate norms
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nEmbedding norms:")
    print(f"  Mean norm: {np.mean(norms):.6f}")
    print(f"  Std norm: {np.std(norms):.6f}")
    print(f"  Min norm: {np.min(norms):.6f}")
    print(f"  Max norm: {np.max(norms):.6f}")
    
    # Memory usage
    memory_mb = embeddings.nbytes / (1024 * 1024)
    print(f"\nMemory usage: {memory_mb:.2f} MB")


def test_similarity(
    embeddings: np.ndarray,
    chunks_df: pd.DataFrame,
    model: SentenceTransformer = None,
    query: str = "I was charged a fee that I did not authorize"
) -> None:
    """
    Test similarity search with a sample query.
    
    Args:
        embeddings: Numpy array of embeddings
        chunks_df: DataFrame containing chunk metadata
        model: SentenceTransformer model (will load if None)
        query: Test query string
    """
    print("\n" + "="*80)
    print("SIMILARITY SEARCH TEST")
    print("="*80)
    
    if model is None:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Encode the query
    print(f"\nQuery: '{query}'")
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    
    # Calculate cosine similarities
    # Normalize embeddings
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    
    # Compute similarities
    similarities = np.dot(embeddings_norm, query_norm)
    
    # Get top 5 results
    top_k = 5
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    print(f"\nTop {top_k} most similar chunks:")
    for i, idx in enumerate(top_indices, 1):
        print(f"\n{i}. Similarity: {similarities[idx]:.4f}")
        print(f"   Product: {chunks_df.iloc[idx]['product_category']}")
        print(f"   Issue: {chunks_df.iloc[idx]['issue']}")
        print(f"   Text: {chunks_df.iloc[idx]['text'][:200]}...")


def main():
    """Main function to generate embeddings."""
    # Define paths
    chunks_file = "data/processed/sample_chunks.csv"
    embeddings_file = "data/processed/sample_embeddings.npy"
    
    # Generate embeddings
    embeddings, chunks_df = generate_embeddings(
        chunks_file=chunks_file,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=32
    )
    
    # Analyze embeddings
    analyze_embeddings(embeddings)
    
    # Save embeddings
    save_embeddings(embeddings, chunks_df, embeddings_file)
    
    # Test similarity search
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    test_similarity(embeddings, chunks_df, model)


if __name__ == "__main__":
    main()