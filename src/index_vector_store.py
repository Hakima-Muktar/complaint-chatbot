"""
FAISS Vector Store Builder

This script builds a FAISS vector store from embeddings and metadata.
Supports similarity search with metadata filtering.
"""

import pandas as pd
import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer


class FAISSVectorStore:
    """FAISS-based vector store with metadata support."""
    
    def __init__(self, dimension: int = 384):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension
        self.index = None
        self.metadata = []
        self.model = None
    
    def build_index(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
        index_type: str = "flat"
    ) -> None:
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Numpy array of embeddings (n_samples, dimension)
            metadata: List of metadata dictionaries
            index_type: Type of FAISS index ('flat', 'ivf', or 'hnsw')
        """
        print(f"Building FAISS index with {len(embeddings)} vectors...")
        
        # Normalize embeddings for cosine similarity
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings_norm.astype('float32')
        
        if index_type == "flat":
            # Flat index (exact search, good for small datasets)
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
        elif index_type == "ivf":
            # IVF index (approximate search, faster for large datasets)
            nlist = min(100, len(embeddings) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.train(embeddings_norm)
        elif index_type == "hnsw":
            # HNSW index (hierarchical navigable small world)
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add vectors to index
        self.index.add(embeddings_norm)
        self.metadata = metadata
        
        print(f"Index built successfully!")
        print(f"  Index type: {index_type}")
        print(f"  Total vectors: {self.index.ntotal}")
        print(f"  Dimension: {self.dimension}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Dict = None
    ) -> List[Dict]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {'product': 'Credit card'})
        
        Returns:
            List of dictionaries containing results and metadata
        """
        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        query_norm = query_norm.astype('float32').reshape(1, -1)
        
        # Search
        if filter_metadata:
            # If filtering, retrieve more results and filter
            k_retrieve = min(k * 10, self.index.ntotal)
        else:
            k_retrieve = k
        
        distances, indices = self.index.search(query_norm, k_retrieve)
        
        # Prepare results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for empty results
                continue
            
            metadata = self.metadata[idx].copy()
            metadata['similarity'] = float(dist)
            metadata['rank'] = i + 1
            
            # Apply metadata filtering
            if filter_metadata:
                match = all(
                    metadata.get(key) == value
                    for key, value in filter_metadata.items()
                )
                if not match:
                    continue
            
            results.append(metadata)
            
            if len(results) >= k:
                break
        
        return results
    
    def save(self, output_dir: str) -> None:
        """
        Save FAISS index and metadata.
        
        Args:
            output_dir: Directory to save the index
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = output_path / "faiss.index"
        faiss.write_index(self.index, str(index_file))
        
        # Save metadata
        metadata_file = output_path / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"\nVector store saved to {output_dir}")
        print(f"  Index file: {index_file}")
        print(f"  Metadata file: {metadata_file}")
    
    @classmethod
    def load(cls, input_dir: str, dimension: int = 384) -> 'FAISSVectorStore':
        """
        Load FAISS index and metadata.
        
        Args:
            input_dir: Directory containing the index
            dimension: Embedding dimension
        
        Returns:
            Loaded FAISSVectorStore instance
        """
        input_path = Path(input_dir)
        
        # Load FAISS index
        index_file = input_path / "faiss.index"
        index = faiss.read_index(str(index_file))
        
        # Load metadata
        metadata_file = input_path / "metadata.pkl"
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        store = cls(dimension=dimension)
        store.index = index
        store.metadata = metadata
        
        print(f"Vector store loaded from {input_dir}")
        print(f"  Total vectors: {store.index.ntotal}")
        
        return store
    
    def load_model(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Load sentence transformer model for encoding queries."""
        self.model = SentenceTransformer(model_name)
    
    def query(self, text: str, k: int = 5, filter_metadata: Dict = None) -> List[Dict]:
        """
        Query the vector store with text.
        
        Args:
            text: Query text
            k: Number of results
            filter_metadata: Optional metadata filters
        
        Returns:
            List of results
        """
        if self.model is None:
            self.load_model()
        
        # Encode query
        query_embedding = self.model.encode([text], convert_to_numpy=True)[0]
        
        # Search
        return self.search(query_embedding, k=k, filter_metadata=filter_metadata)


def build_vector_store(
    embeddings_file: str,
    metadata_file: str,
    output_dir: str,
    index_type: str = "flat"
) -> FAISSVectorStore:
    """
    Build FAISS vector store from embeddings and metadata.
    
    Args:
        embeddings_file: Path to embeddings .npy file
        metadata_file: Path to metadata CSV file
        output_dir: Directory to save the vector store
        index_type: Type of FAISS index
    
    Returns:
        Built FAISSVectorStore instance
    """
    print(f"Loading embeddings from {embeddings_file}...")
    embeddings = np.load(embeddings_file)
    
    print(f"Loading metadata from {metadata_file}...")
    metadata_df = pd.read_csv(metadata_file)
    metadata = metadata_df.to_dict('records')
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Metadata records: {len(metadata)}")
    
    # Build vector store
    store = FAISSVectorStore(dimension=embeddings.shape[1])
    store.build_index(embeddings, metadata, index_type=index_type)
    
    # Save vector store
    store.save(output_dir)
    
    return store


def test_vector_store(store: FAISSVectorStore) -> None:
    """
    Test the vector store with sample queries.
    
    Args:
        store: FAISSVectorStore instance
    """
    print("\n" + "="*80)
    print("VECTOR STORE TESTING")
    print("="*80)
    
    # Load model for encoding
    store.load_model()
    
    # Test queries
    test_queries = [
        "I was charged unauthorized fees on my credit card",
        "My loan application was denied without explanation",
        "I cannot access my savings account online",
        "International money transfer was delayed"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: '{query}'")
        print('='*80)
        
        results = store.query(query, k=3)
        
        for j, result in enumerate(results, 1):
            print(f"\nResult {j}:")
            print(f"  Similarity: {result['similarity']:.4f}")
            print(f"  Product: {result.get('product_category', 'N/A')}")
            print(f"  Issue: {result.get('issue', 'N/A')}")
            print(f"  Company: {result.get('company', 'N/A')}")
            print(f"  Text: {result.get('text', '')[:200]}...")


def main():
    """Main function to build vector store."""
    # Define paths
    embeddings_file = "data/processed/sample_embeddings.npy"
    metadata_file = "data/processed/sample_chunks.csv"
    output_dir = "vector_store/faiss_index"
    
    # Build vector store
    store = build_vector_store(
        embeddings_file=embeddings_file,
        metadata_file=metadata_file,
        output_dir=output_dir,
        index_type="flat"  # Use flat index for exact search
    )
    
    # Test vector store
    test_vector_store(store)
    
    print("\n" + "="*80)
    print("Vector store build complete!")
    print(f"Saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()