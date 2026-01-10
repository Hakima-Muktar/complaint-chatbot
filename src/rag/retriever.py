from typing import List, Dict, Any, Callable
from src.vector_store_builder import FAISSVectorStore

def get_retriever(vector_store: FAISSVectorStore, k: int = 5) -> Callable[[str], List[Dict[str, Any]]]:
    """
    Returns a retriever function that takes a query string and returns top-k chunks.
    
    Args:
        vector_store: The initialized FAISSVectorStore object.
        k: Number of chunks to retrieve.
        
    Returns:
        A callable function: query -> list of results.
    """
    def retrieve(query: str) -> List[Dict[str, Any]]:
        # vector_store.query handles embedding and searching
        results = vector_store.query(query, k=k)
        return results
        
    return retrieve