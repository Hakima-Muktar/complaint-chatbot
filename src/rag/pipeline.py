from typing import Dict, Any, List
from transformers import pipeline
from src.vector_store_builder import FAISSVectorStore
from src.rag.retriever import get_retriever
from src.rag.generator import generate_answer
import os

class RAGSystem:
    def __init__(self, vector_store_path: str, model_name: str = "google/flan-t5-small"):
        """
        Initialize RAG System.
        
        Args:
            vector_store_path: Path to the FAISS index directory.
            model_name: Name of the HF model to use for generation.
        """
        if not os.path.exists(vector_store_path):
             raise ValueError(f"Vector store path does not exist: {vector_store_path}")

        print(f"Loading Vector Store from {vector_store_path}...")
        self.vector_store = FAISSVectorStore.load(vector_store_path)
        self.retriever = get_retriever(self.vector_store)
        
        print(f"Loading LLM: {model_name}...")
        # Use text2text-generation for T5-like models
        self.llm_model = pipeline(
            "text2text-generation",
            model=model_name,
            max_length=512,
            truncation=True
        )
        
    def query(self, user_query: str) -> Dict[str, Any]:
        """
        End-to-end RAG pipeline: Retrieve -> Generate.
        
        Args:
            user_query: The user's question.
            
        Returns:
            Dictionary with query, answer, and retrieved context.
        """
        # Retrieve
        retrieved_docs = self.retriever(user_query)
        
        # Format context
        context_text = "\n\n".join([
            f"[Source: {doc.get('company', 'Unknown')}] {doc.get('text', '')}" 
            for doc in retrieved_docs
        ])
        
        # Generate
        answer = generate_answer(user_query, context_text, self.llm_model)
        
        return {
            "query": user_query,
            "answer": answer,
            "context": retrieved_docs
        }