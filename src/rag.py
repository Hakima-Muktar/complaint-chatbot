"""
Task 3: RAG Pipeline - Retrieval Augmented Generation

Implements:
1. Retriever (FAISS + SentenceTransformers)
2. Prompt engineering
3. Generator (Ollama LLM)

Usage:
    from src.rag import RAGPipeline

    rag = RAGPipeline()
    answer, sources = rag.answer("Why are people unhappy with credit cards?")
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

from src.llm.local_ollama import get_llm_client

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.bin"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.pkl"

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
DEFAULT_TOP_K = 5

PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust Financial.

INSTRUCTIONS:
- Use ONLY the complaint excerpts below
- If the information is insufficient, say so clearly
- Be concise and factual
- Mention relevant product categories when applicable
- Summarize recurring themes if present

COMPLAINT EXCERPTS:
{context}

QUESTION:
{question}

ANSWER:
"""


# ===================================================================
# Retriever
# ===================================================================
class Retriever:
    """Semantic search over FAISS index."""

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.index = self._load_index(index_path)
        self.metadata = self._load_metadata(metadata_path)
        self.model = SentenceTransformer(embedding_model)

        # Safety check: embedding dimensions must match
        assert (
            self.index.d == self.model.get_sentence_embedding_dimension()
        ), "Embedding dimension mismatch between FAISS index and model"

    def _load_index(self, path: Path) -> faiss.Index:
        if not path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {path}\n"
                "👉 Run: python src/index_vector_store.py"
            )
        return faiss.read_index(str(path))

    def _load_metadata(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(
                f"Metadata not found at {path}\n"
                "👉 Run: python src/index_vector_store.py"
            )
        with open(path, "rb") as f:
            return pickle.load(f)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        product_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve most relevant complaint chunks."""

        # Embed query
        query_embedding = self.model.encode(
            [query], convert_to_numpy=True
        ).astype(np.float32)

        # Avoid searching more vectors than exist
        search_k = min(
            top_k * 3 if product_filter else top_k,
            self.index.ntotal,
        )

        distances, indices = self.index.search(query_embedding, search_k)

        results: List[Dict[str, Any]] = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            chunk = self.metadata[idx]
            meta = chunk["metadata"]

            # Product filtering
            if product_filter and meta.get("product") != product_filter:
                continue

            results.append(
                {
                    "text": chunk["text"],
                    "complaint_id": meta.get("complaint_id", ""),
                    "product": meta.get("product", ""),
                    "issue": meta.get("issue", ""),
                    "company": meta.get("company", ""),
                    "distance": float(dist),
                }
            )

            if len(results) >= top_k:
                break

        # Sort by similarity (lower distance = better)
        results.sort(key=lambda x: x["distance"])
        return results


# ===================================================================
# RAG Pipeline
# ===================================================================
class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""

    def __init__(
        self,
        llm_model: str = "mistral:7b-instruct",
        top_k: int = DEFAULT_TOP_K,
    ):
        self.retriever = Retriever()
        self.llm = get_llm_client(model=llm_model)
        self.top_k = top_k

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Complaint {i}] "
                f"Product: {chunk['product']} | Issue: {chunk['issue']}\n"
                f"{chunk['text']}"
            )
        return "\n\n".join(parts)

    def _build_prompt(self, question: str, context: str) -> str:
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def answer(
        self,
        question: str,
        product_filter: Optional[str] = None,
        top_k: Optional[int] = None,
        temperature: float = 0.7,
    ) -> Tuple[str, List[Dict[str, Any]]]:

        k = top_k or self.top_k

        chunks = self.retriever.retrieve(
            query=question,
            top_k=k,
            product_filter=product_filter,
        )

        if not chunks:
            return "No relevant complaints found for this question.", []

        context = self._build_context(chunks)
        prompt = self._build_prompt(question, context)

        answer = self.llm.generate(prompt, temperature=temperature)

        return answer.strip(), chunks

    def retrieve_only(
        self,
        question: str,
        product_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        return self.retriever.retrieve(
            query=question,
            top_k=k,
            product_filter=product_filter,
        )


# ===================================================================
# Local test
# ===================================================================
if __name__ == "__main__":
    print("Initializing RAG pipeline...")
    rag = RAGPipeline()

    print("\nTesting retrieval only...")
    test_chunks = rag.retrieve_only("billing dispute credit card", top_k=3)
    for i, c in enumerate(test_chunks, 1):
        print(f"{i}. [{c['product']}] {c['issue']}")

    print("\n" + "=" * 60)
    print("Testing full RAG pipeline")
    print("=" * 60)

    q = "What are the main complaints about credit cards?"
    answer, sources = rag.answer(q)

    print("\nAnswer:\n", answer)
    print("\nSources:")
    for i, s in enumerate(sources, 1):
        print(f"{i}. [{s['product']}] {s['issue']}")
