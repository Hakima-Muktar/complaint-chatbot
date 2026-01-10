import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ------------------
# CONFIG
# ------------------
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
VECTOR_DIR = "vector_store"
INDEX_PATH = os.path.join(VECTOR_DIR, "faiss_index.bin")
META_PATH = os.path.join(VECTOR_DIR, "metadata.pkl")

os.makedirs(VECTOR_DIR, exist_ok=True)

# ------------------
# LOAD YOUR CHUNKS
# ------------------
# This assumes you already created `chunks`
# If chunks are saved, load them instead
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

texts = [c["text"] for c in chunks]

# ------------------
# EMBEDDINGS
# ------------------
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
).astype(np.float32)

dim = embeddings.shape[1]

# ------------------
# FAISS
# ------------------
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

with open(META_PATH, "wb") as f:
    pickle.dump(chunks, f)

print(f"✅ FAISS index saved to {INDEX_PATH}")
print(f"📦 Vectors: {index.ntotal:,}")
