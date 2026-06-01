"""backend/src/embedding.py"""

import numpy as np
from sentence_transformers import SentenceTransformer

# ✅ USE THIS FOR HIGH ACCURACY
_MODEL_NAME = "all-MiniLM-L6-v2"
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_MODEL_NAME)
    return _embedder


def embed_text(text: str) -> np.ndarray:
    return get_embedder().encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )


def embed_batch(texts: list) -> np.ndarray:
    return get_embedder().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False
    )


def embedding_cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
    return float(np.dot(e1, e2))

