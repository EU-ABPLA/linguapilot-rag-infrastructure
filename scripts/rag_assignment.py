from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from math import sqrt
from typing import Any, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

STORY_SEGMENTS: List[str] = [
    "Anna found an old map hidden between the pages of a dusty book.",
    "The map showed a broken bridge deep in the forest behind the willow tree.",
    "She gathered wood and rope, built a small raft, and carefully crossed the river.",
    "On the other side, Anna discovered a bright cottage filled with books and candles.",
    "At sunset, she turned back home with the map and a plan to return soon.",
]

DEFAULT_QUERY = "How did Anna cross the river?"


class SentenceTransformersEmbedding:
    """Fallback embedding provider using SentenceTransformers."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        """Initialize SentenceTransformers embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for offline embeddings. "
                "Install it with: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model)

    def embed(self, texts: Sequence[str], trace: Optional[Any] = None) -> List[List[float]]:
        """Embed texts using SentenceTransformers."""
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise ValueError("texts must be a sequence of strings")
        
        normalized: List[str] = []
        for item in texts:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("text item must be non-empty string")
            normalized.append(item)
        
        if not normalized:
            raise ValueError("texts must not be empty")
        
        # SentenceTransformers returns numpy arrays, convert to lists
        embeddings = self.model.encode(normalized, convert_to_tensor=False)
        return [embedding.tolist() for embedding in embeddings]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (sqrt(norm_a) * sqrt(norm_b))


def rank_segments(query_vector: Sequence[float], segment_vectors: List[Sequence[float]]) -> List[int]:
    scores = [cosine_similarity(query_vector, vec) for vec in segment_vectors]
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def load_openai_embedding_class() -> type:
    module_path = os.path.join(SRC, "libs", "embedding", "openai_embedding.py")
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Embedding module not found: {module_path}")
    spec = importlib.util.spec_from_file_location("openai_embedding_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load OpenAIEmbedding module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, "OpenAIEmbedding")


def build_embedding_client(api_key: str, model: str):
    """Build embedding client using OpenAI API or SentenceTransformers fallback."""
    if api_key and api_key.strip():
        # Use OpenAI API
        OpenAIEmbedding = load_openai_embedding_class()
        return OpenAIEmbedding(model=model, api_key=api_key)
    else:
        # Use SentenceTransformers as fallback
        print("Using SentenceTransformers for offline embeddings...")
        # Map common OpenAI model names to SentenceTransformers models
        model_mapping = {
            "text-embedding-3-small": "all-MiniLM-L6-v2",
            "text-embedding-3-large": "all-mpnet-base-v2",
            "text-embedding-ada-002": "all-MiniLM-L6-v2",
        }
        st_model = model_mapping.get(model, "all-MiniLM-L6-v2")
        return SentenceTransformersEmbedding(model=st_model)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Embed 5 story segments and find the most similar one to a query."
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Question to match.")
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="OpenAI embedding model or fallback model name.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="OpenAI API key. Leave empty to use offline fallback if available.",
    )
    args = parser.parse_args(argv)
    args.api_key = args.api_key.strip()

    if not args.api_key:
        print(
            "No OpenAI API key detected. Using SentenceTransformers for offline embeddings."
        )

    embedding_client = build_embedding_client(api_key=args.api_key, model=args.model)
    try:
        segment_vectors = embedding_client.embed(STORY_SEGMENTS)
        query_vector = embedding_client.embed([args.query])[0]
    except RuntimeError as exc:
        print(f"Embedding failed: {exc}")
        print(
            "Failed to generate embeddings. Ensure that either:\n"
            "  1. You have a valid OpenAI API key and internet connection, or\n"
            "  2. You have sentence-transformers installed: pip install sentence-transformers"
        )
        return 1

    ranked_indices = rank_segments(query_vector, segment_vectors)
    best_index = ranked_indices[0]

    print("Query:")
    print(args.query)
    print("")
    print("Scores de similarité :")
    for rank, index in enumerate(ranked_indices, start=1):
        score = cosine_similarity(query_vector, segment_vectors[index])
        print(f"{rank}. segment {index + 1}: score={score:.4f}")
    print("")
    print("Meilleur segment :")
    print(f"Segment {best_index + 1}: {STORY_SEGMENTS[best_index]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
