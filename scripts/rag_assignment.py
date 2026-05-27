"""
Week 2: Semantic Similarity Ranking for Story Segments

Demonstrates embedding generation and cosine similarity ranking
using either OpenAI embeddings (online) or SentenceTransformers (offline).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from math import sqrt
from typing import Any, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

STORY_SEGMENTS: List[str] = [
    "Anna found an old map hidden between the pages of a dusty book.",
    "The map showed a broken bridge deep in the forest behind the willow tree.",
    "She gathered wood and rope, built a small raft, and carefully crossed the river.",
    "On the other side, Anna discovered a bright cottage filled with books and candles.",
    "At sunset, she turned back home with the map and a plan to return soon.",
]

DEFAULT_QUERY = "How did Anna cross the river?"

OPENAI_MODEL_TO_SENTENCE_TRANSFORMERS = {
    "text-embedding-3-small": "all-MiniLM-L6-v2",
    "text-embedding-3-large": "all-mpnet-base-v2",
    "text-embedding-ada-002": "all-MiniLM-L6-v2",
}



class OfflineEmbedder:
    """Generate embeddings using SentenceTransformers for offline inference."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "sentence-transformers required. Install with: pip install sentence-transformers"
            ) from error
        
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str], trace: Optional[Any] = None) -> List[List[float]]:
        """Convert texts to embedding vectors."""
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
            raise ValueError("texts must be a sequence of strings")
        
        validated_texts = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("each text must be a non-empty string")
            validated_texts.append(text)
        
        if not validated_texts:
            raise ValueError("at least one text required")
        
        embeddings = self.model.encode(validated_texts, convert_to_tensor=False)
        return [embedding.tolist() for embedding in embeddings]


def compute_cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Returns value in [0, 1] where 1 is perfect similarity.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("vectors must have same dimension")
    
    dot_product = sum(x * y for x, y in zip(vector_a, vector_b))
    magnitude_a = sqrt(sum(x * x for x in vector_a))
    magnitude_b = sqrt(sum(y * y for y in vector_b))
    
    if magnitude_a <= 0.0 or magnitude_b <= 0.0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)


def rank_segments_by_similarity(
    query_vector: Sequence[float],
    segment_vectors: List[Sequence[float]]
) -> List[tuple[int, float]]:
    """
    Rank segments by similarity to query vector.
    
    Returns list of (segment_index, similarity_score) tuples sorted by score descending.
    """
    similarities = [
        (index, compute_cosine_similarity(query_vector, segment_vector))
        for index, segment_vector in enumerate(segment_vectors)
    ]
    return sorted(similarities, key=lambda x: x[1], reverse=True)


def load_openai_embedding_class() -> type:
    """Dynamically load OpenAIEmbedding class from src/libs/embedding/openai_embedding.py"""
    module_path = os.path.join(SRC_PATH, "libs", "embedding", "openai_embedding.py")
    
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Module not found: {module_path}")
    
    spec = importlib.util.spec_from_file_location("openai_embedding_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load openai_embedding module")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    
    return getattr(module, "OpenAIEmbedding")


def create_embedder(api_key: str, model_name: str) -> Any:
    """
    Create embedder using OpenAI API if key provided, otherwise use offline embedder.
    
    Args:
        api_key: OpenAI API key (use empty string for offline mode)
        model_name: Model name (OpenAI or SentenceTransformers)
        
    Returns:
        Embedder instance with embed() method
    """
    if api_key and api_key.strip():
        openai_embedding_class = load_openai_embedding_class()
        return openai_embedding_class(model=model_name, api_key=api_key)
    
    print("Using SentenceTransformers for offline embeddings...")
    offline_model = OPENAI_MODEL_TO_SENTENCE_TRANSFORMERS.get(model_name, "all-MiniLM-L6-v2")
    return OfflineEmbedder(model_name=offline_model)


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rank story segments by similarity to query using embeddings"
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Query text to match against segments"
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="Embedding model name (OpenAI or SentenceTransformers)"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="OpenAI API key (optional; uses offline if empty)"
    )
    
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point for segment ranking demo."""
    args = parse_arguments(argv)
    api_key = args.api_key.strip()
    
    if not api_key:
        print("No OpenAI API key provided. Using offline embeddings.")
    
    try:
        embedder = create_embedder(api_key=api_key, model_name=args.model)
        segment_vectors = embedder.embed(STORY_SEGMENTS)
        query_vector = embedder.embed([args.query])[0]
    except RuntimeError as error:
        print(f"Embedding failed: {error}")
        print(
            "Ensure either:\n"
            "  1. Valid OpenAI API key and internet connection, or\n"
            "  2. sentence-transformers installed: pip install sentence-transformers"
        )
        return 1
    
    rankings = rank_segments_by_similarity(query_vector, segment_vectors)
    
    print(f"Query: {args.query}\n")
    print("Similarity scores:")
    for rank, (segment_index, similarity_score) in enumerate(rankings, start=1):
        print(f"  {rank}. Segment {segment_index + 1}: {similarity_score:.4f}")
    
    best_segment_index, best_score = rankings[0]
    print(f"\nBest match:")
    print(f"  Segment {best_segment_index + 1} (score: {best_score:.4f})")
    print(f"  {STORY_SEGMENTS[best_segment_index]}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
