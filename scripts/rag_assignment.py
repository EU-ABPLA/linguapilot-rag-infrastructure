from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from math import sqrt
from typing import List, Sequence

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
    OpenAIEmbedding = load_openai_embedding_class()
    return OpenAIEmbedding(model=model, api_key=api_key)


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
            "No OpenAI API key was detected. Running with offline fallback embeddings instead."
        )

    embedding_client = build_embedding_client(api_key=args.api_key, model=args.model)
    try:
        segment_vectors = embedding_client.embed(STORY_SEGMENTS)
        query_vector = embedding_client.embed([args.query])[0]
    except RuntimeError as exc:
        print(f"Embedding failed: {exc}")
        print(
            "If you do not have a valid OpenAI API key, set the OPENAI_API_KEY environment variable "
            "or pass --api-key <YOUR_KEY> to use the OpenAI embedding service."
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
