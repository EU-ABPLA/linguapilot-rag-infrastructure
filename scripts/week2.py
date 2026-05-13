
"""Week 2: PDF Ingestion & Chunking Analysis with ChromaDB

This script takes a grammar textbook (PDF), chops it into fixed-size pieces,
and looks for where the chunking breaks semantic meaning.

Then it stores everything in ChromaDB so you can search by meaning.

Usage:
    python scripts/week2.py --pdf "path/to/textbook.pdf"
    python scripts/week2.py --pdf "grammar.pdf" --chunk-size 1024
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@dataclass
class ChunkAnalysis:
    """Holds analysis data for a single chunk - where it came from and whats wrong with it."""
    chunk_id: int
    text: str
    start_position: int
    end_position: int
    issues: List[str]
    
    def __str__(self) -> str:
        issues_str = "\n  → ".join(self.issues) if self.issues else "None detected"
        return (
            f"\nChunk #{self.chunk_id}\n"
            f"  Position: chars {self.start_position:,}-{self.end_position:,}\n"
            f"  Size: {len(self.text):,} characters\n"
            f"  Problems:\n  → {issues_str}\n"
            f"  Preview: {self.text[:80]}...\n"
        )


def extract_text_from_pdf(pdf_path: str) -> str:
    """Pull all the text out of a PDF file."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Need pdfplumber: pip install pdfplumber")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Can't find the PDF: {pdf_path}")
    
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    
    return "\n".join(pages)


def apply_fixed_chunking(text: str, chunk_size: int = 512, overlap: int = 0) -> List[str]:
    """Split text into fixed-size chunks. Simple but often breaks things in the middle."""
    chunks = []
    stride = chunk_size - overlap
    
    for i in range(0, len(text), stride):
        chunk = text[i : i + chunk_size]
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
    
    return chunks


def analyze_chunks_for_issues(
    text: str, chunks: List[str], chunk_size: int
) -> List[ChunkAnalysis]:
    """Look for places where the fixed chunking broke things.
    
    We check for:
    - Words cut in half
    - Unclosed quotes or parentheses
    - Dialogue starting mid-line
    - Table structures split across chunks
    """
    analyses = []
    position = 0
    
    for chunk_id, chunk in enumerate(chunks):
        issues = []
        
        start_pos = text.find(chunk, position)
        end_pos = start_pos + len(chunk)
        position = end_pos
        
        # Mid-word break?
        if chunk and not chunk[-1].isspace() and end_pos < len(text):
            next_char = text[end_pos] if end_pos < len(text) else ""
            if next_char and next_char not in " \n\t.,;:!?\"-":
                issues.append(" Word got cut in half")
        
        # Unclosed quotes?
        if chunk.count("'") % 2 != 0:
            issues.append("Odd number of single quotes (probably split)")
        if chunk.count('"') % 2 != 0:
            issues.append("Odd number of double quotes (probably split)")
        
        # Unclosed brackets?
        open_parens = chunk.count("(") - chunk.count(")")
        open_brackets = chunk.count("[") - chunk.count("]")
        if open_parens > 0:
            issues.append(f"{open_parens} open parenthesis/es")
        if open_brackets > 0:
            issues.append(f"{open_brackets} open bracket/s")
        
        # Dialogue cut off?
        if chunk.startswith('"') or chunk.startswith("'") or chunk.lstrip().startswith("-"):
            if start_pos > 0 and text[start_pos - 1] not in "\n":
                issues.append("Dialogue probably cut mid-conversation")
        
        # Table split?
        if "|" in chunk:
            pipe_count = chunk.count("|")
            if pipe_count % 2 != 0:
                issues.append(f"Table column split ({pipe_count} pipes)")
        
        analysis = ChunkAnalysis(
            chunk_id=chunk_id,
            text=chunk,
            start_position=start_pos,
            end_position=end_pos,
            issues=issues,
        )
        analyses.append(analysis)
    
    return analyses


def initialize_chromadb(collection_name: str = "grammar_rules") -> object:
    """Set up ChromaDB on your local machine (SQLite backend)."""
    try:
        import chromadb
    except ImportError:
        raise ImportError("Need chromadb: pip install chromadb")
    
    return chromadb.Client()


def embed_chunks_with_sentence_transformers(chunks: List[str]) -> List[List[float]]:
    """Turn each chunk into a vector so we can search by meaning."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Need sentence-transformers: pip install sentence-transformers")
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_tensor=False)
    return [embedding.tolist() for embedding in embeddings]


def store_chunks_in_chromadb(
    client: object,
    chunks: List[str],
    embeddings: List[List[float]],
    collection_name: str = "grammar_rules",
) -> object:
    """Save chunks + their vectors to ChromaDB."""
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"chunk_id": i, "size": len(chunk)} for i, chunk in enumerate(chunks)]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    
    return collection


def retrieve_grammar_rules(
    collection: object,
    query: str,
    num_results: int = 3,
) -> List[dict]:
    """Search ChromaDB for chunks that match your query."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Need sentence-transformers: pip install sentence-transformers")
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode([query], convert_to_tensor=False)[0].tolist()
    
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=num_results,
        include=["documents", "distances", "metadatas"]
    )
    
    retrieved = []
    for doc, distance, metadata in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0]
    ):
        similarity = 1 - distance
        retrieved.append({
            "document": doc,
            "similarity_score": similarity,
            "metadata": metadata,
        })
    
    return retrieved


def print_chunk_analysis_report(analyses: List[ChunkAnalysis], top_issues: int = 5) -> None:
    """Show what went wrong with the chunking."""
    print("\n" + "═" * 80)
    print("CHUNKING ANALYSIS REPORT")
    print("═" * 80)
    
    total = len(analyses)
    problematic_count = sum(1 for a in analyses if a.issues)
    
    print(f"\nOverview:")
    print(f"   Total chunks: {total:,}")
    print(f"   Chunks with issues: {problematic_count:,} ({problematic_count / total * 100:.1f}%)")
    
    # Count issue types
    issue_types = {}
    for analysis in analyses:
        for issue in analysis.issues:
            issue_type = issue.split(":")[0]
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
    
    if issue_types:
        print("\nWhat went wrong (by frequency):")
        for issue_type, count in sorted(issue_types.items(), key=lambda x: -x[1]):
            print(f"   {issue_type}: {count}x")
    
    # Show top problematic chunks
    if problematic_count > 0:
        print(f"\nWorst offenders (top {min(top_issues, problematic_count)} chunks):")
        print("─" * 80)
        
        worst = sorted(
            [a for a in analyses if a.issues],
            key=lambda x: len(x.issues),
            reverse=True
        )[:top_issues]
        
        for analysis in worst:
            print(analysis)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Take a grammar textbook PDF, chunk it, and search it"
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Your PDF file (required)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="How many characters per chunk (default: 512)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Overlap between chunks in chars (default: 0)",
    )
    parser.add_argument(
        "--query",
        default="past tense verb conjugation",
        help="What do you want to find? (default: past tense)",
    )
    parser.add_argument(
        "--num-results",
        type=int,
        default=3,
        help="How many results to show (default: 3)",
    )
    parser.add_argument(
        "--collection-name",
        default="grammar_rules",
        help="ChromaDB collection name",
    )
    
    args = parser.parse_args(argv)
    
    # Extract text from PDF
    print("\nReading PDF...")
    text = extract_text_from_pdf(args.pdf)
    print(f"   Got {len(text):,} characters")
    
    #Split into fixed chunks
    print(f"\nSplitting into {args.chunk_size}-char chunks...")
    chunks = apply_fixed_chunking(text, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"   Created {len(chunks):,} chunks")
    
    #Look for problems
    print("\nAnalyzing where the chunking breaks things...")
    analyses = analyze_chunks_for_issues(text, chunks, args.chunk_size)
    print_chunk_analysis_report(analyses)
    
    # Set up ChromaDB
    print("\nStarting ChromaDB...")
    client = initialize_chromadb()
    print("   ✓ Ready")
    
    # Create embeddings
    print("\nConverting chunks to vectors...")
    embeddings = embed_chunks_with_sentence_transformers(chunks)
    print(f"   ✓ {len(embeddings):,} vectors created")
    
    #Store everything
    print(f"\nStoring in ChromaDB collection '{args.collection_name}'...")
    collection = store_chunks_in_chromadb(
        client,
        chunks,
        embeddings,
        collection_name=args.collection_name,
    )
    print(f"   ✓ Stored {len(chunks):,} chunks")
    
    # search!
    print(f"\nSearching for: '{args.query}'")
    retrieved = retrieve_grammar_rules(
        collection,
        args.query,
        num_results=args.num_results,
    )
    
    print("\n" + "═" * 80)
    print("SEARCH RESULTS")
    print("═" * 80)
    for rank, result in enumerate(retrieved, 1):
        print(f"\n#{rank} (match score: {result['similarity_score']:.2%})")
        print(f"{result['document'][:180]}...\n")
    
    print("═" * 80)
    print("All done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


