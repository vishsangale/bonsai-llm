"""
RAG Evaluation — Part 2: RAGAS across three knowledge bases
============================================================
Usage:
    python eval_rag.py --dataset wikipedia
    python eval_rag.py --dataset custom
    python eval_rag.py --dataset hf-model-cards
    python eval_rag.py --dataset all
"""

import os, json, argparse, textwrap
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
CHUNK_SIZE = 256   # tokens (approximated as words for simplicity)
CHUNK_OVERLAP = 32
TOP_K = 3
MAX_NEW_TOKENS = 200


# ── Chunker ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def format_results_table(dataset_name: str, results: dict) -> str:
    """Return a formatted markdown-style table string for a single dataset."""
    header = f"\nDataset: {dataset_name}\n"
    divider = f"{'─' * 50}\n"
    row_fmt = "| {:<22} | {:>6} |\n"
    header_row = row_fmt.format("Metric", "Score")
    sep_row = "|" + "-" * 24 + "|" + "-" * 8 + "|\n"
    rows = "".join(row_fmt.format(k, f"{v:.4f}") for k, v in results.items())
    return header + divider + header_row + sep_row + rows
