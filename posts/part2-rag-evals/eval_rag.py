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
import wikipedia as wiki_api
from huggingface_hub import ModelCard

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
CHUNK_SIZE = 256   # tokens (approximated as words for simplicity)
CHUNK_OVERLAP = 32
TOP_K = 3
MAX_NEW_TOKENS = 200


# ── Chunker ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks."""
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")
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


# ── Corpus Loaders ────────────────────────────────────────────────────────────

def load_custom_corpus() -> list[str]:
    """Load hand-written passages from data/custom_corpus.json."""
    with open(DATA_DIR / "custom_corpus.json") as f:
        items = json.load(f)
    return [item["text"] for item in items]


WIKIPEDIA_ARTICLES = [
    "Photosynthesis", "DNA", "Black hole", "Plate tectonics",
    "Vaccine", "Neuron", "Quantum mechanics", "Evolution"
]

def load_wikipedia_corpus() -> list[str]:
    """Fetch pinned Wikipedia articles and return their summaries."""
    docs = []
    for title in WIKIPEDIA_ARTICLES:
        try:
            page = wiki_api.page(title, auto_suggest=False)
            docs.append(page.content[:3000])  # first ~3000 chars to keep it manageable
        except Exception as e:
            print(f"  Warning: could not fetch '{title}': {e}")
    return docs


HF_MODELS = [
    "bert-base-uncased", "gpt2", "t5-small", "distilbert-base-uncased",
    "roberta-base", "facebook/bart-large-cnn", "openai/whisper-small",
    "google/flan-t5-base"
]

def load_hf_model_cards_corpus() -> list[str]:
    """Fetch model card text from pinned HuggingFace models."""
    docs = []
    for model_id in HF_MODELS:
        try:
            card = ModelCard.load(model_id)
            docs.append(card.text[:3000])
        except Exception as e:
            print(f"  Warning: could not fetch card for '{model_id}': {e}")
    return docs


CORPUS_LOADERS = {
    "wikipedia": load_wikipedia_corpus,
    "custom": load_custom_corpus,
    "hf-model-cards": load_hf_model_cards_corpus,
}
