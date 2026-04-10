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
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

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


# ── Eval Sample Loader ────────────────────────────────────────────────────────

def load_eval_samples(dataset: str) -> list[dict]:
    """Return list of {question, reference} dicts for the given dataset."""
    with open(DATA_DIR / "eval_samples.json") as f:
        all_samples = json.load(f)
    if dataset not in all_samples:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(all_samples)}")
    return all_samples[dataset]


# ── Embedding & FAISS Index ───────────────────────────────────────────────────

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embed_model = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print("Loading embedding model...")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def build_index(docs: list[str]) -> tuple:
    """Chunk all docs, embed them, build a FAISS index.
    Returns (index, chunks) where chunks[i] corresponds to index vector i.
    """
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc))

    embed_model = get_embed_model()
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embed_model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks


def retrieve(question: str, index, chunks: list[str], k: int = TOP_K) -> list[str]:
    """Return top-k chunks most relevant to the question."""
    embed_model = get_embed_model()
    q_vec = embed_model.encode([question], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_vec)
    _, ids = index.search(q_vec, k)
    return [chunks[i] for i in ids[0] if i < len(chunks)]
