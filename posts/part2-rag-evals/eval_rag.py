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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset as HFDataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
    SemanticSimilarity,
)

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

    if not chunks:
        raise ValueError("No chunks produced from docs — corpus may be empty.")

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
    _, ids = index.search(q_vec, min(k, len(chunks)))
    return [chunks[i] for i in ids[0] if i < len(chunks)]


# ── Generator ─────────────────────────────────────────────────────────────────

GENERATOR_MODEL = "google/gemma-3-1b-it"
_gen_model = None
_gen_tokenizer = None

PROMPT_TEMPLATE = (
    "Answer the question using only the context below. "
    "If the answer is not in the context, say 'I don't know'.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

def get_generator():
    global _gen_model, _gen_tokenizer
    if _gen_model is None:
        print(f"Loading generator model {GENERATOR_MODEL!r}...")
        _gen_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
        _gen_model = AutoModelForCausalLM.from_pretrained(
            GENERATOR_MODEL,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if not torch.cuda.is_available():
            _gen_model = _gen_model.to("cpu")
    return _gen_model, _gen_tokenizer

def generate_answer(question: str, contexts: list[str]) -> str:
    """Generate an answer conditioned on retrieved contexts using Gemma-3-1b-it."""
    model, tokenizer = get_generator()
    context_str = "\n\n".join(contexts)
    user_content = PROMPT_TEMPLATE.format(context=context_str, question=question)

    # Use the chat template so the instruction-tuned model follows the prompt correctly
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens (after the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── RAGAS Scoring ─────────────────────────────────────────────────────────────

RAGAS_METRICS = [
    Faithfulness(),
    ResponseRelevancy(),
    ContextPrecision(),
    ContextRecall(),
    AnswerCorrectness(),
    SemanticSimilarity(),
]


def score_with_ragas(samples: list[dict]) -> dict:
    """
    Run RAGAS on a list of dicts with keys:
        user_input, retrieved_contexts (list[str]), response, reference
    Returns dict of metric_name -> float score.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Add it to your .env file. "
            "See .env.example for the format."
        )

    dataset = HFDataset.from_list(samples)
    result = evaluate(dataset=dataset, metrics=RAGAS_METRICS)
    # result.scores is a list of per-sample dicts; average across samples
    scores = {}
    for metric_name in result.scores[0]:
        vals = [s[metric_name] for s in result.scores if s[metric_name] is not None]
        scores[metric_name] = sum(vals) / len(vals) if vals else float("nan")
    return scores


def run_dataset(dataset_name: str) -> dict:
    """Load corpus, build index, generate answers, score with RAGAS."""
    print(f"\n{'='*60}")
    print(f"Running dataset: {dataset_name}")
    print(f"{'='*60}")

    print("Loading corpus...")
    docs = CORPUS_LOADERS[dataset_name]()
    print(f"  {len(docs)} documents loaded")

    index, chunks = build_index(docs)

    print("Loading eval samples...")
    eval_samples = load_eval_samples(dataset_name)
    print(f"  {len(eval_samples)} questions")

    print("Generating answers...")
    ragas_samples = []
    for i, sample in enumerate(eval_samples):
        q = sample["question"]
        ref = sample["reference"]
        contexts = retrieve(q, index, chunks)
        response = generate_answer(q, contexts)
        ragas_samples.append({
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": response,
            "reference": ref,
        })
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(eval_samples)} done")

    print("Scoring with RAGAS...")
    scores = score_with_ragas(ragas_samples)
    return scores
