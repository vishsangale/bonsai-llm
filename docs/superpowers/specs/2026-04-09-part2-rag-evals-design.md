# Part 2: RAG & Generation Evals — Design Spec

**Date:** 2026-04-09
**Series:** Bonsai LLM Evals
**Format:** tutorial.md only (beginner-friendly, step-by-step)

---

## Overview

Part 2 extends the eval series into open-ended generation. Where Part 1 used log-prob scoring on multiple-choice benchmarks, Part 2 evaluates a RAG pipeline using RAGAS — a framework that scores outputs across six dimensions using a mix of LLM-as-judge and embedding-based metrics. The tutorial runs the same pipeline against three different knowledge bases and compares results, giving readers concrete intuition for how knowledge base quality affects eval scores.

---

## Tutorial Structure

Follows Part 1's numbered steps format.

| Step | Title |
|---|---|
| 1 | Why generation evals are different |
| 2 | The three knowledge bases and why they matter |
| 3 | RAG pipeline overview (chunking, retrieval, generation) |
| 4 | What RAGAS is + accept the HF license + environment setup |
| 5 | The six RAGAS metrics, one at a time |
| 6 | Get the script |
| 7 | Run the script across all three knowledge bases |
| 8 | Read the output |
| 9 | Interpret the cross-dataset comparison |
| 10 | Limits: LLM-as-judge blind spots, source leakage, corpus-question coupling |
| 11 | When to use these evals — and when not to |

---

## Script Design

**File:** `posts/part2-rag-evals/eval_rag.py`

**CLI:**
```
python eval_rag.py --dataset wikipedia     # single dataset
python eval_rag.py --dataset custom
python eval_rag.py --dataset hf-model-cards
python eval_rag.py --dataset all           # runs all three, prints comparison table
```

**Internal components:**

- **Corpus loaders** — one function per knowledge base, returns a list of raw document strings for indexing
- **Eval sample sets** — separate from corpus; each sample is `(question, reference_answer)`. Kept distinct so retrieval is real, not lookup.
- **RAG pipeline** — sentence-transformers for embeddings, FAISS for retrieval (top-k=3, chunk size=256 tokens, overlap=32), `gemma-3-1b-it` for generation with a strict "answer only from context; say I don't know if unsupported" prompt template
- **RAGAS scorer** — takes `(question, retrieved_contexts, response, reference)` per sample, runs all 6 metrics, returns results dict. LLM-backed metrics use OpenAI; embedding-backed metrics use `sentence-transformers/all-MiniLM-L6-v2` locally.
- **Reporter** — prints per-dataset table + cross-dataset comparison table, saves `results_<dataset>.json`
- **API key** — `OPENAI_API_KEY` loaded from `.env` via `python-dotenv`, fails fast with a clear error if missing

**Pinned dependency versions:**
```
ragas==0.2.x  sentence-transformers faiss-cpu openai python-dotenv
wikipedia huggingface_hub transformers torch
```
(no `langchain` — use RAGAS native API directly to avoid version drift)

---

## Knowledge Bases

### Corpus + eval sample separation

Each knowledge base has two parts:
- **Corpus documents** — raw text chunks indexed into FAISS for retrieval
- **Eval samples** — separate `(question, reference_answer)` pairs the retriever has not seen directly

| Name | Corpus source | Eval samples | Characteristics |
|---|---|---|---|
| `wikipedia` | 8 fixed science articles (pinned list below) | 20 questions written against those articles | Clean, factual, good baseline |
| `custom` | 10 short hand-written topic passages in the repo | 20 questions written against those passages | Controlled ground truth, best-case scenario |
| `hf-model-cards` | README text from 8 pinned HF models (list below) | 20 questions about those models | Noisy, technical, real-world-ish |

**Pinned Wikipedia articles:** Photosynthesis, DNA, Black hole, Plate tectonics, Vaccine, Neuron, Quantum mechanics, Evolution

**Pinned HF model cards:** `bert-base-uncased`, `gpt2`, `t5-small`, `distilbert-base-uncased`, `roberta-base`, `facebook/bart-large-cnn`, `openai/whisper-small`, `google/flan-t5-base`

All corpus text and eval samples are committed to the repo so results are reproducible.

---

## RAGAS Metrics

Using exact RAGAS API names:

| RAGAS class | Plain-English gloss | Scored by |
|---|---|---|
| `Faithfulness` | Did the answer contradict the context? | LLM (OpenAI) |
| `ResponseRelevancy` | Did the answer address the question? | LLM + embeddings |
| `ContextPrecision` | Is retrieved context signal or noise? | LLM (OpenAI) |
| `ContextRecall` | Did retrieval miss anything needed? | LLM (OpenAI) |
| `AnswerCorrectness` | Is the answer factually right vs. reference? | LLM + embeddings |
| `SemanticSimilarity` | Semantic closeness to reference answer | Embeddings only |

**Generator model:** `google/gemma-3-1b-it` (HuggingFace, gated — readers accept license once at hf.co/google/gemma-3-1b-it)
**Judge model:** OpenAI (`OPENAI_API_KEY` from `.env`)
**Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, no API key)

---

## RAG Pipeline Config

Documented explicitly in the tutorial for reproducibility:

- **Chunking:** 256-token chunks, 32-token overlap, split on sentence boundaries
- **Retrieval:** top-k=3 chunks per question
- **Prompt template:** `"Answer the question using only the context below. If the answer is not in the context, say 'I don't know'.\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"`
- **Generation:** greedy decoding, max 200 new tokens

---

## Expected Output

```
Dataset: wikipedia
| Metric               | Score |
|----------------------|-------|
| Faithfulness         | x.xx  |
| ResponseRelevancy    | x.xx  |
| ContextPrecision     | x.xx  |
| ContextRecall        | x.xx  |
| AnswerCorrectness    | x.xx  |
| SemanticSimilarity   | x.xx  |

...

Cross-dataset comparison (all metrics)
| Metric               | wikipedia | custom | hf-model-cards |
|----------------------|-----------|--------|----------------|
| Faithfulness         | x.xx      | x.xx   | x.xx           |
| ...                  |           |        |                |

Raw results saved to results_<dataset>.json
```

Tutorial includes one worked example showing: question → top-3 retrieved chunks → generated answer → per-metric scores with interpretation.

---

## Narrative Arc

The tutorial's central hypothesis (framed explicitly as a hypothesis, not a guaranteed outcome): the same pipeline, same model, same metrics — but different knowledge bases will likely produce different scores because corpus quality, domain match, and chunk coherence all affect retrieval and generation. Running it and seeing the actual numbers is the point.

---

## Limits Section

- **LLM-as-judge biases** — verbosity bias, self-preference if judge and generator share a model family
- **Source leakage** — if reference answers are written from the same passages indexed in FAISS, context recall is artificially inflated. The custom set is most susceptible; the tutorial flags this explicitly.
- **Corpus-question coupling** — questions too tailored to exact corpus wording inflate precision/recall; questions too far from corpus content floor them. The eval samples are written to be realistic, not adversarial.
- **Ground truth quality** — `AnswerCorrectness` and `ContextRecall` scores are only as good as the reference answers. Poor references produce misleading numbers.
- **HF license gate** — `gemma-3-1b-it` requires accepting terms once at hf.co. Tutorial includes this as an explicit setup step.
