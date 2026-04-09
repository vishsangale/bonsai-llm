# Part 2: RAG & Generation Evals — Design Spec

**Date:** 2026-04-09
**Series:** Bonsai LLM Evals
**Format:** tutorial.md only (beginner-friendly, step-by-step)

---

## Overview

Part 2 extends the eval series into open-ended generation. Where Part 1 used log-prob scoring on multiple-choice benchmarks, Part 2 evaluates a RAG pipeline using RAGAS — an LLM-as-judge framework that scores outputs across six dimensions. The tutorial runs the same pipeline against three different knowledge bases and compares results, giving readers concrete intuition for how knowledge base quality affects eval scores.

---

## Tutorial Structure

Follows Part 1's numbered steps format.

| Step | Title |
|---|---|
| 1 | Why generation evals are different |
| 2 | The three knowledge bases and why they matter |
| 3 | RAG pipeline overview |
| 4 | What RAGAS is + environment setup |
| 5 | The six RAGAS metrics, one at a time |
| 6 | Run the script across all three knowledge bases |
| 7 | Read the output |
| 8 | Interpret the cross-dataset comparison |
| 9 | Limits: GPT-2 as generator, LLM-as-judge blind spots, contamination |
| 10 | When to use these evals — and when not to |

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

- **Dataset loaders** — one function per knowledge base, each returns a list of `(question, context, ground_truth)` tuples
- **RAG pipeline** — sentence-transformers for embeddings, FAISS for retrieval, GPT-2 for generation
- **RAGAS scorer** — takes pipeline outputs, runs all 6 metrics via OpenAI judge, returns results dict
- **Reporter** — prints comparison table (same style as Part 1), saves `results_<dataset>.json`
- **API key** — loaded from `.env` via `python-dotenv`, fails fast with a clear error if missing

**Dependencies:**
```
ragas langchain sentence-transformers faiss-cpu openai python-dotenv wikipedia huggingface_hub transformers torch
```

---

## Knowledge Bases

| Name | Source | Characteristics |
|---|---|---|
| `wikipedia` | `wikipedia` Python package, 5-10 science articles | Clean, factual, good baseline |
| `custom` | ~20 hand-written Q&A pairs in the repo | Controlled ground truth, best-case scenario |
| `hf-model-cards` | README text from 5-10 popular HF models via `huggingface_hub` | Noisy, technical, real-world-ish |

Each dataset includes ground truth answers so all 6 RAGAS metrics have something to score against.

---

## RAGAS Metrics

| Metric | What it catches |
|---|---|
| Faithfulness | Hallucination — did the answer contradict the context? |
| Answer Relevance | Did the answer actually address the question? |
| Context Precision | Is the retrieved context signal or noise? |
| Context Recall | Did retrieval miss anything the answer needed? |
| Answer Correctness | Is the answer factually right vs. ground truth? |
| Answer Similarity | Semantic closeness to ground truth (even if phrased differently) |

**Judge model:** OpenAI (key loaded from `.env`)
**Generator model:** GPT-2 (HuggingFace, no API key required)
**Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`

---

## Expected Output

```
Dataset: wikipedia
| Metric              | Score |
|---------------------|-------|
| Faithfulness        | x.xx  |
| Answer Relevance    | x.xx  |
| Context Precision   | x.xx  |
| Context Recall      | x.xx  |
| Answer Correctness  | x.xx  |
| Answer Similarity   | x.xx  |

...

Cross-dataset comparison (all metrics, acc_norm style table)
Raw results saved to results_<dataset>.json
```

---

## Narrative Arc

The tutorial's central insight: the same pipeline, same model, same metrics — but different knowledge bases produce meaningfully different scores. Custom (controlled) scores highest. Wikipedia (clean but general) sits in the middle. HF model cards (noisy, domain-specific) scores lowest. This teaches readers that eval scores reflect the data as much as the model.

---

## Limits Section

- GPT-2 is not instruction-tuned — generation quality is intentionally poor to make retrieval failures visible
- LLM-as-judge has its own biases (verbosity, self-preference if same model family)
- Ground truth quality caps context recall and answer correctness scores
- Benchmark contamination less of a concern here than in Part 1 (custom eval set)
