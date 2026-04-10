# RAG Evals From Scratch: Measuring What Your Pipeline Gets Wrong

*Part 2 of the Bonsai LLM eval series — Beginner*

> **Runtime note:** First run downloads Gemma-3-1b-it (~2GB) and the embedding model (~90MB). Generating answers and scoring with RAGAS takes another 20–40 minutes depending on your hardware. You will need a Gemini API key for the RAGAS judge metrics (free tier works).

---

## Prerequisites

- Python 3.9+
- A terminal and a virtual environment
- A Gemini API key (for RAGAS judge metrics — free at [aistudio.google.com](https://aistudio.google.com))
- A HuggingFace account with the Gemma-3 license accepted (see Step 4)
- A GPU is helpful but not required — CPU works for 1B models

---

## Step 1 — Why generation evals are different

Part 1 showed how to evaluate a model on multiple-choice benchmarks using log-probability scoring. The process is deterministic: given a question and four options, the model assigns a score to each, and the highest wins.

Generation evals are messier. When a model produces free text, there is no single correct token sequence to compare against. "The sky is blue" and "The sky appears blue" are both correct answers to the same question, but character-by-character comparison treats them as different.

RAG adds another layer: the pipeline has two failure modes, not one. The retriever can fail to surface the right context, or the generator can fail to use the context it was given. A single accuracy number cannot tell you which failed.

RAGAS addresses both problems. It scores six dimensions independently — faithfulness, response relevancy, context precision, context recall, answer correctness, and semantic similarity — so you can separate retrieval failures from generation failures.

---

## Step 2 — The three knowledge bases and why they matter

We run the same pipeline against three corpora:

**Custom** — 10 tutorial-specific passages. Questions are written directly against the text. Best-case scenario: retrieval is easy, answers are verifiable, ground truth is clean.

**Wikipedia** — 8 real Wikipedia articles on science topics (Photosynthesis, DNA, Black hole, Plate tectonics, Vaccine, Neuron, Quantum mechanics, Evolution). Longer text, denser structure, eval questions target facts from the article text.

**HF Model Cards** — README text from 8 popular HuggingFace models (bert-base-uncased, gpt2, t5-small, distilbert-base-uncased, roberta-base, facebook/bart-large-cnn, openai/whisper-small, google/flan-t5-base). Technical, domain-specific; model authors wrote them for practitioners, not eval prompts.

This setup tests how the same RAG pipeline behaves as corpus quality and domain match change.

---

## Step 3 — RAG pipeline overview

The pipeline has four stages:

1. **Chunk** — Split each document into 256-word chunks with 32-word overlap. Overlap reduces boundary loss by repeating 32 words between adjacent chunks.
2. **Embed** — Encode all chunks into dense vectors using `sentence-transformers/all-MiniLM-L6-v2`. These vectors live in a FAISS index built with cosine similarity (L2-normalized inner product).
3. **Retrieve** — For each question, encode it with the same embedding model and return the top-3 most similar chunks.
4. **Generate** — Feed the question and retrieved chunks to Gemma-3-1b-it with a strict prompt: "Answer using only the context below. If the answer is not in the context, say 'I don't know'."

The prompt constraint matters for evaluation. If every claim in the answer is supported by retrieved context, faithfulness stays high. If the model pulls from pretraining instead, faithfulness drops. This makes faithfulness a direct signal for how well the grounding instruction is working.

---

## Step 4 — Setup

**Accept the Gemma-3 license** (one-time):

Visit [hf.co/google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) and accept the terms. Then log in from the terminal:

```bash
huggingface-cli login
```

**Create your environment:**

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install ragas langchain-google-genai langchain-community \
            sentence-transformers faiss-cpu python-dotenv \
            wikipedia huggingface_hub transformers torch datasets
```

**Set your Gemini key:**

```bash
cp .env.example .env
# edit .env and add: GEMINI_API_KEY=your-key-here
```

The script loads the key from `.env` automatically via `python-dotenv`.

---

## Step 5 — The six RAGAS metrics

RAGAS scores each sample individually then averages across the dataset. Some metrics use an LLM judge (Gemini in this setup), others use embedding similarity only.

| Metric | What it measures | How scored |
|---|---|---|
| `Faithfulness` | Are the answer's claims supported by retrieved context? | LLM |
| `ResponseRelevancy` | Does the answer address the question? | LLM + embeddings |
| `ContextPrecision` | Are relevant chunks ranked above irrelevant ones? | LLM |
| `ContextRecall` | Did retrieval cover the claims in the reference answer? | LLM |
| `AnswerCorrectness` | Is the answer right compared to the reference? | LLM + embeddings |
| `SemanticSimilarity` | How close is the answer to the reference in vector space? | Embeddings only |

**Faithfulness** is the hallucination check. If Gemma-3 adds facts from its pretraining that aren't in the retrieved chunks, faithfulness drops. The metric works by asking the judge to verify each claim in the answer against the context.

**ContextPrecision** is rank-sensitive: it penalizes pipelines that retrieve the right content but bury it behind irrelevant chunks. A low score means useful context exists but the retriever ranked noise above it.

**ContextRecall** compares what was retrieved against what the reference answer claims. Low recall means the retriever missed content the generator needed — no prompt change fixes that.

**SemanticSimilarity** is embedding-based only — no LLM judge call needed. Use it as a fast sanity check that avoids the judge's verbosity bias.

---

## Step 6 — Get the script

```bash
git clone https://github.com/vishsangale/bonsai-llm
cd bonsai-llm/posts/part2-rag-evals
```

The repo includes `data/custom_corpus.json` and `data/eval_samples.json`. Corpus documents and eval samples are kept separate so retrieval is real, not lookup.

---

## Step 7 — Run the script

Single dataset (fastest, good for testing your setup):

```bash
python eval_rag.py --dataset custom
```

All three datasets (runs sequentially, saves one JSON per dataset):

```bash
python eval_rag.py --dataset all
```

On first run, the script downloads Gemma-3-1b-it (~2GB) and the embedding model (~90MB). Subsequent runs skip downloads. The Gemini judge calls happen during scoring — expect a few hundred API calls per dataset.

---

## Step 8 — Read the output

Custom corpus output:

```
Dataset: custom
──────────────────────────────────────────────────
| Metric                 |  Score |
|------------------------|--------|
| faithfulness           |    nan |
| answer_relevancy       | 0.3640 |
| context_precision      | 1.0000 |
| context_recall         | 1.0000 |
| answer_correctness     | 0.4897 |
| semantic_similarity    | 0.5256 |
```

The `nan` for faithfulness comes from the metric failing to produce a numeric score in this run. The RAGAS faithfulness metric asks the judge to return structured verdicts for each claim; when the response does not match the expected format, the score is undefined. This happened consistently with Gemini 2.5 Flash across all three datasets. Test your judge model against faithfulness on a small sample before relying on it. The other five metrics are unaffected.

`context_precision` and `context_recall` at 1.0 means the retriever put sufficient relevant context at the top and covered the reference claims — expected when questions are written directly against the indexed passages.

`answer_correctness` at 0.49 and `semantic_similarity` at 0.53 are lower than you might expect given perfect retrieval. Before concluding the generator is just paraphrasing, inspect the raw responses: with a 1B model on a strict "only use context" prompt, incomplete or off-target answers are common and will drag both scores down.

---

## Step 9 — Interpret the cross-dataset comparison

| Metric | custom | wikipedia | hf-model-cards |
|---|---|---|---|
| faithfulness | nan | nan | nan |
| answer_relevancy | 0.3640 | 0.2239 | 0.4997 |
| context_precision | 1.0000 | 0.3250 | 0.5583 |
| context_recall | 1.0000 | 0.3000 | 0.4500 |
| answer_correctness | 0.4897 | 0.1541 | 0.2752 |
| semantic_similarity | 0.5256 | 0.2166 | 0.4257 |

**hf-model-cards outperforms Wikipedia on every non-NaN metric.** That's counterintuitive — model card READMEs mix installation instructions, performance benchmarks, and licensing notes in the same document, while Wikipedia articles are structured prose on a single topic.

One hypothesis: domain alignment. Model card text and ML-related eval questions may sit closer together under this embedding model than Wikipedia science prose does. The chunking also helps: model card sections tend to be short and self-contained, so a 256-word chunk usually contains one coherent idea. Wikipedia paragraphs build across sentences, so a chunk is more likely to contain partial context for any given question.

**Wikipedia's retrieval failure** (context precision 0.33, recall 0.30) compounds into a large correctness drop. `AnswerCorrectness` falls from 0.49 (custom) → 0.15 (Wikipedia) → 0.28 (model cards). When the retriever misses, the generator either says "I don't know" or produces off-target answers — there is no prompt fix for missing context.

**ResponseRelevancy** is the most stable metric (0.36 → 0.22 → 0.50) because it scores whether the answer addresses the question, not whether it's correct. Model card answers score highest here: even when factually incomplete, the generator stays on-topic because the retrieved text is thematically aligned with the question.

Low context precision and recall alongside low answer correctness points to retrieval first. Corpus quality moved the metrics more than any prompt change would.

---

## Step 10 — Limits

**LLM-as-judge biases.** The judge LLM has a verbosity bias — it tends to score longer answers higher even when they are not more faithful. It can also exhibit self-preference if the judge and generator share a model family. Treat scores as relative comparisons between pipeline configurations, not absolute ground truth.

**Faithfulness NaN.** This run returned `NaN` for faithfulness with Gemini 2.5 Flash across all three datasets. Test your chosen judge on a small sample before a full eval run to verify it produces numeric faithfulness scores.

**Source leakage.** The custom dataset is most susceptible: the reference answers were written from the same small artificial corpus that's indexed in FAISS. High recall on custom is less impressive for this reason. The Wikipedia and model card samples are written against web sources and are less coupled to the indexed text.

**Ground truth quality.** `AnswerCorrectness` and `ContextRecall` are only as good as the reference answers. A vague or incomplete reference produces misleading numbers regardless of pipeline quality.

**HuggingFace license gate.** Gemma-3-1b-it requires accepting Google's terms of use once at hf.co. If you hit an authentication error on first run, check that you ran `huggingface-cli login` and that your token has read access.

---

## Step 11 — When to use these evals — and when not to

| Use RAGAS for... | Don't use it for... |
|---|---|
| Comparing two retrieval strategies | Predicting user satisfaction |
| Catching faithfulness regressions after prompt changes | Measuring safety or alignment |
| Diagnosing whether failures are retrieval or generation | Evaluating open-ended creative tasks |
| Reproducible baselines with a fixed corpus | Final product quality decisions alone |

The diagnostic split is the main reason to run all six metrics. High faithfulness but low answer correctness: the generator is using context but the retriever is surfacing the wrong chunks. High context recall but low faithfulness: the right content is retrieved but the generator is going off-script. Different failure modes need different fixes.

---

## What's next

**Part 3** compares the major eval frameworks side-by-side: lm-eval-harness, DeepEval, RAGAS, and Inspect. Same models, same tasks — different tools, different trade-offs.

---

*Full code: [github.com/vishsangale/bonsai-llm/tree/main/posts/part2-rag-evals](https://github.com/vishsangale/bonsai-llm/tree/main/posts/part2-rag-evals)*
