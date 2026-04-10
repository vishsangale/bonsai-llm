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

RAGAS addresses both problems. It scores six dimensions independently — faithfulness, response relevancy, context precision, context recall, answer correctness, and semantic similarity — so you can see exactly where the pipeline breaks down.

---

## Step 2 — The three knowledge bases and why they matter

We run the same pipeline against three corpora:

**Custom** — 10 short passages written specifically for this tutorial. Questions and answers are tightly coupled to the text. This is the best-case scenario: retrieval is easy, answers are verifiable, ground truth is clean.

**Wikipedia** — 8 real Wikipedia articles on science topics (Photosynthesis, DNA, Black hole, Plate tectonics, Vaccine, Neuron, Quantum mechanics, Evolution). The text is longer, more densely structured, and the questions are written against the actual article content.

**HF Model Cards** — README text from 8 popular HuggingFace models (bert-base-uncased, gpt2, t5-small, distilbert-base-uncased, roberta-base, facebook/bart-large-cnn, openai/whisper-small, google/flan-t5-base). Technical, domain-specific, and written for a different audience than the eval questions. The hardest corpus.

The experiment is not about which corpus is "best" in general — it is about watching how the same pipeline responds to different data quality and domain match. That is the main thing RAGAS lets you see.

---

## Step 3 — RAG pipeline overview

The pipeline has four stages:

1. **Chunk** — Split each document into 256-word chunks with 32-word overlap. Overlap ensures sentences at chunk boundaries are not split and lost.
2. **Embed** — Encode all chunks into dense vectors using `sentence-transformers/all-MiniLM-L6-v2`. These vectors live in a FAISS index built with cosine similarity (L2-normalized inner product).
3. **Retrieve** — For each question, encode it with the same embedding model and return the top-3 most similar chunks.
4. **Generate** — Feed the question and retrieved chunks to Gemma-3-1b-it with a strict prompt: "Answer using only the context below. If the answer is not in the context, say 'I don't know'."

The prompt constraint matters for evaluation: if the model generates text that goes beyond the retrieved context, faithfulness drops. If it answers correctly from the context, faithfulness stays high. This makes faithfulness a useful signal for prompt engineering — tightening or loosening the grounding instruction changes the score directly.

---

## Step 4 — What RAGAS is, and setting up your environment

RAGAS (Retrieval Augmented Generation Assessment) is an open-source framework for evaluating RAG pipelines without hand-labeling every output. It uses an LLM-as-judge to score dimensions like faithfulness and relevancy automatically, so you get structured signal instead of eyeballing model outputs.

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

| Metric | What it catches | How scored |
|---|---|---|
| `Faithfulness` | Did the answer contradict the context? | LLM |
| `ResponseRelevancy` | Did the answer address the question? | LLM + embeddings |
| `ContextPrecision` | Is the retrieved context signal or noise? | LLM |
| `ContextRecall` | Did retrieval miss anything the answer needed? | LLM |
| `AnswerCorrectness` | Is the answer right compared to the reference? | LLM + embeddings |
| `SemanticSimilarity` | How close is the answer to the reference semantically? | Embeddings only |

A few things worth knowing before you see numbers:

**Faithfulness** is the hallucination check. A faithful answer only makes claims supported by the retrieved context. If Gemma-3 adds facts from its pretraining that aren't in the retrieved chunks, faithfulness drops. The metric works by asking the judge to verify each claim in the answer against the context.

**ContextRecall** requires a ground truth answer. The judge asks: is everything in the reference answer covered by the retrieved chunks? Low recall means the retriever is missing relevant content — the generator cannot compensate for retrieval failures.

**SemanticSimilarity** is embedding-based only — no LLM judge call needed. It measures how close the generated answer is to the reference in vector space. Useful as a fast sanity check that avoids the judge's verbosity bias.

---

## Step 6 — Get the script

```bash
git clone https://github.com/vishsangale/bonsai-llm
cd bonsai-llm/posts/part2-rag-evals
```

The data files (`data/custom_corpus.json`, `data/eval_samples.json`) are already committed. Corpus documents and eval samples are kept separate so retrieval is real, not lookup.

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

Here is the actual output for the custom corpus (best-case scenario, hand-written passages):

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

The `nan` for faithfulness is not a bug in your setup — it reflects real behavior worth understanding. The RAGAS faithfulness metric works by asking the judge LLM to return structured verdicts for each claim. When the LLM's response does not match the expected format exactly (as happened here with Gemini 2.5 Flash), the metric cannot compute a score and returns `nan`. This is a known rough edge in RAGAS when using judge models other than the default (OpenAI). The other five metrics are unaffected.

The scores on a 0–1 scale. `context_precision` and `context_recall` at 1.0 for the custom corpus tells you the retriever is returning exactly the right chunks — which makes sense when questions are written directly against the indexed passages. This is the best-case retrieval scenario, and the numbers confirm it.

`answer_correctness` at 0.49 and `semantic_similarity` at 0.53 are lower than you might expect given perfect retrieval. The generator is finding the right context but the answers are not semantically identical to the reference. For a 1B instruction-tuned model on a strict "only use context" prompt, this is typical: the model paraphrases rather than reproduces, and paraphrases score lower on embedding similarity than near-verbatim answers.

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

The most striking result is not the expected decline — it is that **hf-model-cards outperforms Wikipedia on every metric**.

Custom dominates as expected (questions written against the exact indexed text, best-case retrieval), but model card READMEs score higher than Wikipedia articles on context precision (0.56 vs 0.33), context recall (0.45 vs 0.30), and semantic similarity (0.43 vs 0.22). This is counterintuitive given that model cards mix installation instructions, performance benchmarks, and licensing notes in the same document.

The likely explanation is **domain alignment**. The embedding model (`all-MiniLM-L6-v2`) was trained on a broad corpus that includes ML documentation. It maps model card text and ML-related questions into a shared region of the embedding space more effectively than it maps Wikipedia's prose-heavy science writing. The chunking also helps: model card sections are naturally short and self-contained, so 256-word chunks tend to contain one coherent idea. Wikipedia paragraphs often build across many sentences, making a single chunk harder to match to a narrow question.

**Wikipedia's poor retrieval** (context precision 0.33, recall 0.30) reflects the opposite problem: science articles use varied vocabulary, passive voice, and cross-referencing that does not align cleanly with question-style queries. "What is the role of ATP in photosynthesis?" and the paragraph containing the answer use different enough phrasing that cosine similarity underperforms.

**AnswerCorrectness** tracks the retrieval results: 0.49 (custom) → 0.15 (Wikipedia) → 0.28 (model cards). When the retriever fails to surface relevant context, the generator produces "I don't know" or off-target answers — there is no prompt engineering fix for missing context.

**ResponseRelevancy** is the most stable metric (0.36 → 0.22 → 0.50) because it evaluates whether the answer addresses the question, regardless of factual correctness. The model card answers score highest here: even when factually incomplete, the generator stays on-topic because the retrieved model card text is thematically aligned with the question.

The key diagnostic insight: if you see low context precision and recall alongside low answer correctness, the problem is retrieval — better chunking, a stronger embedding model, or more corpus preprocessing will help more than prompt engineering. Domain match between the embedding model's training distribution and your corpus is a silent lever that most pipeline audits miss.

---

## Step 10 — Limits

**LLM-as-judge biases.** The judge LLM has a verbosity bias — it tends to score longer, more detailed answers higher even when they are not more faithful. It can also exhibit self-preference if the judge and generator share training data or a model family. Treat scores as relative comparisons between pipeline configurations, not absolute ground truth.

**Faithfulness NaN.** As shown above, the faithfulness metric depends on the judge returning structured verdicts in an exact format. When using non-default judge models (anything other than OpenAI GPT-4), the probability of format mismatch is higher. If faithfulness matters for your use case, test your specific judge model against RAGAS's faithfulness metric on a small sample before running a full eval.

**Source leakage.** The custom dataset is most susceptible: when reference answers are written from the same passages you indexed, `ContextRecall` measures how well you reproduced your own text, not real retrieval quality. The Wikipedia and HF model card samples are less coupled because the questions are written against web sources, not the exact text in the FAISS index.

**Ground truth quality.** `AnswerCorrectness` and `ContextRecall` scores are only as good as the reference answers. A vague or incomplete reference produces misleading numbers regardless of pipeline quality.

**HuggingFace license gate.** Gemma-3-1b-it requires accepting Google's terms of use once at hf.co. If you hit an authentication error on first run, check that you ran `huggingface-cli login` and that your token has read access.

---

## Step 11 — When to use these evals — and when not to

| Use RAGAS for... | Don't use it for... |
|---|---|
| Comparing two retrieval strategies | Predicting user satisfaction |
| Catching faithfulness regressions after prompt changes | Measuring safety or alignment |
| Diagnosing whether failures are retrieval or generation | Evaluating open-ended creative tasks |
| Reproducible baselines with a fixed corpus | Final product quality decisions alone |

The diagnostic split is the main reason to run all six metrics. If faithfulness is high but answer correctness is low, the generator is using the context but the retriever is surfacing the wrong content. If context recall is high but faithfulness is low, the right content is retrieved but the generator is going off-script. Different failure modes need different fixes.

The comparison across corpora is also the most direct way to answer a question practitioners often have: "how much does corpus quality matter?" The answer here is stark — going from hand-written passages to real Wikipedia articles dropped context precision by 70 percentage points, with a corresponding drop in answer correctness. Corpus curation is not preprocessing noise; it is a first-order performance lever.

---

## What's next

**Part 3** compares the major eval frameworks side-by-side: lm-eval-harness, DeepEval, RAGAS, and Inspect. Same models, same tasks — different tools, different trade-offs.

---

*Full code: [github.com/vishsangale/bonsai-llm/tree/main/posts/part2-rag-evals](https://github.com/vishsangale/bonsai-llm/tree/main/posts/part2-rag-evals)*
