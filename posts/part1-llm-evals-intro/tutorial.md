# LLM Evals From Scratch: Run Your First Benchmarks

*Part 1 of the Bonsai LLM eval series — Beginner*

> **Runtime note:** First-run timing depends heavily on your hardware and connection — model and dataset downloads alone can take 10–20 minutes. On an M1 Mac on CPU, the three benchmark runs for GPT-2 took about 18 minutes. Expect more on older hardware.

---

## Prerequisites

- Python 3.9+
- A terminal and a virtual environment
- No GPU required (CPU is fine for GPT-2)

---

## Step 1 — Why training loss isn't enough

Here's a failure mode worth knowing before you spend time training anything. You run a model for a few thousand steps, validation loss drops steadily, you call it converged. Then you throw a basic factual question at it and it hallucinates confidently.

This happens because loss only measures how well the model predicts the next token on your training data. A model that memorizes plausible-sounding token sequences can get low loss while being wrong about nearly everything testable.

An **eval** ties a task, a metric, and a decision together — so the number you get means something beyond "did loss go down."

The taxonomy is worth knowing too, because people mix up format and scoring:

| Format | Examples | How it's typically scored |
|---|---|---|
| Multiple choice (2–4 options) | ARC, PIQA, HellaSwag, MMLU | Log-prob over choices |
| Short answer / math | GSM8K, MATH | Exact match |
| Long-form generation | Summarization, QA, code | ROUGE, BERTScore, LLM-as-judge |

GSM8K asks for a final numeric answer — it's a short-answer benchmark scored with exact match, not a "generation eval" just because the model generates text. This tutorial uses multiple-choice with log-prob scoring.

---

## Step 2 — The three benchmarks we'll run

### ARC-Easy (AI2 Reasoning Challenge)
4-way multiple choice, grade-school science. Here's a real item from the dataset:

> *"Which of the following best describes a physical change? (A) Burning wood (B) Rusting iron (C) Melting ice (D) Digesting food"*

Answer: C. The question requires knowing that melting is reversible and doesn't change chemical composition. Random baseline: 25%.

### PIQA (Physical Intuition QA)
2-way multiple choice about physical tasks. Real item:

> *Goal: Separate egg whites from yolk using a water bottle.*
> *Solution 1: Squeeze the bottle, hold it over the yolk, then release — the suction pulls the yolk in.*
> *Solution 2: Fill the bottle with water and pour it over the egg to wash away the whites.*

Answer: Solution 1. The model has to reason about suction and physical manipulation. Random baseline: 50%.

### HellaSwag
4-way sentence completion, adversarially constructed. Correct completions describe what naturally happens next; wrong options were specifically written to fool models that rely on surface-level patterns rather than understanding. That makes it harder than it looks, and harder than ARC and PIQA for most small models. Random baseline: 25%.

---

## Step 3 — Why we use acc_norm in this tutorial

For multiple-choice tasks, the harness scores each answer choice by its log-probability under the model. The problem: longer choices accumulate more log-probability tokens, creating a length bias toward verbose options.

**acc_norm** divides each choice's log-probability by its token count before comparing. When answer lengths are similar (like PIQA's two short options), the difference from raw `acc` is small. When they vary significantly, it matters — so we use `acc_norm` throughout to stay consistent.

> The harness outputs both `acc` and `acc_norm`. The numbers for PIQA will be nearly identical. For HellaSwag, they can differ by a few points because the wrong answers are sometimes much longer than the correct one.

---

## Step 4 — Set up your environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install lm_eval transformers torch
```

`lm_eval` is the [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

---

## Step 5 — Get the script

Clone the full repo:

```bash
git clone https://github.com/vishsangale/bonsai-llm
cd bonsai-llm/posts/part1-llm-evals-intro
python eval_gpt2.py
```

---

## Step 6 — What the script does

The core call:

```python
results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=gpt2",
    tasks=["arc_easy", "piqa", "hellaswag"],
    num_fewshot=0,
    batch_size=8,
)
```

`num_fewshot=0` means zero-shot: just the question, no worked examples in the prompt.

On the first run, the harness downloads benchmark datasets from HuggingFace into `~/.cache/huggingface/datasets/` — you'll see progress bars for each task. If you're on a slow connection this is where most of the wait time goes. The GPT-2 weights are about 550MB and land in `~/.cache/huggingface/hub/`. Subsequent runs skip all of this.

One thing that trips up beginners: if a task name is wrong or not installed, the harness fails with a `KeyError` rather than a clear error message. If that happens, run `lm_eval --tasks list` to see what's available and check your spelling.

---

## Step 7 — Read the output

```
|   Tasks   |Version|Filter|n-shot|  Metric  | Value |   |Stderr|
|-----------|------:|------|-----:|----------|------:|---|-----:|
|arc_easy   |      1|none  |     0|acc       | 0.4360|±  |0.0102|
|           |       |      |      |acc_norm  | 0.4394|±  |0.0102|
|hellaswag  |      1|none  |     0|acc       | 0.2886|±  |0.0045|
|           |       |      |      |acc_norm  | 0.2951|±  |0.0045|
|piqa       |      1|none  |     0|acc       | 0.6305|±  |0.0112|
|           |       |      |      |acc_norm  | 0.6250|±  |0.0112|
```

The `Stderr` column tells you how reliable the estimate is. For ARC-Easy (~2600 questions) it's ±0.01, meaning the true number is probably within one percentage point of what's shown. For tasks with fewer examples, stderr grows — keep that in mind when comparing small differences between models.

---

## Step 8 — Interpret the results

| Task | acc_norm (GPT-2) | Random |
|---|---|---|
| ARC-Easy | ~44% | 25% |
| PIQA | ~62% | 50% |
| HellaSwag | ~30% | 25% |

The PIQA score is the interesting one. GPT-2 at ~62% on a 50% baseline is a larger gain than ARC-Easy's ~44% on a 25% baseline, in relative terms. That's not because PIQA is easier — it's because GPT-2's training data (web text) is saturated with "how to" content about physical tasks. GPT-2 picked up enough statistical regularity about physical tasks from web text that it transfers to this benchmark.

HellaSwag at ~30% is close to chance. That's expected here: GPT-2 was trained to predict tokens, not to reason about what comes next in a scenario. HellaSwag was designed so that wrong answers score high on surface-level statistics — exactly the thing GPT-2 is good at — which is why small base completion models consistently score poorly on it.

---

## Step 9 — Compare model sizes

Edit the `MODEL` variable at the top of `eval_gpt2.py`:

```python
MODEL = "gpt2"         # 124M parameters
MODEL = "gpt2-medium"  # 355M parameters
MODEL = "gpt2-large"   # 774M parameters
MODEL = "gpt2-xl"      # 1.5B parameters
```

All three scores improve with size. Anecdotally, HellaSwag tends to move more than ARC-Easy as you go up the GPT-2 family — the adversarial task seems to benefit more from additional capacity — but run it yourself and see what you observe. The gap between acc and acc_norm may also narrow at larger sizes; that's worth watching.

The script saves `results_<model>.json` after each run. Diff two JSON files to track which tasks regress after fine-tuning — that's the main practical use of these baselines.

---

## Step 10 — Limits and contamination

**Benchmark contamination** is worth knowing about before you trust leaderboard numbers. If a model's training data contains benchmark test items — even partially — the scores overstate real capability. This is a genuine concern for models trained on large web crawls, many of which postdate these benchmarks. GPT-2 is old enough that contamination is less of a concern here.

The practical implication: public benchmarks are reliable for comparing models trained under similar conditions and for catching regressions. For product decisions, use held-out evals that match your actual use case — ones you control and the model has never seen.

---

## When to use these evals — and when not to

| Use benchmark evals for... | Don't use them for... |
|---|---|
| Comparing two models on the same task | Predicting real-world helpfulness |
| Catching capability regressions after fine-tuning | Detecting hallucination in open-ended output |
| Sanity checks against random baseline | Evaluating instruction-following or chat quality |
| Reproducible baselines in papers/reports | Measuring safety or alignment |

My rule of thumb: if you can't say what decision changes based on the eval result, you probably don't need that eval yet.

---

## What's next

**Part 2** moves into open-ended generation evals. We'll use a small instruction-tuned model, build a RAG pipeline, and measure it with RAGAS — scoring faithfulness, answer relevance, and context recall.

**Part 3** compares the major eval frameworks side-by-side: lm-eval-harness, DeepEval, RAGAS, and Inspect.

---

*Full code: [github.com/vishsangale/bonsai-llm/tree/main/posts/part1-llm-evals-intro](https://github.com/vishsangale/bonsai-llm/tree/main/posts/part1-llm-evals-intro)*
