# Framework Showdown: lm-eval vs DeepEval vs RAGAS vs Inspect

*Part 3 of the Bonsai LLM eval series — Intermediate*

> **Prerequisites:** Python 3.9+, basic understanding of LLMs (covered in Parts 1 & 2). We will use `gpt2` and `gemma-3-1b-it` to explore these frameworks locally.

---

This post compares the four major LLM evaluation frameworks: **lm-eval-harness**, **RAGAS**, **DeepEval**, and **Inspect**. Each targets a different evaluation paradigm: log-prob benchmark scoring, diagnostic RAG metrics, CI/CD unit testing, and agentic workflows respectively.

## 1. lm-eval-harness: Academic Benchmarks

EleutherAI's `lm-eval` is the engine behind the HuggingFace Open LLM Leaderboard. It evaluates base statistical knowledge deterministically by measuring the log-probability of target token sequences (e.g., scoring choice A vs choice B). Use this for pre-training or fine-tuning base models against standardized benchmarks (MMLU, HellaSwag, ARC). It is not designed for open-ended generation or agent evaluation.

### Local Example with GPT-2

While you can run `lm-eval` directly from the command line, you can also use it via Python (as we did in **Part 1**):

```python
import lm_eval

# Evaluates GPT-2 on the ARC-Easy dataset
results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=gpt2",
    tasks=["arc_easy"],
    num_fewshot=0,
    batch_size=8,
)
print(results["results"]["arc_easy"]["acc_norm,none"])
```

## 2. RAGAS: Retrieval Diagnostics

RAGAS breaks RAG evaluation into independent metrics (Faithfulness, Context Precision, Answer Relevancy) using the LLM-as-a-judge paradigm. A single accuracy metric obscures whether a RAG failure was caused by bad retrieval or bad generation; RAGAS isolates the exact failure mode. Because it relies entirely on a judge LLM, metrics will be noisy if the judge model is weak or misaligned. Use RAGAS for tuning chunking strategies or embedding models offline.

### Local Example

RAGAS operates on datasets containing your RAG pipeline's inputs and outputs (as we did in **Part 2**):

```python
from ragas import evaluate
from ragas.metrics import answer_relevancy
from datasets import Dataset

# A dataset representing your RAG pipeline's inputs and outputs
data = {
    "question": ["What is the capital of France?"],
    "answer": ["Paris is the capital of France."],
    "contexts": [["France is a country in Western Europe. Its capital is Paris."]]
}
dataset = Dataset.from_dict(data)

# Evaluates the pipeline using the answer_relevancy metric
results = evaluate(dataset, metrics=[answer_relevancy])
print(results)
```

## 3. DeepEval: CI/CD and Unit Testing

DeepEval brings standard software engineering primitives to LLMs, integrating directly with `pytest`. It packages metrics into test cases (`LLMTestCase`) with pass/fail thresholds, making it trivial to run evaluations in CI/CD pipelines to block regressions.

### Local Example with Gemma-3-1b-it

DeepEval natively supports OpenAI, but you can hook up local models. Here’s a minimal example using `gemma-3-1b-it` (code provided in `deepeval_example.py`):

```python
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

# See deepeval_example.py for the LocalGemmaModel implementation
from deepeval_example import LocalGemmaModel
custom_model = LocalGemmaModel()

def test_answer_relevancy():
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund return policy."
    )

    relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=custom_model)
    assert_test(test_case, [relevancy_metric])
```

Run this with standard pytest: `pytest deepeval_example.py`.

**Expected Output:**
```text
========================= test session starts ==========================
collected 1 item

deepeval_example.py .                                             [100%]

--------------------------- Test passed --------------------------------
Metrics:
- Answer Relevancy:
    Score: 1.0
    Reason: The answer is highly relevant and directly addresses the 
            concern about shoes not fitting by stating the return policy.

========================== 1 passed in 4.23s ===========================
```

## 4. Inspect: The Agentic Sandbox

Developed by the UK AI Safety Institute, Inspect is designed for multi-step agent evaluation. It sandboxes tool use and execution, cleanly separating datasets, solvers (prompt engineering), and scorers. It natively supports HuggingFace and major APIs without boilerplate model wrappers. Use Inspect when evaluating complex reasoning loops or tool-using agents.

### Local Example with GPT-2

Here’s a minimal example running `gpt2` out-of-the-box (code provided in `inspect_example.py`):

```python
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate

@task
def capital_qa():
    return Task(
        dataset=[
            Sample(input="What is the capital of France?", target="Paris")
        ],
        plan=[generate()],
        scorer=exact()
    )
```

Run this via the Inspect CLI: `inspect eval inspect_example.py --model hf/gpt2`.

**Expected Output:**
```text
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ◉ capital_qa                                                           ┃
┠────────────────────────────────────────────────────────────────────────┨
┃ dataset: capital_qa (3 samples)                                        ┃
┃ model: hf/gpt2                                                         ┃
┃ scorer: exact                                                          ┃
┠────────────────────────────────────────────────────────────────────────┨
┃ accuracy: 1.000                                                        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

## Summary Matrix

| Framework | When to Use | When NOT to Use | Typical Workflow | Evaluation Method |
|---|---|---|---|---|
| **lm-eval** | Evaluating base knowledge against standard benchmarks (MMLU, HellaSwag) | Evaluating open-ended chat, RAG systems, or agents | `lm_eval --model hf --tasks mmlu` | Log-probs, Exact match |
| **RAGAS** | Tuning RAG chunk sizes, diagnosing retrieval vs. generation failures | Evaluating general instruction-following or complex tool use | Python scripting with datasets | LLM-as-a-judge |
| **DeepEval** | Blocking regressions in CI/CD, software-engineering-style unit testing | Running standard academic benchmarks | `pytest test_llm.py` | LLM-as-a-judge (metrics) |
| **Inspect** | Evaluating multi-step reasoning, autonomous agents, and tool use | Strictly testing a simple RAG pipeline's chunking strategy | `inspect eval test_agent.py` | Extensible (Exact, LLM, Custom) |

In a production system, you will likely use `lm-eval` for base model validation, `RAGAS` for offline retrieval tuning, and `DeepEval` or `Inspect` in CI/CD to prevent regressions.

---

*Full code: [github.com/vishsangale/bonsai-llm/tree/main/posts/part3-framework-showdown](https://github.com/vishsangale/bonsai-llm/tree/main/posts/part3-framework-showdown)*
