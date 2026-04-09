"""
LLM Evals — Part 1: Running Standard Benchmarks on GPT-2
=========================================================
Self-contained script. Install deps, then run:

    pip install lm_eval transformers torch

    python eval_gpt2.py
"""

import json
import lm_eval
from lm_eval.utils import make_table


# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "gpt2"          # swap for "gpt2-medium", "gpt2-large", or any HF model
TASKS = [
    "arc_easy",         # grade-school science, multiple choice
    "piqa",             # physical intuition, multiple choice
    "hellaswag",        # sentence completion, multiple choice
]
NUM_FEWSHOT = 0         # zero-shot — no examples in the prompt
BATCH_SIZE  = 8         # lower if you run out of memory


# ── Run evals ─────────────────────────────────────────────────────────────────

print(f"\nEvaluating {MODEL!r} on: {', '.join(TASKS)}")
print(f"Mode: {NUM_FEWSHOT}-shot | Batch size: {BATCH_SIZE}\n")

results = lm_eval.simple_evaluate(
    model="hf",
    model_args=f"pretrained={MODEL}",
    tasks=TASKS,
    num_fewshot=NUM_FEWSHOT,
    batch_size=BATCH_SIZE,
)


# ── Pretty-print results ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(make_table(results))
print("=" * 60)


# ── Also dump raw JSON so you can diff runs later ─────────────────────────────

output_path = f"results_{MODEL.replace('/', '_')}.json"
with open(output_path, "w") as f:
    json.dump(results["results"], f, indent=2)

print(f"\nRaw results saved to {output_path}")


# ── Explain what you're seeing ─────────────────────────────────────────────────

# We report acc_norm throughout. For multiple-choice tasks, answer choices often
# differ in token length, so raw log-probability is biased toward longer options.
# acc_norm divides by the number of tokens in each choice before comparing.
# When answer lengths are roughly equal (e.g. PIQA's two short options) the
# difference between acc and acc_norm is small, but using acc_norm consistently
# avoids a common source of confusion when comparing results across tasks.
BASELINES = {
    "arc_easy":  ("acc_norm", 0.25, "4-way multiple choice"),
    "piqa":      ("acc_norm", 0.50, "2-way multiple choice"),
    "hellaswag": ("acc_norm", 0.25, "4-way multiple choice"),
}

print("\n── Sanity check vs. random baselines (acc_norm) ──")
for task, (metric, random_baseline, note) in BASELINES.items():
    val = results["results"][task].get(metric, results["results"][task].get("acc", 0))
    delta = val - random_baseline
    flag = "✓ above chance" if delta > 0.02 else "✗ near random"
    print(f"  {task:12s}  {metric}: {val:.4f}  (random={random_baseline:.2f}, Δ={delta:+.4f})  {flag}  [{note}]")

print()
