"""
Inspect AI Example

This script demonstrates how to create a simple evaluation task using Inspect from UK AISI.
Inspect has native integration with HuggingFace, meaning you don't need to write boilerplate
model wrappers. You define datasets and tasks, and specify the model via the CLI.

Run this with:
inspect eval inspect_example.py --model hf/gpt2
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate, system_message

@task
def capital_qa():
    """
    A simple task that evaluates whether the model can correctly identify capital cities.
    """
    return Task(
        # 1. Dataset: Defines the inputs and the expected targets
        dataset=[
            Sample(
                input="What is the capital of France?",
                target="Paris"
            ),
            Sample(
                input="What is the capital of Japan?",
                target="Tokyo"
            ),
            Sample(
                input="What is the capital of Brazil?",
                target=["Brasilia", "Brasília"] # Multiple valid targets
            )
        ],
        # 2. Plan: Defines how to prompt the model. 
        # `generate()` triggers the model's generation.
        plan=[
            system_message("You are a helpful assistant. Provide only the name of the city, with no other text."),
            generate()
        ],
        # 3. Scorer: Defines how to grade the model's output against the target.
        scorer=exact()
    )
