# Part 2: RAG & Generation Evals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained RAG evaluation script using Gemma-3-1b-it for generation and RAGAS for scoring across three knowledge bases (Wikipedia, custom, HF model cards), accompanied by a beginner-friendly tutorial.

**Architecture:** A single parameterized CLI script (`eval_rag.py`) separates corpus loading, RAG pipeline (chunk → embed → retrieve → generate), and RAGAS scoring. Corpus documents and eval samples are stored as committed JSON files for reproducibility. The tutorial walks through each component step-by-step before showing real numbers.

**Tech Stack:** `transformers`, `sentence-transformers`, `faiss-cpu`, `ragas==0.2.*`, `openai`, `python-dotenv`, `wikipedia`, `huggingface_hub`, `torch`, `pytest`

---

## File Structure

```
posts/part2-rag-evals/
├── eval_rag.py              # Main CLI script
├── tutorial.md              # Step-by-step tutorial
├── .env.example             # Template for OPENAI_API_KEY
└── data/
    ├── custom_corpus.json   # 10 hand-written topic passages
    └── eval_samples.json    # 20 Q&A pairs per dataset (wikipedia, custom, hf-model-cards)

tests/
└── test_eval_rag.py         # Unit tests for chunker, reporter, loader helpers
```

---

### Task 1: Scaffold directory and data files

**Files:**
- Create: `posts/part2-rag-evals/.env.example`
- Create: `posts/part2-rag-evals/data/custom_corpus.json`
- Create: `posts/part2-rag-evals/data/eval_samples.json`

- [ ] **Step 1: Create the directory structure**

```bash
mkdir -p posts/part2-rag-evals/data
```

- [ ] **Step 2: Create `.env.example`**

```
OPENAI_API_KEY=sk-...
```

- [ ] **Step 3: Create `data/custom_corpus.json`**

Ten short topic passages on distinct science subjects. Each passage is 150–250 words so chunking produces 1–2 chunks per passage.

```json
[
  {
    "id": "photosynthesis",
    "title": "Photosynthesis",
    "text": "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy stored as glucose. It occurs primarily in the chloroplasts of plant cells, which contain a pigment called chlorophyll. Chlorophyll absorbs red and blue wavelengths of light while reflecting green, which is why plants appear green. The overall reaction can be summarized as: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2. The process has two main stages. The light-dependent reactions occur in the thylakoid membranes and capture energy from sunlight to produce ATP and NADPH, releasing oxygen as a byproduct. The light-independent reactions, also called the Calvin cycle, take place in the stroma and use ATP and NADPH to fix carbon dioxide into glucose. Photosynthesis is fundamental to life on Earth — it produces the oxygen we breathe and is the primary source of organic matter in most ecosystems."
  },
  {
    "id": "dna",
    "title": "DNA Structure and Replication",
    "text": "Deoxyribonucleic acid (DNA) is the molecule that carries genetic instructions for the development, functioning, growth, and reproduction of all known organisms and many viruses. DNA is a double helix formed by two complementary strands of nucleotides running antiparallel to each other. Each nucleotide consists of a deoxyribose sugar, a phosphate group, and one of four nitrogenous bases: adenine (A), thymine (T), guanine (G), and cytosine (C). Adenine pairs with thymine via two hydrogen bonds, and guanine pairs with cytosine via three hydrogen bonds. During replication, the double helix unwinds and each strand serves as a template for a new complementary strand. DNA polymerase synthesizes the new strand in the 5' to 3' direction and proofreads for errors. Human cells contain approximately 3 billion base pairs organized into 23 pairs of chromosomes."
  },
  {
    "id": "black_hole",
    "title": "Black Holes",
    "text": "A black hole is a region of spacetime where gravity is so strong that nothing — not even light or other electromagnetic waves — can escape once it passes the event horizon. Black holes form when massive stars collapse at the end of their life cycle in a supernova explosion. The boundary of a black hole is called the event horizon; once matter crosses it, no information can return to the outside universe. The central point of infinite density is called a singularity. Black holes are characterized by just three properties: mass, charge, and angular momentum (the no-hair theorem). Supermassive black holes, containing millions to billions of solar masses, reside at the centers of most large galaxies, including the Milky Way. The first direct image of a black hole shadow was captured by the Event Horizon Telescope in 2019, targeting the supermassive black hole in galaxy M87."
  },
  {
    "id": "plate_tectonics",
    "title": "Plate Tectonics",
    "text": "Plate tectonics is the scientific theory describing the large-scale motion of Earth's lithosphere. The lithosphere is divided into about a dozen major and several minor tectonic plates that float on the semi-fluid asthenosphere beneath them. Plates move at rates of 1–10 centimeters per year driven by convection currents in the mantle. At divergent boundaries, plates move apart and new oceanic crust forms from magma. At convergent boundaries, one plate may subduct beneath another, often forming ocean trenches and volcanic arcs. At transform boundaries, plates slide horizontally past each other, producing earthquakes. The theory explains the distribution of earthquakes, volcanoes, and mountain ranges. The supercontinent Pangaea began breaking apart about 175 million years ago, producing the continents in their current positions. Plate tectonics was confirmed in the 1960s through evidence from seafloor spreading, paleomagnetism, and earthquake seismology."
  },
  {
    "id": "vaccines",
    "title": "How Vaccines Work",
    "text": "Vaccines train the immune system to recognize and fight specific pathogens without causing the disease itself. They work by introducing an antigen — which may be a weakened or inactivated pathogen, a protein subunit, or genetic instructions (as in mRNA vaccines) — that stimulates the immune system to produce antibodies and memory cells. If the vaccinated person later encounters the real pathogen, their immune system can mount a rapid response before the infection takes hold. The immune memory created by vaccines can last for years or decades. Herd immunity occurs when a sufficient proportion of a population becomes immune, reducing the pathogen's ability to spread and protecting even unvaccinated individuals. Vaccines have eliminated or drastically reduced diseases such as smallpox, polio, measles, and diphtheria. mRNA vaccine technology, used in COVID-19 vaccines, works by delivering instructions for cells to produce a harmless piece of the pathogen's protein, triggering an immune response without using any live virus."
  },
  {
    "id": "neuron",
    "title": "Neurons and Neural Signaling",
    "text": "Neurons are electrically excitable cells that transmit information throughout the nervous system via electrical and chemical signals. A typical neuron has three main parts: a cell body (soma) containing the nucleus, dendrites that receive incoming signals, and a single axon that transmits signals away from the cell body. When a neuron receives sufficient stimulation, it fires an action potential — a rapid, self-propagating electrical signal caused by the flow of sodium and potassium ions across the cell membrane. Action potentials travel along the axon at speeds up to 120 meters per second in myelinated fibers. At synapses, the junction between two neurons, electrical signals are converted to chemical signals called neurotransmitters. Neurotransmitters such as glutamate, GABA, dopamine, and serotonin cross the synaptic cleft and bind to receptors on the receiving neuron, either exciting or inhibiting it. The human brain contains approximately 86 billion neurons forming trillions of synaptic connections."
  },
  {
    "id": "quantum_mechanics",
    "title": "Quantum Mechanics Basics",
    "text": "Quantum mechanics is the branch of physics describing the behavior of matter and energy at atomic and subatomic scales. Unlike classical mechanics, which describes the world in terms of definite positions and velocities, quantum mechanics describes particles using probability distributions called wave functions. The wave function evolves according to the Schrödinger equation and collapses to a definite value upon measurement. Key principles include wave-particle duality (particles exhibit both wave-like and particle-like properties), the Heisenberg uncertainty principle (position and momentum cannot both be known precisely simultaneously), and quantization (energy is absorbed or emitted in discrete packets called quanta). Quantum entanglement allows particles to be correlated such that measuring one instantly determines the state of another, regardless of distance. Quantum mechanics underpins modern technology including transistors, lasers, and MRI machines, and forms the basis of quantum computing research."
  },
  {
    "id": "evolution",
    "title": "Evolution by Natural Selection",
    "text": "Evolution is the process of change in all forms of life over generations. Charles Darwin and Alfred Russel Wallace independently proposed the mechanism of natural selection: individuals with heritable traits better suited to their environment tend to survive and reproduce more successfully, passing those traits to offspring. Over many generations, advantageous traits become more common in the population. Genetic variation, the raw material for evolution, arises through mutations, genetic recombination during sexual reproduction, and gene flow between populations. When populations become reproductively isolated, they may diverge into separate species — a process called speciation. Evidence for evolution comes from the fossil record, comparative anatomy, molecular biology (shared DNA sequences across species), direct observation of evolution in bacteria and viruses, and biogeography. The common ancestor of all life on Earth existed approximately 3.5 to 4 billion years ago."
  },
  {
    "id": "entropy",
    "title": "Entropy and the Second Law of Thermodynamics",
    "text": "Entropy is a measure of the disorder or randomness of a system. The second law of thermodynamics states that in any spontaneous process, the total entropy of a closed system always increases or remains constant — it never decreases. This gives time a direction: processes naturally evolve from ordered states to disordered states. For example, heat flows from hot objects to cold ones, gases expand to fill available space, and mixed substances do not spontaneously unmix. Entropy is defined in statistical mechanics as proportional to the logarithm of the number of microstates consistent with a macrostate (S = k ln W, where k is Boltzmann's constant). Living organisms appear to decrease local entropy by building ordered structures, but they do so by consuming energy and increasing entropy in their surroundings. The concept of entropy has applications in information theory, where it measures the uncertainty in a message, and in cosmology, where the ultimate fate of the universe may be a state of maximum entropy called heat death."
  },
  {
    "id": "crispr",
    "title": "CRISPR-Cas9 Gene Editing",
    "text": "CRISPR-Cas9 is a molecular tool that allows scientists to edit DNA sequences in living organisms with high precision. Originally discovered as part of the immune system of bacteria, CRISPR (Clustered Regularly Interspaced Short Palindromic Repeats) uses a guide RNA to direct the Cas9 protein to a specific location in the genome. Once there, Cas9 acts as molecular scissors, cutting both strands of the DNA double helix. The cell's own repair mechanisms then either disable the gene (knockout) or insert a new sequence provided by researchers. CRISPR-Cas9 is faster, cheaper, and more accurate than previous gene-editing techniques. It has applications in treating genetic diseases such as sickle cell anemia and certain cancers, developing disease-resistant crops, and basic research into gene function. The 2020 Nobel Prize in Chemistry was awarded to Jennifer Doudna and Emmanuelle Charpentier for developing the CRISPR-Cas9 method."
  }
]
```

- [ ] **Step 4: Create `data/eval_samples.json`**

20 questions per dataset. Wikipedia and HF model card questions are written against the pinned sources; custom questions are written against the custom corpus passages above.

```json
{
  "wikipedia": [
    {"question": "What molecule do plants use to absorb light during photosynthesis?", "reference": "Chlorophyll is the pigment plants use to absorb light during photosynthesis."},
    {"question": "What are the two main stages of photosynthesis?", "reference": "The light-dependent reactions and the Calvin cycle (light-independent reactions)."},
    {"question": "What base pairs with adenine in DNA?", "reference": "Thymine pairs with adenine via two hydrogen bonds."},
    {"question": "In which direction does DNA polymerase synthesize new strands?", "reference": "DNA polymerase synthesizes new strands in the 5' to 3' direction."},
    {"question": "What is the event horizon of a black hole?", "reference": "The event horizon is the boundary of a black hole beyond which nothing, not even light, can escape."},
    {"question": "What telescope captured the first image of a black hole in 2019?", "reference": "The Event Horizon Telescope captured the first image of a black hole shadow in 2019."},
    {"question": "At what rate do tectonic plates typically move?", "reference": "Tectonic plates move at rates of 1 to 10 centimeters per year."},
    {"question": "What drives tectonic plate movement?", "reference": "Convection currents in the mantle drive tectonic plate movement."},
    {"question": "How do mRNA vaccines work?", "reference": "mRNA vaccines deliver instructions for cells to produce a harmless piece of the pathogen's protein, triggering an immune response without using any live virus."},
    {"question": "What is herd immunity?", "reference": "Herd immunity occurs when enough of a population becomes immune to reduce a pathogen's ability to spread, protecting even unvaccinated individuals."},
    {"question": "What causes an action potential in a neuron?", "reference": "An action potential is caused by the rapid flow of sodium and potassium ions across the cell membrane."},
    {"question": "What are neurotransmitters?", "reference": "Neurotransmitters are chemical signals released at synapses that cross the synaptic cleft and bind to receptors on the receiving neuron."},
    {"question": "What does the Heisenberg uncertainty principle state?", "reference": "The Heisenberg uncertainty principle states that position and momentum of a particle cannot both be known precisely at the same time."},
    {"question": "What is quantum entanglement?", "reference": "Quantum entanglement is a phenomenon where particles are correlated such that measuring one instantly determines the state of another regardless of distance."},
    {"question": "Who proposed natural selection independently?", "reference": "Charles Darwin and Alfred Russel Wallace independently proposed natural selection."},
    {"question": "What is speciation?", "reference": "Speciation is the process by which reproductively isolated populations diverge into separate species."},
    {"question": "What does the second law of thermodynamics state about entropy?", "reference": "The second law states that in any spontaneous process, the total entropy of a closed system always increases or remains constant."},
    {"question": "Who won the 2020 Nobel Prize in Chemistry for CRISPR?", "reference": "Jennifer Doudna and Emmanuelle Charpentier won the 2020 Nobel Prize in Chemistry for developing CRISPR-Cas9."},
    {"question": "What does Cas9 do in the CRISPR-Cas9 system?", "reference": "Cas9 acts as molecular scissors, cutting both strands of the DNA double helix at a location specified by the guide RNA."},
    {"question": "What is the formula for entropy in statistical mechanics?", "reference": "Entropy is defined as S = k ln W, where k is Boltzmann's constant and W is the number of microstates."}
  ],
  "custom": [
    {"question": "What pigment do plants use to absorb light?", "reference": "Chlorophyll is the pigment plants use to absorb light."},
    {"question": "What is the overall chemical equation for photosynthesis?", "reference": "6CO2 + 6H2O + light energy → C6H12O6 + 6O2"},
    {"question": "What are the four nitrogenous bases in DNA?", "reference": "Adenine, thymine, guanine, and cytosine."},
    {"question": "How many base pairs does the human genome contain?", "reference": "Approximately 3 billion base pairs."},
    {"question": "What is a singularity in a black hole?", "reference": "A singularity is the central point of infinite density inside a black hole."},
    {"question": "What three properties characterize a black hole?", "reference": "Mass, charge, and angular momentum, according to the no-hair theorem."},
    {"question": "What happens at a divergent tectonic plate boundary?", "reference": "At divergent boundaries, plates move apart and new oceanic crust forms from magma."},
    {"question": "When did Pangaea begin breaking apart?", "reference": "Pangaea began breaking apart about 175 million years ago."},
    {"question": "How long can immune memory from vaccines last?", "reference": "Immune memory created by vaccines can last for years or decades."},
    {"question": "What is the soma of a neuron?", "reference": "The soma is the cell body of a neuron, which contains the nucleus."},
    {"question": "How fast can action potentials travel in myelinated fibers?", "reference": "Action potentials can travel up to 120 meters per second in myelinated fibers."},
    {"question": "What equation describes how a quantum wave function evolves?", "reference": "The Schrödinger equation describes how a wave function evolves over time."},
    {"question": "What is the raw material for evolution?", "reference": "Genetic variation is the raw material for evolution."},
    {"question": "How old is the common ancestor of all life on Earth?", "reference": "Approximately 3.5 to 4 billion years old."},
    {"question": "What is the statistical mechanics formula for entropy?", "reference": "S = k ln W, where k is Boltzmann's constant and W is the number of microstates."},
    {"question": "What is the ultimate fate of the universe according to thermodynamics?", "reference": "A state of maximum entropy called heat death."},
    {"question": "What immune system did CRISPR originate from?", "reference": "The bacterial immune system."},
    {"question": "What does the guide RNA do in CRISPR-Cas9?", "reference": "The guide RNA directs the Cas9 protein to a specific location in the genome."},
    {"question": "What disease was treated using CRISPR mentioned in the text?", "reference": "Sickle cell anemia."},
    {"question": "What is wave-particle duality?", "reference": "Wave-particle duality is the quantum principle that particles exhibit both wave-like and particle-like properties."}
  ],
  "hf-model-cards": [
    {"question": "What architecture is BERT based on?", "reference": "BERT is based on the Transformer architecture, specifically the encoder."},
    {"question": "What does BERT stand for?", "reference": "Bidirectional Encoder Representations from Transformers."},
    {"question": "What type of model is GPT-2?", "reference": "GPT-2 is an autoregressive language model based on the Transformer decoder."},
    {"question": "How many parameters does the base GPT-2 model have?", "reference": "The base GPT-2 model has 117 million parameters."},
    {"question": "What task is T5 primarily designed for?", "reference": "T5 is designed for text-to-text transfer, framing all NLP tasks as text generation."},
    {"question": "What does T5 stand for?", "reference": "Text-To-Text Transfer Transformer."},
    {"question": "What is DistilBERT?", "reference": "DistilBERT is a smaller, faster, lighter version of BERT trained via knowledge distillation."},
    {"question": "How much smaller is DistilBERT compared to BERT?", "reference": "DistilBERT is 40% smaller than BERT while retaining 97% of its language understanding."},
    {"question": "What pre-training objective does RoBERTa use?", "reference": "RoBERTa uses masked language modeling, the same as BERT, but with improved training procedures."},
    {"question": "What is BART primarily used for?", "reference": "BART is primarily used for sequence-to-sequence tasks like summarization and translation."},
    {"question": "What does BART stand for?", "reference": "Bidirectional and Auto-Regressive Transformers."},
    {"question": "What is Whisper designed to do?", "reference": "Whisper is designed for automatic speech recognition and translation."},
    {"question": "How was Whisper trained?", "reference": "Whisper was trained on 680,000 hours of multilingual and multitask supervised data collected from the web."},
    {"question": "What is Flan-T5?", "reference": "Flan-T5 is a version of T5 fine-tuned on a large collection of tasks using instruction tuning."},
    {"question": "What improvement does Flan-T5 have over T5?", "reference": "Flan-T5 achieves stronger zero-shot and few-shot performance than T5 by training on diverse instruction-formatted tasks."},
    {"question": "What tokenizer does BERT use?", "reference": "BERT uses WordPiece tokenization."},
    {"question": "What language was BERT originally trained on?", "reference": "BERT was originally trained on English text from Wikipedia and BookCorpus."},
    {"question": "What is the context length of GPT-2?", "reference": "GPT-2 has a maximum context length of 1024 tokens."},
    {"question": "What license is the RoBERTa model released under?", "reference": "RoBERTa is released under the MIT License."},
    {"question": "What is the input format for Flan-T5?", "reference": "Flan-T5 takes text as input and produces text as output in a text-to-text format."}
  ]
}
```

- [ ] **Step 5: Commit scaffold**

```bash
git add posts/part2-rag-evals/
git commit -m "feat: scaffold part2-rag-evals directory with data files"
```

---

### Task 2: Tests for chunker and reporter

**Files:**
- Create: `tests/test_eval_rag.py`

- [ ] **Step 1: Write failing tests for `chunk_text`**

```python
# tests/test_eval_rag.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'posts', 'part2-rag-evals'))

from eval_rag import chunk_text

def test_chunk_text_short_passage_returns_one_chunk():
    text = "Short passage with fewer than 256 tokens. " * 3
    chunks = chunk_text(text, chunk_size=256, overlap=32)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)

def test_chunk_text_long_passage_returns_multiple_chunks():
    # ~300 words → should produce at least 2 chunks at chunk_size=256
    text = "word " * 300
    chunks = chunk_text(text, chunk_size=256, overlap=32)
    assert len(chunks) >= 2

def test_chunk_text_overlap_means_chunks_share_content():
    text = " ".join([f"word{i}" for i in range(300)])
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    if len(chunks) > 1:
        # Last tokens of chunk[0] should appear at start of chunk[1]
        end_of_first = chunks[0].split()[-5:]
        start_of_second = chunks[1].split()[:10]
        assert any(w in start_of_second for w in end_of_first)

def test_chunk_text_no_empty_chunks():
    text = "Some text. " * 100
    chunks = chunk_text(text, chunk_size=256, overlap=32)
    assert all(len(c.strip()) > 0 for c in chunks)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /path/to/bonsai-llm
source venv/bin/activate
pytest tests/test_eval_rag.py -v
```

Expected: `ImportError: cannot import name 'chunk_text' from 'eval_rag'` (file doesn't exist yet).

- [ ] **Step 3: Write failing tests for `format_results_table`**

Add to `tests/test_eval_rag.py`:

```python
from eval_rag import format_results_table

def test_format_results_table_contains_metric_names():
    results = {
        "Faithfulness": 0.82,
        "ResponseRelevancy": 0.75,
        "ContextPrecision": 0.68,
        "ContextRecall": 0.71,
        "AnswerCorrectness": 0.65,
        "SemanticSimilarity": 0.79,
    }
    table = format_results_table("wikipedia", results)
    assert "Faithfulness" in table
    assert "wikipedia" in table
    assert "0.82" in table

def test_format_results_table_all_metrics_present():
    results = {k: 0.5 for k in [
        "Faithfulness", "ResponseRelevancy", "ContextPrecision",
        "ContextRecall", "AnswerCorrectness", "SemanticSimilarity"
    ]}
    table = format_results_table("custom", results)
    for metric in results:
        assert metric in table
```

- [ ] **Step 4: Run tests again to confirm they still fail**

```bash
pytest tests/test_eval_rag.py -v
```

Expected: `ImportError` — `eval_rag.py` doesn't exist yet.

- [ ] **Step 5: Commit tests**

```bash
git add tests/test_eval_rag.py
git commit -m "test: add failing tests for chunk_text and format_results_table"
```

---

### Task 3: Implement `chunk_text` and `format_results_table`

**Files:**
- Create: `posts/part2-rag-evals/eval_rag.py`

- [ ] **Step 1: Create `eval_rag.py` with `chunk_text`**

```python
# posts/part2-rag-evals/eval_rag.py
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

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
CHUNK_SIZE = 256   # tokens (approximated as words for simplicity)
CHUNK_OVERLAP = 32
TOP_K = 3
MAX_NEW_TOKENS = 200


# ── Chunker ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks."""
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
```

- [ ] **Step 2: Add `format_results_table`**

```python
# append to eval_rag.py

def format_results_table(dataset_name: str, results: dict) -> str:
    """Return a formatted markdown-style table string for a single dataset."""
    header = f"\nDataset: {dataset_name}\n"
    divider = f"{'─' * 50}\n"
    row_fmt = "| {:<22} | {:>6} |\n"
    header_row = row_fmt.format("Metric", "Score")
    sep_row = "|" + "-" * 24 + "|" + "-" * 8 + "|\n"
    rows = "".join(row_fmt.format(k, f"{v:.4f}") for k, v in results.items())
    return header + divider + header_row + sep_row + rows
```

- [ ] **Step 3: Run tests — they should pass now**

```bash
pytest tests/test_eval_rag.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add posts/part2-rag-evals/eval_rag.py
git commit -m "feat: implement chunk_text and format_results_table, tests passing"
```

---

### Task 4: Corpus loaders

**Files:**
- Modify: `posts/part2-rag-evals/eval_rag.py`

- [ ] **Step 1: Add `load_custom_corpus`**

```python
# append to eval_rag.py

def load_custom_corpus() -> list[str]:
    """Load hand-written passages from data/custom_corpus.json."""
    with open(DATA_DIR / "custom_corpus.json") as f:
        items = json.load(f)
    return [item["text"] for item in items]
```

- [ ] **Step 2: Add `load_wikipedia_corpus`**

```python
# append to eval_rag.py
import wikipedia as wiki_api

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
```

- [ ] **Step 3: Add `load_hf_model_cards_corpus`**

```python
# append to eval_rag.py
from huggingface_hub import HfApi

HF_MODELS = [
    "bert-base-uncased", "gpt2", "t5-small", "distilbert-base-uncased",
    "roberta-base", "facebook/bart-large-cnn", "openai/whisper-small",
    "google/flan-t5-base"
]

def load_hf_model_cards_corpus() -> list[str]:
    """Fetch model card text from pinned HuggingFace models."""
    from huggingface_hub import ModelCard
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
```

- [ ] **Step 4: Smoke-test corpus loaders manually**

```bash
cd posts/part2-rag-evals
source ../../venv/bin/activate
python -c "
from eval_rag import load_custom_corpus, load_hf_model_cards_corpus
docs = load_custom_corpus()
print(f'custom: {len(docs)} docs, first 100 chars: {docs[0][:100]}')
docs = load_hf_model_cards_corpus()
print(f'hf-model-cards: {len(docs)} docs')
"
```

Expected: `custom: 10 docs`, `hf-model-cards: 8 docs` (or fewer if any cards fail to load).

- [ ] **Step 5: Commit**

```bash
git add posts/part2-rag-evals/eval_rag.py
git commit -m "feat: add corpus loaders for wikipedia, custom, and hf-model-cards"
```

---

### Task 5: Eval sample loader and FAISS index

**Files:**
- Modify: `posts/part2-rag-evals/eval_rag.py`

- [ ] **Step 1: Add `load_eval_samples`**

```python
# append to eval_rag.py

def load_eval_samples(dataset: str) -> list[dict]:
    """Return list of {question, reference} dicts for the given dataset."""
    with open(DATA_DIR / "eval_samples.json") as f:
        all_samples = json.load(f)
    if dataset not in all_samples:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(all_samples)}")
    return all_samples[dataset]
```

- [ ] **Step 2: Add `build_index` using FAISS + sentence-transformers**

```python
# append to eval_rag.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

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
```

- [ ] **Step 3: Smoke-test index and retrieval**

```bash
python -c "
from eval_rag import load_custom_corpus, build_index, retrieve
docs = load_custom_corpus()
index, chunks = build_index(docs)
print(f'Index built: {index.ntotal} vectors')
results = retrieve('What pigment absorbs light in plants?', index, chunks)
print(f'Top chunk: {results[0][:200]}')
"
```

Expected: index built with >10 vectors, top chunk mentions chlorophyll.

- [ ] **Step 4: Commit**

```bash
git add posts/part2-rag-evals/eval_rag.py
git commit -m "feat: add eval sample loader, FAISS index builder, and retriever"
```

---

### Task 6: Generator (Gemma-3-1b-it)

**Files:**
- Modify: `posts/part2-rag-evals/eval_rag.py`

- [ ] **Step 1: Add generator loader and `generate_answer`**

```python
# append to eval_rag.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    prompt = PROMPT_TEMPLATE.format(context=context_str, question=question)

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
```

- [ ] **Step 2: Smoke-test generation**

```bash
python -c "
from eval_rag import load_custom_corpus, build_index, retrieve, generate_answer
docs = load_custom_corpus()
index, chunks = build_index(docs)
q = 'What pigment do plants use to absorb light?'
contexts = retrieve(q, index, chunks)
answer = generate_answer(q, contexts)
print('Answer:', answer)
"
```

Expected: a coherent sentence mentioning chlorophyll (may be imperfect — that's fine and intentional for the tutorial).

- [ ] **Step 3: Commit**

```bash
git add posts/part2-rag-evals/eval_rag.py
git commit -m "feat: add Gemma-3-1b-it generator with context-grounded prompt template"
```

---

### Task 7: RAGAS scorer

**Files:**
- Modify: `posts/part2-rag-evals/eval_rag.py`

- [ ] **Step 1: Add `score_with_ragas`**

RAGAS expects a `Dataset` with fields: `user_input`, `retrieved_contexts`, `response`, `reference`.

```python
# append to eval_rag.py
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
```

- [ ] **Step 2: Add `run_dataset` to orchestrate the full pipeline for one dataset**

```python
# append to eval_rag.py

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
```

- [ ] **Step 3: Commit**

```bash
git add posts/part2-rag-evals/eval_rag.py
git commit -m "feat: add RAGAS scorer and run_dataset orchestrator"
```

---

### Task 8: Reporter and CLI

**Files:**
- Modify: `posts/part2-rag-evals/eval_rag.py`

- [ ] **Step 1: Add `print_comparison_table` and `save_results`**

```python
# append to eval_rag.py

def print_comparison_table(all_results: dict):
    """Print a cross-dataset comparison table."""
    datasets = list(all_results.keys())
    metrics = list(next(iter(all_results.values())).keys())

    col_w = 14
    header = "| {:<22} |".format("Metric") + "".join(f" {d:<{col_w}} |" for d in datasets)
    sep = "|" + "-"*24 + "|" + ("".join("-"*(col_w+2) + "|" for _ in datasets))
    print(f"\n{'='*60}")
    print("Cross-dataset comparison (acc_norm-style)")
    print(f"{'='*60}")
    print(header)
    print(sep)
    for metric in metrics:
        row = f"| {metric:<22} |"
        for d in datasets:
            val = all_results[d].get(metric, float("nan"))
            row += f" {val:<{col_w}.4f} |"
        print(row)

def save_results(dataset_name: str, scores: dict):
    """Save results dict to results_<dataset>.json."""
    path = Path(__file__).parent / f"results_{dataset_name}.json"
    with open(path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nRaw results saved to {path.name}")
```

- [ ] **Step 2: Add `main` and argparse CLI**

```python
# append to eval_rag.py

def main():
    parser = argparse.ArgumentParser(description="RAG eval with RAGAS across knowledge bases")
    parser.add_argument(
        "--dataset",
        choices=list(CORPUS_LOADERS) + ["all"],
        default="custom",
        help="Which knowledge base to evaluate (default: custom)"
    )
    args = parser.parse_args()

    datasets = list(CORPUS_LOADERS) if args.dataset == "all" else [args.dataset]
    all_results = {}

    for ds in datasets:
        scores = run_dataset(ds)
        print(format_results_table(ds, scores))
        save_results(ds, scores)
        all_results[ds] = scores

    if len(datasets) > 1:
        print_comparison_table(all_results)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the full test suite**

```bash
pytest tests/test_eval_rag.py -v
```

Expected: all tests PASS.

- [ ] **Step 4: End-to-end smoke run on custom dataset (fastest)**

```bash
python eval_rag.py --dataset custom
```

Expected: prints per-metric scores table and saves `results_custom.json`.

- [ ] **Step 5: Commit**

```bash
git add posts/part2-rag-evals/eval_rag.py
git commit -m "feat: add reporter, CLI, and main — eval_rag.py complete"
```

---

### Task 9: Run all three datasets and capture results

**Files:**
- Create: `posts/part2-rag-evals/results_wikipedia.json`
- Create: `posts/part2-rag-evals/results_custom.json`
- Create: `posts/part2-rag-evals/results_hf-model-cards.json`

- [ ] **Step 1: Run all datasets**

```bash
python eval_rag.py --dataset all
```

Expected: scores for all three datasets printed, three JSON files saved.

- [ ] **Step 2: Note the actual numbers**

Record the cross-dataset comparison table output — these numbers go directly into the tutorial in Task 10.

- [ ] **Step 3: Commit results**

```bash
git add posts/part2-rag-evals/results_*.json
git commit -m "chore: add RAGAS benchmark results for all three knowledge bases"
```

---

### Task 10: Write `tutorial.md`

**Files:**
- Create: `posts/part2-rag-evals/tutorial.md`

Write the full tutorial following the 11-step structure from the spec. Fill in the real numbers from Task 9 results wherever the tutorial shows scores.

- [ ] **Step 1: Write Steps 1–4 (motivation, knowledge bases, pipeline overview, RAGAS intro + setup)**

```markdown
# RAG Evals From Scratch: Measuring What Your Pipeline Gets Wrong

*Part 2 of the Bonsai LLM eval series — Beginner*

> **Runtime note:** First run downloads Gemma-3-1b-it (~2GB) and three datasets. Expect 10–20 minutes on first run. You will need an OpenAI API key for the RAGAS judge metrics.

---

## Prerequisites

- Python 3.9+
- A terminal and a virtual environment
- An OpenAI API key (for RAGAS judge metrics)
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

We will run the same pipeline against three corpora:

**Custom** — 10 short passages written specifically for this tutorial. Questions and answers are tightly coupled to the text. This is the best-case scenario: retrieval is easy, answers are verifiable, ground truth is clean.

**Wikipedia** — 8 real Wikipedia articles on science topics. The text is longer, has more structure, and the questions are written against the actual content. A realistic baseline.

**HF Model Cards** — README text from 8 popular HuggingFace models. Technical, domain-specific, and written for a different audience than the eval questions. The hardest corpus.

The experiment is not about which corpus is "best" in general — it is about watching how the same pipeline responds to different data quality and domain match. That is the main thing RAGAS lets you see.

---

## Step 3 — RAG pipeline overview

The pipeline has four stages:

1. **Chunk** — Split each document into 256-word chunks with 32-word overlap. Overlap ensures sentences at chunk boundaries are not lost.
2. **Embed** — Encode all chunks into dense vectors using `sentence-transformers/all-MiniLM-L6-v2`. These vectors live in a FAISS index.
3. **Retrieve** — For each question, encode it with the same embedding model and return the top-3 most similar chunks by cosine similarity.
4. **Generate** — Feed the question and retrieved chunks to Gemma-3-1b-it with a strict prompt: "Answer using only the context below. If the answer is not in the context, say 'I don't know'."

The prompt constraint is important for RAGAS: if the model generates text that goes beyond the context, faithfulness drops. If it answers correctly from the context, faithfulness stays high.

---

## Step 4 — What RAGAS is, and setting up your environment

RAGAS (Retrieval Augmented Generation Assessment) is an open-source framework for evaluating RAG pipelines without hand-labeling every output. It uses an LLM-as-judge to score dimensions like faithfulness and relevancy automatically, so you get structured signal instead of eyeballing outputs.

**Accept the Gemma-3 license** (one-time):

Visit [hf.co/google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) and accept the terms. Then log in from the terminal:

```bash
huggingface-cli login
```

**Create your environment:**

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install ragas sentence-transformers faiss-cpu openai python-dotenv \
            wikipedia huggingface_hub transformers torch datasets
```

**Set your OpenAI key:**

```bash
cp .env.example .env
# edit .env and add your key: OPENAI_API_KEY=sk-...
```
```

- [ ] **Step 2: Write Steps 5–7 (metrics, get the script, run it)**

```markdown
---

## Step 5 — The six RAGAS metrics

RAGAS scores each sample individually then averages across the dataset. Some metrics use an LLM judge (OpenAI), others use embeddings.

| Metric | What it catches | How scored |
|---|---|---|
| `Faithfulness` | Did the answer contradict the context? | LLM |
| `ResponseRelevancy` | Did the answer address the question? | LLM + embeddings |
| `ContextPrecision` | Is the retrieved context signal or noise? | LLM |
| `ContextRecall` | Did retrieval miss anything the answer needed? | LLM |
| `AnswerCorrectness` | Is the answer right compared to the reference? | LLM + embeddings |
| `SemanticSimilarity` | How close is the answer to the reference semantically? | Embeddings only |

A few things worth knowing before you see numbers:

**Faithfulness** is the hallucination check. A faithful answer only makes claims that are supported by the retrieved context. If Gemma-3 adds facts from its training data that aren't in the chunks it was given, faithfulness drops.

**ContextRecall** requires a ground truth answer. The judge asks: is everything in the reference answer covered by the retrieved chunks? Low recall means the retriever is missing relevant content — the generator cannot fix this no matter how good it is.

**SemanticSimilarity** is embedding-based only — no OpenAI call needed. It measures how close the generated answer is to the reference in vector space. Useful as a fast sanity check and does not share the LLM judge's verbosity bias.

---

## Step 6 — Get the script

```bash
git clone https://github.com/vishsangale/bonsai-llm
cd bonsai-llm/posts/part2-rag-evals
```

---

## Step 7 — Run the script

Single dataset (fastest, good for testing):

```bash
python eval_rag.py --dataset custom
```

All three datasets (runs sequentially, saves one JSON per dataset):

```bash
python eval_rag.py --dataset all
```

On first run, the script downloads Gemma-3-1b-it (~2GB) and the embedding model (~90MB). Subsequent runs skip downloads. The OpenAI judge calls happen during scoring — expect a few hundred API calls per dataset run.
```

- [ ] **Step 3: Write Steps 8–11 (read output, interpret, limits, when to use)**

Fill in `ACTUAL_*` placeholders with the real numbers from Task 9.

```markdown
---

## Step 8 — Read the output

```
Dataset: custom
──────────────────────────────────────────────────
| Metric                 |  Score |
|------------------------|--------|
| Faithfulness           | ACTUAL_FAITH_CUSTOM |
| ResponseRelevancy      | ACTUAL_RR_CUSTOM    |
| ContextPrecision       | ACTUAL_CP_CUSTOM    |
| ContextRecall          | ACTUAL_CR_CUSTOM    |
| AnswerCorrectness      | ACTUAL_AC_CUSTOM    |
| SemanticSimilarity     | ACTUAL_SS_CUSTOM    |
```

Each metric is on a 0–1 scale. The `Faithfulness` score tells you what fraction of the generated answer's claims are supported by the retrieved context. A score of 0.85 means roughly 85% of claims check out; the remaining 15% either went beyond the context or contradicted it.

The stderr equivalent here is not printed by default, but RAGAS scores are sensitive to sample size — with 20 questions, treat differences smaller than ~0.05 as noise.

---

## Step 9 — Interpret the cross-dataset comparison

| Metric | custom | wikipedia | hf-model-cards |
|---|---|---|---|
| Faithfulness | ACTUAL_FAITH_CUSTOM | ACTUAL_FAITH_WIKI | ACTUAL_FAITH_HF |
| ResponseRelevancy | ACTUAL_RR_CUSTOM | ACTUAL_RR_WIKI | ACTUAL_RR_HF |
| ContextPrecision | ACTUAL_CP_CUSTOM | ACTUAL_CP_WIKI | ACTUAL_CP_HF |
| ContextRecall | ACTUAL_CR_CUSTOM | ACTUAL_CR_WIKI | ACTUAL_CR_HF |
| AnswerCorrectness | ACTUAL_AC_CUSTOM | ACTUAL_AC_WIKI | ACTUAL_AC_HF |
| SemanticSimilarity | ACTUAL_SS_CUSTOM | ACTUAL_SS_WIKI | ACTUAL_SS_HF |

[Fill in interpretation paragraph based on actual results — note which metrics diverge most across datasets, which metric shows the starkest gap between custom and hf-model-cards, and what that tells you about retrieval vs generation failures.]

---

## Step 10 — Limits

**LLM-as-judge biases.** The OpenAI judge has a verbosity bias — it tends to score longer, more detailed answers higher even when they are not more faithful. It can also have self-preference if the judge and generator share training data. Treat scores as relative comparisons between pipeline configurations, not absolute ground truth.

**Source leakage.** The custom dataset is the most susceptible: if your reference answers are written from the same passages you indexed, `ContextRecall` is measuring how well you copied your own text, not real retrieval quality. The Wikipedia and HF model card samples are less coupled because they are written against web sources, not the exact text in the index.

**Ground truth quality.** `AnswerCorrectness` and `ContextRecall` scores are only as good as the reference answers. A vague or incomplete reference will produce misleading numbers regardless of pipeline quality.

**HuggingFace license gate.** Gemma-3-1b-it requires accepting Google's terms of use once at hf.co. If you hit an authentication error on first run, check that you ran `huggingface-cli login`.

---

## Step 11 — When to use these evals — and when not to

| Use RAGAS for... | Don't use it for... |
|---|---|
| Comparing two retrieval strategies | Predicting user satisfaction |
| Catching faithfulness regressions after prompt changes | Measuring safety or alignment |
| Diagnosing whether failures are retrieval or generation | Evaluating open-ended creative tasks |
| Reproducible baselines with a fixed corpus | Final product quality decisions alone |

The diagnostic split is the main reason to run all six metrics. If faithfulness is high but answer correctness is low, the generator is using the context but the retriever is surfacing the wrong content. If context recall is high but faithfulness is low, the right content is retrieved but the generator is going off-script. Different failure modes need different fixes.

---

## What's next

**Part 3** compares the major eval frameworks side-by-side: lm-eval-harness, DeepEval, RAGAS, and Inspect. Same models, same tasks — different tools, different trade-offs.

---

*Full code: [github.com/vishsangale/bonsai-llm/tree/main/posts/part2-rag-evals](https://github.com/vishsangale/bonsai-llm/tree/main/posts/part2-rag-evals)*
```

- [ ] **Step 4: Replace all `ACTUAL_*` placeholders with real numbers from Task 9 results**

Open `results_wikipedia.json`, `results_custom.json`, `results_hf-model-cards.json` and substitute each placeholder. Also write the interpretation paragraph in Step 9 based on actual patterns observed.

- [ ] **Step 5: Commit**

```bash
git add posts/part2-rag-evals/tutorial.md
git commit -m "docs: add Part 2 tutorial draft with real RAGAS numbers"
```

---

### Task 11: Push and update README

**Files:**
- Modify: `posts/README.md`

- [ ] **Step 1: Update series table in `posts/README.md`**

Change Part 2 status from `Planned` to `Ready`:

```markdown
| [Part 2](part2-rag-evals/) | RAG & Generation Evals (RAGAS, LLM-as-judge) | Ready |
```

- [ ] **Step 2: Commit and push**

```bash
git add posts/README.md
git commit -m "docs: mark Part 2 as ready in README"
git push origin dev
```
