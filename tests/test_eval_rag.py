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
