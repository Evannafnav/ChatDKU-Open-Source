# ChatDKU Candidate Task — Mini Agentic RAG

A small, bilingual, page-cited question-answering pipeline. **Status: implementation draft.** The keyword retrieval path was exercised locally; the vector/DSPy/local-LLM path still needs a machine with installed dependencies and an allowed local model server. No answer-quality or model-comparison result is claimed yet.

This directory is a self-contained candidate exercise inside a fork of the open-source ChatDKU repository. It does not replace ChatDKU's production agent. Do not add private student records, internal documents, `.env`, model weights, or API credentials to this public repository.

## Architecture

`PDF page → overlapping passage with (document, page, chunk) → BM25 keyword tool + sentence-transformer vector tool → reciprocal-rank fusion → DSPy planner and synthesizer → answer with retrieved-page citations`

The retrieval tools are callable separately via `search --mode keyword|vector|hybrid`. DSPy `Plan` proposes a query, then `Synthesize` receives selected passages and returns the evidence IDs it used. The program validates those IDs and renders filenames/pages from passage metadata, never from model-written citations. This still does **not** guarantee every answer claim is supported, so manual support evaluation remains necessary.

## Setup

Python 3.11+ is required. From the repository root:

```bash
cd candidate_task
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e '.[test]'
```

Dependencies: DSPy 3.x, pypdf 5–6, sentence-transformers 3–5, NumPy 1–2; pytest is optional for tests. See `pyproject.toml` for constraints. The first embedding run downloads the chosen model. PDFs must contain extractable text; scanned pages need OCR. Put your own authorized PDFs under `local_docs/` (gitignored).

Run a local OpenAI-compatible **vLLM** server in another terminal on a suitable GPU machine:

```bash
vllm serve Qwen/Qwen3-8B --host 127.0.0.1 --port 8000
```

The model is a selectable example, not a tested configuration. An MLX server on compatible Apple Silicon or SGLang can be used instead, provided it exposes a compatible `/v1` endpoint. Ollama is excluded by the assignment. Model downloads, memory requirements, and chat-template compatibility must be checked on your hardware.

## Entry points

```bash
python -m candidate_rag search --docs local_docs --mode keyword 'library hours'
python -m candidate_rag search --docs local_docs --mode hybrid --embedding-model BAAI/bge-small-en-v1.5 '图书馆开放时间'
python -m candidate_rag ask --docs local_docs --embedding-model BAAI/bge-small-en-v1.5 --lm-model Qwen/Qwen3-8B --base-url http://127.0.0.1:8000/v1 'What are the library hours?'
python -m candidate_rag benchmark --docs local_docs --questions local_questions.jsonl --mode keyword
python -m candidate_rag benchmark --docs local_docs --questions local_questions.jsonl --mode hybrid --embedding-model Qwen/Qwen3-Embedding-0.6B
python -m pytest -q
```

`local_questions.jsonl` has one JSON object per line: `{"question":"...", "expected_document":"guide.pdf", "expected_page":2}`. The benchmark reports retrieval hit@5, not answer accuracy. For an actual comparison, run at least two embedding configurations on the same document set and questions, then two allowed local LLM configurations with manually labeled answer correctness, source support, Chinese/English quality, latency, and failure cases. Record hardware, versions, model revisions, seed/temperature, and exact data split. Do not infer answer quality from hit@5.

## Limitations and design choices

- CJK overlapping bigrams provide only a basic keyword baseline; terms and English/CJK mixed queries can miss relevant passages.
- A predominantly English embedding such as `bge-small-en-v1.5` may underperform on Chinese. Compare a multilingual embedding.
- Retrieved hits may be irrelevant. The answer stage may still make an unsupported claim or select a weak source. Claim-level support verification is a planned improvement.
- The simple corpus is held in memory and re-embedded at each run; no OCR, persistent index, reranking, telemetry, access controls, or conversation state.
- A planner failure falls back to the raw question; a synthesis or model-server failure is surfaced rather than masked.

## Evaluation record

| Configuration | Retrieval hit@5 | Answer correctness | Source support | Status |
| --- | --- | --- | --- | --- |
| Keyword-only smoke checks | Not a labeled benchmark | Not measured | Not measured | Unit checks pass |
| Embedding model A vs B | Pending | — | — | Requires model downloads and labeled corpus |
| Local LLM A vs B | — | Pending | Pending | Requires supported local server and hardware |

The candidate-task PDF requests locally hosted SGLang, vLLM, or MLX (not Ollama), DSPy, two retrieval tools, citations with document and page, bilingual use, and empirical comparison. This directory implements the pipeline and benchmark entry points, while the model-hosted comparison remains pending. The parent repository's metadata normalization fix is a separate, small codebase improvement.
