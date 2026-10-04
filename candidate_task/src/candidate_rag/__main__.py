"""python -m candidate_rag {ask,search,benchmark} --help"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import KeywordSearch, VectorSearch, fuse, load_pages


def main() -> None:
    parser = argparse.ArgumentParser(description="ChatDKU bilingual local agentic RAG")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ask", "search", "benchmark"):
        p = sub.add_parser(name)
        p.add_argument("--docs", type=Path, required=True, help="Directory of PDFs")
        p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
        if name == "ask":
            p.add_argument("question")
            p.add_argument("--lm-model", default="Qwen/Qwen3-8B")
            p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
        elif name == "search":
            p.add_argument("query")
            p.add_argument("--mode", choices=["keyword", "vector", "hybrid"], default="hybrid")
        else:
            p.add_argument("--questions", type=Path, required=True, help="JSONL: question, expected_document, expected_page")
            p.add_argument("--mode", choices=["keyword", "vector", "hybrid"], default="hybrid")
    args = parser.parse_args()
    passages = load_pages(args.docs)
    if not passages:
        parser.error("No extractable PDF text found. Scanned PDFs need OCR first.")
    if args.command == "ask":
        from .agent import Agent
        result = Agent(passages, args.embedding_model, args.lm_model, args.base_url).ask(args.question)
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
        return
    keyword = KeywordSearch(passages)
    vector = VectorSearch(passages, args.embedding_model) if args.mode != "keyword" else None

    def retrieve(query: str):
        kw = keyword.search(query)
        if args.mode == "keyword":
            return [p for p, _ in kw]
        ve = vector.search(query)
        return [p for p, _ in ve] if args.mode == "vector" else fuse(kw, ve)

    if args.command == "search":
        print(json.dumps([p.dict() for p in retrieve(args.query)], ensure_ascii=False, indent=2))
        return
    cases = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        parser.error("No benchmark cases")
    results = []
    for case in cases:
        hits = retrieve(case["question"])
        hit = any(p.document == case["expected_document"] and p.page == case["expected_page"] for p in hits)
        results.append({"question": case["question"], "hit_at_5": hit, "retrieved": [p.key for p in hits]})
    print(json.dumps({"mode": args.mode, "embedding_model": args.embedding_model if vector else None,
                      "hit_at_5": sum(x["hit_at_5"] for x in results) / len(results), "n": len(results),
                      "cases": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
