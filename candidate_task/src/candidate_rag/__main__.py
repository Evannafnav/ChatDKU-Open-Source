"""python -m candidate_rag {ask,search,benchmark,evaluate} --help"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import KeywordSearch, VectorSearch, fuse, load_pages


def main() -> None:
    parser = argparse.ArgumentParser(description="ChatDKU bilingual local agentic RAG")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ask", "search", "benchmark", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--docs", type=Path, required=True, help="Directory of PDFs")
        p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
        if name in ("ask", "evaluate"):
            if name == "ask":
                p.add_argument("question")
            else:
                p.add_argument("--questions", type=Path, required=True)
                p.add_argument("--output", type=Path, required=True)
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
    if args.command == "evaluate":
        from .agent import Agent
        from .evaluation import evaluate_cases
        cases = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
        report = evaluate_cases(Agent(passages, args.embedding_model, args.lm_model, args.base_url), cases)
        report["lm_model"] = args.lm_model
        report["embedding_model"] = args.embedding_model
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(cases)} cases to {args.output}")
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
        matches = [p.document == case["expected_document"] and p.page == case["expected_page"] for p in hits]
        results.append({"case_id": case.get("case_id"), "language": case.get("language"),
                        "question": case["question"], "hit_at_1": bool(matches and matches[0]),
                        "hit_at_5": any(matches[:5]), "retrieved": [p.key for p in hits]})
    languages = sorted({r["language"] for r in results if r["language"]})
    by_language = {lang: {"n": sum(r["language"] == lang for r in results),
                          "hit_at_1": sum(r["hit_at_1"] for r in results if r["language"] == lang) /
                          sum(r["language"] == lang for r in results),
                          "hit_at_5": sum(r["hit_at_5"] for r in results if r["language"] == lang) /
                          sum(r["language"] == lang for r in results)} for lang in languages}
    print(json.dumps({"mode": args.mode, "embedding_model": args.embedding_model if vector else None,
                      "hit_at_1": sum(x["hit_at_1"] for x in results) / len(results),
                      "hit_at_5": sum(x["hit_at_5"] for x in results) / len(results), "n": len(results),
                      "by_language": by_language,
                      "cases": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
