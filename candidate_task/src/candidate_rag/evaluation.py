"""Record answer-level observations without inventing quality judgments."""
from __future__ import annotations

from dataclasses import asdict
from statistics import median
from time import perf_counter
from typing import Callable, Iterable


def evaluate_cases(agent, cases: Iterable[dict], clock: Callable[[], float] = perf_counter) -> dict:
    """Run a fixed question set and leave claim support for human annotation."""
    rows = []
    for index, case in enumerate(cases, 1):
        question = case["question"]
        expected_document = case["expected_document"]
        expected_page = int(case["expected_page"])
        start = clock()
        try:
            answer = agent.ask(question)
            elapsed = clock() - start
            sources = answer.sources
            row = {
                "case_id": case.get("case_id", index),
                "language": case.get("language", "zh" if any("\u3400" <= c <= "\u9fff" for c in question) else "en"),
                "question": question,
                "expected_document": expected_document,
                "expected_page": expected_page,
                "reference_answer": case.get("reference_answer"),
                "answer": asdict(answer),
                "expected_page_cited": any(
                    source["document"] == expected_document and source["page"] == expected_page
                    for source in sources
                ),
                "latency_seconds": round(elapsed, 3),
                "error": None,
                "manual_answer_correct": None,
                "manual_claim_support": None,
            }
        except Exception as exc:
            elapsed = clock() - start
            row = {
                "case_id": case.get("case_id", index),
                "language": case.get("language", "unknown"),
                "question": question,
                "expected_document": expected_document,
                "expected_page": expected_page,
                "reference_answer": case.get("reference_answer"),
                "answer": None,
                "expected_page_cited": False,
                "latency_seconds": round(elapsed, 3),
                "error": f"{type(exc).__name__}: {exc}",
                "manual_answer_correct": None,
                "manual_claim_support": None,
            }
        rows.append(row)
    if not rows:
        raise ValueError("No evaluation cases")
    successful = [row for row in rows if row["error"] is None]
    return {
        "n": len(rows),
        "successful": len(successful),
        "expected_page_citation_rate": sum(row["expected_page_cited"] for row in rows) / len(rows),
        "median_latency_seconds": median(row["latency_seconds"] for row in rows),
        "note": "Citation match is metadata-level only. Annotate answer correctness and claim support manually.",
        "cases": rows,
    }
