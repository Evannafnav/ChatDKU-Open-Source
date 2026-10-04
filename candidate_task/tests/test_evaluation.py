from candidate_rag.agent import Answer
from candidate_rag.evaluation import evaluate_cases


def test_evaluation_records_citation_and_defers_human_judgment():
    class FakeAgent:
        def ask(self, question):
            return Answer("A grounded-looking answer", [{"document": "guide.pdf", "page": 2}], question)

    times = iter([1.0, 1.5])
    report = evaluate_cases(
        FakeAgent(),
        [{"case_id": "en-1", "language": "en", "question": "When?", "expected_document": "guide.pdf", "expected_page": 2}],
        clock=lambda: next(times),
    )
    assert report["expected_page_citation_rate"] == 1.0
    assert report["median_latency_seconds"] == 0.5
    assert report["cases"][0]["manual_claim_support"] is None
    assert report["cases"][0]["manual_answer_correct"] is None
