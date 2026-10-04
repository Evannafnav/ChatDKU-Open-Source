"""DSPy planner and synthesizer with deterministic citation validation."""
from __future__ import annotations

from dataclasses import dataclass
import re

from .core import KeywordSearch, Passage, VectorSearch, fuse


@dataclass
class Answer:
    text: str
    sources: list[dict]
    plan: str


class Agent:
    def __init__(self, passages: list[Passage], embedding_model: str, lm_model: str, base_url: str):
        import dspy

        class Plan(dspy.Signature):
            """Choose a short search query from the user's question. Keep proper nouns verbatim."""
            question: str = dspy.InputField()
            search_query: str = dspy.OutputField(desc="Compact retrieval query in the user's language")

        class Synthesize(dspy.Signature):
            """Answer in the question's language using only supplied evidence. If insufficient, explicitly say so. Do not invent citations."""
            question: str = dspy.InputField()
            evidence: str = dspy.InputField()
            answer: str = dspy.OutputField(desc="Concise, grounded answer without citation markers")
            source_ids: str = dspy.OutputField(desc="Comma-separated numeric evidence IDs actually used; empty if no answer can be supported")

        dspy.configure(lm=dspy.LM(f"openai/{lm_model}", api_base=base_url, api_key="local", temperature=0))
        self.planner = dspy.Predict(Plan)
        self.synthesizer = dspy.Predict(Synthesize)
        self.keyword = KeywordSearch(passages)
        self.vector = VectorSearch(passages, embedding_model)

    def ask(self, question: str, k: int = 4) -> Answer:
        if not question.strip():
            raise ValueError("question cannot be empty")
        # Planner is allowed to fail; raw question remains a safe retrieval query.
        try:
            plan = str(self.planner(question=question).search_query).strip() or question
        except Exception:
            plan = question
        hits = fuse(self.keyword.search(plan, k), self.vector.search(plan, k), k)
        if not hits:
            return Answer("未找到相关文档证据。" if _chinese(question) else "No relevant document evidence was found.", [], plan)
        evidence = "\n\n".join(f"[{i}] {p.document}, page {p.page}: {p.text}" for i, p in enumerate(hits, 1))
        prediction = self.synthesizer(question=question, evidence=evidence)
        response = str(prediction.answer).strip()
        # The LM selects IDs, but filenames and pages come only from trusted metadata.
        # A cited page is still not a verified claim-level support judgment.
        ids = list(dict.fromkeys(int(s) for s in re.findall(r"\d+", str(prediction.source_ids))
                                 if 1 <= int(s) <= len(hits)))
        selected = [hits[i - 1] for i in ids]
        sources = [{"document": p.document, "page": p.page, "chunk": p.chunk} for p in selected]
        marker = "来源" if _chinese(question) else "Sources"
        citations = "; ".join(f"{p.document} p.{p.page}" for p in selected)
        return Answer(f"{response}\n\n{marker}: {citations or ('未选取可核验来源' if _chinese(question) else 'No verifiable source selected')}", sources, plan)


def _chinese(text: str) -> bool:
    return any("\u3400" <= c <= "\u9fff" for c in text)
