"""Page-aware indexing and two explicitly callable retrieval tools."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import re


def tokens(text: str) -> list[str]:
    # English words and overlapping CJK bigrams; deliberately simple baseline.
    latin = re.findall(r"[a-z0-9]+", text.lower())
    cjk = re.findall(r"[\u3400-\u9fff]+", text)
    return latin + [part[i:i + 2] for part in cjk for i in range(max(0, len(part) - 1))]


@dataclass(frozen=True)
class Passage:
    document: str
    page: int
    chunk: int
    text: str

    @property
    def key(self) -> str:
        return f"{self.document}:p{self.page}:c{self.chunk}"

    def dict(self) -> dict:
        return asdict(self) | {"key": self.key}


def load_pages(root: Path, max_chars: int = 1100, overlap: int = 160) -> list[Passage]:
    if max_chars <= overlap or overlap < 0:
        raise ValueError("max_chars must exceed nonnegative overlap")
    from pypdf import PdfReader

    passages: list[Passage] = []
    for path in sorted(root.rglob("*.pdf")):
        for page_no, page in enumerate(PdfReader(str(path)).pages, 1):
            text = " ".join((page.extract_text() or "").split())
            if not text:
                continue
            step = max_chars - overlap
            for chunk_no, start in enumerate(range(0, len(text), step), 1):
                snippet = text[start:start + max_chars]
                if snippet:
                    passages.append(Passage(str(path.relative_to(root)), page_no, chunk_no, snippet))
                if start + max_chars >= len(text):
                    break
    return passages


class KeywordSearch:
    """Small BM25 implementation with no remote service."""

    def __init__(self, passages: list[Passage]):
        self.passages = passages
        self.docs = [tokens(p.text) for p in passages]
        self.avg = sum(map(len, self.docs)) / len(self.docs) if self.docs else 0
        self.df = {term: sum(term in doc for doc in self.docs)
                   for term in set(t for doc in self.docs for t in doc)}

    def search(self, query: str, k: int = 5) -> list[tuple[Passage, float]]:
        n = len(self.docs)
        out = []
        for passage, doc in zip(self.passages, self.docs):
            counts = {term: doc.count(term) for term in set(tokens(query))}
            score = 0.0
            for term, freq in counts.items():
                if not freq:
                    continue
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                score += idf * freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * len(doc) / (self.avg or 1)))
            if score > 0:
                out.append((passage, score))
        return sorted(out, key=lambda x: (-x[1], x[0].key))[:k]


class VectorSearch:
    """Sentence-transformer embeddings, normalized cosine similarity."""

    def __init__(self, passages: list[Passage], model_name: str):
        from sentence_transformers import SentenceTransformer
        self.passages = passages
        self.model = SentenceTransformer(model_name)
        self.vectors = self.model.encode([p.text for p in passages], normalize_embeddings=True)

    def search(self, query: str, k: int = 5) -> list[tuple[Passage, float]]:
        if not self.passages:
            return []
        vector = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.vectors @ vector
        indices = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.passages[i].key))[:k]
        return [(self.passages[i], float(scores[i])) for i in indices]


def fuse(keyword: list[tuple[Passage, float]], vector: list[tuple[Passage, float]], k: int = 5) -> list[Passage]:
    """Reciprocal rank fusion; retains results unique to either tool."""
    scores: dict[str, float] = {}
    passages: dict[str, Passage] = {}
    for results in (keyword, vector):
        for rank, (passage, _) in enumerate(results, 1):
            passages[passage.key] = passage
            scores[passage.key] = scores.get(passage.key, 0) + 1 / (60 + rank)
    return [passages[key] for key in sorted(scores, key=lambda key: (-scores[key], key))[:k]]
