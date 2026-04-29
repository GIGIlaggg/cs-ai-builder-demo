"""In-memory knowledge base with BM25 retrieval.

Production swap: replace `KB.search` with a call to pgvector / Pinecone /
Qdrant / Vespa. The interface is intentionally narrow.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

KB_PATH = Path(__file__).parent.parent / "data" / "kb.jsonl"


@dataclass
class KBChunk:
    id: str
    title: str
    intent: str
    text: str

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "intent": self.intent, "text": self.text}


def _tokenize(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 1]


class KB:
    def __init__(self, chunks: list[KBChunk]):
        self.chunks = chunks
        self._tokenized = [_tokenize(c.text + " " + c.title) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized)

    @classmethod
    def load(cls, path: Path = KB_PATH) -> "KB":
        chunks: list[KBChunk] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            chunks.append(KBChunk(id=d["id"], title=d["title"], intent=d["intent"], text=d["text"]))
        return cls(chunks)

    def search(self, query: str, intent: str | None = None, k: int = 4) -> list[KBChunk]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(scores, self.chunks), key=lambda x: -x[0])
        results: list[KBChunk] = []
        for _, chunk in ranked:
            if intent and chunk.intent not in (intent, "all"):
                continue
            results.append(chunk)
            if len(results) >= k:
                break
        return results
