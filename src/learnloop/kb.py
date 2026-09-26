"""The knowledge base the bot answers from: approved Q&A entries plus a local embedding index."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_embedder = None


def embed(texts: list[str]) -> np.ndarray:
    """Normalized embeddings, computed locally with fastembed (no API call, no cost)."""
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding

        _embedder = TextEmbedding(EMBED_MODEL)
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    v = np.array(list(_embedder.embed(texts)), dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@dataclass
class Entry:
    id: str
    question: str
    answer: str
    source: str  # "seed" or the conversation the loop learned it from
    approved_by: str  # "seed", "judge-only" or the reviewer's name


class KB:
    def __init__(self, entries: list[Entry]):
        self.entries = entries
        # Question and answer together: a vague question still matches an answer about the same thing.
        self.vectors = embed([f"{e.question} {e.answer}" for e in entries])

    @classmethod
    def load(cls, path: Path) -> "KB":
        return cls([Entry(**json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(asdict(e), ensure_ascii=False) + "\n" for e in self.entries), encoding="utf-8")

    def search(self, query: str, k: int = 3) -> list[tuple[Entry, float]]:
        if not self.entries:
            return []
        scores = self.vectors @ embed([query])[0]
        idx = np.argsort(-scores)[:k]
        return [(self.entries[i], float(scores[i])) for i in idx]

    def add(self, new: list[Entry]) -> "KB":
        return KB(self.entries + new)
