from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .types import KnowledgeItem, SearchResult


class KnowledgeRetriever:
    """Small, reproducible Chinese vector store backed by TF-IDF vectors."""

    def __init__(self, items: Iterable[KnowledgeItem]) -> None:
        self.items = list(items)
        if not self.items:
            raise ValueError("Knowledge base is empty")
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=1,
            max_features=30000,
            sublinear_tf=True,
        )
        texts = [self._search_text(item) for item in self.items]
        self.matrix = self.vectorizer.fit_transform(texts)

    @staticmethod
    def _search_text(item: KnowledgeItem) -> str:
        return " ".join([item.title, item.content, " ".join(item.tags)])

    @classmethod
    def from_jsonl(cls, paths: Iterable[str | Path]) -> "KnowledgeRetriever":
        items: list[KnowledgeItem] = []
        for path_value in paths:
            path = Path(path_value)
            if not path.exists():
                raise FileNotFoundError(f"Knowledge file not found: {path}")
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        items.append(KnowledgeItem.from_dict(json.loads(line)))
        return cls(items)

    def search(
        self,
        query: str,
        top_k: int = 5,
        item_types: set[str] | None = None,
    ) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []
        query_vector = self.vectorizer.transform([query])
        similarities = linear_kernel(query_vector, self.matrix).ravel()
        candidate_indices = np.argsort(similarities)[::-1]
        results: list[SearchResult] = []
        for index in candidate_indices:
            item = self.items[int(index)]
            if item_types and item.item_type not in item_types:
                continue
            results.append(SearchResult(item=item, score=float(similarities[index])))
            if len(results) >= top_k:
                break
        return results

