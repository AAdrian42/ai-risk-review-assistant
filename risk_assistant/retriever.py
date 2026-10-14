from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import linear_kernel
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from .types import KnowledgeItem, SearchResult


class LocalEmbeddings(Embeddings):
    """Normalized lexical vectors; no remote model or API key required."""

    def __init__(self) -> None:
        self.encoder = HashingVectorizer(analyzer="char", ngram_range=(2, 4),
                                        n_features=1024, alternate_sign=False, norm="l2")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.encoder.transform(texts).toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class KnowledgeRetriever:
    """LangChain vector retrieval combined with TF-IDF lexical reranking."""

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
        self.vector_store = InMemoryVectorStore(LocalEmbeddings())
        self.vector_store.add_documents([
            Document(page_content=text, metadata={"index": i, "item_type": item.item_type})
            for i, (item, text) in enumerate(zip(self.items, texts))
        ])

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
        if top_k <= 0:
            return []
        vector_hits = self.vector_store.similarity_search_with_score(
            query, k=max(top_k * 4, 20),
            filter=(lambda doc: doc.metadata["item_type"] in item_types) if item_types else None,
        )
        query_vector = self.vectorizer.transform([query])
        similarities = linear_kernel(query_vector, self.matrix).ravel()
        # Union candidates preserves rare exact terms while adding vector neighbors.
        lexical = [int(i) for i in np.argsort(similarities)[::-1]
                   if not item_types or self.items[int(i)].item_type in item_types][:max(top_k * 4, 20)]
        vector_scores = {doc.metadata["index"]: float(score) for doc, score in vector_hits}
        candidates = set(lexical) | set(vector_scores)
        candidate_indices = sorted(candidates, key=lambda i: (
            0.85 * float(similarities[i]) + 0.15 * vector_scores.get(i, 0)), reverse=True)
        results: list[SearchResult] = []
        for index in candidate_indices:
            item = self.items[int(index)]
            if item_types and item.item_type not in item_types:
                continue
            results.append(SearchResult(item=item, score=float(similarities[index])))
            if len(results) >= top_k:
                break
        return results

