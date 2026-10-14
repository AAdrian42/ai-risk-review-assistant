from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class KnowledgeItem:
    item_id: str
    item_type: str
    title: str
    content: str
    tags: list[str]
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KnowledgeItem":
        return cls(
            item_id=str(value["item_id"]),
            item_type=str(value["item_type"]),
            title=str(value["title"]),
            content=str(value["content"]),
            tags=list(value.get("tags", [])),
            source=str(value.get("source", "unknown")),
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    item: KnowledgeItem
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, **self.item.to_dict()}


@dataclass
class WorkflowResult:
    intent: str
    answer: str
    evidence: list[SearchResult]
    risk_assessment: dict[str, Any] | None
    generation_mode: str
    elapsed_ms: float
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "answer": self.answer,
            "evidence": [item.to_dict() for item in self.evidence],
            "risk_assessment": self.risk_assessment,
            "generation_mode": self.generation_mode,
            "elapsed_ms": self.elapsed_ms,
            "warnings": self.warnings,
        }
