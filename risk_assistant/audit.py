from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @staticmethod
    def _fingerprint(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    def write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "event_time": datetime.now(timezone.utc).isoformat(),
            "question_fingerprint": self._fingerprint(str(payload.get("question", ""))),
            "intent": payload.get("intent"),
            "evidence_ids": payload.get("evidence_ids", []),
            "generation_mode": payload.get("generation_mode"),
            "risk_band": payload.get("risk_band"),
            "elapsed_ms": payload.get("elapsed_ms"),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
