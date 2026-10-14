from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

from .audit import AuditLogger
from .generator import AnswerGenerator
from .retriever import KnowledgeRetriever
from .risk_engine import RiskEngine
from .types import SearchResult, WorkflowResult


INTENT_KEYWORDS = {
    "metric_query": (
        "auc",
        "ks",
        "召回",
        "精确",
        "f1",
        "校准",
        "模型效果",
        "指标",
        "stacking",
        "lightgbm",
        "集成模型",
    ),
    "case_search": ("相似案例", "历史案例", "案例", "类似客户"),
    "case_review": ("审核", "风险", "违约概率", "是否通过", "申请"),
    "rule_query": ("规则", "阈值", "准入", "政策", "如何处理", "怎么办"),
}


class RiskReviewWorkflow:
    def __init__(
        self,
        knowledge_dir: str | Path | None = None,
        provider: str | None = None,
        audit_path: str | Path | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        knowledge = Path(knowledge_dir or root / "data" / "knowledge")
        self.retriever = KnowledgeRetriever.from_jsonl(
            [knowledge / "rules.jsonl", knowledge / "cases.jsonl"]
        )
        self.risk_engine = RiskEngine(project_root=root.parent)
        self.generator = AnswerGenerator(provider=provider)
        log_path = audit_path or os.getenv(
            "AUDIT_LOG_PATH", str(root / "data" / "audit" / "review_events.jsonl")
        )
        self.audit = AuditLogger(log_path)
        self.top_k = int(os.getenv("RAG_TOP_K", "5"))

    @staticmethod
    def classify_intent(question: str, application: dict[str, Any] | None = None) -> str:
        normalized = question.lower().strip()
        if application:
            return "case_review"
        scores = {
            intent: sum(1 for keyword in keywords if keyword in normalized)
            for intent, keywords in INTENT_KEYWORDS.items()
        }
        best_intent = max(scores, key=scores.get)
        return best_intent if scores[best_intent] > 0 else "rule_query"

    @staticmethod
    def _case_query(application: dict[str, Any]) -> str:
        return (
            f"信用等级 {application.get('grade', '')} {application.get('subGrade', '')}，"
            f"DTI {application.get('dti', '')}，FICO {application.get('ficoRangeHigh', '')}，"
            f"贷款金额 {application.get('loanAmnt', '')}，期限 {application.get('term', '')} 年，"
            f"利率 {application.get('interestRate', '')}，逾期 {application.get('delinquency_2years', 0)}"
        )

    @staticmethod
    def _rule_query(application: dict[str, Any]) -> str:
        parts = ["信用评级风险"]
        if application.get("dti") is not None:
            parts.append("DTI 负债收入比 偿债压力")
        if application.get("ficoRangeHigh") is not None:
            parts.append("FICO 信用评分")
        if application.get("loanAmnt") is not None and application.get("annualIncome") is not None:
            parts.append("贷款收入比 贷款金额 年收入")
        if int(application.get("term", 3)) >= 5:
            parts.append("长期贷款 贷款期限")
        if float(application.get("delinquency_2years", 0)) > 0:
            parts.append("历史逾期 征信核验")
        if float(application.get("pubRec", 0)) > 0:
            parts.append("公共记录 破产 人工核实")
        return " ".join(parts)

    def run(
        self,
        question: str,
        application: dict[str, Any] | None = None,
        log_event: bool = True,
    ) -> WorkflowResult:
        started = time.perf_counter()
        intent = self.classify_intent(question, application)
        risk_assessment = self.risk_engine.assess(application) if application else None

        if intent == "case_search":
            evidence_types = {"case"}
            retrieval_query = question
        elif intent == "case_review" and application:
            retrieval_query = self._case_query(application)
            rule_evidence = self.retriever.search(
                self._rule_query(application),
                top_k=3,
                item_types={"rule", "metric", "boundary"},
            )
            case_evidence = self.retriever.search(
                retrieval_query,
                top_k=2,
                item_types={"case"},
            )
            evidence = rule_evidence + case_evidence
            evidence_types = None
        elif intent == "case_review":
            evidence_types = {"rule", "metric", "boundary"}
            retrieval_query = question + " " + self._case_query(application or {})
        else:
            evidence_types = {"rule", "metric", "boundary"}
            retrieval_query = question

        if evidence_types is not None:
            evidence = self.retriever.search(
                retrieval_query,
                top_k=self.top_k,
                item_types=evidence_types,
            )
        answer, mode = self.generator.generate(question, evidence, risk_assessment)
        answer, warnings = self._guardrail(answer, evidence)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        result = WorkflowResult(
            intent=intent,
            answer=answer,
            evidence=evidence,
            risk_assessment=risk_assessment,
            generation_mode=mode,
            elapsed_ms=elapsed_ms,
            warnings=warnings,
        )
        if log_event:
            self.audit.write(
                {
                    "question": question,
                    "intent": intent,
                    "evidence_ids": [entry.item.item_id for entry in evidence],
                    "generation_mode": mode,
                    "risk_band": risk_assessment.get("risk_band") if risk_assessment else None,
                    "elapsed_ms": elapsed_ms,
                }
            )
        return result

    @staticmethod
    def _guardrail(answer: str, evidence: list[SearchResult]) -> tuple[str, list[str]]:
        warnings: list[str] = []
        cited = set(re.findall(r"\[([A-Z]+-[A-Z0-9-]+)\]", answer))
        evidence_ids = {result.item.item_id for result in evidence}
        if evidence and not (cited & evidence_ids):
            warnings.append("回答缺少有效证据编号，系统已补充引用。")
            answer += "\n\n参考证据：" + "、".join(f"[{value}]" for value in sorted(evidence_ids)[:3])
        if evidence and evidence[0].score < 0.08:
            warnings.append("最高检索得分较低，建议人工确认。")
        if "仅用于辅助审核" not in answer:
            answer += "\n\n本结果仅用于辅助审核，不构成自动授信、拒贷或合规决定。"
        return answer, warnings
