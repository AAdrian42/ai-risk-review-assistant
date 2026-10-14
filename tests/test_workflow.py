from __future__ import annotations

import json
from pathlib import Path

import pytest

from risk_assistant.retriever import KnowledgeRetriever
from risk_assistant.risk_engine import RiskEngine
from risk_assistant.workflow import RiskReviewWorkflow
from risk_assistant.generator import AnswerGenerator
from langchain_core.vectorstores import InMemoryVectorStore


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = ROOT / "data" / "knowledge"


@pytest.fixture(scope="module")
def workflow() -> RiskReviewWorkflow:
    return RiskReviewWorkflow(provider="local", audit_path=ROOT / "data" / "audit" / "test.jsonl")


def test_manifest_declares_synthetic_cases() -> None:
    manifest = json.loads((KNOWLEDGE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["synthetic_case_count"] >= 2000
    assert "合成" in manifest["data_statement"]


def test_bundled_model_loads_without_parent_project(tmp_path) -> None:
    engine = RiskEngine(project_root=tmp_path)
    assert engine.mode == "phase6_platt_xgboost"
    assert engine.service.model_path.is_relative_to(ROOT)


def test_retriever_finds_auc_metric() -> None:
    retriever = KnowledgeRetriever.from_jsonl([KNOWLEDGE / "rules.jsonl", KNOWLEDGE / "cases.jsonl"])
    results = retriever.search("模型 AUC 和 KS", top_k=3, item_types={"metric"})
    assert "METRIC-AUC" in [result.item.item_id for result in results]


def test_workflow_includes_citation_and_boundary(workflow: RiskReviewWorkflow) -> None:
    result = workflow.run("SHAP 是否能够证明因果关系", log_event=False)
    assert "[BOUNDARY-SHAP]" in result.answer
    assert "仅用于辅助审核" in result.answer


def test_case_review_returns_explanation(workflow: RiskReviewWorkflow) -> None:
    result = workflow.run(
        "请审核这笔申请",
        application={
            "loanAmnt": 30000,
            "term": 5,
            "interestRate": 19.5,
            "installment": 500,
            "grade": "E",
            "subGrade": "E4",
            "annualIncome": 60000,
            "dti": 32,
            "ficoRangeLow": 630,
            "ficoRangeHigh": 634,
            "delinquency_2years": 2,
            "pubRec": 1,
        },
        log_event=False,
    )
    assert result.risk_assessment is not None
    assert result.risk_assessment["risk_band"] in {"high", "very_high"}
    assert len(result.risk_assessment["risk_signals"]) >= 3


def test_vector_store_executes_filtered_search(workflow, monkeypatch) -> None:
    store = workflow.retriever.vector_store
    assert isinstance(store, InMemoryVectorStore)
    calls = []
    original = store.similarity_search_with_score
    def capture(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(store, "similarity_search_with_score", capture)
    results = workflow.retriever.search("信用等级", item_types={"rule"})
    assert calls and results and all(r.item.item_type == "rule" for r in results)
    assert workflow.retriever.search("信用等级", top_k=0) == []


def test_missing_key_uses_local_evidence(workflow, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    evidence = workflow.retriever.search("AUC", item_types={"metric"})
    answer, mode = AnswerGenerator("openai").generate("AUC", evidence)
    assert mode == "local_grounded" and "[METRIC-AUC]" in answer


def test_provider_failure_falls_back(workflow, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-a-real-key")
    generator = AnswerGenerator("openai")
    def fail(*args):
        raise TimeoutError("simulated failure")
    monkeypatch.setattr(generator, "_openai", fail)
    evidence = workflow.retriever.search("AUC", item_types={"metric"})
    answer, mode = generator.generate("AUC", evidence)
    assert mode == "local_fallback" and "[METRIC-AUC]" in answer


def test_unknown_citation_is_replaced(workflow) -> None:
    evidence = workflow.retriever.search("AUC", item_types={"metric"})
    answer, warnings = workflow._guardrail("错误依据 [RULE-INVENTED]", evidence)
    assert "RULE-INVENTED" not in answer and warnings
