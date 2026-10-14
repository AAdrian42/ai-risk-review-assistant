from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk_assistant.workflow import RiskReviewWorkflow


EVAL_SET = [
    {"query": "最终模型的 AUC 和 KS 是多少", "intent": "metric_query", "target": "METRIC-AUC"},
    {"query": "0.22 阈值下召回率和精确率怎么样", "intent": "metric_query", "target": "METRIC-RECALL"},
    {"query": "Platt 校准后概率可信度如何", "intent": "metric_query", "target": "METRIC-CALIBRATION"},
    {"query": "信用评级差为什么风险更高", "intent": "case_review", "target": "RULE-GRADE"},
    {"query": "DTI 偏高时审核人员要核验什么", "intent": "case_review", "target": "RULE-DTI"},
    {"query": "FICO 评分低应该怎么看", "intent": "rule_query", "target": "RULE-FICO"},
    {"query": "贷款金额相对收入太高怎么办", "intent": "rule_query", "target": "RULE-LOAN-INCOME"},
    {"query": "五年期长期贷款要关注什么", "intent": "rule_query", "target": "RULE-TERM"},
    {"query": "近两年存在逾期记录如何审核", "intent": "case_review", "target": "RULE-DELINQUENCY"},
    {"query": "查到公共记录和破产记录应该怎么处理", "intent": "rule_query", "target": "RULE-PUBLIC-RECORD"},
    {"query": "检索不到可靠规则时怎么办", "intent": "rule_query", "target": "RULE-LOW-CONFIDENCE"},
    {"query": "助手能不能直接自动拒贷", "intent": "rule_query", "target": "BOUNDARY-NO-AUTO-DECISION"},
    {"query": "SHAP 是否说明了因果关系", "intent": "rule_query", "target": "BOUNDARY-SHAP"},
    {"query": "模型迁移到别的贷款产品需要重新验证吗", "intent": "rule_query", "target": "BOUNDARY-DATA"},
    {"query": "客户收入缺失应该怎么处理", "intent": "rule_query", "target": "RULE-MISSING"},
    {"query": "滚动时间窗口的 AUC 稳定吗", "intent": "metric_query", "target": "METRIC-STABILITY"},
    {"query": "为什么没有选择 Stacking 集成模型", "intent": "metric_query", "target": "METRIC-ENSEMBLE"},
]


def main() -> None:
    workflow = RiskReviewWorkflow(provider="local")
    details: list[dict] = []
    intent_hits = 0
    retrieval_hits = 0
    citation_hits = 0

    for item in EVAL_SET:
        result = workflow.run(item["query"], log_event=False)
        top3 = [entry.item.item_id for entry in result.evidence[:3]]
        intent_ok = result.intent == item["intent"]
        retrieval_ok = item["target"] in top3
        cited_ids = set(re.findall(r"\[([A-Z]+-[A-Z0-9-]+)\]", result.answer))
        citation_ok = bool(cited_ids.intersection(top3))
        intent_hits += int(intent_ok)
        retrieval_hits += int(retrieval_ok)
        citation_hits += int(citation_ok)
        details.append(
            {
                "query": item["query"],
                "expected_intent": item["intent"],
                "predicted_intent": result.intent,
                "target_id": item["target"],
                "top3": ", ".join(top3),
                "intent_ok": intent_ok,
                "retrieval_ok": retrieval_ok,
                "citation_ok": citation_ok,
                "elapsed_ms": result.elapsed_ms,
            }
        )

    count = len(EVAL_SET)
    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": count,
        "metrics": {
            "intent_accuracy": intent_hits / count,
            "retrieval_hit_at_3": retrieval_hits / count,
            "citation_completeness": citation_hits / count,
        },
        "details": details,
        "limitations": [
            "评测集由项目作者构建，规模较小，可能高估泛化效果。",
            "案例全部为合成数据，尚未进行真实审核人员用户测试。",
            "效率提升比例尚未测量，不能据此声称提升 60%。",
        ],
    }
    output_dir = ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation_report.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
