from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "knowledge"
SEED = 20260924
CASE_COUNT = 2048


RULES = [
    {
        "item_id": "METRIC-AUC",
        "item_type": "metric",
        "title": "最终模型区分能力",
        "content": "独立测试集 AUC 为 0.7368，KS 为 0.3458。AUC 衡量整体排序能力，不代表某一阈值下的准确率，也不能单独决定是否通过。",
        "tags": ["AUC", "KS", "模型效果", "排序能力"],
        "source": "京东信贷项目第六阶段实验结果",
    },
    {
        "item_id": "METRIC-RECALL",
        "item_type": "metric",
        "title": "最终阈值下的识别效果",
        "content": "在 0.22 阈值下，独立测试集召回率约 64.07%，精确率约 35.02%，F1 约 0.4528。阈值应结合漏检和误判成本共同选择。",
        "tags": ["召回率", "精确率", "F1", "阈值", "0.22"],
        "source": "京东信贷项目第六阶段实验结果",
    },
    {
        "item_id": "METRIC-CALIBRATION",
        "item_type": "metric",
        "title": "概率校准结果",
        "content": "Platt 校准后测试集 Brier Score 为 0.1403，10 分箱 ECE 为 0.0028。校准改善概率可信度，但上线后仍需持续监控。",
        "tags": ["Platt", "概率校准", "Brier", "ECE"],
        "source": "京东信贷项目第六阶段实验结果",
    },
    {
        "item_id": "RULE-GRADE",
        "item_type": "rule",
        "title": "信用评级风险核验",
        "content": "历史样本中评级从 A 到 G 的违约率总体递增。D 及以下评级应重点结合收入、DTI、FICO 和历史逾期核验，不可仅凭评级直接拒绝。",
        "tags": ["grade", "subGrade", "信用评级", "违约率"],
        "source": "京东信贷项目 EDA 与 SHAP 分析",
    },
    {
        "item_id": "RULE-DTI",
        "item_type": "rule",
        "title": "负债收入比核验",
        "content": "DTI 越高通常代表偿债压力越大。DTI 偏高时应核验收入真实性、其他负债和月供压力，并与贷款收入比共同判断。",
        "tags": ["DTI", "负债收入比", "偿债压力", "收入核验"],
        "source": "京东信贷项目特征分析",
    },
    {
        "item_id": "RULE-FICO",
        "item_type": "rule",
        "title": "FICO 信用评分核验",
        "content": "FICO 与违约风险总体呈负相关，评分较低时风险通常更高。评分只是风险信号之一，需要与评级、逾期、收入和期限联合审核。",
        "tags": ["FICO", "信用评分", "征信", "负相关"],
        "source": "京东信贷项目 EDA 与 SHAP 分析",
    },
    {
        "item_id": "RULE-LOAN-INCOME",
        "item_type": "rule",
        "title": "贷款收入比核验",
        "content": "贷款金额相对年收入越高，潜在还款负担越大。贷款收入比较高时应核验收入稳定性、资金用途和现有负债。",
        "tags": ["贷款收入比", "loanIncomeRatio", "贷款金额", "年收入"],
        "source": "京东信贷项目衍生特征实验",
    },
    {
        "item_id": "RULE-TERM",
        "item_type": "rule",
        "title": "贷款期限核验",
        "content": "历史样本中长期贷款的违约风险更高。期限超过 36 个月时，应关注总利息负担、收入稳定性以及月供覆盖能力。",
        "tags": ["term", "贷款期限", "36个月", "60个月"],
        "source": "京东信贷项目分组分析",
    },
    {
        "item_id": "RULE-DELINQUENCY",
        "item_type": "rule",
        "title": "历史逾期核验",
        "content": "近两年存在逾期记录时，应核验逾期次数、金额、持续时间和是否已经结清。历史逾期不能脱离其他信息直接作为拒绝依据。",
        "tags": ["逾期", "delinquency", "历史记录", "征信"],
        "source": "信贷审核演示规则",
    },
    {
        "item_id": "RULE-PUBLIC-RECORD",
        "item_type": "rule",
        "title": "公共记录与破产记录核验",
        "content": "命中公共记录或破产记录时，应转人工核实记录类型、发生时间和处置状态，并检查数据是否存在同名或录入错误。",
        "tags": ["公共记录", "破产", "pubRec", "人工核实"],
        "source": "信贷审核演示规则",
    },
    {
        "item_id": "RULE-INCOME-VERIFY",
        "item_type": "rule",
        "title": "收入真实性核验",
        "content": "当申报收入与职业、地区、历史流水或贷款金额明显不匹配时，应补充工资流水、纳税或其他可验证材料。不得仅依据模型推测收入造假。",
        "tags": ["收入", "真实性", "工资流水", "材料"],
        "source": "信贷审核演示规则",
    },
    {
        "item_id": "RULE-LOW-CONFIDENCE",
        "item_type": "rule",
        "title": "低置信度处理",
        "content": "当知识检索得分低、证据相互冲突或关键字段缺失时，系统应停止给出确定结论，提示补充信息并转人工确认。",
        "tags": ["低置信度", "冲突", "缺失", "人工确认"],
        "source": "助手产品安全规范",
    },
    {
        "item_id": "BOUNDARY-NO-AUTO-DECISION",
        "item_type": "boundary",
        "title": "不得自动审批",
        "content": "助手输出仅用于信息检索和辅助审核，不得自动作出授信、拒贷、额度或定价决定。最终决策必须进入现有授权与人工复核流程。",
        "tags": ["产品边界", "自动审批", "人工复核", "合规"],
        "source": "助手产品安全规范",
    },
    {
        "item_id": "BOUNDARY-SHAP",
        "item_type": "boundary",
        "title": "SHAP 解释边界",
        "content": "SHAP 描述特征对当前模型输出的贡献方向和相对程度，不证明特征与违约之间存在因果关系。",
        "tags": ["SHAP", "解释性", "因果", "特征贡献"],
        "source": "京东信贷项目 limitations",
    },
    {
        "item_id": "BOUNDARY-DATA",
        "item_type": "boundary",
        "title": "数据适用边界",
        "content": "当前模型基于历史离线样本，样本结构、标签成熟度和产品客群变化都可能导致效果漂移。迁移到其他产品前必须重新验证。",
        "tags": ["数据边界", "样本偏差", "模型漂移", "重新验证"],
        "source": "京东信贷项目 limitations",
    },
    {
        "item_id": "RULE-PRIVACY",
        "item_type": "rule",
        "title": "个人信息保护",
        "content": "演示系统不得录入姓名、身份证号、手机号、银行卡号等直接身份信息。真实接入前需要完成最小化采集、脱敏、权限和留痕设计。",
        "tags": ["隐私", "个人信息", "脱敏", "权限"],
        "source": "助手产品安全规范",
    },
    {
        "item_id": "RULE-THRESHOLD",
        "item_type": "rule",
        "title": "审核阈值使用方式",
        "content": "项目验证阈值为 0.22，用于平衡召回率和精确率。该阈值不是固定业务红线，上线前应根据坏账损失、人工审核成本和通过率重新测算。",
        "tags": ["阈值", "0.22", "坏账损失", "通过率"],
        "source": "京东信贷项目阈值实验",
    },
    {
        "item_id": "RULE-MISSING",
        "item_type": "rule",
        "title": "关键字段缺失处理",
        "content": "关键字段缺失既需要按字段性质填充，也应保留缺失指示变量。审核侧遇到收入、FICO 或历史逾期字段缺失时，应提示补充材料。",
        "tags": ["缺失值", "缺失指示", "收入", "FICO"],
        "source": "京东信贷项目特征工程实验",
    },
    {
        "item_id": "METRIC-STABILITY",
        "item_type": "metric",
        "title": "时间稳定性结果",
        "content": "滚动时间窗口共评估 20 个窗口，AUC 均值约 0.7261、标准差约 0.0226，稳定性结论为需要关注。应按月监控 AUC、KS、校准和客群分布。",
        "tags": ["时间稳定性", "滚动窗口", "AUC波动", "监控"],
        "source": "京东信贷项目第六阶段稳定性实验",
    },
    {
        "item_id": "METRIC-ENSEMBLE",
        "item_type": "metric",
        "title": "集成模型选择结论",
        "content": "LightGBM、加权投票和 Stacking 的提升均未达到 0.5 个百分点替换标准，因此保留更稳定、易解释和易部署的单一 XGBoost。",
        "tags": ["LightGBM", "投票", "Stacking", "模型选择"],
        "source": "京东信贷项目第六阶段模型比较",
    },
]


def risk_band(grade: str, dti: float, fico: int, term: int, delinquency: int) -> str:
    score = "ABCDEFG".index(grade) * 1.1 + max(dti - 15, 0) / 8 + max(680 - fico, 0) / 35
    score += 0.8 if term == 5 else 0
    score += min(delinquency, 2) * 0.7
    if score < 2.0:
        return "low"
    if score < 4.0:
        return "medium"
    if score < 6.0:
        return "high"
    return "very_high"


def build_cases(count: int, rng: random.Random) -> list[dict]:
    cases: list[dict] = []
    purposes = ["债务整合", "消费", "教育", "装修", "医疗", "小微经营"]
    outcomes = {"low": "按常规流程通过", "medium": "补充材料后通过", "high": "转人工复核", "very_high": "加强核验并升级复核"}
    for index in range(1, count + 1):
        grade = rng.choices(list("ABCDEFG"), weights=[10, 18, 25, 21, 14, 8, 4])[0]
        subgrade = f"{grade}{rng.randint(1, 5)}"
        dti = round(rng.uniform(3, 42), 1)
        fico = rng.randrange(600, 786, 5)
        term = rng.choices([3, 5], weights=[75, 25])[0]
        delinquency = rng.choices([0, 1, 2, 3], weights=[76, 16, 6, 2])[0]
        income = rng.randrange(36000, 300001, 2000)
        amount = rng.randrange(3000, min(50000, max(5000, int(income * 0.6))), 1000)
        rate = round(rng.uniform(5.0 + "ABCDEFG".index(grade), 12.0 + "ABCDEFG".index(grade) * 2.2), 2)
        band = risk_band(grade, dti, fico, term, delinquency)
        default_observed = rng.random() < {"low": 0.06, "medium": 0.16, "high": 0.31, "very_high": 0.47}[band]
        content = (
            f"合成案例：信用等级 {grade}、子等级 {subgrade}，年收入 {income} 元，"
            f"贷款金额 {amount} 元，期限 {term} 年，利率 {rate}%，DTI {dti}，"
            f"FICO {fico}-{fico + 4}，近两年逾期 {delinquency} 次，"
            f"用途为{rng.choice(purposes)}。辅助风险等级 {band}，审核结果为{outcomes[band]}，"
            f"观察标签为{'发生违约' if default_observed else '未发生违约'}。"
        )
        cases.append(
            {
                "item_id": f"CASE-{index:05d}",
                "item_type": "case",
                "title": f"合成信贷案例 {index:05d}",
                "content": content,
                "tags": [grade, subgrade, band, f"term_{term}", f"fico_{fico // 20 * 20}"],
                "source": "固定随机种子生成的脱敏合成案例",
                "metadata": {
                    "synthetic": True,
                    "grade": grade,
                    "subGrade": subgrade,
                    "annualIncome": income,
                    "loanAmnt": amount,
                    "term": term,
                    "interestRate": rate,
                    "dti": dti,
                    "ficoRangeLow": fico,
                    "ficoRangeHigh": fico + 4,
                    "delinquency_2years": delinquency,
                    "risk_band": band,
                    "observed_default": int(default_observed),
                },
            }
        )
    return cases


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    cases = build_cases(CASE_COUNT, rng)
    write_jsonl(OUTPUT_DIR / "rules.jsonl", RULES)
    write_jsonl(OUTPUT_DIR / "cases.jsonl", cases)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "rule_and_metric_count": len(RULES),
        "synthetic_case_count": len(cases),
        "total_knowledge_items": len(RULES) + len(cases),
        "data_statement": "规则来自项目实验结论与演示性审核规范；案例全部为合成数据，不代表真实客户。",
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
