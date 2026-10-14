from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from risk_assistant.workflow import RiskReviewWorkflow


ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="AI 智能风控审核助手",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .stApp { background: #f6f7f9; color: #17202a; }
      [data-testid="stSidebar"] { background: #e8eef2; }
      [data-testid="stSidebar"] * { color: #17202a; }
      [data-testid="stSidebar"] div[data-testid="stMetric"] {
        background: #ffffff; border-color: #cbd5dd;
      }
      [data-testid="stSidebar"] div[data-testid="stMetric"] * { color: #17202a; }
      [data-testid="stTextArea"] textarea,
      [data-testid="stNumberInput"] input {
        background-color: #ffffff !important; color: #17202a !important;
        -webkit-text-fill-color: #17202a !important; caret-color: #147d92;
      }
      [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background-color: #ffffff !important; color: #17202a !important;
      }
      [data-testid="stSelectbox"] [data-baseweb="select"] * { color: #17202a !important; }
      [data-testid="stMetric"] * { color: #17202a; }
      .block-container { padding-top: 1.35rem; max-width: 1500px; }
      h1, h2, h3 { letter-spacing: 0; }
      div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #dfe3e8; border-radius: 6px; padding: 12px;
      }
      div[data-testid="stExpander"] {
        background: #ffffff; border: 1px solid #dfe3e8; border-radius: 6px;
      }
      .risk-note {
        border-left: 4px solid #b54708; background: #fff7ed; padding: 12px 14px; margin: 8px 0 16px;
      }
      .evidence { color: #475467; font-size: 0.92rem; }
      @media (max-width: 1100px) {
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
          min-width: 210px; flex: 1 1 210px;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_workflow() -> RiskReviewWorkflow:
    return RiskReviewWorkflow()


def ensure_data() -> None:
    manifest = ROOT / "data" / "knowledge" / "manifest.json"
    if manifest.exists():
        return
    st.error("知识库尚未生成，请先运行：python scripts/build_demo_corpus.py")
    st.stop()


def show_evidence(result) -> None:
    st.subheader("证据与相似案例")
    for entry in result.evidence:
        label = f"{entry.item.item_id} | {entry.item.title} | 相似度 {entry.score:.3f}"
        with st.expander(label):
            st.write(entry.item.content)
            st.caption(f"来源：{entry.item.source}")


ensure_data()
workflow = load_workflow()
manifest = json.loads((ROOT / "data" / "knowledge" / "manifest.json").read_text(encoding="utf-8"))

with st.sidebar:
    st.title("风控审核台")
    st.caption("个人项目原型")
    st.divider()
    st.metric("知识条目", f"{manifest['total_knowledge_items']:,}")
    st.metric("合成案例", f"{manifest['synthetic_case_count']:,}")
    st.metric("预测引擎", workflow.risk_engine.mode)
    st.divider()
    st.caption("不录入姓名、身份证号、手机号或银行卡号。")

st.title("AI 智能风控审核助手")
st.markdown(
    '<div class="risk-note">输出仅用于规则检索与辅助审核，最终授信决定须进入人工授权流程。</div>',
    unsafe_allow_html=True,
)

review_tab, qa_tab, system_tab = st.tabs(["申请审核", "规则问答", "评测与系统"])

with review_tab:
    left, right = st.columns([0.92, 1.08], gap="large")
    with left:
        st.subheader("申请信息")
        with st.form("credit_application"):
            row1a, row1b, row1c = st.columns(3)
            grade = row1a.selectbox("信用等级", list("ABCDEFG"), index=2)
            sub_num = row1b.selectbox("子等级", [1, 2, 3, 4, 5], index=2)
            term = row1c.segmented_control("期限", options=[3, 5], default=3, format_func=lambda x: f"{x} 年")

            row2a, row2b = st.columns(2)
            loan_amount = row2a.number_input("贷款金额（元）", 1000, 500000, 20000, 1000)
            annual_income = row2b.number_input("年收入（元）", 10000, 3000000, 100000, 5000)

            row3a, row3b = st.columns(2)
            interest_rate = row3a.number_input("年利率（%）", 0.0, 40.0, 12.0, 0.1)
            dti = row3b.number_input("DTI", 0.0, 100.0, 18.0, 0.5)

            row4a, row4b, row4c = st.columns(3)
            fico_low = row4a.number_input("FICO 下限", 300, 850, 680, 5)
            delinquency = row4b.number_input("近两年逾期次数", 0, 20, 0, 1)
            public_records = row4c.number_input("公共记录数", 0, 20, 0, 1)
            submitted = st.form_submit_button("开始辅助审核", type="primary", width="stretch")

    with right:
        st.subheader("审核结果")
        if submitted:
            application = {
                "loanAmnt": loan_amount,
                "term": term or 3,
                "interestRate": interest_rate,
                "installment": round(loan_amount / ((term or 3) * 12), 2),
                "grade": grade,
                "subGrade": f"{grade}{sub_num}",
                "annualIncome": annual_income,
                "dti": dti,
                "ficoRangeLow": fico_low,
                "ficoRangeHigh": min(fico_low + 4, 900),
                "delinquency_2years": delinquency,
                "pubRec": public_records,
            }
            result = workflow.run("请审核这笔申请并解释主要风险依据", application=application)
            assessment = result.risk_assessment or {}
            m1, m2, m3 = st.columns(3)
            m1.metric("风险等级", str(assessment.get("risk_band", "-")).upper())
            m2.metric("参考违约概率", f"{assessment.get('probability', 0):.1%}")
            m3.metric("审核阈值", f"{assessment.get('threshold', 0):.0%}")
            st.progress(min(float(assessment.get("probability", 0)), 1.0))
            st.markdown(result.answer)
            if result.warnings:
                st.warning("；".join(result.warnings))
            st.caption(f"引擎：{assessment.get('engine_mode')} | 耗时：{result.elapsed_ms:.0f} ms")
            show_evidence(result)
        else:
            st.info("填写左侧信息后开始审核。")

with qa_tab:
    st.subheader("规则与案例查询")
    question = st.text_area(
        "输入问题",
        value="模型为什么把长期贷款和较高 DTI 视为风险信号？",
        height=90,
    )
    if st.button("查询知识库", type="primary"):
        result = workflow.run(question)
        c1, c2, c3 = st.columns(3)
        c1.metric("识别意图", result.intent)
        c2.metric("证据数量", len(result.evidence))
        c3.metric("响应耗时", f"{result.elapsed_ms:.0f} ms")
        st.markdown(result.answer)
        if result.warnings:
            st.warning("；".join(result.warnings))
        show_evidence(result)

with system_tab:
    st.subheader("知识库")
    c1, c2, c3 = st.columns(3)
    c1.metric("规则与指标", manifest["rule_and_metric_count"])
    c2.metric("合成案例", f"{manifest['synthetic_case_count']:,}")
    c3.metric("总条目", f"{manifest['total_knowledge_items']:,}")
    st.caption(manifest["data_statement"])

    report_path = ROOT / "reports" / "evaluation_report.json"
    st.subheader("离线评测")
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = report["metrics"]
        e1, e2, e3 = st.columns(3)
        e1.metric("意图准确率", f"{metrics['intent_accuracy']:.1%}")
        e2.metric("检索 Hit@3", f"{metrics['retrieval_hit_at_3']:.1%}")
        e3.metric("引用完整率", f"{metrics['citation_completeness']:.1%}")
        with st.expander("查看评测明细"):
            st.dataframe(pd.DataFrame(report["details"]), width="stretch", hide_index=True)
    else:
        st.info("运行 python scripts/evaluate.py 后显示实测结果。")

    st.subheader("数据和决策边界")
    st.write("案例全部为固定随机种子生成的合成数据；规则主要来自项目实验结论和演示性审核规范。")
    st.write("模型分数、SHAP 或相似案例都不能单独作为拒贷、额度或定价依据。")
