# AI 智能风控审核助手

这是一个由信贷违约预测项目延伸出的个人产品原型。它帮助风控审核人员完成规则查询、相似案例检索和单笔申请初审，并为每条结论保留证据引用与审计记录。

**在线体验：** [aadrian42-risk-review-assistant.streamlit.app](https://aadrian42-risk-review-assistant.streamlit.app/)

> 项目边界：这是离线原型，不连接京东内部系统，不使用企业生产数据，也不自动作出授信决定。演示案例均为合成数据。

## 已实现能力

- 规则问答：用自然语言查询准入、风险分层、模型指标和审核要点。
- 相似案例：根据申请特征召回相近的历史演示案例。
- 辅助审核：识别评级、负债率、FICO、期限、利率和历史逾期等风险信号。
- 可追溯回答：所有结论展示知识条目编号、检索得分和引用原文。
- 多节点工作流：意图识别 -> 知识检索 -> 风险分析 -> 回答生成 -> 合规校验 -> 审计留痕。
- 可选大模型：默认本地抽取式回答；配置密钥后可切换 GPT 或 Claude。
- 离线评测：覆盖意图分类、规则检索和引用完整性，输出可复核 JSON 报告。

## 项目结构

```text
ai_risk_review_assistant/
|-- app.py                         # Streamlit 演示界面
|-- risk_assistant/                # 检索、工作流、模型适配与日志
|-- scripts/build_demo_corpus.py   # 构造公开口径规则与合成案例
|-- scripts/evaluate.py            # 离线评测
|-- data/knowledge/                # 规则与案例知识库
|-- docs/PRD.md                    # 产品需求文档
|-- docs/ARCHITECTURE.md           # 架构与数据流
|-- tests/                         # 单元测试
`-- reports/evaluation_report.json # 实测结果
```

## 快速开始

```powershell
cd ai_risk_review_assistant
python -m pip install -r requirements.txt
python scripts\build_demo_corpus.py
python scripts\evaluate.py
streamlit run app.py
```

默认使用本地检索与模板化回答，不需要联网或 API Key。

## 云端部署

在 Streamlit Community Cloud 中选择本仓库、`main` 分支和 `app.py`，Python 版本选择 3.12。
仓库包含合成案例、知识条目和校准模型，无需访问其他项目目录。
审核引擎正常加载时显示 `phase6_platt_xgboost`；其他引擎标识表示使用了规则回退。

## 可选大模型

将 `.env.example` 中的配置写入环境变量，并把 `LLM_PROVIDER` 设置为 `openai` 或 `anthropic`。代码只会把当前问题和已召回的证据片段发送给模型，不会发送完整知识库；真实业务接入前仍需完成隐私、脱敏和供应商合规评审。

## 数据说明

知识库由两部分组成：

1. 根据本项目已有实验结果整理的模型指标、风险因素、阈值和使用边界。
2. 用固定随机种子生成的合成信贷案例，用于演示相似案例召回。

合成案例不代表真实客户，也不能用于生产审批。`data/knowledge/manifest.json` 会记录规则条数、合成案例条数、生成参数和数据声明。

## 简历口径

项目完成后可以如实描述为：独立搭建风控规则问答和案例检索原型，完成 RAG 检索、多节点审核工作流、可追溯引用、离线评测及 Streamlit 产品演示。知识库规模和准确率应引用 `manifest.json` 与 `evaluation_report.json` 中的实测结果。

“效率提升 60%”目前没有真实用户实验支持，不应直接写入简历。后续需要邀请实际用户完成对照任务并记录平均耗时后再计算。
