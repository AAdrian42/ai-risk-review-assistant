from __future__ import annotations

import os
from typing import Any

from .types import SearchResult


SYSTEM_PROMPT = """你是信贷风控审核辅助助手。只能依据给定证据回答，不得编造规则，
不得作出自动授信或拒贷决定。结论必须引用证据编号，信息不足时明确要求人工确认。"""


class AnswerGenerator:
    def __init__(self, provider: str | None = None) -> None:
        self.provider = (provider or os.getenv("LLM_PROVIDER", "local")).lower()

    def generate(
        self,
        question: str,
        evidence: list[SearchResult],
        risk_assessment: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        try:
            if self.provider == "openai" and os.getenv("OPENAI_API_KEY"):
                return self._openai(question, evidence, risk_assessment), "openai"
            if self.provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
                return self._anthropic(question, evidence, risk_assessment), "anthropic"
        except Exception:
            # Do not surface credentials or provider error payloads to the UI.
            return self._local(question, evidence, risk_assessment), "local_fallback"
        return self._local(question, evidence, risk_assessment), "local_grounded"

    @staticmethod
    def _context(evidence: list[SearchResult], risk_assessment: dict[str, Any] | None) -> str:
        blocks = [
            f"[{result.item.item_id}] {result.item.title}: {result.item.content}"
            for result in evidence
        ]
        if risk_assessment:
            blocks.append(f"[RISK_ENGINE] {risk_assessment}")
        return "\n".join(blocks)

    def _local(
        self,
        question: str,
        evidence: list[SearchResult],
        risk_assessment: dict[str, Any] | None,
    ) -> str:
        lines: list[str] = []
        if risk_assessment:
            probability = risk_assessment["probability"]
            lines.append(
                f"该申请的辅助风险结果为 {risk_assessment['risk_band']}，"
                f"参考违约概率为 {probability:.1%}，审核阈值为 {risk_assessment['threshold']:.0%}。"
            )
            lines.append("主要信号：" + "；".join(risk_assessment["risk_signals"]) + "。")
            if risk_assessment["protective_signals"]:
                lines.append("保护因素：" + "；".join(risk_assessment["protective_signals"]) + "。")
            lines.append(risk_assessment["decision_support"])

        if evidence:
            lines.append("依据如下：")
            for result in evidence[:3]:
                lines.append(
                    f"[{result.item.item_id}] {result.item.title}：{result.item.content}"
                )
        else:
            lines.append("当前知识库没有检索到足够依据，请转人工确认或补充更具体的问题。")

        lines.append("本结果仅用于辅助审核，不构成自动授信、拒贷或合规决定。")
        return "\n\n".join(lines)

    def _openai(
        self,
        question: str,
        evidence: list[SearchResult],
        risk_assessment: dict[str, Any] | None,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=20, max_retries=1)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"问题：{question}\n证据：\n{self._context(evidence, risk_assessment)}",
                },
            ],
        )
        return response.choices[0].message.content or ""

    def _anthropic(
        self,
        question: str,
        evidence: list[SearchResult],
        risk_assessment: dict[str, Any] | None,
    ) -> str:
        from anthropic import Anthropic

        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=20, max_retries=1)
        response = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
            max_tokens=800,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"问题：{question}\n证据：\n{self._context(evidence, risk_assessment)}",
                }
            ],
        )
        return "".join(block.text for block in response.content if hasattr(block, "text"))
