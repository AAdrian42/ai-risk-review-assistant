from __future__ import annotations

import math
from pathlib import Path
from typing import Any


GRADE_RISK = {grade: index for index, grade in enumerate("ABCDEFG")}


def _bounded(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


class RiskEngine:
    """Use the calibrated project model when available, otherwise a transparent fallback."""

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2])
        self.service = None
        self.mode = "transparent_heuristic"
        try:
            from .model_service import CreditRiskService

            bundled_model = Path(__file__).resolve().parents[1] / "models" / "phase6" / "phase6_calibrated_model_bundle.joblib"
            model_path = bundled_model if bundled_model.exists() else self.project_root / "models" / "phase6" / "phase6_calibrated_model_bundle.joblib"
            self.service = CreditRiskService(model_path=model_path)
            self.mode = "phase6_platt_xgboost"
        except Exception:
            self.service = None

    def assess(self, application: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize(application)
        if self.service is not None:
            try:
                prediction = self.service.predict([normalized])[0]
                probability = float(prediction["calibrated_default_probability"])
                band = str(prediction["risk_band"])
                threshold = float(prediction["threshold"])
                mode = self.mode
            except Exception:
                probability, band, threshold = self._heuristic(normalized)
                mode = "transparent_heuristic_fallback"
        else:
            probability, band, threshold = self._heuristic(normalized)
            mode = self.mode

        signals, protections = self._signals(normalized)
        if (
            band in {"high", "very_high"}
            and len(signals) == 1
            and signals[0].startswith("未发现明显单项")
        ):
            signals = ["模型综合评分已超过审核阈值，但单项强风险信号不突出，需核验特征组合影响"]
        recommendation = self._recommendation(band, signals)
        return {
            "probability": round(probability, 4),
            "risk_band": band,
            "threshold": threshold,
            "decision_support": recommendation,
            "risk_signals": signals,
            "protective_signals": protections,
            "engine_mode": mode,
            "disclaimer": "仅供辅助审核，不构成自动授信、拒贷或合规决定。",
        }

    @staticmethod
    def _normalize(application: dict[str, Any]) -> dict[str, Any]:
        grade = str(application.get("grade", "C")).strip().upper() or "C"
        subgrade = str(application.get("subGrade", f"{grade}3")).strip().upper()
        normalized = {
            "id": application.get("id"),
            "loanAmnt": float(application.get("loanAmnt", 15000)),
            "term": int(application.get("term", 3)),
            "interestRate": float(application.get("interestRate", 12.0)),
            "installment": float(application.get("installment", 500)),
            "grade": grade,
            "subGrade": subgrade,
            "annualIncome": float(application.get("annualIncome", 80000)),
            "dti": float(application.get("dti", 18.0)),
            "ficoRangeLow": float(application.get("ficoRangeLow", 680)),
            "ficoRangeHigh": float(application.get("ficoRangeHigh", 684)),
            "delinquency_2years": float(application.get("delinquency_2years", 0)),
            "pubRec": float(application.get("pubRec", 0)),
            "pubRecBankruptcies": float(application.get("pubRecBankruptcies", 0)),
            "issueDate": str(application.get("issueDate", "2018-01-01")),
            "earliesCreditLine": str(application.get("earliesCreditLine", "Jan-2010")),
            "employmentLength": str(application.get("employmentLength", "5 years")),
            "employmentTitle": application.get("employmentTitle", "demo"),
            "homeOwnership": application.get("homeOwnership", "RENT"),
            "verificationStatus": application.get("verificationStatus", "Verified"),
            "purpose": application.get("purpose", "debt_consolidation"),
            "postCode": application.get("postCode", "000"),
            "regionCode": application.get("regionCode", "0"),
            "openAcc": float(application.get("openAcc", 8)),
            "revolBal": float(application.get("revolBal", 10000)),
            "revolUtil": float(application.get("revolUtil", 45)),
            "totalAcc": float(application.get("totalAcc", 18)),
        }
        return normalized

    @staticmethod
    def _heuristic(application: dict[str, Any]) -> tuple[float, str, float]:
        grade = GRADE_RISK.get(application["grade"], 2)
        fico = (application["ficoRangeLow"] + application["ficoRangeHigh"]) / 2
        income = max(application["annualIncome"], 1.0)
        loan_income = application["loanAmnt"] / income
        logit = (
            -2.55
            + 0.38 * grade
            + 0.028 * max(application["dti"] - 15, 0)
            + 0.55 * max(loan_income - 0.2, 0)
            + 0.025 * max(application["interestRate"] - 10, 0)
            + 0.42 * (application["term"] >= 5)
            + 0.35 * (application["delinquency_2years"] > 0)
            + 0.25 * (application["pubRec"] > 0)
            - 0.009 * max(fico - 650, 0)
        )
        probability = 1 / (1 + math.exp(-_bounded(logit, -8, 8)))
        threshold = 0.22
        if probability < 0.10:
            band = "low"
        elif probability < threshold:
            band = "medium"
        elif probability < 0.40:
            band = "high"
        else:
            band = "very_high"
        return probability, band, threshold

    @staticmethod
    def _signals(application: dict[str, Any]) -> tuple[list[str], list[str]]:
        signals: list[str] = []
        protections: list[str] = []
        fico = (application["ficoRangeLow"] + application["ficoRangeHigh"]) / 2
        loan_income = application["loanAmnt"] / max(application["annualIncome"], 1)

        if GRADE_RISK.get(application["grade"], 2) >= 3:
            signals.append(f"信用等级为 {application['grade']}，历史样本中较低评级违约率更高")
        elif GRADE_RISK.get(application["grade"], 2) <= 1:
            protections.append(f"信用等级为 {application['grade']}，属于相对较低风险等级")
        if application["dti"] >= 25:
            signals.append(f"DTI 为 {application['dti']:.1f}，偿债压力偏高")
        elif application["dti"] <= 12:
            protections.append(f"DTI 为 {application['dti']:.1f}，偿债压力相对较低")
        if fico < 660:
            signals.append(f"FICO 均值约 {fico:.0f}，信用评分偏低")
        elif fico >= 720:
            protections.append(f"FICO 均值约 {fico:.0f}，信用评分较好")
        if loan_income >= 0.35:
            signals.append(f"贷款收入比为 {loan_income:.2f}，贷款负担偏高")
        if application["term"] >= 5:
            signals.append("贷款期限为 5 年，长期贷款在历史样本中风险更高")
        if application["delinquency_2years"] > 0:
            signals.append("近两年存在逾期记录")
        if application["pubRec"] > 0 or application["pubRecBankruptcies"] > 0:
            signals.append("存在公共记录或破产记录，需人工核验")
        if not signals:
            signals.append("未发现明显单项高风险信号，仍需结合完整材料审核")
        return signals, protections

    @staticmethod
    def _recommendation(band: str, signals: list[str]) -> str:
        if band in {"high", "very_high"}:
            return "建议进入人工复核队列，重点核验收入真实性、负债情况和历史逾期材料。"
        if band == "medium":
            return "建议补充核验关键材料后再决定，避免仅依据单一模型分数。"
        return "可进入常规审核流程，同时保留基础反欺诈与身份核验。"
