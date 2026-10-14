from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("phase6_api.service")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "phase6" / "phase6_calibrated_model_bundle.joblib"


def parse_employment_length(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text == "< 1 year":
        return 0.0
    match = re.search(r"\d+", text)
    return float(match.group(0)) if match else np.nan


class CreditRiskService:
    def __init__(self, model_path: str | Path | None = None) -> None:
        configured = model_path or os.getenv("CREDIT_MODEL_PATH") or DEFAULT_MODEL_PATH
        self.model_path = Path(configured)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model bundle not found: {self.model_path}")
        self.bundle = joblib.load(self.model_path)
        self.model = self.bundle["model"]
        self.calibrator = self.bundle["calibrator"]
        self.preprocessor = self.bundle["preprocessor"]
        self.threshold = float(self.bundle["threshold"])
        self.feature_names = list(self.bundle["feature_names"])
        self.model_version = "phase6-20260619"
        LOGGER.info("Loaded model bundle from %s", self.model_path)

    @staticmethod
    def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        for column in columns:
            if column not in frame.columns:
                frame[column] = np.nan
        return frame

    def _add_derived_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        required = [
            "issueDate",
            "earliesCreditLine",
            "employmentLength",
            "grade",
            "subGrade",
            "annualIncome",
            "loanAmnt",
            "installment",
            "revolBal",
            "openAcc",
            "totalAcc",
            "pubRec",
            "pubRecBankruptcies",
            "delinquency_2years",
            "ficoRangeLow",
            "ficoRangeHigh",
        ]
        output = self._ensure_columns(output, required)
        output["issueDate"] = pd.to_datetime(output["issueDate"], errors="coerce")
        output["earliesCreditLineDate"] = pd.to_datetime(
            output["earliesCreditLine"], format="%b-%Y", errors="coerce"
        )
        output["issueYear"] = output["issueDate"].dt.year
        output["issueMonth"] = output["issueDate"].dt.month
        output["employmentLengthYears"] = output["employmentLength"].map(parse_employment_length)
        output["ficoMean"] = output[["ficoRangeLow", "ficoRangeHigh"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        output["creditHistoryYears"] = (
            output["issueDate"] - output["earliesCreditLineDate"]
        ).dt.days / 365.25
        output.loc[output["creditHistoryYears"] < 0, "creditHistoryYears"] = np.nan
        grade_map = {grade: index + 1 for index, grade in enumerate("ABCDEFG")}
        output["gradeOrdinal"] = output["grade"].map(grade_map)

        def subgrade_number(value: Any) -> float:
            if pd.isna(value) or not re.match(r"^[A-G][1-5]$", str(value)):
                return np.nan
            return (grade_map[str(value)[0]] - 1) * 5 + int(str(value)[1])

        output["subGradeOrdinal"] = output["subGrade"].map(subgrade_number)
        output["monthlyIncome"] = pd.to_numeric(output["annualIncome"], errors="coerce") / 12
        output["loanIncomeRatio"] = pd.to_numeric(output["loanAmnt"], errors="coerce") / pd.to_numeric(output["annualIncome"], errors="coerce").replace(0, np.nan)
        output["installmentIncomeRatio"] = pd.to_numeric(output["installment"], errors="coerce") / output["monthlyIncome"].replace(0, np.nan)
        output["revolBalIncomeRatio"] = pd.to_numeric(output["revolBal"], errors="coerce") / pd.to_numeric(output["annualIncome"], errors="coerce").replace(0, np.nan)
        output["openAccTotalAccRatio"] = pd.to_numeric(output["openAcc"], errors="coerce") / pd.to_numeric(output["totalAcc"], errors="coerce").replace(0, np.nan)
        output["pubRecFlag"] = (pd.to_numeric(output["pubRec"], errors="coerce").fillna(0) > 0).astype(np.int8)
        output["bankruptcyFlag"] = (pd.to_numeric(output["pubRecBankruptcies"], errors="coerce").fillna(0) > 0).astype(np.int8)
        output["delinquencyFlag"] = (pd.to_numeric(output["delinquency_2years"], errors="coerce").fillna(0) > 0).astype(np.int8)
        return output

    def transform(self, records: list[dict[str, Any]]) -> np.ndarray:
        frame = self._add_derived_features(pd.DataFrame(records))
        metadata = self.preprocessor["metadata"]
        fill_values = metadata["fill_values"]
        mappings = metadata["category_mappings"]
        all_names = self.preprocessor["all_feature_names"]
        columns: list[np.ndarray] = []
        for name in all_names:
            if name in fill_values:
                if name not in frame:
                    series = pd.Series(np.nan, index=frame.index, dtype=float)
                else:
                    series = pd.to_numeric(frame[name], errors="coerce")
                values = series.replace([np.inf, -np.inf], np.nan).fillna(fill_values[name]).to_numpy(dtype=np.float32)
            elif name.endswith("_missing"):
                source = name.removesuffix("_missing")
                if source not in frame:
                    values = np.ones(len(frame), dtype=np.float32)
                else:
                    values = pd.to_numeric(frame[source], errors="coerce").isna().to_numpy(dtype=np.float32)
            elif name.endswith("_target"):
                source = name.removesuffix("_target")
                mapping_info = mappings[source]
                if source not in frame:
                    series = pd.Series("Unknown", index=frame.index, dtype="string")
                else:
                    series = frame[source].astype("string").fillna("Unknown")
                values = series.astype(str).map(mapping_info["mapping"]).fillna(mapping_info["fallback"]).to_numpy(dtype=np.float32)
            else:
                raise ValueError(f"Unsupported pipeline feature: {name}")
            columns.append(values)
        all_matrix = np.column_stack(columns).astype(np.float32)
        selected = all_matrix[:, self.preprocessor["selected_indices"]]
        if self.bundle.get("uses_interactions"):
            selected = np.hstack([selected, self._interactions(frame)])
        if selected.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Feature count mismatch: transformed={selected.shape[1]}, expected={len(self.feature_names)}"
            )
        return selected

    def _interactions(self, frame: pd.DataFrame) -> np.ndarray:
        spec = self.bundle["interaction_spec"]
        grade_map = {grade: index + 1 for index, grade in enumerate("ABCDEFG")}
        annual_income = pd.to_numeric(frame["annualIncome"], errors="coerce")
        loan_amount = pd.to_numeric(frame["loanAmnt"], errors="coerce").replace(0, np.nan)
        term_months = pd.to_numeric(frame["term"], errors="coerce") * 12
        grade_ordinal = frame["grade"].map(grade_map).astype(float).replace(0, np.nan)
        interactions = pd.DataFrame(
            {
                "incomeLoanCoverage": annual_income / loan_amount,
                "termGradeRatio": term_months / grade_ordinal,
                "termGradeInteraction": term_months * grade_ordinal,
            }
        ).replace([np.inf, -np.inf], np.nan)
        for name, limits in spec["limits"].items():
            interactions[name] = interactions[name].clip(limits["lower"], limits["upper"]).fillna(limits["fill"])
        return interactions[spec["names"]].to_numpy(dtype=np.float32)

    def _calibrate(self, matrix: np.ndarray, raw_probability: np.ndarray) -> np.ndarray:
        if self.bundle["calibration_method"] == "Platt":
            margin = self.model.predict(matrix, output_margin=True)
            return self.calibrator.predict_proba(margin.reshape(-1, 1))[:, 1]
        return np.asarray(self.calibrator.predict(raw_probability), dtype=float)

    def predict(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        matrix = self.transform(records)
        raw_probability = self.model.predict_proba(matrix)[:, 1]
        calibrated = self._calibrate(matrix, raw_probability)
        results = []
        for record, raw_value, calibrated_value in zip(records, raw_probability, calibrated):
            predicted_default = int(calibrated_value >= self.threshold)
            if calibrated_value < 0.10:
                risk_band = "low"
            elif calibrated_value < self.threshold:
                risk_band = "medium"
            elif calibrated_value < 0.40:
                risk_band = "high"
            else:
                risk_band = "very_high"
            results.append(
                {
                    "application_id": record.get("id"),
                    "calibrated_default_probability": round(float(calibrated_value), 8),
                    "raw_default_probability": round(float(raw_value), 8),
                    "threshold": self.threshold,
                    "predicted_default": predicted_default,
                    "risk_band": risk_band,
                    "model_version": self.model_version,
                }
            )
        return results
