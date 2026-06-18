from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CounterfactualContext:
    """Reference statistics needed to recompute engineered features consistently."""

    amount_q75: float = np.nan
    income_q25: float = np.nan


def _safe_quantile(df: pd.DataFrame, column: str, q: float) -> float:
    if column not in df.columns:
        return np.nan
    return float(pd.to_numeric(df[column], errors="coerce").quantile(q))


def build_counterfactual_context(reference_df: pd.DataFrame) -> CounterfactualContext:
    return CounterfactualContext(
        amount_q75=_safe_quantile(reference_df, "amount", 0.75),
        income_q25=_safe_quantile(reference_df, "total_income_pa", 0.25),
    )


def _band(value: float, bins: list[float], labels: list[str], missing_label: str = "Missing") -> str:
    if pd.isna(value):
        return missing_label
    for upper, label in zip(bins[1:], labels):
        if value <= upper:
            return label
    return labels[-1]


def recompute_engineered_features(row_df: pd.DataFrame, context: CounterfactualContext) -> pd.DataFrame:
    """Recompute common engineered variables after scenario changes."""
    output = row_df.copy()
    amount = pd.to_numeric(output["amount"], errors="coerce") if "amount" in output.columns else pd.Series(np.nan, index=output.index)
    income = pd.to_numeric(output["total_income_pa"], errors="coerce") if "total_income_pa" in output.columns else pd.Series(np.nan, index=output.index)
    interest_rate = pd.to_numeric(output["interest_rate"], errors="coerce") if "interest_rate" in output.columns else pd.Series(np.nan, index=output.index)

    if "amount" in output.columns:
        if "amount_log1p" in output.columns:
            output["amount_log1p"] = np.log1p(amount.clip(lower=0))
        if "high_amount_flag" in output.columns and np.isfinite(context.amount_q75):
            output["high_amount_flag"] = (amount >= context.amount_q75).astype("int64")
        if "amount_band" in output.columns:
            output["amount_band"] = amount.apply(lambda x: _band(x, [-np.inf, 25000, 50000, 100000, 250000, np.inf], ["<=25K", "25K-50K", "50K-100K", "100K-250K", ">250K"]))

    if "total_income_pa" in output.columns:
        if "total_income_pa_log1p" in output.columns:
            output["total_income_pa_log1p"] = np.log1p(income.clip(lower=0))
        if "low_income_flag" in output.columns and np.isfinite(context.income_q25):
            output["low_income_flag"] = (income <= context.income_q25).astype("int64")
        if "income_band" in output.columns:
            output["income_band"] = income.apply(lambda x: _band(x, [-np.inf, 40000, 60000, 90000, 120000, np.inf], ["<=40K", "40K-60K", "60K-90K", "90K-120K", ">120K"]))

    if {"amount", "total_income_pa", "loan_to_income_ratio"}.issubset(output.columns):
        output["loan_to_income_ratio"] = np.where(income > 0, amount / income, np.nan)

    if "loan_to_income_ratio" in output.columns:
        lti = pd.to_numeric(output["loan_to_income_ratio"], errors="coerce")
        if "loan_to_income_ratio_log1p" in output.columns:
            output["loan_to_income_ratio_log1p"] = np.log1p(lti.clip(lower=0))
        if "high_loan_to_income_flag" in output.columns:
            output["high_loan_to_income_flag"] = (lti >= 2.0).astype("int64")
        if "very_high_loan_to_income_flag" in output.columns:
            output["very_high_loan_to_income_flag"] = (lti >= 4.0).astype("int64")
        if "loan_to_income_band" in output.columns:
            output["loan_to_income_band"] = lti.apply(lambda x: _band(x, [-np.inf, 0.5, 1.0, 2.0, 4.0, np.inf], ["<=0.5", "0.5-1.0", "1.0-2.0", "2.0-4.0", ">4.0"]))

    if {"interest_rate", "loan_to_income_ratio", "interest_rate_x_lti"}.issubset(output.columns):
        output["interest_rate_x_lti"] = interest_rate * pd.to_numeric(output["loan_to_income_ratio"], errors="coerce")

    if "interest_rate" in output.columns:
        if "high_interest_flag" in output.columns:
            output["high_interest_flag"] = (interest_rate >= 16).astype("int64")
        if "interest_rate_band" in output.columns:
            output["interest_rate_band"] = interest_rate.apply(lambda x: _band(x, [-np.inf, 8, 12, 16, np.inf], ["<=8%", "8%-12%", "12%-16%", ">16%"]))

    if "delinq_2yrs" in output.columns and "has_prior_delinquency_flag" in output.columns:
        output["has_prior_delinquency_flag"] = (pd.to_numeric(output["delinq_2yrs"], errors="coerce") > 0).astype("int64")

    if "number_of_loans" in output.columns:
        nloans = pd.to_numeric(output["number_of_loans"], errors="coerce")
        if "has_existing_loans_flag" in output.columns:
            output["has_existing_loans_flag"] = (nloans > 0).astype("int64")
        if "multiple_loans_flag" in output.columns:
            output["multiple_loans_flag"] = (nloans > 1).astype("int64")

    if "dependents" in output.columns and "dependents_band" in output.columns:
        dep = pd.to_numeric(output["dependents"], errors="coerce")
        output["dependents_band"] = dep.apply(lambda x: _band(x, [-np.inf, 0, 2, np.inf], ["0", "1-2", "3+"]))

    if "tenure_years" in output.columns and "tenure_band" in output.columns:
        tenure = pd.to_numeric(output["tenure_years"], errors="coerce")
        output["tenure_band"] = tenure.apply(lambda x: _band(x, [-np.inf, 3, 5, np.inf], ["<=3 years", "4-5 years", "6+ years"]))

    return output


def _scenario_rows(base_row: pd.Series) -> list[tuple[str, str, dict[str, Any], pd.Series]]:
    """Create practical and diagnostic counterfactual scenario rows."""
    scenarios: list[tuple[str, str, dict[str, Any], pd.Series]] = []

    def add(name: str, action_type: str, changes: dict[str, float | int]) -> None:
        row = base_row.copy()
        applied: dict[str, Any] = {}
        for col, value in changes.items():
            if col in row.index:
                row[col] = value
                applied[col] = value
        if applied:
            scenarios.append((name, action_type, applied, row))

    amount = pd.to_numeric(pd.Series([base_row.get("amount", np.nan)]), errors="coerce").iloc[0]
    income = pd.to_numeric(pd.Series([base_row.get("total_income_pa", np.nan)]), errors="coerce").iloc[0]
    rate = pd.to_numeric(pd.Series([base_row.get("interest_rate", np.nan)]), errors="coerce").iloc[0]
    delinq = pd.to_numeric(pd.Series([base_row.get("delinq_2yrs", 0)]), errors="coerce").fillna(0).iloc[0]
    nloans = pd.to_numeric(pd.Series([base_row.get("number_of_loans", 0)]), errors="coerce").fillna(0).iloc[0]

    if np.isfinite(rate):
        for target_rate in [16.0, 12.0, 8.0]:
            if rate > target_rate:
                add(f"Reduce pricing rate to {target_rate:.0f}%", "pricing_scenario", {"interest_rate": target_rate})

    if np.isfinite(amount) and amount > 0:
        for pct in [0.10, 0.20, 0.30]:
            add(f"Reduce exposure/request amount by {pct:.0%}", "affordability_scenario", {"amount": amount * (1 - pct)})

    if np.isfinite(income) and income > 0:
        for pct in [0.10, 0.20, 0.30]:
            add(f"Increase verified annual income by {pct:.0%}", "affordability_scenario", {"total_income_pa": income * (1 + pct)})

    if np.isfinite(amount) and amount > 0 and np.isfinite(income) and income > 0:
        add("Reduce exposure 20% and increase verified income 20%", "combined_affordability_scenario", {"amount": amount * 0.80, "total_income_pa": income * 1.20})

    if np.isfinite(delinq) and delinq > 0:
        add("No prior delinquency history", "diagnostic_non_actionable", {"delinq_2yrs": 0})

    if np.isfinite(nloans) and nloans > 1:
        add("One existing loan instead of multiple loans", "diagnostic_portfolio_scenario", {"number_of_loans": 1})
    elif np.isfinite(nloans) and nloans > 0:
        add("No existing loans", "diagnostic_portfolio_scenario", {"number_of_loans": 0})

    return scenarios


def generate_counterfactual_scenarios(
    pipeline: object,
    X_candidates: pd.DataFrame,
    baseline_scores: pd.Series,
    threshold: float,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate and score counterfactual scenario diagnostics.

    These scenarios are diagnostic explanations for analysts and regulators. They are not
    approval guarantees, adverse-action notices, or customer instructions.
    """
    context = build_counterfactual_context(reference_df)
    rows: list[dict[str, Any]] = []
    model_features = list(X_candidates.columns)
    for idx, base_row in X_candidates.iterrows():
        if idx not in baseline_scores.index:
            continue
        base_score = float(baseline_scores.loc[idx])
        scenario_inputs = _scenario_rows(base_row)
        if not scenario_inputs:
            continue
        scenario_df = pd.DataFrame([row for _, _, _, row in scenario_inputs])
        scenario_df = recompute_engineered_features(scenario_df, context)
        scenario_scores = np.asarray(pipeline.predict_proba(scenario_df[model_features]))[:, 1]
        for (scenario_name, action_type, applied_changes, _), scenario_score in zip(scenario_inputs, scenario_scores):
            rows.append(
                {
                    "row_index": idx,
                    "scenario": scenario_name,
                    "action_type": action_type,
                    "changed_features": "; ".join(applied_changes.keys()),
                    "baseline_probability": base_score,
                    "scenario_probability": float(scenario_score),
                    "probability_change": float(scenario_score - base_score),
                    "absolute_probability_reduction": float(base_score - scenario_score),
                    "operating_threshold": float(threshold),
                    "baseline_above_threshold": bool(base_score >= threshold),
                    "scenario_below_threshold": bool(scenario_score < threshold),
                    "crosses_below_threshold": bool(base_score >= threshold and scenario_score < threshold),
                    "governance_note": "Diagnostic model-sensitivity scenario; not a customer-level promise or instruction.",
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["row_index", "crosses_below_threshold", "absolute_probability_reduction"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def best_counterfactual_per_account(counterfactuals: pd.DataFrame) -> pd.DataFrame:
    if counterfactuals.empty:
        return counterfactuals
    ordered = counterfactuals.sort_values(
        ["row_index", "crosses_below_threshold", "absolute_probability_reduction"],
        ascending=[True, False, False],
    )
    return ordered.groupby("row_index", as_index=False).head(1).reset_index(drop=True)
