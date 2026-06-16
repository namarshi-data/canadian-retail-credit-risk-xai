from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CounterfactualContext:
    """Reference statistics needed to recompute engineered features consistently."""

    amount_q75: float
    income_q25: float


def build_counterfactual_context(reference_df: pd.DataFrame) -> CounterfactualContext:
    """Build reference quantiles used by engineered binary flags."""
    return CounterfactualContext(
        amount_q75=float(pd.to_numeric(reference_df["amount"], errors="coerce").quantile(0.75)),
        income_q25=float(pd.to_numeric(reference_df["total_income_pa"], errors="coerce").quantile(0.25)),
    )


def _band(value: float, bins: list[float], labels: list[str], missing_label: str = "Missing") -> str:
    if pd.isna(value):
        return missing_label
    for upper, label in zip(bins[1:], labels):
        if value <= upper:
            return label
    return labels[-1]


def recompute_engineered_features(row_df: pd.DataFrame, context: CounterfactualContext) -> pd.DataFrame:
    """Recompute engineered variables after a scenario changes core risk drivers."""
    output = row_df.copy()
    amount = pd.to_numeric(output.get("amount"), errors="coerce")
    income = pd.to_numeric(output.get("total_income_pa"), errors="coerce")
    interest_rate = pd.to_numeric(output.get("interest_rate"), errors="coerce")

    if "amount" in output.columns:
        output["amount_log1p"] = np.log1p(amount.clip(lower=0))
        output["high_amount_flag"] = (amount >= context.amount_q75).astype("int64")
        output["amount_band"] = amount.apply(lambda x: _band(x, [-np.inf, 25000, 50000, 100000, 250000, np.inf], ["<=25K", "25K-50K", "50K-100K", "100K-250K", ">250K"]))

    if "total_income_pa" in output.columns:
        output["total_income_pa_log1p"] = np.log1p(income.clip(lower=0))
        output["low_income_flag"] = (income <= context.income_q25).astype("int64")
        output["income_band"] = income.apply(lambda x: _band(x, [-np.inf, 40000, 60000, 90000, 120000, np.inf], ["<=40K", "40K-60K", "60K-90K", "90K-120K", ">120K"]))

    if {"amount", "total_income_pa"}.issubset(output.columns):
        output["loan_to_income_ratio"] = np.where(income > 0, amount / income, np.nan)

    if "loan_to_income_ratio" in output.columns:
        lti = pd.to_numeric(output["loan_to_income_ratio"], errors="coerce")
        output["loan_to_income_ratio_log1p"] = np.log1p(lti.clip(lower=0))
        output["high_loan_to_income_flag"] = (lti >= 2.0).astype("int64")
        output["very_high_loan_to_income_flag"] = (lti >= 4.0).astype("int64")
        output["loan_to_income_band"] = lti.apply(lambda x: _band(x, [-np.inf, 0.5, 1.0, 2.0, 4.0, np.inf], ["<=0.5", "0.5-1.0", "1.0-2.0", "2.0-4.0", ">4.0"]))

    if {"interest_rate", "loan_to_income_ratio"}.issubset(output.columns):
        output["interest_rate_x_lti"] = interest_rate * pd.to_numeric(output["loan_to_income_ratio"], errors="coerce")

    if "interest_rate" in output.columns:
        output["high_interest_flag"] = (interest_rate >= 16).astype("int64")
        output["interest_rate_band"] = interest_rate.apply(lambda x: _band(x, [-np.inf, 8, 12, 16, np.inf], ["<=8%", "8%-12%", "12%-16%", ">16%"] ))

    if "delinq_2yrs" in output.columns:
        output["has_prior_delinquency_flag"] = (pd.to_numeric(output["delinq_2yrs"], errors="coerce") > 0).astype("int64")

    if "number_of_loans" in output.columns:
        nloans = pd.to_numeric(output["number_of_loans"], errors="coerce")
        output["has_existing_loans_flag"] = (nloans > 0).astype("int64")
        output["multiple_loans_flag"] = (nloans > 1).astype("int64")

    if "dependents" in output.columns:
        dep = pd.to_numeric(output["dependents"], errors="coerce")
        output["dependents_band"] = dep.apply(lambda x: _band(x, [-np.inf, 0, 2, np.inf], ["0", "1-2", "3+"]))

    if "tenure_years" in output.columns:
        tenure = pd.to_numeric(output["tenure_years"], errors="coerce")
        output["tenure_band"] = tenure.apply(lambda x: _band(x, [-np.inf, 3, 5, np.inf], ["<=3 years", "4-5 years", "6+ years"]))

    return output


def _scenario_rows(base_row: pd.Series) -> list[tuple[str, str, pd.Series]]:
    """Create practical and diagnostic counterfactual candidate rows."""
    scenarios: list[tuple[str, str, pd.Series]] = []

    def add(name: str, action_type: str, changes: dict[str, float | int]) -> None:
        row = base_row.copy()
        for col, value in changes.items():
            if col in row.index:
                row[col] = value
        scenarios.append((name, action_type, row))

    amount = float(base_row.get("amount", np.nan)) if pd.notna(base_row.get("amount", np.nan)) else np.nan
    income = float(base_row.get("total_income_pa", np.nan)) if pd.notna(base_row.get("total_income_pa", np.nan)) else np.nan
    rate = float(base_row.get("interest_rate", np.nan)) if pd.notna(base_row.get("interest_rate", np.nan)) else np.nan
    delinq = float(base_row.get("delinq_2yrs", 0) or 0)
    nloans = float(base_row.get("number_of_loans", 0) or 0)

    if np.isfinite(rate):
        for target_rate in [16.0, 12.0, 8.0]:
            if rate > target_rate:
                add(f"Reduce interest rate to {target_rate:.0f}%", "pricing_scenario", {"interest_rate": target_rate})

    if np.isfinite(amount) and amount > 0:
        for pct in [0.10, 0.20, 0.30]:
            add(f"Reduce requested/exposure amount by {pct:.0%}", "affordability_scenario", {"amount": amount * (1 - pct)})

    if np.isfinite(income) and income > 0:
        for pct in [0.10, 0.20, 0.30]:
            add(f"Increase verified annual income by {pct:.0%}", "affordability_scenario", {"total_income_pa": income * (1 + pct)})

    if np.isfinite(amount) and amount > 0 and np.isfinite(income) and income > 0:
        add("Reduce exposure 20% and increase verified income 20%", "combined_affordability_scenario", {"amount": amount * 0.80, "total_income_pa": income * 1.20})

    if delinq > 0:
        add("No prior delinquency history", "diagnostic_non_actionable", {"delinq_2yrs": 0})

    if nloans > 1:
        add("One existing loan instead of multiple loans", "diagnostic_portfolio_scenario", {"number_of_loans": 1})
    elif nloans > 0:
        add("No existing loans", "diagnostic_portfolio_scenario", {"number_of_loans": 0})

    return scenarios


def generate_counterfactual_scenarios(
    pipeline,
    X_candidates: pd.DataFrame,
    baseline_scores: pd.Series,
    threshold: float,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate and score counterfactual scenarios for selected high-risk accounts."""
    context = build_counterfactual_context(reference_df)
    rows = []
    for idx, base_row in X_candidates.iterrows():
        base_score = float(baseline_scores.loc[idx])
        scenario_inputs = _scenario_rows(base_row)
        if not scenario_inputs:
            continue
        scenario_df = pd.DataFrame([row for _, _, row in scenario_inputs], index=range(len(scenario_inputs)))
        scenario_df = recompute_engineered_features(scenario_df, context)
        scenario_scores = np.asarray(pipeline.predict_proba(scenario_df[X_candidates.columns]))[:, 1]

        for (scenario_name, action_type, _), scenario_score in zip(scenario_inputs, scenario_scores):
            rows.append(
                {
                    "row_index": idx,
                    "scenario": scenario_name,
                    "action_type": action_type,
                    "baseline_probability": base_score,
                    "scenario_probability": float(scenario_score),
                    "probability_change": float(scenario_score - base_score),
                    "absolute_probability_reduction": float(base_score - scenario_score),
                    "operating_threshold": float(threshold),
                    "crosses_below_threshold": bool(scenario_score < threshold),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["row_index", "crosses_below_threshold", "absolute_probability_reduction"], ascending=[True, False, False]).reset_index(drop=True)


def best_counterfactual_per_account(counterfactuals: pd.DataFrame) -> pd.DataFrame:
    """Return the strongest scenario per account, preferring threshold-crossing cases."""
    if counterfactuals.empty:
        return counterfactuals
    ordered = counterfactuals.sort_values(["row_index", "crosses_below_threshold", "absolute_probability_reduction"], ascending=[True, False, False])
    return ordered.groupby("row_index", as_index=False).head(1).reset_index(drop=True)
