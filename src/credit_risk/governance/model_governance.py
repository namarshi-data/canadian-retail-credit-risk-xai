from __future__ import annotations

"""Professional model-governance, monitoring, and reporting utilities for Notebook 09.

Notebook 09 consolidates outputs from Notebooks 01-08 into a model card,
validation summary, control register, monitoring plan, stakeholder brief, and
production-readiness checks. The model remains a decision-support/manual-review
prioritization model, not an automated decline engine.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

TARGET_COLUMN = "defaulter"


@dataclass
class GovernanceInputs:
    """Container for governance-stage inputs created by previous notebooks."""

    modeling_df: pd.DataFrame
    feature_catalog: pd.DataFrame
    feature_policy: pd.DataFrame
    split_summary: pd.DataFrame
    validation_results: pd.DataFrame
    test_results: pd.DataFrame
    all_model_operational_comparison: pd.DataFrame
    operational_recommendation: pd.DataFrame
    test_confirmation: pd.DataFrame
    policy_options: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    cost_assumptions: pd.DataFrame
    threshold_readiness_gate: pd.DataFrame
    xai_grouped_importance: pd.DataFrame
    xai_global_importance: pd.DataFrame
    xai_regulator_summary: pd.DataFrame
    xai_anchor_rules: pd.DataFrame
    xai_best_counterfactuals: pd.DataFrame
    xai_counterfactuals: pd.DataFrame
    xai_probability_deciles: pd.DataFrame
    xai_metrics: pd.DataFrame
    xai_stakeholder_summary: pd.DataFrame
    xai_weakness_diagnostics: pd.DataFrame
    xai_recommendations: pd.DataFrame
    xai_readiness_gate: pd.DataFrame
    xai_manifest: pd.DataFrame


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    if required:
        raise FileNotFoundError(
            f"Required governance input not found: {path}. Run prior notebooks before Notebook 09."
        )
    return pd.DataFrame()


def read_first_existing_csv(paths: Sequence[Path], required: bool = True) -> pd.DataFrame:
    """Read the first existing CSV from a list of compatible filenames."""
    for path in paths:
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    if required:
        expected = "\n".join(f"- {p}" for p in paths)
        raise FileNotFoundError(f"None of these expected governance inputs exist:\n{expected}")
    return pd.DataFrame()


def load_governance_inputs(processed_dir: Path, table_dir: Path) -> GovernanceInputs:
    """Load outputs from Notebooks 05-08 using current and backward-compatible names."""
    return GovernanceInputs(
        modeling_df=read_first_existing_csv([processed_dir / "credit_risk_modeling_dataset.csv"]),
        feature_catalog=read_first_existing_csv(
            [
                table_dir / "05_modeling_feature_catalog.csv",
                table_dir / "modeling_feature_catalog.csv",
                table_dir / "model_feature_policy.csv",
                table_dir / "model_feature_catalog.csv",
            ],
            required=False,
        ),
        feature_policy=read_first_existing_csv(
            [
                table_dir / "05_feature_leakage_and_usage_policy.csv",
                table_dir / "feature_leakage_and_usage_policy.csv",
                table_dir / "model_feature_policy.csv",
            ],
            required=False,
        ),
        split_summary=read_first_existing_csv(
            [
                table_dir / "05_modeling_split_distribution.csv",
                table_dir / "modeling_split_distribution.csv",
                table_dir / "06_modeling_split_distribution.csv",
            ],
            required=False,
        ),
        validation_results=read_first_existing_csv(
            [table_dir / "06_model_validation_results_default_threshold.csv", table_dir / "model_validation_results_default_threshold.csv"],
            required=False,
        ),
        test_results=read_first_existing_csv(
            [table_dir / "06_model_test_results_default_threshold.csv", table_dir / "model_test_results_default_threshold.csv"],
            required=False,
        ),
        all_model_operational_comparison=read_first_existing_csv(
            [
                table_dir / "07_all_model_operational_threshold_comparison_validation.csv",
                table_dir / "06_all_model_operational_threshold_comparison_validation.csv",
            ],
            required=False,
        ),
        operational_recommendation=read_first_existing_csv(
            [
                table_dir / "07_operational_threshold_recommendation_validation.csv",
                table_dir / "06_operational_model_threshold_recommendation.csv",
                table_dir / "06_recommended_threshold_summary.csv",
                table_dir / "recommended_threshold_summary.csv",
            ],
            required=True,
        ),
        test_confirmation=read_first_existing_csv(
            [
                table_dir / "07_test_confirmation_selected_operational_model.csv",
                table_dir / "06_test_selected_threshold_results.csv",
                table_dir / "06_test_selected_threshold_results_all_models.csv",
                table_dir / "champion_threshold_shortlist_test.csv",
            ],
            required=False,
        ),
        policy_options=read_first_existing_csv([table_dir / "07_policy_option_table_validation.csv"], required=False),
        cost_sensitivity=read_first_existing_csv([table_dir / "07_business_cost_sensitivity_validation.csv"], required=False),
        cost_assumptions=read_first_existing_csv(
            [table_dir / "07_business_cost_assumptions.csv", table_dir / "business_cost_assumptions.csv"],
            required=False,
        ),
        threshold_readiness_gate=read_first_existing_csv([table_dir / "07_threshold_readiness_gate.csv"], required=False),
        xai_grouped_importance=read_first_existing_csv(
            [table_dir / "08_xai_grouped_shap_importance.csv", table_dir / "xai_grouped_shap_importance.csv"],
            required=False,
        ),
        xai_global_importance=read_first_existing_csv(
            [
                table_dir / "08_xai_global_shap_importance_transformed.csv",
                table_dir / "08_xai_global_shap_importance.csv",
                table_dir / "xai_global_shap_importance.csv",
            ],
            required=False,
        ),
        xai_regulator_summary=read_first_existing_csv([table_dir / "08_xai_business_regulator_summary_model_insights.csv"], required=False),
        xai_anchor_rules=read_first_existing_csv(
            [table_dir / "08_xai_anchor_style_rules.csv", table_dir / "xai_anchor_like_rules.csv"], required=False
        ),
        xai_best_counterfactuals=read_first_existing_csv(
            [table_dir / "08_xai_best_counterfactual_per_account.csv", table_dir / "xai_best_counterfactual_per_account.csv"],
            required=False,
        ),
        xai_counterfactuals=read_first_existing_csv([table_dir / "08_xai_counterfactual_scenarios.csv"], required=False),
        xai_probability_deciles=read_first_existing_csv(
            [table_dir / "08_xai_probability_decile_profile.csv", table_dir / "xai_probability_decile_profile.csv"],
            required=False,
        ),
        xai_metrics=read_first_existing_csv([table_dir / "08_xai_classification_metrics_at_operating_threshold.csv"], required=False),
        xai_stakeholder_summary=read_first_existing_csv([table_dir / "08_xai_stakeholder_metric_impact_summary.csv"], required=False),
        xai_weakness_diagnostics=read_first_existing_csv(
            [table_dir / "08_deepchecks_fallback_model_weakness_diagnostics.csv"], required=False
        ),
        xai_recommendations=read_first_existing_csv([table_dir / "08_model_enhancement_recommendations.csv"], required=False),
        xai_readiness_gate=read_first_existing_csv([table_dir / "08_xai_readiness_gate.csv"], required=False),
        xai_manifest=read_first_existing_csv([table_dir / "08_xai_output_manifest.csv"], required=False),
    )


def _first_row(df: pd.DataFrame) -> pd.Series:
    return df.iloc[0] if isinstance(df, pd.DataFrame) and not df.empty else pd.Series(dtype="object")


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _format_pct(value: Any, digits: int = 2) -> str:
    value = _safe_float(value)
    return "Not available" if value is None else f"{value * 100:.{digits}f}%"


def _format_number(value: Any, digits: int = 4) -> str:
    value = _safe_float(value)
    return "Not available" if value is None else f"{value:.{digits}f}"


def _format_currency(value: Any, digits: int = 0) -> str:
    value = _safe_float(value)
    return "Not available" if value is None else f"${value:,.{digits}f}"


def _model_name_from_row(row: pd.Series) -> str:
    return str(row.get("model_name", row.get("champion_model", "not_available")))


def operational_model_name(inputs: GovernanceInputs) -> str:
    return _model_name_from_row(_first_row(inputs.operational_recommendation))


def operating_threshold(inputs: GovernanceInputs) -> float | None:
    return _safe_float(_first_row(inputs.operational_recommendation).get("threshold"))


def threshold_objective(inputs: GovernanceInputs) -> str:
    return str(_first_row(inputs.operational_recommendation).get("objective", "not_available"))


def _best_test_confirmation(inputs: GovernanceInputs) -> pd.Series:
    if not inputs.test_confirmation.empty:
        return _first_row(inputs.test_confirmation)
    return pd.Series(dtype="object")


def _default_rate(inputs: GovernanceInputs) -> float | None:
    if TARGET_COLUMN in inputs.modeling_df.columns:
        return _safe_float(inputs.modeling_df[TARGET_COLUMN].mean())
    return None


def _top_feature_column(df: pd.DataFrame) -> str | None:
    for col in ["raw_feature", "feature", "feature_name", "feature_label", "transformed_feature"]:
        if col in df.columns:
            return col
    return df.columns[0] if not df.empty else None


def build_model_inventory(inputs: GovernanceInputs) -> pd.DataFrame:
    """Create model inventory and intended-use documentation."""
    rows = [
        ("business_use", "Early-warning retail credit default-risk ranking and manual-review prioritization"),
        ("model_scope", "Portfolio monitoring and decision-support; not an automated credit-decline engine"),
        ("target", "Defaulter indicator"),
        ("operational_model", operational_model_name(inputs)),
        ("operating_threshold", operating_threshold(inputs)),
        ("threshold_objective", threshold_objective(inputs)),
        ("modeling_rows", len(inputs.modeling_df)),
        ("portfolio_default_rate", _default_rate(inputs)),
        ("feature_count", len(inputs.feature_catalog) if not inputs.feature_catalog.empty else "not_available"),
        ("sensitive_proxy_use", "Excluded from baseline model; available only for permitted governance review"),
        ("leakage_control", "Repayment-derived variables excluded from modelling features"),
        ("explainability_assets", "SHAP, anchor-style rules, counterfactual diagnostics, Deepchecks/fallback diagnostics"),
    ]
    return pd.DataFrame([{"item": item, "value": value} for item, value in rows])


def build_validation_test_summary(inputs: GovernanceInputs) -> pd.DataFrame:
    """Create model performance summary at default and operating thresholds."""
    model = operational_model_name(inputs)
    frames: list[dict[str, Any]] = []

    if not inputs.validation_results.empty and "model_name" in inputs.validation_results.columns:
        match = inputs.validation_results.loc[inputs.validation_results["model_name"].eq(model)]
        if not match.empty:
            row = match.iloc[0].to_dict()
            row.update({"evaluation_view": "validation_default_0_50", "selected_operating_threshold": False})
            frames.append(row)

    if not inputs.test_results.empty and "model_name" in inputs.test_results.columns:
        match = inputs.test_results.loc[inputs.test_results["model_name"].eq(model)]
        if not match.empty:
            row = match.iloc[0].to_dict()
            row.update({"evaluation_view": "test_default_0_50", "selected_operating_threshold": False})
            frames.append(row)

    if not inputs.operational_recommendation.empty:
        row = _first_row(inputs.operational_recommendation).to_dict()
        row.update({"evaluation_view": "validation_selected_operating_threshold", "selected_operating_threshold": True})
        frames.append(row)

    if not inputs.test_confirmation.empty:
        row = _first_row(inputs.test_confirmation).to_dict()
        row.update({"evaluation_view": "test_selected_operating_threshold", "selected_operating_threshold": True})
        frames.append(row)

    summary = pd.DataFrame(frames)
    wanted = [
        "evaluation_view", "model_name", "dataset", "threshold", "roc_auc", "pr_auc", "brier_score",
        "recall", "precision", "f1", "balanced_accuracy", "mcc", "review_rate", "business_cost",
        "false_negative", "false_positive", "true_positive", "true_negative", "selected_operating_threshold",
    ]
    return summary[[c for c in wanted if c in summary.columns]] if not summary.empty else summary


def build_feature_governance_summary(inputs: GovernanceInputs) -> pd.DataFrame:
    """Summarize feature usage, leakage controls, and governance decisions."""
    if inputs.feature_policy.empty:
        if inputs.feature_catalog.empty:
            return pd.DataFrame()
        return pd.DataFrame([{"governance_group": "feature_catalog_available", "feature_count": len(inputs.feature_catalog)}])

    policy = inputs.feature_policy.copy()
    group_cols = [c for c in ["model_usage", "governance_decision", "reason", "feature_status"] if c in policy.columns]
    if not group_cols:
        return pd.DataFrame([{"governance_group": "policy_rows", "feature_count": len(policy)}])
    return (
        policy.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="feature_count")
        .sort_values("feature_count", ascending=False)
        .reset_index(drop=True)
    )


def build_xai_governance_summary(inputs: GovernanceInputs, top_n: int = 12) -> pd.DataFrame:
    """Convert top SHAP drivers into governance-readable interpretation notes."""
    xai = inputs.xai_grouped_importance.copy()
    if xai.empty:
        xai = inputs.xai_regulator_summary.copy()
    if xai.empty:
        return pd.DataFrame()

    feature_col = _top_feature_column(xai)
    xai = xai.head(top_n).copy()

    def note(feature: Any) -> str:
        f = str(feature).lower()
        if "missing" in f or "data_quality" in f or "quality" in f:
            return "Data-quality signal: monitor for drift and avoid treating this as borrower behaviour alone."
        if "interest" in f or "pricing" in f:
            return "Pricing/risk signal: explain carefully because pricing can reflect prior underwriting risk."
        if "income" in f or "amount" in f or "loan_to_income" in f or "afford" in f:
            return "Affordability/exposure signal: suitable for portfolio risk interpretation."
        if "category" in f or "home" in f or "tenure" in f or "employment" in f:
            return "Segment/product signal: monitor stability and business reasonableness."
        return "Review for business reasonableness, stability, and possible proxy risk."

    if feature_col:
        xai["governance_note"] = xai[feature_col].map(note)
    xai["governance_action"] = "Monitor as top driver; include in periodic drift and reason-code review."
    return xai


def build_control_register() -> pd.DataFrame:
    """Professional model-risk control register."""
    rows = [
        ("Data ingestion", "Many-to-many joins or duplicate keys alter target rate.", "Use stable borrower-record key and duplicate checks.", "Notebook 01/05 record-key and split outputs", "Risk analytics", "Each refresh"),
        ("Data quality", "Missing or inconsistent inputs distort model scores.", "Track missingness, range, and logical consistency; open data-quality issues for breaches.", "Notebook 02/03/08 diagnostics", "Data owner", "Each refresh/monthly"),
        ("Leakage prevention", "Repayment-derived variables leak future outcome information.", "Exclude repayment-derived variables from modelling features and document exceptions.", "Feature policy", "Model developer", "Before release"),
        ("Sensitive/proxy governance", "Sensitive or proxy variables may cause unfair outcomes.", "Exclude direct sensitive fields from baseline model; use only for approved audit review.", "Feature policy and fairness/proxy review", "Model governance/compliance", "Before release and annually"),
        ("Model performance", "Ranking performance deteriorates after portfolio changes.", "Monitor PR-AUC, ROC-AUC, recall, precision, F1, Brier score when labels mature.", "Notebook 06/07 outputs", "Model owner", "Monthly/quarterly"),
        ("Threshold governance", "Threshold creates too many reviews or too many missed defaults.", "Use validation-selected threshold with review-rate cap and test confirmation.", "Notebook 07 outputs", "Credit strategy", "Quarterly/material change"),
        ("Explainability", "Model decisions cannot be understood by stakeholders.", "Maintain SHAP, local reasons, anchor-style rules, and counterfactual diagnostics.", "Notebook 08 outputs", "Risk analytics", "Each release"),
        ("Monitoring", "Score, feature, or data-quality drift changes model behaviour.", "Track score distribution, top-feature drift, review rate, realized default outcomes.", "Notebook 09 monitoring plan", "Model monitoring team", "Monthly"),
        ("Change management", "Uncontrolled model/file changes break reproducibility.", "Version model artifact, training code, features, threshold, and governance outputs together.", "Git + artifact manifest", "Model owner", "Each release"),
    ]
    return pd.DataFrame(rows, columns=["control_area", "risk", "control", "evidence", "owner", "frequency"])


def build_model_risk_limit_register(inputs: GovernanceInputs) -> pd.DataFrame:
    """Initial model risk limits and escalation triggers."""
    test = _best_test_confirmation(inputs)
    review_rate = test.get("review_rate")
    recall = test.get("recall")
    precision = test.get("precision")

    def floor(value: Any, offset: float) -> str:
        v = _safe_float(value)
        return "Define after production baseline" if v is None else f"< {max(v - offset, 0):.2%}"

    def ceil(value: Any, offset: float) -> str:
        v = _safe_float(value)
        return "Define after production baseline" if v is None else f"> {min(v + offset, 1):.2%}"

    return pd.DataFrame(
        [
            {"metric": "Score PSI", "baseline": "Training/test score distribution", "warning_limit": "> 0.10", "breach_limit": "> 0.25", "frequency": "Monthly", "action": "Investigate population shift, recalibration, or retraining need."},
            {"metric": "Top SHAP driver PSI", "baseline": "Top XAI feature distributions", "warning_limit": "> 0.10", "breach_limit": "> 0.25", "frequency": "Monthly", "action": "Review feature drift and data source changes."},
            {"metric": "Review rate", "baseline": _format_pct(review_rate), "warning_limit": ceil(review_rate, 0.05), "breach_limit": ceil(review_rate, 0.10), "frequency": "Weekly/monthly", "action": "Review threshold capacity and staffing impact."},
            {"metric": "Recall on matured labels", "baseline": _format_pct(recall), "warning_limit": floor(recall, 0.05), "breach_limit": floor(recall, 0.10), "frequency": "After labels mature", "action": "Assess missed-default concentration and refresh need."},
            {"metric": "Precision on reviewed accounts", "baseline": _format_pct(precision), "warning_limit": floor(precision, 0.03), "breach_limit": floor(precision, 0.05), "frequency": "After labels mature", "action": "Review false-positive burden and threshold."},
            {"metric": "Critical feature missingness", "baseline": "Notebook 02/03 profile", "warning_limit": "+25% relative", "breach_limit": "+50% relative", "frequency": "Each refresh", "action": "Open data-quality incident and assess model use pause."},
        ]
    )


def build_monitoring_kpi_snapshot(inputs: GovernanceInputs) -> pd.DataFrame:
    """Initial KPI snapshot for governance reporting."""
    test = _best_test_confirmation(inputs)
    xai_col = _top_feature_column(inputs.xai_grouped_importance)
    top_features = []
    if xai_col:
        top_features = inputs.xai_grouped_importance[xai_col].head(5).astype(str).tolist()

    rows = [
        ("Operational model", operational_model_name(inputs), "Model-threshold pair selected by validation business policy."),
        ("Portfolio default rate", _format_pct(_default_rate(inputs)), "Base rate for interpreting precision and review workload."),
        ("Threshold objective", threshold_objective(inputs), "Business rule used for threshold selection."),
        ("Operating threshold", _format_number(operating_threshold(inputs), 3), "Probability cutoff for manual-review flag."),
        ("Test recall", _format_pct(test.get("recall")), "Share of defaults captured at selected threshold."),
        ("Test precision", _format_pct(test.get("precision")), "Share of reviewed accounts that defaulted."),
        ("Test review rate", _format_pct(test.get("review_rate")), "Operational workload from the selected threshold."),
        ("Test business cost", _format_currency(test.get("business_cost")), "Scenario cost from false positives and false negatives."),
        ("Top SHAP drivers", ", ".join(top_features), "Main explanation drivers to monitor for drift."),
    ]
    return pd.DataFrame([{"kpi": k, "value": v, "interpretation": i} for k, v, i in rows])


def build_governance_summary(inputs: GovernanceInputs) -> pd.DataFrame:
    """One-row governance summary for README/reporting."""
    test = _best_test_confirmation(inputs)
    return pd.DataFrame(
        [
            {
                "operational_model": operational_model_name(inputs),
                "modeling_rows": len(inputs.modeling_df),
                "portfolio_default_rate": _default_rate(inputs),
                "operating_threshold": operating_threshold(inputs),
                "threshold_objective": threshold_objective(inputs),
                "test_recall": test.get("recall"),
                "test_precision": test.get("precision"),
                "test_review_rate": test.get("review_rate"),
                "test_business_cost": test.get("business_cost"),
                "primary_governance_decision": "Use as decision-support/manual-review prioritization model, not as automated credit-decline engine.",
            }
        ]
    )


def build_governance_readiness_gate(inputs: GovernanceInputs) -> pd.DataFrame:
    checks = [
        ("modeling_dataset_available", not inputs.modeling_df.empty, "Processed modelling dataset loaded."),
        ("threshold_recommendation_available", not inputs.operational_recommendation.empty, "Notebook 07 operational threshold loaded."),
        ("test_confirmation_available", not inputs.test_confirmation.empty, "Selected threshold confirmed on test set."),
        ("xai_outputs_available", not inputs.xai_grouped_importance.empty, "Notebook 08 SHAP outputs loaded."),
        ("control_register_created", True, "Notebook 09 control register created."),
        ("monitoring_limits_created", True, "Notebook 09 model risk limits created."),
        ("model_card_created", True, "Model card will be saved as markdown."),
        ("stakeholder_brief_created", True, "Stakeholder brief will be saved as markdown."),
    ]
    return pd.DataFrame([{"check": c, "passed": bool(p), "note": n} for c, p, n in checks])


def _markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "Not available."
    work = df.head(max_rows) if max_rows else df
    try:
        return work.to_markdown(index=False)
    except Exception:
        return work.to_csv(index=False)


def write_model_card(inputs: GovernanceInputs, output_path: Path) -> None:
    inventory = build_model_inventory(inputs)
    perf = build_validation_test_summary(inputs)
    xai = build_xai_governance_summary(inputs, top_n=10)
    limits = build_model_risk_limit_register(inputs)
    controls = build_control_register()

    text = f"""# Model Card - Canadian Retail Credit Risk XAI

## Model overview

- **Business purpose:** Early-warning retail credit default-risk ranking and manual-review prioritization.
- **Operational model:** `{operational_model_name(inputs)}`
- **Operating threshold:** `{_format_number(operating_threshold(inputs), 3)}`
- **Threshold objective:** `{threshold_objective(inputs)}`
- **Target:** borrower default indicator.
- **Intended use:** decision support for portfolio monitoring and credit-risk review.
- **Out-of-scope use:** automated credit decline, pricing decisioning, adverse-action communication, or production use without independent validation, legal/privacy review, and fairness assessment.

## Model inventory

{_markdown_table(inventory)}

## Validation and test performance

{_markdown_table(perf)}

## Top explainability drivers and governance notes

{_markdown_table(xai, max_rows=10)}

## Key controls

{_markdown_table(controls, max_rows=12)}

## Monitoring limits

{_markdown_table(limits)}

## Limitations

- This is a portfolio project built on available/synthetic project data and must not be treated as production banking advice.
- Counterfactuals are diagnostic scenario analysis, not customer instructions.
- Threshold cost assumptions are illustrative scenario assumptions, not accounting estimates.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def write_validation_summary(inputs: GovernanceInputs, output_path: Path) -> None:
    text = f"""# Model Validation Summary

## Executive summary

This model is positioned as a **manual-review prioritization and portfolio-monitoring tool**. It is not approved as a standalone automated credit decision engine.

{_markdown_table(build_governance_summary(inputs))}

## Performance evidence

{_markdown_table(build_validation_test_summary(inputs))}

## Feature governance

{_markdown_table(build_feature_governance_summary(inputs))}

## XAI governance summary

{_markdown_table(build_xai_governance_summary(inputs, top_n=12))}

## Validation decision

The model is acceptable for portfolio analytics and decision-support demonstration purposes, subject to the documented monitoring plan and limitations. Before production use, it would require independent model validation, data lineage review, calibration review, fairness testing, privacy/legal review, and user-acceptance testing with credit-risk stakeholders.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def write_stakeholder_brief(inputs: GovernanceInputs, output_path: Path) -> None:
    summary = build_governance_summary(inputs).iloc[0]
    text = f"""# Stakeholder Brief - Retail Credit Default-Risk Model

## What the model does

The model ranks borrowers by estimated default risk so that a credit-risk team can prioritize manual review and portfolio monitoring.

## Recommended operating point

- **Operational model:** {summary.get('operational_model')}
- **Operating threshold:** {_format_number(summary.get('operating_threshold'), 3)}
- **Test recall:** {_format_pct(summary.get('test_recall'))}
- **Test precision:** {_format_pct(summary.get('test_precision'))}
- **Test review rate:** {_format_pct(summary.get('test_review_rate'))}

## Business interpretation

At the selected threshold, the model captures a meaningful share of future defaults while keeping the review population close to the operational cap used in this project.

## Main model drivers

{_markdown_table(build_xai_governance_summary(inputs, top_n=5), max_rows=5)}

## How this should be used

Use the score to support analyst review, portfolio segmentation, and monitoring. Do not use it as a standalone automated lending decision without additional validation, fairness testing, compliance review, and production monitoring.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def write_monitoring_plan(inputs: GovernanceInputs, output_path: Path) -> None:
    text = f"""# Model Monitoring Plan

## Monitoring objective

Ensure the credit-risk model remains stable, explainable, and operationally useful after deployment or future data refreshes.

## KPI snapshot

{_markdown_table(build_monitoring_kpi_snapshot(inputs))}

## Risk limits and escalation triggers

{_markdown_table(build_model_risk_limit_register(inputs))}

## Suggested cadence

- **Each data refresh:** schema checks, duplicate-key checks, missingness checks, and leakage-policy checks.
- **Monthly:** score distribution, review rate, feature drift, data-quality drift, top-SHAP-driver drift.
- **After labels mature:** realized default rate, recall, precision, false negatives, and false positives.
- **Quarterly or material-change event:** threshold review, challenger review, governance sign-off.

## Escalation actions

If a breach occurs, pause automated refreshes if necessary, document the issue, identify root cause, quantify borrower/business impact, and decide whether remediation, recalibration, threshold adjustment, or retraining is required.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def build_release_artifact_manifest(table_dir: Path, governance_dir: Path, project_root: Path | None = None) -> pd.DataFrame:
    rows = []
    for folder, kind in [(table_dir, "table"), (governance_dir, "governance_document")]:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*")):
            if path.is_file() and (path.name.startswith("09_") or path.suffix.lower() in {".md", ".csv"}):
                rel = path.relative_to(project_root) if project_root and path.is_relative_to(project_root) else path
                rows.append({"artifact_type": kind, "path": str(rel), "exists": path.exists(), "size_bytes": path.stat().st_size})
    return pd.DataFrame(rows)


def save_governance_outputs(inputs: GovernanceInputs, table_dir: Path, governance_dir: Path, project_root: Path | None = None) -> dict[str, str]:
    """Save Notebook 09 tables and markdown governance documents."""
    table_dir.mkdir(parents=True, exist_ok=True)
    governance_dir.mkdir(parents=True, exist_ok=True)

    table_outputs: dict[str, pd.DataFrame] = {
        "09_model_inventory.csv": build_model_inventory(inputs),
        "09_model_validation_test_summary.csv": build_validation_test_summary(inputs),
        "09_feature_governance_summary.csv": build_feature_governance_summary(inputs),
        "09_xai_governance_summary.csv": build_xai_governance_summary(inputs),
        "09_model_control_register.csv": build_control_register(),
        "09_model_risk_limit_register.csv": build_model_risk_limit_register(inputs),
        "09_monitoring_kpi_snapshot.csv": build_monitoring_kpi_snapshot(inputs),
        "09_model_governance_summary.csv": build_governance_summary(inputs),
        "09_governance_readiness_gate.csv": build_governance_readiness_gate(inputs),
    }

    saved: dict[str, str] = {}
    for filename, df in table_outputs.items():
        path = table_dir / filename
        df.to_csv(path, index=False)
        saved[filename] = str(path)

    markdown_outputs = {
        "09_model_card.md": governance_dir / "09_model_card.md",
        "09_model_validation_summary.md": governance_dir / "09_model_validation_summary.md",
        "09_stakeholder_brief.md": governance_dir / "09_stakeholder_brief.md",
        "09_model_monitoring_plan.md": governance_dir / "09_model_monitoring_plan.md",
    }
    write_model_card(inputs, markdown_outputs["09_model_card.md"])
    write_validation_summary(inputs, markdown_outputs["09_model_validation_summary.md"])
    write_stakeholder_brief(inputs, markdown_outputs["09_stakeholder_brief.md"])
    write_monitoring_plan(inputs, markdown_outputs["09_model_monitoring_plan.md"])
    saved.update({name: str(path) for name, path in markdown_outputs.items()})

    manifest = build_release_artifact_manifest(table_dir, governance_dir, project_root=project_root)
    manifest_path = table_dir / "09_governance_output_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    saved["09_governance_output_manifest.csv"] = str(manifest_path)
    return saved
