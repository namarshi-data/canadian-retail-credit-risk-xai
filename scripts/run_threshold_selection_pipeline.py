
from __future__ import annotations

"""Run Notebook 07 threshold-selection and business-cost pipeline.

This script expects Notebook 06 to have already saved validation and test prediction
files in reports/tables. Thresholds are selected on validation data only; the test
set is used only for final confirmation of the selected operating rule.
"""

import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from credit_risk.config import TABLE_DIR, ensure_project_directories  # noqa: E402
from credit_risk.models.thresholding import (  # noqa: E402
    ThresholdCostAssumptions,
    apply_recommendation_to_threshold_grid,
    build_all_model_operational_threshold_comparison,
    build_cost_sensitivity_scenarios,
    build_metric_stakeholder_impact_summary,
    build_model_handoff_for_explainability,
    build_policy_option_table,
    build_threshold_grid,
    build_threshold_policy_decision_table,
    build_threshold_readiness_gate,
    cost_assumptions_frame,
    evaluate_threshold_grid_all_models,
    make_confusion_matrix_long,
    read_first_existing_csv,
    recommend_operational_model_threshold,
    run_business_cost_sensitivity_analysis,
    save_threshold_outputs,
)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be numeric. Received: {value!r}") from exc


def main() -> None:
    ensure_project_directories()

    false_negative_cost = _env_float("THRESHOLD_FALSE_NEGATIVE_COST", 5_000)
    false_positive_cost = _env_float("THRESHOLD_FALSE_POSITIVE_COST", 500)
    review_rate_cap = _env_float("THRESHOLD_REVIEW_RATE_CAP", 0.30)
    min_recall = _env_float("THRESHOLD_MIN_RECALL", 0.60)
    threshold_step = _env_float("THRESHOLD_GRID_STEP", 0.005)

    validation_predictions = read_first_existing_csv(
        [
            TABLE_DIR / "06_validation_predictions_default_threshold.csv",
            TABLE_DIR / "validation_predictions_default_threshold.csv",
        ]
    )
    test_predictions = read_first_existing_csv(
        [
            TABLE_DIR / "06_test_predictions_default_threshold.csv",
            TABLE_DIR / "test_predictions_default_threshold.csv",
        ]
    )
    validation_results = read_first_existing_csv(
        [
            TABLE_DIR / "06_model_validation_results_default_threshold.csv",
            TABLE_DIR / "model_validation_results_default_threshold.csv",
        ],
        required=False,
    )

    cost_assumptions = ThresholdCostAssumptions(
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    thresholds = build_threshold_grid(start=0.01, stop=0.99, step=threshold_step)

    validation_grid = evaluate_threshold_grid_all_models(
        validation_predictions,
        thresholds=thresholds,
        dataset_name="validation",
        cost_assumptions=cost_assumptions,
    )
    test_grid = evaluate_threshold_grid_all_models(
        test_predictions,
        thresholds=thresholds,
        dataset_name="test",
        cost_assumptions=cost_assumptions,
    )

    all_model_operational_comparison = build_all_model_operational_threshold_comparison(
        threshold_grid=validation_grid,
        validation_results=validation_results,
        review_rate_cap=review_rate_cap,
        min_recall=min_recall,
        objective_name=f"minimum_cost_review_rate_le_{int(review_rate_cap * 100)}pct",
    )
    operational_recommendation = recommend_operational_model_threshold(all_model_operational_comparison)
    test_confirmation = apply_recommendation_to_threshold_grid(
        recommendation=operational_recommendation,
        threshold_grid=test_grid,
        dataset_name="test",
    )

    policy_options = build_policy_option_table(validation_grid, validation_results)
    cost_sensitivity = run_business_cost_sensitivity_analysis(
        validation_predictions=validation_predictions,
        validation_results=validation_results,
        thresholds=thresholds,
        review_rate_cap=review_rate_cap,
        min_recall=min_recall,
        scenarios=build_cost_sensitivity_scenarios(),
    )

    validation_confusion = make_confusion_matrix_long(operational_recommendation, dataset_name="validation")
    test_confusion = make_confusion_matrix_long(test_confirmation, dataset_name="test")
    stakeholder_summary = build_metric_stakeholder_impact_summary(test_confirmation)
    decision_table = build_threshold_policy_decision_table(operational_recommendation, test_confirmation)
    explainability_handoff = build_model_handoff_for_explainability(operational_recommendation, test_confirmation)
    readiness_gate = build_threshold_readiness_gate(validation_grid, test_grid, operational_recommendation, test_confirmation)
    assumptions_table = cost_assumptions_frame(cost_assumptions)

    outputs = {
        "07_business_cost_assumptions.csv": assumptions_table,
        "07_threshold_grid_validation_all_models.csv": validation_grid,
        "07_threshold_grid_test_all_models.csv": test_grid,
        "07_all_model_operational_threshold_comparison_validation.csv": all_model_operational_comparison,
        "07_operational_threshold_recommendation_validation.csv": operational_recommendation,
        "07_test_confirmation_selected_operational_model.csv": test_confirmation,
        "07_policy_option_table_validation.csv": policy_options,
        "07_business_cost_sensitivity_validation.csv": cost_sensitivity,
        "07_confusion_matrix_selected_validation.csv": validation_confusion,
        "07_confusion_matrix_selected_test.csv": test_confusion,
        "07_stakeholder_metric_impact_summary.csv": stakeholder_summary,
        "07_threshold_policy_decision_table.csv": decision_table,
        "07_model_handoff_for_explainability.csv": explainability_handoff,
        "07_threshold_readiness_gate.csv": readiness_gate,
    }
    save_threshold_outputs(TABLE_DIR, outputs)

    print("Notebook 07 threshold-selection pipeline completed.")
    print(f"Cost assumptions: FN=${false_negative_cost:,.0f}, FP=${false_positive_cost:,.0f}")
    print(f"Policy: review_rate <= {review_rate_cap:.0%}, minimum recall >= {min_recall:.0%}")
    print("\nRecommended operational model/threshold from validation:")
    display_cols = [
        "model_name", "threshold", "pr_auc", "roc_auc", "recall", "precision",
        "review_rate", "business_cost", "false_negative", "false_positive",
    ]
    print(operational_recommendation[[c for c in display_cols if c in operational_recommendation.columns]].to_string(index=False))
    print("\nTest confirmation:")
    print(test_confirmation[[c for c in display_cols if c in test_confirmation.columns]].to_string(index=False))
    print("\nSaved Notebook 07 outputs to reports/tables/ with 07_ prefixes.")


if __name__ == "__main__":
    main()
