from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from credit_risk.config import TABLE_DIR, ensure_project_directories
from credit_risk.models.thresholding import (
    ThresholdCostAssumptions,
    apply_thresholds_to_dataset,
    build_threshold_grid,
    build_threshold_shortlist,
    evaluate_threshold_grid_all_models,
    recommend_operating_threshold,
    save_threshold_outputs,
)


def main() -> None:
    ensure_project_directories()

    validation_predictions_path = TABLE_DIR / "validation_predictions_default_threshold.csv"
    test_predictions_path = TABLE_DIR / "test_predictions_default_threshold.csv"
    selection_summary_path = TABLE_DIR / "model_selection_summary.csv"

    for path in [validation_predictions_path, test_predictions_path, selection_summary_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}. Run scripts/run_model_training_pipeline.py first.")

    validation_predictions = pd.read_csv(validation_predictions_path, low_memory=False)
    test_predictions = pd.read_csv(test_predictions_path, low_memory=False)
    selection_summary = pd.read_csv(selection_summary_path)

    champion_model_name = str(selection_summary.sort_values("selection_rank").iloc[0]["model_name"])
    cost_assumptions = ThresholdCostAssumptions(false_negative_cost=5_000, false_positive_cost=500)
    thresholds = build_threshold_grid(start=0.01, stop=0.99, step=0.005)

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

    validation_shortlist = build_threshold_shortlist(validation_grid, champion_model_name)
    recommendation = recommend_operating_threshold(validation_shortlist)
    test_shortlist = apply_thresholds_to_dataset(
        threshold_shortlist=validation_shortlist,
        threshold_grid=test_grid,
        model_name=champion_model_name,
        dataset_name="test",
    )

    save_threshold_outputs(
        table_dir=TABLE_DIR,
        validation_grid=validation_grid,
        test_grid=test_grid,
        validation_shortlist=validation_shortlist,
        test_shortlist=test_shortlist,
        recommendation=recommendation,
        cost_assumptions=cost_assumptions,
    )

    print("Threshold-selection pipeline completed")
    print(f"Champion model: {champion_model_name}")
    print(f"Cost assumptions: FN=${cost_assumptions.false_negative_cost:,.0f}, FP=${cost_assumptions.false_positive_cost:,.0f}")
    print("Recommended validation operating threshold:")
    display_cols = [
        "objective",
        "threshold",
        "business_cost",
        "recall",
        "precision",
        "review_rate",
        "false_negative",
        "false_positive",
    ]
    print(pd.DataFrame([recommendation])[display_cols].to_string(index=False))
    print("Saved threshold outputs to reports/tables/")


if __name__ == "__main__":
    main()
