from __future__ import annotations

"""Run Notebook 06 model-training pipeline from terminal.

This script expects Notebook 05 to have already created:
    data/processed/credit_risk_modeling_dataset.csv

It trains leakage-safe Logistic Regression benchmark, Random Forest baseline/tuned,
and XGBoost baseline/tuned models. All preprocessing, skewness treatment,
winsorization, encoding, scaling, optional resampling, and model fitting happen
inside train-only pipelines.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from credit_risk.config import MODEL_ARTIFACT_DIR, PROCESSED_DIR, TABLE_DIR, ensure_project_directories  # noqa: E402
from credit_risk.models.train import TrainingConfig, run_training_workflow_from_file  # noqa: E402


def main() -> None:
    ensure_project_directories()

    input_path = PROCESSED_DIR / "credit_risk_modeling_dataset.csv"
    config = TrainingConfig.from_environment()

    artifacts = run_training_workflow_from_file(
        modeling_dataset_path=input_path,
        table_dir=TABLE_DIR,
        model_artifact_dir=MODEL_ARTIFACT_DIR,
        config=config,
    )

    display_cols = [
        "model_name",
        "dataset",
        "pr_auc",
        "roc_auc",
        "brier_score",
        "recall",
        "precision",
        "f1",
        "balanced_accuracy",
        "mcc",
        "review_rate",
        "business_cost",
        "threshold",
        "false_negative",
        "false_positive",
        "true_positive",
        "true_negative",
    ]

    print("Notebook 06 model training completed.")
    print(f"Input dataset: {input_path}")
    print(f"Numeric features: {len(artifacts.numeric_features)}")
    print(f"Categorical features: {len(artifacts.categorical_features)}")
    print(f"Ranking champion by validation PR-AUC: {artifacts.ranking_champion_model_name}")
    print(f"Operational champion after validation thresholding: {artifacts.operational_champion_model_name}")
    print(f"Saved champion model artifact: {artifacts.champion_model_name}")
    print("\nValidation results at default 0.50 threshold:")
    print(artifacts.validation_results[[c for c in display_cols if c in artifacts.validation_results.columns]].to_string(index=False))
    print("\nAll-model operational threshold comparison on validation:")
    cols = [
        "operational_rank", "model_name", "threshold", "pr_auc", "roc_auc",
        "recall", "precision", "review_rate", "business_cost",
        "validation_pr_auc_at_default_threshold",
    ]
    print(artifacts.all_model_threshold_shortlist[[c for c in cols if c in artifacts.all_model_threshold_shortlist.columns]].to_string(index=False))

    print("\nRecommended operational model/threshold from validation:")
    print(artifacts.operational_model_threshold_recommendation.to_string(index=False))
    print("\nTest confirmation for selected operational champion:")
    print(artifacts.test_selected_threshold_results.to_string(index=False))


if __name__ == "__main__":
    main()
