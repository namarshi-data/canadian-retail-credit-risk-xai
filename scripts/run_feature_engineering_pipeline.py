from __future__ import annotations

"""Run Notebook 05 feature-engineering artifacts from the terminal.

This script is leakage-safe:
- It creates deterministic, row-level engineered features.
- It creates a leakage-reviewed unencoded modelling dataset.
- It saves feature policy, leakage review, lineage, split distribution,
  rare-category review, train-only screening, and preprocessing design artifacts.
- It does not fit encoders, imputers, scalers, resamplers, or models.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from credit_risk.config import PROCESSED_DIR, TABLE_DIR, ensure_project_directories
except Exception:  # Allows the script to run in a minimal copied folder.
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    TABLE_DIR = PROJECT_ROOT / "reports" / "tables"

    def ensure_project_directories() -> None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        TABLE_DIR.mkdir(parents=True, exist_ok=True)

from credit_risk.features.engineering import save_feature_engineering_outputs

INPUT_FILE = PROCESSED_DIR / "credit_risk_cleaned.csv"
MODELING_FILE = PROCESSED_DIR / "credit_risk_modeling_dataset.csv"


def main() -> None:
    """Execute deterministic feature engineering and leakage-safe dataset design."""
    ensure_project_directories()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {INPUT_FILE}. Run scripts/run_cleaning_pipeline.py first."
        )

    cleaned_df = pd.read_csv(INPUT_FILE, low_memory=False)
    artifacts = save_feature_engineering_outputs(
        cleaned_df=cleaned_df,
        processed_dir=PROCESSED_DIR,
        table_dir=TABLE_DIR,
        random_state=42,
    )

    print("Feature-engineering pipeline completed successfully.")
    print(f"Input cleaned dataset shape: {cleaned_df.shape}")
    print(f"Modelling dataset shape: {artifacts['modeling_df'].shape}")
    print(f"Modelling dataset saved: {MODELING_FILE}")
    print(f"Feature policy rows: {artifacts['feature_policy'].shape[0]}")
    print(f"Feature catalog rows: {artifacts['feature_catalog'].shape[0]}")

    print("\nSplit distribution:")
    print(artifacts["split_distribution"].to_string(index=False))

    print("\nFeature policy summary:")
    print(
        artifacts["feature_policy"]
        .groupby(["decision", "baseline_model_policy"], dropna=False)
        .agg(feature_count=("feature", "count"))
        .reset_index()
        .sort_values("feature_count", ascending=False)
        .to_string(index=False)
    )

    failed_qa = artifacts["qa_checks"].query("status == 'fail'")
    if not failed_qa.empty:
        print("\nWARNING: Feature-engineering QA failures found:")
        print(failed_qa.to_string(index=False))
    else:
        print("\nFeature-engineering QA checks passed or are review-only.")

    print("\nReminder: fit imputers, encoders, scalers, resamplers, and models in Notebook 06 on the training split only.")


if __name__ == "__main__":
    main()
