from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from credit_risk.config import PROCESSED_DIR, TABLE_DIR, ensure_project_directories
from credit_risk.features.engineering import save_feature_engineering_outputs


def main() -> None:
    ensure_project_directories()

    input_path = PROCESSED_DIR / "credit_risk_cleaned.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing cleaned dataset: {input_path}. Run scripts/run_cleaning_pipeline.py first."
        )

    cleaned_df = pd.read_csv(input_path, low_memory=False)
    artifacts = save_feature_engineering_outputs(cleaned_df, PROCESSED_DIR, TABLE_DIR)

    modeling_path = PROCESSED_DIR / "credit_risk_modeling_dataset.csv"
    print("Feature engineering completed")
    print(f"Input shape: {cleaned_df.shape}")
    print(f"Modeling dataset saved: {modeling_path}")
    print(f"Feature catalog rows: {artifacts['feature_catalog'].shape[0]}")
    print("Split distribution:")
    print(artifacts["split_distribution"].to_string(index=False))


if __name__ == "__main__":
    main()
