from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from credit_risk.config import INTERIM_DIR, PROCESSED_DIR, TABLE_DIR, ensure_project_directories
from credit_risk.data.cleaning import clean_credit_risk_dataset


def main() -> None:
    ensure_project_directories()

    interim_path = INTERIM_DIR / "credit_risk_merged_interim.csv"
    processed_path = PROCESSED_DIR / "credit_risk_cleaned.csv"

    if not interim_path.exists():
        raise FileNotFoundError(
            f"Interim dataset not found at {interim_path}. Run scripts/run_data_pipeline.py first."
        )

    df = pd.read_csv(interim_path, low_memory=False)
    result = clean_credit_risk_dataset(df)

    result.cleaned.to_csv(processed_path, index=False)
    result.audit_summary.to_csv(TABLE_DIR / "cleaning_audit_summary.csv", index=False)
    result.flag_summary.to_csv(TABLE_DIR / "cleaning_flag_summary.csv", index=False)
    result.model_feature_policy.to_csv(TABLE_DIR / "model_feature_policy.csv", index=False)

    print("Cleaning pipeline completed successfully.")
    print(f"Input shape: {df.shape}")
    print(f"Output shape: {result.cleaned.shape}")
    print(f"Processed dataset: {processed_path}")
    print("\nTop data-quality flags:")
    print(result.flag_summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
