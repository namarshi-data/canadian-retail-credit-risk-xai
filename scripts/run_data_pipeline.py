from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from credit_risk.data.ingestion import (
    merge_credit_risk_sheets,
    normalize_id_columns,
    read_source_workbook,
    standardize_workbook_sheets,
)
from credit_risk.utils.paths import RAW_DATA_DIR, INTERIM_DATA_DIR, TABLES_DIR
from credit_risk.data.validation import validate_target


def main() -> None:
    raw_path = RAW_DATA_DIR / "Credit_Risk_Dataset.xlsx"

    sheets = read_source_workbook(raw_path)
    sheets = standardize_workbook_sheets(sheets)
    sheets = normalize_id_columns(sheets)

    merged = merge_credit_risk_sheets(sheets)

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = INTERIM_DATA_DIR / "credit_risk_merged_interim.csv"
    merged.to_csv(output_path, index=False)

    target_check = validate_target(merged, target="defaulter")

    print(f"Saved merged interim dataset: {output_path}")
    print(f"Shape: {merged.shape}")
    print(f"Default rate: {target_check.get('default_rate'):.4%}")
    print(f"Record-key duplicate count: {merged[['user_id', 'record_sequence']].duplicated().sum()}")


if __name__ == "__main__":
    main()
