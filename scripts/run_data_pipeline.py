from __future__ import annotations

"""Run Notebook 01 data ingestion and schema-review pipeline.

This script reads the raw multi-sheet workbook, standardizes schemas, safely
merges on ``user_id + record_sequence``, validates the merge grain, and saves the
interim dataset plus audit tables.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from credit_risk.config import INTERIM_DIR, RAW_DIR, TABLE_DIR, ensure_project_directories
except ImportError:  # pragma: no cover - portability fallback
    RAW_DIR = PROJECT_ROOT / "data" / "raw"
    INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
    TABLE_DIR = PROJECT_ROOT / "reports" / "tables"

    def ensure_project_directories() -> None:
        for path in [RAW_DIR, INTERIM_DIR, TABLE_DIR]:
            path.mkdir(parents=True, exist_ok=True)

from credit_risk.data.ingestion import (  # noqa: E402
    build_merge_audit,
    build_sheet_grain_summary,
    merge_credit_risk_sheets,
    normalize_id_columns,
    read_source_workbook,
    source_sheet_overview,
    standardize_workbook_sheets,
    validate_required_sheets,
)
from credit_risk.data.schema import (  # noqa: E402
    build_cardinality_review,
    build_data_dictionary,
    build_dataset_inventory,
    duplicate_id_summary,
    summarize_workbook,
)
from credit_risk.data.validation import (  # noqa: E402
    build_leakage_review_table,
    build_readiness_gate,
    validate_expected_columns,
    validate_merge_integrity,
    validate_target,
)

RAW_PATH = RAW_DIR / "Credit_Risk_Dataset.xlsx"
INTERIM_OUTPUT_PATH = INTERIM_DIR / "credit_risk_merged_interim.csv"
TARGET_COL = "defaulter"


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    ensure_project_directories()
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw workbook not found at {RAW_PATH}. Place Credit_Risk_Dataset.xlsx in data/raw/.")

    raw_sheets = read_source_workbook(RAW_PATH)
    raw_sheet_overview = source_sheet_overview(raw_sheets)

    sheets = normalize_id_columns(standardize_workbook_sheets(raw_sheets))
    standardized_sheet_overview = source_sheet_overview(sheets)

    required_sheet_check = validate_required_sheets(sheets)
    if not required_sheet_check["present"].all():
        missing = required_sheet_check.loc[~required_sheet_check["present"], "sheet_name"].tolist()
        raise KeyError(f"Missing required standardized sheets: {missing}")

    schema_summary = summarize_workbook(sheets, target_column=TARGET_COL)
    data_dictionary = build_data_dictionary(schema_summary)
    dataset_inventory = build_dataset_inventory(sheets)
    expected_column_check = validate_expected_columns(sheets)
    duplicate_summary = duplicate_id_summary(sheets)
    sheet_grain_summary = build_sheet_grain_summary(sheets)

    cardinality_tables = [build_cardinality_review(df, name) for name, df in sheets.items()]
    categorical_cardinality_review = pd.concat(cardinality_tables, ignore_index=True) if cardinality_tables else pd.DataFrame()

    expected_row_count = sheets["loan_information"].shape[0]
    merged = merge_credit_risk_sheets(sheets)

    merge_audit = build_merge_audit(sheets, merged, target_column=TARGET_COL)
    merge_integrity = validate_merge_integrity(merged, expected_row_count=expected_row_count, target=TARGET_COL)
    target_check = validate_target(merged, target=TARGET_COL)
    leakage_review = build_leakage_review_table(merged.columns)
    readiness_gate = build_readiness_gate(
        {
            "required_sheet_check": required_sheet_check,
            "expected_column_check": expected_column_check,
            "merge_integrity": merge_integrity,
        }
    )

    merged.to_csv(INTERIM_OUTPUT_PATH, index=False)

    outputs = {
        "01_source_sheet_overview.csv": raw_sheet_overview,
        "01_standardized_sheet_overview.csv": standardized_sheet_overview,
        "01_required_sheet_check.csv": required_sheet_check,
        "01_dataset_inventory.csv": dataset_inventory,
        "01_schema_summary.csv": schema_summary,
        "01_data_dictionary_starter.csv": data_dictionary,
        "01_expected_column_check.csv": expected_column_check,
        "01_duplicate_user_id_summary.csv": duplicate_summary,
        "01_sheet_grain_summary.csv": sheet_grain_summary,
        "01_categorical_cardinality_review.csv": categorical_cardinality_review,
        "01_merge_audit.csv": merge_audit,
        "01_merge_integrity_checks.csv": merge_integrity,
        "01_target_validation.csv": pd.DataFrame([target_check]),
        "01_leakage_review_initial.csv": leakage_review,
        "01_ingestion_readiness_gate.csv": readiness_gate,
    }

    for filename, table in outputs.items():
        save_table(table, TABLE_DIR / filename)

    print("Data ingestion pipeline completed successfully.")
    print(f"Saved interim dataset: {INTERIM_OUTPUT_PATH}")
    print(f"Shape: {merged.shape}")
    print(f"Default rate: {target_check.get('default_rate'):.4%}")
    print(f"Record-key duplicate count: {merge_audit.loc[merge_audit['check'].eq('record_key_duplicate_count'), 'result'].iloc[0]}")

    print("\nReadiness gate:")
    print(readiness_gate.to_string(index=False))


if __name__ == "__main__":
    main()
