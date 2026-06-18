from __future__ import annotations

"""Run Notebook 04 portfolio monitoring from the terminal.

Input:
    data/processed/credit_risk_cleaned.csv

Outputs:
    reports/tables/04_*.csv
    reports/figures/*.png
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from credit_risk.config import FIGURE_DIR, PROCESSED_DIR, TABLE_DIR, ensure_project_directories
except Exception:  # pragma: no cover
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
    FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

    def ensure_project_directories() -> None:
        for path in [PROCESSED_DIR, TABLE_DIR, FIGURE_DIR]:
            path.mkdir(parents=True, exist_ok=True)

from credit_risk.monitoring.portfolio_kpis import save_portfolio_monitoring_tables

TARGET_COL = "defaulter"
INPUT_FILE = PROCESSED_DIR / "credit_risk_cleaned.csv"


def main() -> None:
    ensure_project_directories()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {INPUT_FILE}. Run scripts/run_cleaning_pipeline.py first."
        )

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column {TARGET_COL!r} not found in {INPUT_FILE}.")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    tables = save_portfolio_monitoring_tables(
        df,
        output_dir=TABLE_DIR,
        target=TARGET_COL,
        figure_dir=FIGURE_DIR,
    )

    overview = tables["04_portfolio_overview"]
    segment_risk = tables["04_segment_risk_all"]
    high_risk = tables["04_high_risk_segments"]

    print("Notebook 04 portfolio monitoring pipeline completed.")
    print(f"Input file: {INPUT_FILE}")
    print(f"Input shape: {df.shape}")
    print(f"Tables saved to: {TABLE_DIR}")
    print(f"Figures saved to: {FIGURE_DIR}")

    print("\nPortfolio overview:")
    print(overview.to_string(index=False))

    print("\nGenerated tables:")
    for name, table in tables.items():
        print(f"- {name}: {table.shape}")

    if not high_risk.empty:
        print("\nTop high-risk segments:")
        show_cols = ["segment_column", "segment_value", "row_count", "default_rate_pct", "default_rate_lift"]
        print(high_risk[[c for c in show_cols if c in high_risk.columns]].head(10).to_string(index=False))

    print(f"\nSegment risk rows: {len(segment_risk):,}")


if __name__ == "__main__":
    main()
