from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from credit_risk.config import PROCESSED_DIR, TABLE_DIR, ensure_project_directories
from credit_risk.monitoring.portfolio_kpis import save_portfolio_monitoring_tables


def main() -> None:
    ensure_project_directories()

    processed_path = PROCESSED_DIR / "credit_risk_cleaned.csv"
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {processed_path}. Run scripts/run_cleaning_pipeline.py first."
        )

    df = pd.read_csv(processed_path, low_memory=False)
    tables = save_portfolio_monitoring_tables(df, TABLE_DIR)

    print("Portfolio monitoring tables generated")
    print(f"Input shape: {df.shape}")
    print(f"Tables saved to: {TABLE_DIR}")
    print("Generated tables:")
    for name, table in tables.items():
        print(f"- {name}: {table.shape}")


if __name__ == "__main__":
    main()
