from __future__ import annotations

import pandas as pd


def summarize_dataframe(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Create a compact data-quality summary for one DataFrame."""
    return pd.DataFrame(
        {
            "dataset": dataset_name,
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "non_null_count": df.notna().sum().values,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean().values * 100).round(2),
            "unique_values": df.nunique(dropna=True).values,
        }
    )


def summarize_workbook(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a schema summary for every workbook sheet."""
    summaries = [summarize_dataframe(df, name) for name, df in sheets.items()]
    return pd.concat(summaries, ignore_index=True)


def duplicate_id_summary(sheets: dict[str, pd.DataFrame], id_column: str = "user_id") -> pd.DataFrame:
    """Report duplicate borrower ID counts by sheet."""
    rows = []
    for name, df in sheets.items():
        if id_column not in df.columns:
            rows.append({"dataset": name, "id_column_present": False, "duplicate_id_count": None})
            continue
        rows.append(
            {
                "dataset": name,
                "id_column_present": True,
                "row_count": len(df),
                "unique_id_count": df[id_column].nunique(dropna=True),
                "duplicate_id_count": int(df[id_column].duplicated().sum()),
            }
        )
    return pd.DataFrame(rows)
