from __future__ import annotations

"""Schema profiling utilities for Notebook 01 and Notebook 02."""

import numpy as np
import pandas as pd

SENSITIVE_OR_PROXY_COLUMNS = {
    "gender",
    "married",
    "pincode",
    "social_profile",
    "has_social_profile",
    "home",
}

LEAKAGE_PRONE_COLUMNS = {
    "total_payment",
    "received_principal",
    "interest_received",
    "payment_to_amount_ratio",
    "principal_to_amount_ratio",
    "interest_to_amount_ratio",
}

IDENTIFIER_COLUMNS = {"user_id", "customer_id", "borrower_id", "record_sequence"}


def infer_column_role(column: str, series: pd.Series | None = None, target_column: str = "defaulter") -> str:
    """Infer a governance-oriented schema role for one column."""
    lower = column.lower()
    if lower == target_column:
        return "target"
    if lower in IDENTIFIER_COLUMNS or lower.endswith("_id"):
        return "identifier"
    if lower in LEAKAGE_PRONE_COLUMNS or any(token in lower for token in ["payment", "principal", "interest_received"]):
        return "repayment_or_leakage_monitoring_only"
    if lower in SENSITIVE_OR_PROXY_COLUMNS:
        return "sensitive_or_proxy_review"
    if lower.endswith("_flag") or lower.startswith("has_"):
        return "binary_flag_or_quality_signal"
    if series is not None and pd.api.types.is_numeric_dtype(series):
        return "numeric_feature"
    if series is not None and (pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series)):
        return "categorical_feature"
    return "unknown_or_mixed"


def summarize_dataframe(df: pd.DataFrame, dataset_name: str, target_column: str = "defaulter") -> pd.DataFrame:
    """Create a compact but governance-aware schema summary for one DataFrame."""
    rows: list[dict[str, object]] = []
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        unique_values = int(non_null.nunique(dropna=True))
        top_value = non_null.value_counts(dropna=True).index[0] if not non_null.empty else np.nan
        top_share = float(non_null.value_counts(normalize=True, dropna=True).iloc[0] * 100) if not non_null.empty else np.nan
        rows.append(
            {
                "dataset": dataset_name,
                "column": col,
                "dtype": str(series.dtype),
                "column_role": infer_column_role(col, series, target_column=target_column),
                "row_count": int(len(df)),
                "non_null_count": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "missing_pct": round(float(series.isna().mean() * 100), 4),
                "unique_values": unique_values,
                "unique_ratio": round(float(unique_values / max(series.notna().sum(), 1)), 6),
                "top_value": top_value,
                "top_value_share_pct": round(top_share, 4) if pd.notna(top_share) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_workbook(sheets: dict[str, pd.DataFrame], target_column: str = "defaulter") -> pd.DataFrame:
    """Create a schema summary for every standardized workbook sheet."""
    summaries = [summarize_dataframe(df, name, target_column=target_column) for name, df in sheets.items()]
    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()


def duplicate_id_summary(sheets: dict[str, pd.DataFrame], id_column: str = "user_id") -> pd.DataFrame:
    """Report duplicate borrower ID counts by sheet."""
    rows = []
    for name, df in sheets.items():
        if id_column not in df.columns:
            rows.append(
                {
                    "dataset": name,
                    "id_column_present": False,
                    "row_count": int(len(df)),
                    "unique_id_count": None,
                    "duplicate_id_count": None,
                    "max_records_per_id": None,
                }
            )
            continue
        counts = df[id_column].value_counts(dropna=False)
        rows.append(
            {
                "dataset": name,
                "id_column_present": True,
                "row_count": int(len(df)),
                "unique_id_count": int(df[id_column].nunique(dropna=True)),
                "duplicate_id_count": int(df[id_column].duplicated().sum()),
                "max_records_per_id": int(counts.max()) if not counts.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def build_dataset_inventory(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a one-row-per-sheet inventory table."""
    return pd.DataFrame(
        [
            {
                "dataset": name,
                "row_count": int(df.shape[0]),
                "column_count": int(df.shape[1]),
                "missing_value_count": int(df.isna().sum().sum()),
                "full_duplicate_rows": int(df.duplicated().sum()),
            }
            for name, df in sheets.items()
        ]
    )


def build_data_dictionary(schema_summary: pd.DataFrame) -> pd.DataFrame:
    """Create a starter data dictionary from schema summary."""
    if schema_summary.empty:
        return pd.DataFrame()
    dictionary = schema_summary[["dataset", "column", "dtype", "column_role", "missing_pct", "unique_values"]].copy()
    dictionary["business_definition"] = dictionary["column"].str.replace("_", " ", regex=False).str.title()
    dictionary["governance_note"] = dictionary["column_role"].map(
        {
            "identifier": "Join/audit key only; exclude from modelling.",
            "target": "Outcome variable; never use as predictor.",
            "repayment_or_leakage_monitoring_only": "Monitoring only unless prediction timing proves no leakage.",
            "sensitive_or_proxy_review": "Exclude from baseline model; retain for permitted fairness/governance review.",
            "binary_flag_or_quality_signal": "Review business meaning and stability before modelling.",
            "numeric_feature": "Candidate analytical feature subject to leakage and quality review.",
            "categorical_feature": "Candidate categorical feature subject to cardinality and governance review.",
        }
    ).fillna("Review before use.")
    return dictionary


def build_cardinality_review(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Profile categorical cardinality and recommend handling."""
    cat_cols = df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    rows = []
    for col in cat_cols:
        unique = int(df[col].nunique(dropna=True))
        if unique <= 10:
            recommendation = "low_cardinality_one_hot_candidate"
        elif unique <= 50:
            recommendation = "medium_cardinality_review_grouping"
        else:
            recommendation = "high_cardinality_group_or_exclude"
        rows.append(
            {
                "dataset": dataset_name,
                "column": col,
                "unique_values": unique,
                "missing_pct": round(float(df[col].isna().mean() * 100), 4),
                "cardinality_recommendation": recommendation,
            }
        )
    return pd.DataFrame(rows)
