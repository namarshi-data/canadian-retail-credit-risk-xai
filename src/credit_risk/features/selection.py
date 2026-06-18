from __future__ import annotations

"""Train-only feature screening utilities for Notebook 05.

These functions produce diagnostics only. They do not remove features
automatically, and they do not inspect validation/test outcomes for supervised
screening.
"""

import numpy as np
import pandas as pd

TARGET_COL = "defaulter"
SPLIT_COL = "split"
ID_COLS = ["user_id", "record_sequence"]


def cramers_v_from_table(table: pd.DataFrame) -> float:
    """Compute Cramer's V for a contingency table."""
    try:
        from scipy.stats import chi2_contingency

        chi2 = chi2_contingency(table, correction=False)[0]
        n = table.to_numpy().sum()
        r, k = table.shape
        denom = n * (min(k - 1, r - 1))
        return float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan
    except Exception:
        return np.nan


def make_rare_category_review(
    modeling_df: pd.DataFrame,
    categorical_features: list[str],
    min_share_pct: float = 1.0,
) -> pd.DataFrame:
    """Review rare categorical levels before modelling pipeline design."""
    rows = []
    for col in categorical_features:
        if col not in modeling_df.columns:
            continue
        counts = modeling_df[col].astype("object").where(modeling_df[col].notna(), "Missing").value_counts(dropna=False)
        for value, count in counts.items():
            share = count / len(modeling_df) * 100 if len(modeling_df) else np.nan
            rows.append(
                {
                    "feature": col,
                    "category_value": value,
                    "row_count": int(count),
                    "share_pct": round(float(share), 4) if pd.notna(share) else np.nan,
                    "rare_category_flag": bool(pd.notna(share) and share < min_share_pct),
                    "recommended_action": "group_or_handle_unknown_in_pipeline"
                    if pd.notna(share) and share < min_share_pct
                    else "retain_category",
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["feature", "row_count"], ascending=[True, False]).reset_index(drop=True)


def make_multicollinearity_review(
    modeling_df: pd.DataFrame,
    numeric_features: list[str],
    threshold: float = 0.80,
) -> pd.DataFrame:
    """Return strongly correlated numeric feature pairs."""
    numeric_features = [col for col in numeric_features if col in modeling_df.columns]
    if len(numeric_features) < 2:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation", "abs_correlation", "recommended_action"])

    corr = modeling_df[numeric_features].corr(numeric_only=True)
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs = (
        corr.where(mask)
        .stack()
        .reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "correlation"})
    )
    pairs["abs_correlation"] = pairs["correlation"].abs()
    pairs = pairs.query("abs_correlation >= @threshold").copy()
    if pairs.empty:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation", "abs_correlation", "recommended_action"])
    pairs["recommended_action"] = "review_redundancy_before_linear_modeling"
    return pairs.sort_values("abs_correlation", ascending=False).round(4).reset_index(drop=True)


def make_train_only_univariate_screening(
    modeling_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target_col: str = TARGET_COL,
    split_col: str = SPLIT_COL,
) -> pd.DataFrame:
    """Run univariate screening on the training split only.

    This table is for diagnostics and business review. It should not be used as
    a blind feature-drop rule.
    """
    if split_col not in modeling_df.columns:
        raise KeyError(f"Split column {split_col!r} not found.")
    if target_col not in modeling_df.columns:
        raise KeyError(f"Target column {target_col!r} not found.")

    train = modeling_df[modeling_df[split_col] == "train"].copy()
    if train.empty:
        raise ValueError("Training split is empty; cannot run train-only screening.")

    rows = []
    try:
        from scipy import stats
    except Exception:  # pragma: no cover
        stats = None

    y = pd.to_numeric(train[target_col], errors="coerce")

    for col in numeric_features:
        if col not in train.columns:
            continue
        x = pd.to_numeric(train[col], errors="coerce")
        x0 = x[y == 0].dropna()
        x1 = x[y == 1].dropna()

        if len(x0) == 0 or len(x1) == 0:
            p_value = np.nan
            effect = np.nan
            test = "not_available"
        elif set(x.dropna().unique()).issubset({0, 1}):
            table = pd.crosstab(x.fillna("Missing"), y)
            p_value = stats.chi2_contingency(table)[1] if stats is not None and table.shape[0] > 1 and table.shape[1] > 1 else np.nan
            effect = cramers_v_from_table(table)
            test = "chi_square_binary_flag"
        else:
            p_value = stats.mannwhitneyu(x0, x1, alternative="two-sided").pvalue if stats is not None else np.nan
            pooled = np.sqrt((x0.var(ddof=1) + x1.var(ddof=1)) / 2) if len(x0) > 1 and len(x1) > 1 else np.nan
            effect = (x1.mean() - x0.mean()) / pooled if pd.notna(pooled) and pooled != 0 else np.nan
            test = "mann_whitney_u_and_cohens_d"

        rows.append(
            {
                "feature": col,
                "feature_type": "numeric_or_binary",
                "screening_test": test,
                "non_default_mean_or_rate": round(float(x0.mean()), 6) if len(x0) else np.nan,
                "default_mean_or_rate": round(float(x1.mean()), 6) if len(x1) else np.nan,
                "effect_size": round(float(effect), 6) if pd.notna(effect) else np.nan,
                "p_value": round(float(p_value), 8) if pd.notna(p_value) else np.nan,
                "highest_risk_category_on_train": np.nan,
                "leakage_safe_scope": "training_split_only",
            }
        )

    for col in categorical_features:
        if col not in train.columns:
            continue
        temp = train[[col, target_col]].copy()
        temp[col] = temp[col].astype("object").where(temp[col].notna(), "Missing")
        table = pd.crosstab(temp[col], temp[target_col])
        p_value = stats.chi2_contingency(table)[1] if stats is not None and table.shape[0] > 1 and table.shape[1] > 1 else np.nan
        effect = cramers_v_from_table(table)
        default_rates = temp.groupby(col)[target_col].mean().sort_values(ascending=False)
        rows.append(
            {
                "feature": col,
                "feature_type": "categorical",
                "screening_test": "chi_square_and_cramers_v",
                "non_default_mean_or_rate": np.nan,
                "default_mean_or_rate": round(float(default_rates.iloc[0]), 6) if not default_rates.empty else np.nan,
                "effect_size": round(float(effect), 6) if pd.notna(effect) else np.nan,
                "p_value": round(float(p_value), 8) if pd.notna(p_value) else np.nan,
                "highest_risk_category_on_train": str(default_rates.index[0]) if not default_rates.empty else np.nan,
                "leakage_safe_scope": "training_split_only",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["absolute_effect_size"] = out["effect_size"].abs()
    out["screening_recommendation"] = np.select(
        [
            out["absolute_effect_size"].ge(0.10),
            out["absolute_effect_size"].between(0.03, 0.10, inclusive="left"),
        ],
        ["stronger_univariate_signal_review_in_model", "modest_signal_keep_if_business_relevant"],
        default="weak_univariate_signal_do_not_drop_blindly",
    )
    return out.sort_values(["absolute_effect_size", "feature"], ascending=[False, True]).reset_index(drop=True)


def make_feature_family_summary(feature_catalog: pd.DataFrame) -> pd.DataFrame:
    """Summarize feature count by family and policy."""
    if feature_catalog.empty:
        return pd.DataFrame()
    return (
        feature_catalog.groupby(["feature_family", "baseline_model_policy"], dropna=False)
        .agg(feature_count=("feature", "count"), max_missing_pct=("missing_pct", "max"))
        .reset_index()
        .sort_values(["feature_count", "feature_family"], ascending=[False, True])
    )
