from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

try:  # optional dependency: Notebook 08 should explain how to install when missing
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None

TARGET_COLUMN = "defaulter"
SPLIT_COLUMN = "split"
ID_COLUMNS = ["user_id", "record_sequence"]
DEFAULT_FALSE_NEGATIVE_COST = 5_000
DEFAULT_FALSE_POSITIVE_COST = 500


@dataclass
class ExplainabilityArtifacts:
    """Loaded artefacts needed by the explainability workflow."""

    champion_model: object
    metadata: dict[str, Any]
    modeling_df: pd.DataFrame
    recommended_threshold: float
    champion_model_name: str
    model_path: Path
    metadata_path: Path
    threshold_path: Path


def _first_existing(paths: Sequence[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = "\n - ".join(str(p) for p in paths)
    raise FileNotFoundError(f"Could not find {label}. Checked:\n - {joined}")


def _default_dirs(project_root: Path) -> tuple[Path, Path, Path]:
    table_dir = project_root / "reports" / "tables"
    model_dir = project_root / "reports" / "model_artifacts"
    data_dir = project_root / "data" / "processed"
    return table_dir, model_dir, data_dir


def load_explainability_artifacts(
    project_root: Path | None = None,
    model_path: Path | None = None,
    metadata_path: Path | None = None,
    modeling_dataset_path: Path | None = None,
    recommended_threshold_path: Path | None = None,
) -> ExplainabilityArtifacts:
    """Load the operational champion model, metadata, modelling data, and threshold.

    The function accepts explicit paths, but also searches common Notebook 06/07 output
    filenames so the workflow is robust across portfolio iterations.
    """
    project_root = Path.cwd() if project_root is None else Path(project_root)
    table_dir, model_dir, data_dir = _default_dirs(project_root)

    model_path = model_path or _first_existing(
        [
            model_dir / "champion_model.joblib",
            model_dir / "06_champion_model.joblib",
            model_dir / "operational_champion_model.joblib",
            model_dir / "06_operational_champion_model.joblib",
            model_dir / "xgboost_weighted_baseline.joblib",
        ],
        "champion model artefact",
    )
    metadata_path = metadata_path or _first_existing(
        [
            model_dir / "model_feature_metadata.joblib",
            model_dir / "06_model_feature_metadata.joblib",
            model_dir / "operational_model_feature_metadata.joblib",
            model_dir / "06_operational_model_feature_metadata.joblib",
        ],
        "model feature metadata",
    )
    modeling_dataset_path = modeling_dataset_path or _first_existing(
        [
            data_dir / "credit_risk_modeling_dataset.csv",
            data_dir / "credit_risk_feature_engineered.csv",
            data_dir / "credit_risk_model_ready.csv",
        ],
        "modelling dataset",
    )
    recommended_threshold_path = recommended_threshold_path or _first_existing(
        [
            table_dir / "recommended_threshold_summary.csv",
            table_dir / "recommended_operational_threshold_summary.csv",
            table_dir / "07_recommended_threshold_summary.csv",
            table_dir / "07_recommended_operational_threshold_summary.csv",
            table_dir / "06_recommended_operational_threshold_summary.csv",
        ],
        "recommended threshold summary",
    )

    champion_model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": metadata}

    modeling_df = pd.read_csv(modeling_dataset_path, low_memory=False)
    threshold_df = pd.read_csv(recommended_threshold_path, low_memory=False)
    if threshold_df.empty or "threshold" not in threshold_df.columns:
        raise ValueError(f"{recommended_threshold_path} must contain a non-empty threshold column.")

    champion_model_name = str(
        threshold_df.iloc[0].get(
            "model_name",
            metadata.get("operational_champion_model_name", metadata.get("champion_model_name", model_path.stem)),
        )
    )

    return ExplainabilityArtifacts(
        champion_model=champion_model,
        metadata=metadata,
        modeling_df=modeling_df,
        recommended_threshold=float(threshold_df.iloc[0]["threshold"]),
        champion_model_name=champion_model_name,
        model_path=Path(model_path),
        metadata_path=Path(metadata_path),
        threshold_path=Path(recommended_threshold_path),
    )


def feature_columns_from_metadata(metadata: dict[str, Any], modeling_df: pd.DataFrame | None = None) -> list[str]:
    """Return model feature columns from metadata, with a safe fallback."""
    for key in ["feature_columns", "model_features", "features"]:
        if key in metadata and metadata[key]:
            return list(metadata[key])
    numeric = list(metadata.get("numeric_features", []))
    categorical = list(metadata.get("categorical_features", []))
    if numeric or categorical:
        return numeric + categorical
    if modeling_df is None:
        raise ValueError("No feature metadata found and no modelling_df provided for fallback inference.")
    excluded = set(ID_COLUMNS + [TARGET_COLUMN, SPLIT_COLUMN])
    return [c for c in modeling_df.columns if c not in excluded]


def _safe_numeric(series: pd.Series | None, index: pd.Index) -> pd.Series:
    """Return a numeric series aligned to index, using NaN when source is unavailable."""
    if series is None:
        return pd.Series(np.nan, index=index, dtype="float64")
    return pd.to_numeric(series.reindex(index), errors="coerce")


def _fixed_band(value: float, bins: list[float], labels: list[str], missing_label: str = "Missing") -> str:
    """Create stable portfolio bands without fitting on validation/test data."""
    if pd.isna(value):
        return missing_label
    for upper, label in zip(bins[1:], labels):
        if float(value) <= upper:
            return label
    return labels[-1]


def _source_column(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series | None:
    """Find the first available source column among common naming variants."""
    for col in candidates:
        if col in df.columns:
            return df[col]
    return None


def ensure_expected_model_features(
    modeling_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure the dataframe contains every feature expected by the fitted model.

    Notebook 08 must explain the exact model saved by Notebook 06. In some local
    projects the saved model metadata points to engineered feature columns while the
    currently loaded ``data/processed/credit_risk_modeling_dataset.csv`` is an older
    or thinner file. This helper rebuilds deterministic leakage-safe engineered
    columns when their raw inputs are available and creates explicit missing-value
    placeholder columns when a feature cannot be recovered. The fitted model pipeline
    then applies the imputation learned during Notebook 06 training.

    Quantile-based flags are calculated from the train split only when a train split
    is available. This preserves train-only logic for explainability scoring.
    """
    df = modeling_df.copy()
    index = df.index
    status_rows: list[dict[str, Any]] = []

    train_mask = df[SPLIT_COLUMN].astype(str).eq("train") if SPLIT_COLUMN in df.columns else pd.Series(True, index=index)
    if not bool(train_mask.any()):
        train_mask = pd.Series(True, index=index)

    # Source columns with common naming variants.
    amount = _safe_numeric(_source_column(df, ["amount", "loan_amount", "loan_amount_cad"]), index)
    income = _safe_numeric(_source_column(df, ["total_income_pa", "annual_income", "annual_income_cad", "income"]), index)
    interest = _safe_numeric(_source_column(df, ["interest_rate", "interest_rate_pct"]), index)
    dependents = _safe_numeric(_source_column(df, ["dependents", "number_of_dependents"]), index)
    delinq = _safe_numeric(_source_column(df, ["delinq_2yrs", "delinq_2_years", "delinquency_2yrs", "delinquencies_2yrs"]), index)
    nloans = _safe_numeric(_source_column(df, ["number_of_loans", "num_loans", "existing_loans", "loan_count"]), index)

    # Recreate raw alias columns if the model expects them.
    if "delinq_2yrs" in feature_cols and "delinq_2yrs" not in df.columns:
        df["delinq_2yrs"] = delinq
        status_rows.append({"feature": "delinq_2yrs", "action": "rebuilt_from_alias_or_missing_placeholder", "available_non_null": int(df["delinq_2yrs"].notna().sum())})
    if "number_of_loans" in feature_cols and "number_of_loans" not in df.columns:
        df["number_of_loans"] = nloans
        status_rows.append({"feature": "number_of_loans", "action": "rebuilt_from_alias_or_missing_placeholder", "available_non_null": int(df["number_of_loans"].notna().sum())})

    # Core derived affordability/pricing features.
    if "loan_to_income_ratio" in df.columns:
        lti = _safe_numeric(df["loan_to_income_ratio"], index)
    else:
        lti = pd.Series(np.where(income > 0, amount / income, np.nan), index=index)
        if "loan_to_income_ratio" in feature_cols:
            df["loan_to_income_ratio"] = lti
            status_rows.append({"feature": "loan_to_income_ratio", "action": "rebuilt_from_amount_and_income", "available_non_null": int(pd.Series(lti).notna().sum())})

    if "amount_log1p" in feature_cols and "amount_log1p" not in df.columns:
        df["amount_log1p"] = np.log1p(amount.clip(lower=0))
        status_rows.append({"feature": "amount_log1p", "action": "rebuilt_from_amount", "available_non_null": int(df["amount_log1p"].notna().sum())})
    if "total_income_pa_log1p" in feature_cols and "total_income_pa_log1p" not in df.columns:
        df["total_income_pa_log1p"] = np.log1p(income.clip(lower=0))
        status_rows.append({"feature": "total_income_pa_log1p", "action": "rebuilt_from_total_income_pa", "available_non_null": int(df["total_income_pa_log1p"].notna().sum())})
    if "loan_to_income_ratio_log1p" in feature_cols and "loan_to_income_ratio_log1p" not in df.columns:
        df["loan_to_income_ratio_log1p"] = np.log1p(pd.Series(lti, index=index).clip(lower=0))
        status_rows.append({"feature": "loan_to_income_ratio_log1p", "action": "rebuilt_from_loan_to_income_ratio", "available_non_null": int(df["loan_to_income_ratio_log1p"].notna().sum())})
    if "interest_rate_x_lti" in feature_cols and "interest_rate_x_lti" not in df.columns:
        df["interest_rate_x_lti"] = interest * pd.Series(lti, index=index)
        status_rows.append({"feature": "interest_rate_x_lti", "action": "rebuilt_from_interest_rate_and_lti", "available_non_null": int(df["interest_rate_x_lti"].notna().sum())})

    # Binary flags. Quantile thresholds are learned from train split only.
    amount_q75 = float(amount.loc[train_mask].quantile(0.75)) if amount.loc[train_mask].notna().any() else np.nan
    income_q25 = float(income.loc[train_mask].quantile(0.25)) if income.loc[train_mask].notna().any() else np.nan
    if "has_prior_delinquency_flag" in feature_cols and "has_prior_delinquency_flag" not in df.columns:
        df["has_prior_delinquency_flag"] = (df.get("delinq_2yrs", delinq).fillna(0).astype(float) > 0).astype("int64")
        status_rows.append({"feature": "has_prior_delinquency_flag", "action": "rebuilt_from_delinq_2yrs", "available_non_null": int(df["has_prior_delinquency_flag"].notna().sum())})
    if "has_existing_loans_flag" in feature_cols and "has_existing_loans_flag" not in df.columns:
        df["has_existing_loans_flag"] = (df.get("number_of_loans", nloans).fillna(0).astype(float) > 0).astype("int64")
        status_rows.append({"feature": "has_existing_loans_flag", "action": "rebuilt_from_number_of_loans", "available_non_null": int(df["has_existing_loans_flag"].notna().sum())})
    if "multiple_loans_flag" in feature_cols and "multiple_loans_flag" not in df.columns:
        df["multiple_loans_flag"] = (df.get("number_of_loans", nloans).fillna(0).astype(float) > 1).astype("int64")
        status_rows.append({"feature": "multiple_loans_flag", "action": "rebuilt_from_number_of_loans", "available_non_null": int(df["multiple_loans_flag"].notna().sum())})
    if "very_high_loan_to_income_flag" in feature_cols and "very_high_loan_to_income_flag" not in df.columns:
        df["very_high_loan_to_income_flag"] = (pd.Series(lti, index=index) >= 4.0).fillna(False).astype("int64")
        status_rows.append({"feature": "very_high_loan_to_income_flag", "action": "rebuilt_from_lti_threshold_4", "available_non_null": int(df["very_high_loan_to_income_flag"].notna().sum())})
    if "low_income_flag" in feature_cols and "low_income_flag" not in df.columns:
        df["low_income_flag"] = (income <= income_q25).fillna(False).astype("int64") if np.isfinite(income_q25) else np.nan
        status_rows.append({"feature": "low_income_flag", "action": "rebuilt_using_train_q25_income", "available_non_null": int(pd.Series(df["low_income_flag"]).notna().sum())})
    if "high_amount_flag" in feature_cols and "high_amount_flag" not in df.columns:
        df["high_amount_flag"] = (amount >= amount_q75).fillna(False).astype("int64") if np.isfinite(amount_q75) else np.nan
        status_rows.append({"feature": "high_amount_flag", "action": "rebuilt_using_train_q75_amount", "available_non_null": int(pd.Series(df["high_amount_flag"]).notna().sum())})

    # Categorical business bands with fixed thresholds.
    if "amount_band" in feature_cols and "amount_band" not in df.columns:
        df["amount_band"] = amount.apply(lambda x: _fixed_band(x, [-np.inf, 25000, 50000, 100000, 250000, np.inf], ["<=25K", "25K-50K", "50K-100K", "100K-250K", ">250K"]))
        status_rows.append({"feature": "amount_band", "action": "rebuilt_from_amount_fixed_bins", "available_non_null": int(df["amount_band"].notna().sum())})
    if "income_band" in feature_cols and "income_band" not in df.columns:
        df["income_band"] = income.apply(lambda x: _fixed_band(x, [-np.inf, 40000, 60000, 90000, 120000, np.inf], ["<=40K", "40K-60K", "60K-90K", "90K-120K", ">120K"]))
        status_rows.append({"feature": "income_band", "action": "rebuilt_from_income_fixed_bins", "available_non_null": int(df["income_band"].notna().sum())})
    if "dependents_band" in feature_cols and "dependents_band" not in df.columns:
        df["dependents_band"] = dependents.apply(lambda x: _fixed_band(x, [-np.inf, 0, 2, np.inf], ["0", "1-2", "3+"]))
        status_rows.append({"feature": "dependents_band", "action": "rebuilt_from_dependents_fixed_bins", "available_non_null": int(df["dependents_band"].notna().sum())})
    if "loan_to_income_band" in feature_cols and "loan_to_income_band" not in df.columns:
        df["loan_to_income_band"] = pd.Series(lti, index=index).apply(lambda x: _fixed_band(x, [-np.inf, 0.5, 1.0, 2.0, 4.0, np.inf], ["<=0.5", "0.5-1.0", "1.0-2.0", "2.0-4.0", ">4.0"]))
        status_rows.append({"feature": "loan_to_income_band", "action": "rebuilt_from_lti_fixed_bins", "available_non_null": int(df["loan_to_income_band"].notna().sum())})
    if "interest_rate_band" in feature_cols and "interest_rate_band" not in df.columns:
        df["interest_rate_band"] = interest.apply(lambda x: _fixed_band(x, [-np.inf, 8, 12, 16, np.inf], ["<=8%", "8%-12%", "12%-16%", ">16%"]))
        status_rows.append({"feature": "interest_rate_band", "action": "rebuilt_from_interest_rate_fixed_bins", "available_non_null": int(df["interest_rate_band"].notna().sum())})

    # Last-resort placeholders. The trained pipeline's train-only imputers will handle them.
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
            status_rows.append({"feature": col, "action": "missing_placeholder_created_for_pipeline_imputation", "available_non_null": 0})

    audit = pd.DataFrame(status_rows)
    if audit.empty:
        audit = pd.DataFrame([{"feature": "all_expected_features", "action": "already_available", "available_non_null": int(len(df))}])
    return df, audit


def get_model_split(
    modeling_df: pd.DataFrame,
    metadata: dict[str, Any],
    split_name: str = "test",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return raw X, target y, and identity columns for one saved modelling split.

    Any missing engineered feature columns expected by the saved Notebook 06 model are
    rebuilt before the split is selected. This prevents Notebook 08 from failing when
    the local modelling dataset is a thinner copy than the one used for training.
    """
    feature_cols = feature_columns_from_metadata(metadata, modeling_df)
    modeling_df, _ = ensure_expected_model_features(modeling_df, feature_cols)
    missing = [col for col in feature_cols if col not in modeling_df.columns]
    if missing:
        raise ValueError(f"Unable to rebuild expected model features: {missing[:20]}")
    if SPLIT_COLUMN not in modeling_df.columns:
        raise ValueError(f"The modelling dataset must include a {SPLIT_COLUMN!r} column.")

    split_df = modeling_df.loc[modeling_df[SPLIT_COLUMN].astype(str).eq(split_name)].copy()
    if split_df.empty:
        raise ValueError(f"No rows found for split={split_name!r}.")

    identity_cols = [c for c in ID_COLUMNS + [TARGET_COLUMN, SPLIT_COLUMN] if c in split_df.columns]
    return split_df[feature_cols].copy(), split_df[TARGET_COLUMN].astype(int).copy(), split_df[identity_cols].copy()


def predict_scores(model: object, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class default probabilities for a fitted classifier or pipeline."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("Champion model must expose predict_proba for explainability and threshold analysis.")
    return np.asarray(model.predict_proba(X))[:, 1]


def classification_metrics_at_threshold(
    y_true: Iterable[int],
    y_score: Iterable[float],
    threshold: float,
    false_negative_cost: float = DEFAULT_FALSE_NEGATIVE_COST,
    false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST,
) -> dict[str, float | int]:
    """Calculate classification, ranking, calibration, and business-cost metrics."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)
    y_pred = (y_score_arr >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_arr, y_pred)),
        "roc_auc": float(roc_auc_score(y_true_arr, y_score_arr)),
        "pr_auc": float(average_precision_score(y_true_arr, y_score_arr)),
        "brier_score": float(brier_score_loss(y_true_arr, y_score_arr)),
        "review_rate": float(y_pred.mean()),
        "business_cost": float(fn * false_negative_cost + fp * false_positive_cost),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "default_count": int(y_true_arr.sum()),
        "non_default_count": int((1 - y_true_arr).sum()),
    }


def confusion_matrix_summary(metrics: dict[str, float | int]) -> pd.DataFrame:
    """Convert confusion-matrix counts into stakeholder-friendly rows."""
    rows = [
        {
            "cell": "True Negative",
            "count": metrics["true_negative"],
            "business_meaning": "Account predicted lower risk and did not default.",
            "stakeholder_impact": "No manual review needed; operational workload is avoided.",
        },
        {
            "cell": "False Positive",
            "count": metrics["false_positive"],
            "business_meaning": "Account predicted high risk but did not default.",
            "stakeholder_impact": "Creates review workload and possible customer friction.",
        },
        {
            "cell": "False Negative",
            "count": metrics["false_negative"],
            "business_meaning": "Account predicted lower risk but later defaulted.",
            "stakeholder_impact": "Potential credit loss because the account was missed by the early-warning rule.",
        },
        {
            "cell": "True Positive",
            "count": metrics["true_positive"],
            "business_meaning": "Account predicted high risk and later defaulted.",
            "stakeholder_impact": "Useful early-warning identification for credit-risk review.",
        },
    ]
    return pd.DataFrame(rows)


def stakeholder_metric_impact_summary(metrics: dict[str, float | int]) -> pd.DataFrame:
    """Explain accuracy, recall, precision, and F1 for business users."""
    return pd.DataFrame(
        [
            {
                "metric": "Accuracy",
                "value": metrics["accuracy"],
                "interpretation": "Share of all accounts classified correctly at the operating threshold.",
                "stakeholder_impact": "Can be misleading when defaults are rare; should not be the primary credit-risk metric.",
            },
            {
                "metric": "Recall",
                "value": metrics["recall"],
                "interpretation": "Share of actual defaulters captured by the high-risk rule.",
                "stakeholder_impact": "Higher recall reduces missed default risk but can increase manual review volume.",
            },
            {
                "metric": "Precision",
                "value": metrics["precision"],
                "interpretation": "Share of reviewed/high-risk accounts that actually defaulted.",
                "stakeholder_impact": "Higher precision improves reviewer efficiency and reduces customer friction.",
            },
            {
                "metric": "F1 Score",
                "value": metrics["f1"],
                "interpretation": "Harmonic mean of precision and recall.",
                "stakeholder_impact": "Useful single-number trade-off metric, but business-cost and review capacity still matter.",
            },
            {
                "metric": "False Negatives",
                "value": metrics["false_negative"],
                "interpretation": "Defaults missed by the model threshold.",
                "stakeholder_impact": "Usually more expensive in credit risk because missed defaults can become losses.",
            },
            {
                "metric": "False Positives",
                "value": metrics["false_positive"],
                "interpretation": "Non-defaulters flagged for review.",
                "stakeholder_impact": "Creates operational cost and can affect customer experience.",
            },
            {
                "metric": "Review Rate",
                "value": metrics["review_rate"],
                "interpretation": "Share of accounts routed to manual/high-risk review.",
                "stakeholder_impact": "Connects model output to staffing capacity and operational feasibility.",
            },
            {
                "metric": "Business Cost",
                "value": metrics["business_cost"],
                "interpretation": "Illustrative cost using false-negative and false-positive assumptions.",
                "stakeholder_impact": "Supports threshold choice and model governance discussion.",
            },
        ]
    )


def _is_generic_feature_names(names: Sequence[str] | None) -> bool:
    """Return True when feature names are only placeholders like feature_0, x0, f0."""
    if not names:
        return True
    generic_prefixes = ("feature_", "x", "f")
    generic_count = 0
    for i, name in enumerate(names):
        value = str(name)
        if value == f"feature_{i}" or value == f"x{i}" or value == f"f{i}":
            generic_count += 1
        elif any(value.startswith(prefix) and value.replace(prefix, "", 1).isdigit() for prefix in generic_prefixes):
            generic_count += 1
    return generic_count == len(names)


def _preprocessor_feature_names(preprocessor: object, X: pd.DataFrame, n_features: int) -> list[str] | None:
    """Best-effort extraction of business-readable post-transform feature names."""
    # ColumnTransformer and most sklearn transformers support input_features.
    for call in (
        lambda: preprocessor.get_feature_names_out(list(X.columns)),
        lambda: preprocessor.get_feature_names_out(),
    ):
        try:
            names = [str(x) for x in list(call())]
            if len(names) == n_features and not _is_generic_feature_names(names):
                return names
        except Exception:
            pass

    # Some fitted ColumnTransformers expose transformers_ even when get_feature_names_out fails.
    try:
        names: list[str] = []
        for transformer_name, transformer, columns in getattr(preprocessor, "transformers_", []):
            if transformer == "drop":
                continue
            if columns is None:
                continue
            if isinstance(columns, slice):
                raw_cols = list(X.columns[columns])
            elif isinstance(columns, (list, tuple, np.ndarray, pd.Index)):
                raw_cols = [str(X.columns[c]) if isinstance(c, (int, np.integer)) else str(c) for c in columns]
            else:
                raw_cols = [str(columns)]

            if transformer == "passthrough":
                names.extend([f"{transformer_name}__{col}" for col in raw_cols])
                continue

            try:
                out = list(transformer.get_feature_names_out(raw_cols))
                names.extend([str(x) for x in out])
            except Exception:
                # If the transformer does not expand columns, keep the raw feature names.
                names.extend([f"{transformer_name}__{col}" for col in raw_cols])
        if len(names) == n_features and not _is_generic_feature_names(names):
            return names
    except Exception:
        pass

    return None


def get_transformed_feature_names(
    pipeline: object,
    X: pd.DataFrame | None = None,
    transformed_n_features: int | None = None,
    fallback_n: int | None = None,
) -> list[str]:
    """Return post-preprocessing feature names without falling back to generic names too early.

    The previous implementation returned feature_0, feature_1, ... whenever sklearn
    could not expose names from the fitted preprocessor. That made Notebook 08
    unacceptable for business explainability. This version first tries sklearn names,
    then uses raw modelling columns when the transformed matrix has the same width as
    the raw feature frame, which is common with ordinal-encoded XGBoost pipelines.
    """
    n_features = transformed_n_features or fallback_n

    if hasattr(pipeline, "named_steps") and "preprocess" in pipeline.named_steps and X is not None and n_features is not None:
        names = _preprocessor_feature_names(pipeline.named_steps["preprocess"], X, int(n_features))
        if names is not None:
            return names

    # Critical business-readable fallback for ordinal/no-expansion preprocessing.
    if X is not None and n_features is not None and int(n_features) == len(X.columns):
        return [str(c) for c in X.columns]

    if n_features is None:
        raise ValueError("Could not infer transformed feature names because feature count is unknown.")
    return [f"feature_{i}" for i in range(int(n_features))]


def transformed_frame(pipeline: object, X: pd.DataFrame) -> pd.DataFrame:
    """Transform raw features into the numerical feature matrix seen by the estimator."""
    if not hasattr(pipeline, "named_steps") or "preprocess" not in pipeline.named_steps:
        return X.copy()
    preprocessor = pipeline.named_steps["preprocess"]
    values = preprocessor.transform(X)
    if hasattr(values, "toarray"):
        values = values.toarray()
    values = np.asarray(values)
    names = get_transformed_feature_names(pipeline, X=X, transformed_n_features=values.shape[1])

    if len(names) != values.shape[1]:
        names = [f"feature_{i}" for i in range(values.shape[1])]

    return pd.DataFrame(values, index=X.index, columns=names)


def raw_feature_from_transformed_name(feature_name: str, raw_feature_names: Sequence[str] | None = None) -> str:
    """Map sklearn-transformed feature names back to raw business features."""
    name = str(feature_name)
    for prefix in ["numeric__", "num__", "remainder__"]:
        if name.startswith(prefix):
            return name.replace(prefix, "", 1)
    for prefix in ["categorical__", "cat__"]:
        if name.startswith(prefix):
            body = name.replace(prefix, "", 1)
            if raw_feature_names:
                for raw in sorted(raw_feature_names, key=len, reverse=True):
                    if body == raw or body.startswith(f"{raw}_"):
                        return raw
            return body
    if raw_feature_names:
        for raw in sorted(raw_feature_names, key=len, reverse=True):
            if name == raw or name.startswith(f"{raw}_") or name.endswith(f"__{raw}"):
                return raw
    return name


def humanize_feature_name(feature_name: str, raw_feature_names: Sequence[str] | None = None) -> str:
    raw = raw_feature_from_transformed_name(feature_name, raw_feature_names)
    return raw.replace("_", " ").title()


def compute_tree_shap_values(pipeline: object, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Tree SHAP values for the estimator inside a fitted sklearn pipeline."""
    if shap is None:
        raise ImportError("The shap package is required for Notebook 08. Install it with: pip install shap")
    X_transformed = transformed_frame(pipeline, X)
    estimator = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else pipeline
    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(X_transformed)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    shap_df = pd.DataFrame(values, index=X.index, columns=X_transformed.columns)
    return shap_df, X_transformed


def summarize_global_shap(shap_df: pd.DataFrame, raw_feature_names: Sequence[str] | None = None) -> pd.DataFrame:
    """Rank transformed features by mean absolute SHAP magnitude."""
    rows: list[dict[str, Any]] = []
    for col in shap_df.columns:
        values = shap_df[col].astype(float)
        raw_feature = raw_feature_from_transformed_name(col, raw_feature_names)
        rows.append(
            {
                "transformed_feature": col,
                "raw_feature": raw_feature,
                "feature_label": raw_feature.replace("_", " ").title(),
                "mean_abs_shap": float(np.abs(values).mean()),
                "mean_shap": float(values.mean()),
                "positive_contribution_share": float((values > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def summarize_grouped_shap(global_importance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one-hot/transformed SHAP values back to raw feature groups."""
    return (
        global_importance.groupby("raw_feature", as_index=False)
        .agg(
            feature_label=("feature_label", "first"),
            mean_abs_shap=("mean_abs_shap", "sum"),
            mean_shap=("mean_shap", "sum"),
            transformed_feature_count=("transformed_feature", "nunique"),
            positive_contribution_share=("positive_contribution_share", "mean"),
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def summarize_probability_deciles(y_true: pd.Series, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    """Summarize model scores and observed default rates by decile."""
    df = pd.DataFrame({"defaulter": y_true.astype(int), "score": scores}, index=y_true.index)
    df["score_decile"] = pd.qcut(df["score"], q=10, duplicates="drop")
    out = (
        df.groupby("score_decile", observed=False)
        .agg(
            row_count=("defaulter", "size"),
            average_score=("score", "mean"),
            min_score=("score", "min"),
            max_score=("score", "max"),
            observed_default_rate=("defaulter", "mean"),
            review_rate_at_threshold=("score", lambda s: float((s >= threshold).mean())),
        )
        .reset_index()
    )
    out["score_decile"] = out["score_decile"].astype(str)
    return out


def select_explanation_sample(
    X: pd.DataFrame,
    y: pd.Series,
    scores: np.ndarray,
    threshold: float,
    max_rows: int = 1500,
    high_risk_rows: int = 250,
    near_threshold_rows: int = 250,
    random_state: int = 42,
) -> list[int]:
    """Select representative, high-risk, near-threshold, FP, and FN rows for explanations."""
    rng = np.random.default_rng(random_state)
    score_series = pd.Series(scores, index=X.index)
    pred = score_series.ge(threshold).astype(int)
    high_risk = score_series.sort_values(ascending=False).head(min(high_risk_rows, len(score_series))).index
    near_threshold = (score_series - threshold).abs().sort_values().head(min(near_threshold_rows, len(score_series))).index
    false_positive = score_series.loc[(pred.eq(1)) & (y.eq(0))].sort_values(ascending=False).head(100).index
    false_negative = score_series.loc[(pred.eq(0)) & (y.eq(1))].sort_values(ascending=False).head(100).index
    selected = pd.Index(high_risk.tolist() + near_threshold.tolist() + false_positive.tolist() + false_negative.tolist()).drop_duplicates()
    remaining = pd.Index(X.index).difference(selected)
    random_n = max(0, min(max_rows - len(selected), len(remaining)))
    if random_n:
        selected = selected.append(pd.Index(rng.choice(remaining.to_numpy(), size=random_n, replace=False)))
    return selected.drop_duplicates().tolist()[:max_rows]


def individual_top_contributions(
    X_raw: pd.DataFrame,
    identity: pd.DataFrame,
    shap_df: pd.DataFrame,
    scores: pd.Series,
    candidate_indices: Iterable[int],
    threshold: float,
    raw_feature_names: Sequence[str] | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return local top positive and negative SHAP drivers for selected accounts."""
    rows: list[dict[str, Any]] = []
    for idx in candidate_indices:
        if idx not in shap_df.index:
            continue
        contribs = shap_df.loc[idx].sort_values(ascending=False)
        positive = contribs.head(top_n)
        negative = contribs.tail(top_n).sort_values()
        identity_row = identity.loc[idx].to_dict() if idx in identity.index else {"row_index": idx}
        rows.append(
            {
                **identity_row,
                "row_index": idx,
                "predicted_default_probability": float(scores.loc[idx]),
                "operating_threshold": float(threshold),
                "predicted_high_risk": int(scores.loc[idx] >= threshold),
                "top_positive_drivers": "; ".join(
                    f"{humanize_feature_name(k, raw_feature_names)} ({v:+.4f})" for k, v in positive.items()
                ),
                "top_negative_drivers": "; ".join(
                    f"{humanize_feature_name(k, raw_feature_names)} ({v:+.4f})" for k, v in negative.items()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_shap_by_segment(
    X_raw: pd.DataFrame,
    shap_df: pd.DataFrame,
    grouped_importance: pd.DataFrame,
    segment_columns: Sequence[str],
    raw_feature_names: Sequence[str] | None = None,
    top_n_features: int = 8,
    min_segment_rows: int = 100,
) -> pd.DataFrame:
    """Aggregate SHAP drivers by business subsegments such as loan category or employment type."""
    top_raw_features = grouped_importance.head(top_n_features)["raw_feature"].tolist()
    raw_map = {col: raw_feature_from_transformed_name(col, raw_feature_names) for col in shap_df.columns}
    rows: list[dict[str, Any]] = []
    for segment_col in segment_columns:
        if segment_col not in X_raw.columns:
            continue
        segment_series = X_raw.loc[shap_df.index, segment_col].fillna("Missing").astype(str)
        for segment_value, segment_idx in segment_series.groupby(segment_series).groups.items():
            idx = pd.Index(segment_idx)
            if len(idx) < min_segment_rows:
                continue
            for raw_feature in top_raw_features:
                transformed_cols = [col for col, raw in raw_map.items() if raw == raw_feature]
                if not transformed_cols:
                    continue
                values = shap_df.loc[idx, transformed_cols].sum(axis=1)
                rows.append(
                    {
                        "segment_column": segment_col,
                        "segment_value": segment_value,
                        "segment_row_count": len(idx),
                        "raw_feature": raw_feature,
                        "feature_label": raw_feature.replace("_", " ").title(),
                        "mean_abs_shap": float(np.abs(values).mean()),
                        "mean_shap": float(values.mean()),
                        "positive_contribution_share": float((values > 0).mean()),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["segment_column", "segment_value", "mean_abs_shap"], ascending=[True, True, False])


def business_regulator_summary_model_insights(grouped_importance: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Translate top SHAP drivers into stakeholder/regulator-friendly language."""
    guidance = {
        "interest_rate": "Pricing/risk-based rate signal. Review for correlation with affordability and risk tiering.",
        "amount": "Exposure size signal. Higher exposure can increase potential loss and affordability pressure.",
        "loan_to_income_ratio": "Affordability signal comparing exposure to income. Strong business-interpretable driver.",
        "total_income_pa": "Capacity-to-pay signal. Use carefully; ensure missingness and income verification are documented.",
        "delinq_2yrs": "Past delinquency signal. Strong credit-risk rationale, but ensure timing is prior to prediction date.",
        "number_of_loans": "Existing obligation signal. Review for relationship to debt burden and product exposure.",
        "loan_category": "Product/portfolio segment signal. Use for monitoring and fairness/proxy review.",
        "employment_type": "Income stability proxy. Requires careful governance and proxy-bias discussion.",
        "home": "Housing/asset stability proxy. Requires fairness and reason-code sensitivity review.",
    }
    rows: list[dict[str, Any]] = []
    for _, row in grouped_importance.head(top_n).iterrows():
        raw = str(row["raw_feature"])
        rows.append(
            {
                "raw_feature": raw,
                "feature_label": str(row["feature_label"]),
                "mean_abs_shap": float(row["mean_abs_shap"]),
                "directional_note": "Positive mean SHAP increases model score" if row["mean_shap"] > 0 else "Negative mean SHAP lowers model score on average",
                "business_interpretation": guidance.get(raw, "Model uses this feature as a predictive signal; validate business rationale and data lineage."),
                "regulatory_governance_note": "Document lineage, timing, proxy-risk assessment, and whether the feature can support explainable adverse-action style communication.",
            }
        )
    return pd.DataFrame(rows)


def plot_global_shap_bar(global_importance: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = global_importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df["feature_label"], plot_df["mean_abs_shap"])
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Top global default-risk drivers")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_shap_beeswarm(shap_df: pd.DataFrame, X_transformed: pd.DataFrame, output_path: Path, max_display: int = 20) -> None:
    if shap is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_df.values, X_transformed, feature_names=X_transformed.columns, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_numeric_shap_dependence(
    shap_df: pd.DataFrame,
    X_raw: pd.DataFrame,
    raw_feature: str,
    output_path: Path,
    raw_feature_names: Sequence[str] | None = None,
) -> None:
    """Save a dependency-style scatter plot for one raw numeric feature."""
    if raw_feature not in X_raw.columns:
        return
    candidate_cols = [col for col in shap_df.columns if raw_feature_from_transformed_name(col, raw_feature_names) == raw_feature]
    if not candidate_cols:
        return
    x = pd.to_numeric(X_raw.loc[shap_df.index, raw_feature], errors="coerce")
    if x.notna().sum() == 0:
        return
    y = shap_df[candidate_cols].sum(axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, s=12, alpha=0.45)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel(raw_feature.replace("_", " ").title())
    ax.set_ylabel("SHAP contribution")
    ax.set_title(f"SHAP dependence: {raw_feature.replace('_', ' ').title()}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_explainability_readiness_gate(status_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a checklist table for Notebook 08 completion."""
    return pd.DataFrame(status_rows)
