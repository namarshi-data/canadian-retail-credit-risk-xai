from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
except Exception:  # pragma: no cover
    shap = None

TARGET_COLUMN = "defaulter"
SPLIT_COLUMN = "split"
ID_COLUMNS = ["user_id", "record_sequence"]


@dataclass
class ExplainabilityArtifacts:
    """Loaded artefacts required for explainability analysis."""

    champion_model: object
    metadata: dict
    modeling_df: pd.DataFrame
    recommended_threshold: float
    champion_model_name: str


def load_explainability_artifacts(
    model_path: Path,
    metadata_path: Path,
    modeling_dataset_path: Path,
    recommended_threshold_path: Path,
) -> ExplainabilityArtifacts:
    """Load model, feature metadata, modelling data, and operating threshold."""
    champion_model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    modeling_df = pd.read_csv(modeling_dataset_path, low_memory=False)
    threshold_df = pd.read_csv(recommended_threshold_path)

    if "threshold" not in threshold_df.columns or threshold_df.empty:
        raise ValueError("recommended_threshold_summary.csv must contain a threshold column.")

    return ExplainabilityArtifacts(
        champion_model=champion_model,
        metadata=metadata,
        modeling_df=modeling_df,
        recommended_threshold=float(threshold_df.iloc[0]["threshold"]),
        champion_model_name=str(metadata.get("champion_model_name", "champion_model")),
    )


def feature_columns_from_metadata(metadata: dict) -> list[str]:
    """Return model feature columns from saved metadata."""
    return list(metadata.get("numeric_features", [])) + list(metadata.get("categorical_features", []))


def get_model_split(
    modeling_df: pd.DataFrame,
    metadata: dict,
    split_name: str = "test",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X, y, and identity columns for a saved modelling split."""
    feature_cols = feature_columns_from_metadata(metadata)
    missing = [col for col in feature_cols if col not in modeling_df.columns]
    if missing:
        raise ValueError(f"Missing expected model features: {missing[:10]}")

    split_df = modeling_df.loc[modeling_df[SPLIT_COLUMN].eq(split_name)].copy()
    if split_df.empty:
        raise ValueError(f"No rows found for split={split_name!r}.")

    X = split_df[feature_cols].copy()
    y = split_df[TARGET_COLUMN].astype(int).copy()
    identity = split_df[ID_COLUMNS + [TARGET_COLUMN, SPLIT_COLUMN]].copy()
    return X, y, identity


def predict_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class default probabilities."""
    return np.asarray(model.predict_proba(X))[:, 1]


def select_explanation_sample(
    X: pd.DataFrame,
    y: pd.Series,
    scores: np.ndarray,
    max_rows: int = 1500,
    high_risk_rows: int = 250,
    random_state: int = 42,
) -> list[int]:
    """Select a representative explanation sample plus high-risk accounts."""
    rng = np.random.default_rng(random_state)
    index_array = np.asarray(X.index)
    score_series = pd.Series(scores, index=X.index)

    high_risk_idx = score_series.sort_values(ascending=False).head(min(high_risk_rows, len(score_series))).index.to_numpy()
    remaining = np.setdiff1d(index_array, high_risk_idx, assume_unique=False)
    random_n = max(0, min(max_rows - len(high_risk_idx), len(remaining)))
    random_idx = rng.choice(remaining, size=random_n, replace=False) if random_n else np.array([], dtype=index_array.dtype)
    return pd.Index(np.concatenate([high_risk_idx, random_idx])).drop_duplicates().tolist()


def get_transformed_feature_names(pipeline) -> list[str]:
    """Return post-preprocessing feature names from a fitted sklearn pipeline."""
    preprocessor = pipeline.named_steps.get("preprocess")
    if preprocessor is None:
        raise ValueError("Pipeline must contain a 'preprocess' step.")
    return list(preprocessor.get_feature_names_out())


def transformed_frame(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Transform raw modelling features into the numeric matrix used by the estimator."""
    preprocessor = pipeline.named_steps["preprocess"]
    values = preprocessor.transform(X)
    if hasattr(values, "toarray"):
        values = values.toarray()
    return pd.DataFrame(values, index=X.index, columns=get_transformed_feature_names(pipeline))


def compute_tree_shap_values(pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute SHAP values for the fitted tree estimator inside the model pipeline."""
    if shap is None:
        raise ImportError("The shap package is required. Install requirements.txt and rerun Notebook 08.")

    X_transformed = transformed_frame(pipeline, X)
    estimator = pipeline.named_steps.get("model")
    if estimator is None:
        raise ValueError("Pipeline must contain a 'model' step.")

    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(X_transformed)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]

    shap_df = pd.DataFrame(values, index=X.index, columns=X_transformed.columns)
    return shap_df, X_transformed


def raw_feature_from_transformed_name(feature_name: str) -> str:
    """Map sklearn transformed names back to raw feature names where possible."""
    for prefix in ["numeric__", "categorical__"]:
        if feature_name.startswith(prefix):
            return feature_name.replace(prefix, "", 1)
    return feature_name


def humanize_feature_name(feature_name: str) -> str:
    """Convert technical feature names into report-friendly labels."""
    raw = raw_feature_from_transformed_name(feature_name)
    return raw.replace("_", " ").title()


def summarize_global_shap(shap_df: pd.DataFrame) -> pd.DataFrame:
    """Return global SHAP importance ranked by mean absolute contribution."""
    rows = []
    for col in shap_df.columns:
        values = shap_df[col].astype(float)
        rows.append(
            {
                "transformed_feature": col,
                "raw_feature": raw_feature_from_transformed_name(col),
                "feature_label": humanize_feature_name(col),
                "mean_abs_shap": float(np.abs(values).mean()),
                "mean_shap": float(values.mean()),
                "positive_contribution_share": float((values > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def summarize_grouped_shap(global_importance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transformed features back to raw business feature groups."""
    return (
        global_importance.groupby("raw_feature", as_index=False)
        .agg(
            feature_label=("feature_label", "first"),
            mean_abs_shap=("mean_abs_shap", "sum"),
            mean_shap=("mean_shap", "sum"),
            transformed_feature_count=("transformed_feature", "nunique"),
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def summarize_probability_deciles(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Summarize observed default rate by model-score decile."""
    df = pd.DataFrame({"defaulter": y_true.astype(int), "score": scores}, index=y_true.index)
    df["score_decile"] = pd.qcut(df["score"], q=10, duplicates="drop")
    summary = (
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
    summary["score_decile"] = summary["score_decile"].astype(str)
    return summary


def individual_top_contributions(
    X_raw: pd.DataFrame,
    identity: pd.DataFrame,
    shap_df: pd.DataFrame,
    scores: pd.Series,
    candidate_indices: Iterable[int],
    threshold: float,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return top positive and negative SHAP drivers for selected accounts."""
    rows = []
    for idx in candidate_indices:
        if idx not in shap_df.index:
            continue
        contribs = shap_df.loc[idx].sort_values(ascending=False)
        positive = contribs.head(top_n)
        negative = contribs.tail(top_n).sort_values()
        identity_row = identity.loc[idx].to_dict() if idx in identity.index else {}
        rows.append(
            {
                **identity_row,
                "predicted_default_probability": float(scores.loc[idx]),
                "operating_threshold": float(threshold),
                "predicted_high_risk": int(scores.loc[idx] >= threshold),
                "top_positive_drivers": "; ".join(f"{humanize_feature_name(k)} ({v:+.4f})" for k, v in positive.items()),
                "top_negative_drivers": "; ".join(f"{humanize_feature_name(k)} ({v:+.4f})" for k, v in negative.items()),
            }
        )
    return pd.DataFrame(rows)


def plot_global_shap_bar(global_importance: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    """Save a horizontal bar chart of top global SHAP features."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = global_importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df["feature_label"], plot_df["mean_abs_shap"])
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Top global default-risk drivers")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_numeric_shap_dependence(
    shap_df: pd.DataFrame,
    X_raw: pd.DataFrame,
    raw_feature: str,
    output_path: Path,
) -> None:
    """Save a dependence-style scatter plot for one numeric raw feature."""
    transformed_name = f"numeric__{raw_feature}"
    if transformed_name not in shap_df.columns or raw_feature not in X_raw.columns:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X_raw[raw_feature], shap_df[transformed_name], s=12, alpha=0.45)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel(humanize_feature_name(raw_feature))
    ax.set_ylabel("SHAP contribution")
    ax.set_title(f"SHAP dependence: {humanize_feature_name(raw_feature)}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
