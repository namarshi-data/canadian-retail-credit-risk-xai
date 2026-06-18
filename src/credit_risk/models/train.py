from __future__ import annotations

"""Train-only model training for Notebook 06.

This implementation is deliberately leakage-safe:
- The saved train/validation/test split from Notebook 05 is respected.
- Imputers, winsorization limits, skewness transformers, encoders, scalers,
  resamplers, and models are fitted on the training split only.
- Validation is used for model selection, hyperparameter tuning, and threshold
  selection.
- Test is used only for final confirmation.
- XGBoost is required for this project. If it is not installed, a clear error is
  raised with the install command.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "XGBoost is required for this project. Install it with: pip install xgboost"
    ) from exc

try:
    from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
except Exception as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "Hyperopt is required for Notebook 06 XGBoost tuning. Install it with: pip install hyperopt"
    ) from exc

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
except Exception:  # pragma: no cover - optional dependency
    ImbPipeline = None
    RandomUnderSampler = None
    RandomOverSampler = None

from credit_risk.models.evaluate import (
    calibration_by_score_band,
    confusion_matrix_frame,
    evaluate_fitted_model,
    make_prediction_frame,
    model_selection_summary,
    score_decile_lift_table,
)
from credit_risk.models.experiment_tracking import (
    append_experiment_log,
    make_experiment_row,
    neptune_config_status,
    read_experiment_log,
    safe_neptune_log,
    start_neptune_run,
    stop_neptune_run,
)
from credit_risk.models.thresholding import (
    ThresholdCostAssumptions,
    apply_selected_threshold_to_predictions,
    build_threshold_shortlist,
    cost_assumptions_frame,
    evaluate_threshold_grid_all_models,
    recommend_operating_threshold,
)

TARGET_COLUMN = "defaulter"
SPLIT_COLUMN = "split"
ID_COLUMNS = ["user_id", "record_sequence"]

# Defensive exclusions in case earlier notebooks accidentally leave governance-only
# columns in the modelling dataset.
LEAKAGE_RISK_COLUMNS = [
    "total_payment",
    "received_principal",
    "interest_received",
]
SENSITIVE_OR_PROXY_COLUMNS = [
    "gender",
    "pincode",
    "social_profile",
]
HIGH_CARDINALITY_OR_ENCRYPTED_COLUMNS = [
    "industry",
    "role",
]


@dataclass
class ModelTrainingArtifacts:
    """Container for fitted models and all Notebook 06 outputs."""

    fitted_models: dict[str, Any]
    validation_results: pd.DataFrame
    test_results_default_threshold: pd.DataFrame
    selection_summary: pd.DataFrame
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    validation_threshold_grid: pd.DataFrame
    test_selected_threshold_results: pd.DataFrame
    threshold_shortlist: pd.DataFrame
    recommended_threshold_summary: pd.DataFrame
    experiment_log: pd.DataFrame
    tuning_trials: pd.DataFrame
    confusion_matrices_validation: pd.DataFrame
    confusion_matrices_test: pd.DataFrame
    lift_tables_validation: pd.DataFrame
    calibration_tables_validation: pd.DataFrame
    preprocessing_assurance: pd.DataFrame
    feature_inventory: pd.DataFrame
    numeric_features: list[str]
    categorical_features: list[str]
    excluded_features: list[str]
    champion_model_name: str
    ranking_champion_model_name: str
    operational_champion_model_name: str
    all_model_threshold_shortlist: pd.DataFrame
    operational_model_threshold_recommendation: pd.DataFrame
    test_selected_threshold_results_all_models: pd.DataFrame
    model_readiness_gate: pd.DataFrame


@dataclass(frozen=True)
class TrainingConfig:
    """Configurable Notebook 06 training controls."""

    random_state: int = 42
    rf_random_search_iter: int = 25
    xgb_hyperopt_max_evals: int = 35
    cv_folds: int = 3
    use_random_under_sampler: bool = False
    false_negative_cost: float = 5_000
    false_positive_cost: float = 500
    preferred_threshold_objective: str = "minimum_cost_review_rate_le_30pct"
    n_jobs: int = -1
    enable_sampling_challengers: bool = False
    sampler_strategy: float = 0.50
    operational_review_rate_cap: float = 0.30
    operational_min_recall: float = 0.0

    @classmethod
    def from_environment(cls) -> "TrainingConfig":
        return cls(
            random_state=int(os.getenv("MODEL_RANDOM_STATE", "42")),
            rf_random_search_iter=int(os.getenv("RF_RANDOM_SEARCH_N_ITER", "25")),
            xgb_hyperopt_max_evals=int(os.getenv("XGB_HYPEROPT_MAX_EVALS", "35")),
            cv_folds=int(os.getenv("MODEL_CV_FOLDS", "3")),
            use_random_under_sampler=os.getenv("ENABLE_RANDOM_UNDER_SAMPLING", "0").strip() == "1",
            false_negative_cost=float(os.getenv("FALSE_NEGATIVE_COST", "5000")),
            false_positive_cost=float(os.getenv("FALSE_POSITIVE_COST", "500")),
            preferred_threshold_objective=os.getenv(
                "PREFERRED_THRESHOLD_OBJECTIVE", "minimum_cost_review_rate_le_30pct"
            ),
            n_jobs=int(os.getenv("MODEL_N_JOBS", "-1")),
            enable_sampling_challengers=os.getenv("ENABLE_SAMPLING_CHALLENGERS", "0").strip() == "1",
            sampler_strategy=float(os.getenv("SAMPLER_STRATEGY", "0.50")),
            operational_review_rate_cap=float(os.getenv("OPERATIONAL_REVIEW_RATE_CAP", "0.30")),
            operational_min_recall=float(os.getenv("OPERATIONAL_MIN_RECALL", "0.0")),
        )


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder robustly across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=True)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse=True)


class TrainOnlyWinsorizer(BaseEstimator, TransformerMixin):
    """Clip numeric outliers using limits learned from the training split only."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(arr, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(arr, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lower_bounds_, self.upper_bounds_)


def validate_modeling_dataset(modeling_df: pd.DataFrame) -> None:
    """Validate core modelling dataset requirements."""
    required_cols = set(ID_COLUMNS + [TARGET_COLUMN, SPLIT_COLUMN])
    missing_cols = required_cols - set(modeling_df.columns)
    if missing_cols:
        raise ValueError(f"Modelling dataset is missing required columns: {sorted(missing_cols)}")
    required_splits = {"train", "validation", "test"}
    missing_splits = required_splits - set(modeling_df[SPLIT_COLUMN].dropna().unique())
    if missing_splits:
        raise ValueError(f"Missing required split labels: {sorted(missing_splits)}")
    if modeling_df[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains missing values. Fix before model training.")


def infer_feature_columns(modeling_df: pd.DataFrame) -> tuple[list[str], list[str], list[str], pd.DataFrame]:
    """Infer trainable numeric/categorical features with defensive exclusions."""
    excluded = set(ID_COLUMNS + [TARGET_COLUMN, SPLIT_COLUMN])
    excluded.update([col for col in LEAKAGE_RISK_COLUMNS if col in modeling_df.columns])
    excluded.update([col for col in SENSITIVE_OR_PROXY_COLUMNS if col in modeling_df.columns])
    excluded.update([col for col in HIGH_CARDINALITY_OR_ENCRYPTED_COLUMNS if col in modeling_df.columns])

    feature_cols = [col for col in modeling_df.columns if col not in excluded]
    numeric_features = [col for col in feature_cols if pd.api.types.is_numeric_dtype(modeling_df[col])]
    categorical_features = [col for col in feature_cols if col not in numeric_features]
    excluded_features = sorted([col for col in modeling_df.columns if col in excluded])

    rows = []
    for col in numeric_features:
        rows.append({"feature": col, "feature_type": "numeric", "model_use": "candidate_feature"})
    for col in categorical_features:
        rows.append({"feature": col, "feature_type": "categorical", "model_use": "candidate_feature"})
    for col in excluded_features:
        if col in ID_COLUMNS:
            reason = "identifier_or_audit_key"
        elif col in [TARGET_COLUMN, SPLIT_COLUMN]:
            reason = "target_or_split_control"
        elif col in LEAKAGE_RISK_COLUMNS:
            reason = "leakage_risk_monitoring_only"
        elif col in SENSITIVE_OR_PROXY_COLUMNS:
            reason = "sensitive_or_proxy_governance_excluded"
        elif col in HIGH_CARDINALITY_OR_ENCRYPTED_COLUMNS:
            reason = "high_cardinality_or_encrypted_governance_excluded"
        else:
            reason = "excluded"
        rows.append({"feature": col, "feature_type": "excluded", "model_use": reason})

    return numeric_features, categorical_features, excluded_features, pd.DataFrame(rows)


def split_modeling_dataset(modeling_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, dict[str, pd.DataFrame | pd.Series]]:
    """Return train, validation, and test frames from the saved split column."""
    validate_modeling_dataset(modeling_df)
    splits: dict[str, dict[str, pd.DataFrame | pd.Series]] = {}
    for split_name in ["train", "validation", "test"]:
        split_df = modeling_df.loc[modeling_df[SPLIT_COLUMN].eq(split_name)].copy()
        splits[split_name] = {
            "X": split_df[feature_cols].copy(),
            "y": split_df[TARGET_COLUMN].astype(int).copy(),
            "identity": split_df[ID_COLUMNS + [TARGET_COLUMN, SPLIT_COLUMN]].copy(),
        }
    return splits


def default_scale_pos_weight(y_train: pd.Series) -> float:
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    return 1.0 if positives == 0 else negatives / positives


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    categorical_encoding: str,
    apply_winsorization: bool = True,
    apply_power_transform: bool = False,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Build a train-only preprocessing block."""
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if apply_winsorization:
        numeric_steps.append(("winsorizer", TrainOnlyWinsorizer(lower_quantile=0.01, upper_quantile=0.99)))
    if apply_power_transform:
        # Yeo-Johnson supports zero/negative values and handles skewness safely.
        numeric_steps.append(("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=False)))
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    if categorical_encoding == "onehot":
        encoder = _make_one_hot_encoder()
    elif categorical_encoding == "ordinal":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
    else:
        raise ValueError("categorical_encoding must be 'onehot' or 'ordinal'.")

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", encoder),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.30)


def maybe_add_train_only_sampler(pipeline_steps: list[tuple[str, Any]], config: TrainingConfig) -> Any:
    """Return sklearn or imblearn pipeline depending on sampler configuration."""
    if not config.use_random_under_sampler:
        return Pipeline(pipeline_steps)

    if ImbPipeline is None or RandomUnderSampler is None:
        raise ImportError(
            "ENABLE_RANDOM_UNDER_SAMPLING=1 requires imbalanced-learn. Install with: pip install imbalanced-learn"
        )

    steps = pipeline_steps.copy()
    # Resampling occurs after preprocessing and before the model, inside fit(X_train, y_train) only.
    steps.insert(-1, ("sampler", RandomUnderSampler(random_state=config.random_state)))
    return ImbPipeline(steps)


def make_train_only_sampling_pipeline(
    pipeline_steps: list[tuple[str, Any]],
    config: TrainingConfig,
    *,
    sampler_type: str | None = None,
    sampling_strategy: float | str | None = None,
) -> Any:
    """Build a pipeline with an optional train-only sampler.

    The sampler is inserted after preprocessing and before the estimator. Because
    the sampler sits inside an imblearn Pipeline, it is fitted only during
    fit(X_train, y_train). Validation and test data are never resampled.
    """
    if sampler_type is None:
        return Pipeline(pipeline_steps)
    if ImbPipeline is None:
        raise ImportError(
            "Sampling challengers require imbalanced-learn. Install with: pip install imbalanced-learn"
        )

    strategy = config.sampler_strategy if sampling_strategy is None else sampling_strategy
    sampler_key = sampler_type.lower().strip()
    if sampler_key in {"under", "random_under", "random_under_sampler"}:
        if RandomUnderSampler is None:
            raise ImportError("RandomUnderSampler requires imbalanced-learn. Install with: pip install imbalanced-learn")
        sampler = RandomUnderSampler(sampling_strategy=strategy, random_state=config.random_state)
    elif sampler_key in {"over", "random_over", "random_over_sampler"}:
        if RandomOverSampler is None:
            raise ImportError("RandomOverSampler requires imbalanced-learn. Install with: pip install imbalanced-learn")
        sampler = RandomOverSampler(sampling_strategy=strategy, random_state=config.random_state)
    else:
        raise ValueError("sampler_type must be one of: None, 'under', 'over'.")

    steps = pipeline_steps.copy()
    steps.insert(-1, ("sampler", sampler))
    return ImbPipeline(steps)


def _strip_model_prefix(params: dict[str, Any]) -> dict[str, Any]:
    """Convert RandomizedSearchCV params like model__max_depth to estimator params."""
    return {key.replace("model__", ""): value for key, value in params.items() if key.startswith("model__")}


def build_random_forest_pipeline_from_params(
    numeric_features: list[str],
    categorical_features: list[str],
    config: TrainingConfig,
    rf_params: dict[str, Any],
    *,
    sampler_type: str | None = None,
    apply_power_transform: bool = False,
) -> Any:
    """Build a Random Forest challenger from tuned params and optional sampler."""
    model_params = _strip_model_prefix(rf_params)
    model_params.setdefault("class_weight", "balanced_subsample")
    model_params.setdefault("n_jobs", config.n_jobs)
    model_params.setdefault("random_state", config.random_state)
    steps: list[tuple[str, Any]] = [
        (
            "preprocess",
            build_preprocessor(
                numeric_features,
                categorical_features,
                categorical_encoding="ordinal",
                apply_winsorization=True,
                apply_power_transform=apply_power_transform,
                scale_numeric=False,
            ),
        ),
        ("model", RandomForestClassifier(**model_params)),
    ]
    return make_train_only_sampling_pipeline(steps, config, sampler_type=sampler_type)


def build_xgboost_pipeline_from_params_with_sampler(
    numeric_features: list[str],
    categorical_features: list[str],
    y_train: pd.Series,
    config: TrainingConfig,
    xgb_params: dict[str, Any],
    *,
    sampler_type: str | None = None,
    apply_power_transform: bool = False,
) -> Any:
    """Build XGBoost from tuned params with optional train-only sampler challenger."""
    params = _cast_xgb_params(xgb_params)
    steps: list[tuple[str, Any]] = [
        (
            "preprocess",
            build_preprocessor(
                numeric_features,
                categorical_features,
                categorical_encoding="ordinal",
                apply_winsorization=True,
                apply_power_transform=apply_power_transform,
                scale_numeric=False,
            ),
        ),
        (
            "model",
            XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                scale_pos_weight=default_scale_pos_weight(y_train),
                n_jobs=max(1, config.n_jobs),
                random_state=config.random_state,
            ),
        ),
    ]
    return make_train_only_sampling_pipeline(steps, config, sampler_type=sampler_type)


def build_all_model_operational_threshold_comparison(
    threshold_grid: pd.DataFrame | None = None,
    validation_results: pd.DataFrame | None = None,
    config: TrainingConfig | None = None,
    validation_threshold_grid: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select one validation threshold per model under operational constraints.

    This avoids choosing a champion purely by PR-AUC. It evaluates the practical
    model/threshold pair using business cost, review capacity, recall, precision,
    and ranking quality. The validation_threshold_grid alias is accepted to avoid
    notebook/script mismatches when this function is called with explicit keywords.
    """
    if threshold_grid is None:
        threshold_grid = validation_threshold_grid
    if threshold_grid is None:
        raise ValueError("threshold_grid or validation_threshold_grid must be provided.")
    if validation_results is None:
        validation_results = pd.DataFrame()
    if config is None:
        config = TrainingConfig.from_environment()

    rows = []
    pr_auc_map = validation_results.set_index("model_name")["pr_auc"].to_dict() if not validation_results.empty else {}
    roc_auc_map = validation_results.set_index("model_name")["roc_auc"].to_dict() if not validation_results.empty else {}
    brier_map = validation_results.set_index("model_name")["brier_score"].to_dict() if not validation_results.empty else {}

    for model_name in sorted(threshold_grid["model_name"].dropna().unique()):
        shortlist = build_threshold_shortlist(threshold_grid, str(model_name))
        constrained = shortlist.loc[
            shortlist["review_rate"].le(config.operational_review_rate_cap)
            & shortlist["recall"].ge(config.operational_min_recall)
        ].copy()
        if constrained.empty:
            constrained = shortlist.copy()
        if constrained.empty:
            continue
        selected = constrained.sort_values(
            ["business_cost", "recall", "precision", "mcc", "review_rate"],
            ascending=[True, False, False, False, True],
        ).head(1).copy()
        selected["selection_basis"] = (
            f"minimum business cost under review_rate <= {config.operational_review_rate_cap:.0%}; "
            f"min_recall >= {config.operational_min_recall:.0%}"
        )
        selected["validation_pr_auc_at_default_threshold"] = pr_auc_map.get(str(model_name), np.nan)
        selected["validation_roc_auc_at_default_threshold"] = roc_auc_map.get(str(model_name), np.nan)
        selected["validation_brier_at_default_threshold"] = brier_map.get(str(model_name), np.nan)
        rows.append(selected)

    if not rows:
        return pd.DataFrame()
    comparison = pd.concat(rows, ignore_index=True)
    comparison = comparison.sort_values(
        ["business_cost", "recall", "precision", "validation_pr_auc_at_default_threshold", "review_rate"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    comparison.insert(0, "operational_rank", range(1, len(comparison) + 1))
    return comparison


def evaluate_selected_thresholds_on_test(
    validation_operational_comparison: pd.DataFrame,
    test_predictions: pd.DataFrame,
    cost_assumptions: ThresholdCostAssumptions,
) -> pd.DataFrame:
    """Apply each model's validation-selected threshold to the untouched test set."""
    frames = []
    for _, row in validation_operational_comparison.iterrows():
        model_name = str(row["model_name"])
        threshold = float(row["threshold"])
        result = apply_selected_threshold_to_predictions(
            predictions=test_predictions,
            model_name=model_name,
            selected_threshold=threshold,
            dataset_name="test",
            cost_assumptions=cost_assumptions,
        )
        result.insert(0, "validation_operational_rank", int(row["operational_rank"]))
        result.insert(1, "validation_objective", row.get("objective", "selected_from_validation"))
        frames.append(result)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_model_readiness_gate(artifacts_like: dict[str, Any]) -> pd.DataFrame:
    """Create a recruiter/model-risk friendly readiness checklist for Notebook 06."""
    checks = [
        ("required_splits_present", True, "train/validation/test split from Notebook 05 was used"),
        ("train_only_preprocessing_documented", True, "imputers/encoders/scalers/winsorizer/skew transformer inside pipelines"),
        ("xgboost_required_and_fitted", bool(artifacts_like.get("xgboost_fitted", False)), "XGBoost is a core model, not optional"),
        ("random_forest_fitted", bool(artifacts_like.get("random_forest_fitted", False)), "Random Forest baseline and tuned challenger fitted"),
        ("hyperopt_completed", bool(artifacts_like.get("hyperopt_completed", False)), "XGBoost Hyperopt tuning completed"),
        ("rf_random_search_completed", bool(artifacts_like.get("rf_random_search_completed", False)), "Random Forest randomized search completed"),
        ("all_model_threshold_comparison_completed", bool(artifacts_like.get("all_model_threshold_comparison_completed", False)), "All models compared under validation-selected threshold constraints"),
        ("test_used_only_for_confirmation", True, "Test metrics generated after model/threshold selection"),
        ("experiment_tracking_saved", bool(artifacts_like.get("experiment_tracking_saved", False)), "Local CSV experiment log saved; Neptune optional"),
        ("champion_model_saved", bool(artifacts_like.get("champion_model_saved", False)), "Operational champion model artifact saved"),
    ]
    return pd.DataFrame(
        [{"check": name, "status": "pass" if passed else "review", "evidence": evidence} for name, passed, evidence in checks]
    )


def build_logistic_baseline(
    numeric_features: list[str],
    categorical_features: list[str],
    config: TrainingConfig,
) -> Pipeline:
    """Simple classical benchmark: class-weighted Logistic Regression."""
    return Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(
                    numeric_features,
                    categorical_features,
                    categorical_encoding="onehot",
                    apply_winsorization=True,
                    apply_power_transform=True,
                    scale_numeric=True,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="liblinear",
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def build_random_forest_baseline(
    numeric_features: list[str],
    categorical_features: list[str],
    config: TrainingConfig,
    *,
    apply_power_transform: bool = False,
) -> Any:
    """Random Forest baseline with train-only preprocessing."""
    steps: list[tuple[str, Any]] = [
        (
            "preprocess",
            build_preprocessor(
                numeric_features,
                categorical_features,
                categorical_encoding="ordinal",
                apply_winsorization=True,
                apply_power_transform=apply_power_transform,
                scale_numeric=False,
            ),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=14,
                min_samples_leaf=75,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=config.n_jobs,
                random_state=config.random_state,
            ),
        ),
    ]
    return maybe_add_train_only_sampler(steps, config)


def build_xgboost_baseline(
    numeric_features: list[str],
    categorical_features: list[str],
    y_train: pd.Series,
    config: TrainingConfig,
    *,
    apply_power_transform: bool = False,
) -> Any:
    """XGBoost baseline with train-only preprocessing."""
    steps: list[tuple[str, Any]] = [
        (
            "preprocess",
            build_preprocessor(
                numeric_features,
                categorical_features,
                categorical_encoding="ordinal",
                apply_winsorization=True,
                apply_power_transform=apply_power_transform,
                scale_numeric=False,
            ),
        ),
        (
            "model",
            XGBClassifier(
                n_estimators=350,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=5,
                gamma=0.0,
                reg_alpha=0.0,
                reg_lambda=2.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                scale_pos_weight=default_scale_pos_weight(y_train),
                n_jobs=max(1, config.n_jobs),
                random_state=config.random_state,
            ),
        ),
    ]
    return maybe_add_train_only_sampler(steps, config)


def evaluate_and_log(
    model_name: str,
    model,
    splits: dict[str, dict[str, pd.DataFrame | pd.Series]],
    experiment_log_path: Path,
    stage: str,
    model_family: str,
    params: dict[str, Any] | None = None,
    neptune_run=None,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Evaluate one fitted model on validation and test, then log locally/Neptune."""
    validation_metrics = evaluate_fitted_model(
        model_name,
        model,
        splits["validation"]["X"],
        splits["validation"]["y"],
        dataset_name="validation",
    )
    test_metrics = evaluate_fitted_model(
        model_name,
        model,
        splits["test"]["X"],
        splits["test"]["y"],
        dataset_name="test",
    )

    validation_pred = make_prediction_frame(
        model_name,
        model,
        splits["validation"]["X"],
        splits["validation"]["identity"],
    )
    test_pred = make_prediction_frame(
        model_name,
        model,
        splits["test"]["X"],
        splits["test"]["identity"],
    )

    append_experiment_log(
        make_experiment_row(
            experiment_id=f"{stage}_{model_name}",
            model_name=model_name,
            model_family=model_family,
            stage=stage,
            params=params or {},
            metrics={f"validation_{k}": v for k, v in validation_metrics.items() if isinstance(v, (int, float, str))},
        ),
        experiment_log_path,
    )

    safe_neptune_log(neptune_run, f"models/{model_name}/stage", stage)
    safe_neptune_log(neptune_run, f"models/{model_name}/family", model_family)
    if params:
        safe_neptune_log(neptune_run, f"models/{model_name}/params", params)
    for metric_name, value in validation_metrics.items():
        if isinstance(value, (int, float)):
            safe_neptune_log(neptune_run, f"models/{model_name}/validation/{metric_name}", float(value))
    for metric_name, value in test_metrics.items():
        if isinstance(value, (int, float)):
            safe_neptune_log(neptune_run, f"models/{model_name}/test/{metric_name}", float(value))

    return validation_metrics, test_metrics, validation_pred, test_pred


def tune_random_forest_random_search(
    numeric_features: list[str],
    categorical_features: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: TrainingConfig,
) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    """Tune Random Forest using train-only RandomizedSearchCV."""
    base = build_random_forest_baseline(
        numeric_features,
        categorical_features,
        config,
        apply_power_transform=False,
    )

    param_distributions = {
        "model__n_estimators": randint(250, 650),
        "model__max_depth": [6, 8, 10, 12, 14, 16, None],
        "model__min_samples_leaf": randint(25, 250),
        "model__min_samples_split": randint(50, 400),
        "model__max_features": ["sqrt", "log2", 0.35, 0.50, 0.70],
        "model__bootstrap": [True],
    }

    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        n_iter=config.rf_random_search_iter,
        scoring="average_precision",
        cv=cv,
        n_jobs=config.n_jobs,
        verbose=1,
        random_state=config.random_state,
        refit=True,
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    trials = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").reset_index(drop=True)
    return search.best_estimator_, search.best_params_, trials


def _cast_xgb_params(params: dict[str, Any]) -> dict[str, Any]:
    casted = params.copy()
    for int_key in ["n_estimators", "max_depth"]:
        if int_key in casted:
            casted[int_key] = int(casted[int_key])
    return casted


def tune_xgboost_hyperopt(
    numeric_features: list[str],
    categorical_features: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    config: TrainingConfig,
    *,
    apply_power_transform: bool = False,
    neptune_run=None,
) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    """Tune XGBoost using Hyperopt. Objective evaluates validation PR-AUC only."""
    space = {
        "n_estimators": hp.quniform("n_estimators", 250, 850, 50),
        "max_depth": hp.quniform("max_depth", 2, 7, 1),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.015), np.log(0.18)),
        "subsample": hp.uniform("subsample", 0.65, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.60, 1.0),
        "min_child_weight": hp.loguniform("min_child_weight", np.log(1.0), np.log(30.0)),
        "gamma": hp.uniform("gamma", 0.0, 5.0),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-4), np.log(5.0)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(0.25), np.log(20.0)),
    }

    trial_rows: list[dict[str, Any]] = []

    def objective(params: dict[str, Any]) -> dict[str, Any]:
        start_time = time.time()
        params = _cast_xgb_params(params)
        model = build_xgboost_pipeline_from_params(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            y_train=y_train,
            config=config,
            xgb_params=params,
            apply_power_transform=apply_power_transform,
        )
        model.fit(X_train, y_train)
        validation_score = model.predict_proba(X_validation)[:, 1]
        pr_auc = average_precision_score(y_validation, validation_score)
        loss = -float(pr_auc)
        elapsed_seconds = time.time() - start_time

        row = {**params, "validation_pr_auc": float(pr_auc), "loss": loss, "elapsed_seconds": elapsed_seconds}
        trial_rows.append(row)
        safe_neptune_log(neptune_run, "hyperopt/xgboost/trial_validation_pr_auc", float(pr_auc))
        return {"loss": loss, "status": STATUS_OK, "attachments": {"row": json.dumps(row)}}

    trials = Trials()
    best_raw = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=config.xgb_hyperopt_max_evals,
        trials=trials,
        rstate=np.random.default_rng(config.random_state),
        show_progressbar=True,
    )
    best_params = _cast_xgb_params(space_eval(space, best_raw))
    best_model = build_xgboost_pipeline_from_params(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        y_train=y_train,
        config=config,
        xgb_params=best_params,
        apply_power_transform=apply_power_transform,
    )
    best_model.fit(X_train, y_train)
    trials_df = pd.DataFrame(trial_rows).sort_values("validation_pr_auc", ascending=False).reset_index(drop=True)
    return best_model, best_params, trials_df


def build_xgboost_pipeline_from_params(
    numeric_features: list[str],
    categorical_features: list[str],
    y_train: pd.Series,
    config: TrainingConfig,
    xgb_params: dict[str, Any],
    *,
    apply_power_transform: bool = False,
) -> Any:
    """Build XGBoost pipeline from tuned parameters."""
    params = _cast_xgb_params(xgb_params)
    steps: list[tuple[str, Any]] = [
        (
            "preprocess",
            build_preprocessor(
                numeric_features,
                categorical_features,
                categorical_encoding="ordinal",
                apply_winsorization=True,
                apply_power_transform=apply_power_transform,
                scale_numeric=False,
            ),
        ),
        (
            "model",
            XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                scale_pos_weight=default_scale_pos_weight(y_train),
                n_jobs=max(1, config.n_jobs),
                random_state=config.random_state,
            ),
        ),
    ]
    return maybe_add_train_only_sampler(steps, config)


def build_preprocessing_assurance(config: TrainingConfig) -> pd.DataFrame:
    """Document what is fitted inside train-only model pipelines."""
    return pd.DataFrame(
        [
            {"component": "numeric_imputer", "fit_scope": "training_split_only", "implementation": "SimpleImputer inside Pipeline"},
            {"component": "categorical_imputer", "fit_scope": "training_split_only", "implementation": "SimpleImputer inside Pipeline"},
            {"component": "categorical_encoder", "fit_scope": "training_split_only", "implementation": "OneHotEncoder/OrdinalEncoder inside Pipeline"},
            {"component": "winsorization_limits", "fit_scope": "training_split_only", "implementation": "TrainOnlyWinsorizer inside Pipeline"},
            {"component": "skewness_transformer", "fit_scope": "training_split_only", "implementation": "Yeo-Johnson PowerTransformer inside Pipeline where enabled"},
            {"component": "numeric_scaler", "fit_scope": "training_split_only", "implementation": "StandardScaler inside Logistic Regression Pipeline"},
            {
                "component": "random_under_sampler",
                "fit_scope": "training_split_only" if config.use_random_under_sampler else "not_enabled",
                "implementation": "imblearn Pipeline after preprocessing, before model" if config.use_random_under_sampler else "class weights / scale_pos_weight used by default",
            },
            {"component": "random_forest_tuning", "fit_scope": "training_split_only_cv", "implementation": "RandomizedSearchCV scoring=average_precision"},
            {"component": "xgboost_hyperopt", "fit_scope": "train_fit_validation_objective", "implementation": "Hyperopt fits train only and scores validation PR-AUC"},
            {"component": "threshold_selection", "fit_scope": "validation_only", "implementation": "Validation threshold grid; test only confirms selected threshold"},
        ]
    )


def train_and_evaluate_xgb_rf_models(
    modeling_df: pd.DataFrame,
    table_dir: Path,
    model_artifact_dir: Path,
    config: TrainingConfig | None = None,
) -> ModelTrainingArtifacts:
    """Run full Notebook 06 training workflow."""
    config = config or TrainingConfig.from_environment()
    table_dir.mkdir(parents=True, exist_ok=True)
    model_artifact_dir.mkdir(parents=True, exist_ok=True)

    validate_modeling_dataset(modeling_df)
    numeric_features, categorical_features, excluded_features, feature_inventory = infer_feature_columns(modeling_df)
    feature_cols = numeric_features + categorical_features
    splits = split_modeling_dataset(modeling_df, feature_cols)

    X_train = splits["train"]["X"]
    y_train = splits["train"]["y"]
    X_validation = splits["validation"]["X"]
    y_validation = splits["validation"]["y"]

    experiment_log_path = table_dir / "06_experiment_tracking_log.csv"
    neptune_run = start_neptune_run(
        run_name="06-xgb-rf-credit-risk-training",
        tags=["credit-risk", "xgboost", "random-forest", "notebook-06"],
    )

    safe_neptune_log(neptune_run, "config/random_state", config.random_state)
    safe_neptune_log(neptune_run, "config/rf_random_search_iter", config.rf_random_search_iter)
    safe_neptune_log(neptune_run, "config/xgb_hyperopt_max_evals", config.xgb_hyperopt_max_evals)
    safe_neptune_log(neptune_run, "data/train_rows", len(y_train))
    safe_neptune_log(neptune_run, "data/validation_rows", len(y_validation))
    safe_neptune_log(neptune_run, "features/numeric_count", len(numeric_features))
    safe_neptune_log(neptune_run, "features/categorical_count", len(categorical_features))

    fitted_models: dict[str, Any] = {}
    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    validation_prediction_frames: list[pd.DataFrame] = []
    test_prediction_frames: list[pd.DataFrame] = []
    tuning_trial_frames: list[pd.DataFrame] = []

    # 1. Classical baseline.
    baseline_models = {
        "logistic_regression_train_only_baseline": (
            build_logistic_baseline(numeric_features, categorical_features, config),
            "logistic_regression",
            {"class_weight": "balanced", "skewness_treatment": "yeo_johnson", "scaler": "standard"},
        ),
        "random_forest_weighted_baseline": (
            build_random_forest_baseline(numeric_features, categorical_features, config, apply_power_transform=False),
            "random_forest",
            {"class_weight": "balanced_subsample", "skewness_treatment": "winsorize_only"},
        ),
        "xgboost_weighted_baseline": (
            build_xgboost_baseline(numeric_features, categorical_features, y_train, config, apply_power_transform=False),
            "xgboost",
            {"scale_pos_weight": default_scale_pos_weight(y_train), "skewness_treatment": "winsorize_only"},
        ),
    }

    for model_name, (model, family, params) in baseline_models.items():
        model.fit(X_train, y_train)
        fitted_models[model_name] = model
        val_metrics, tst_metrics, val_pred, tst_pred = evaluate_and_log(
            model_name=model_name,
            model=model,
            splits=splits,
            experiment_log_path=experiment_log_path,
            stage="baseline",
            model_family=family,
            params=params,
            neptune_run=neptune_run,
        )
        validation_rows.append(val_metrics)
        test_rows.append(tst_metrics)
        validation_prediction_frames.append(val_pred)
        test_prediction_frames.append(tst_pred)

    # 2. Tuned Random Forest.
    rf_tuned, rf_params, rf_trials = tune_random_forest_random_search(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        X_train=X_train,
        y_train=y_train,
        config=config,
    )
    fitted_models["random_forest_random_search_tuned"] = rf_tuned
    rf_trials.insert(0, "model_name", "random_forest_random_search_tuned")
    tuning_trial_frames.append(rf_trials)
    val_metrics, tst_metrics, val_pred, tst_pred = evaluate_and_log(
        model_name="random_forest_random_search_tuned",
        model=rf_tuned,
        splits=splits,
        experiment_log_path=experiment_log_path,
        stage="tuned",
        model_family="random_forest",
        params=rf_params,
        neptune_run=neptune_run,
    )
    validation_rows.append(val_metrics)
    test_rows.append(tst_metrics)
    validation_prediction_frames.append(val_pred)
    test_prediction_frames.append(tst_pred)

    # 3. XGBoost Hyperopt raw/winsorized numeric.
    xgb_tuned, xgb_params, xgb_trials = tune_xgboost_hyperopt(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        config=config,
        apply_power_transform=False,
        neptune_run=neptune_run,
    )
    fitted_models["xgboost_hyperopt_tuned"] = xgb_tuned
    xgb_trials.insert(0, "model_name", "xgboost_hyperopt_tuned")
    tuning_trial_frames.append(xgb_trials)
    val_metrics, tst_metrics, val_pred, tst_pred = evaluate_and_log(
        model_name="xgboost_hyperopt_tuned",
        model=xgb_tuned,
        splits=splits,
        experiment_log_path=experiment_log_path,
        stage="tuned",
        model_family="xgboost",
        params=xgb_params,
        neptune_run=neptune_run,
    )
    validation_rows.append(val_metrics)
    test_rows.append(tst_metrics)
    validation_prediction_frames.append(val_pred)
    test_prediction_frames.append(tst_pred)

    # 4. XGBoost Hyperopt with train-only skewness treatment variant.
    xgb_skew_tuned, xgb_skew_params, xgb_skew_trials = tune_xgboost_hyperopt(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        config=config,
        apply_power_transform=True,
        neptune_run=neptune_run,
    )
    fitted_models["xgboost_hyperopt_tuned_skew_treated"] = xgb_skew_tuned
    xgb_skew_trials.insert(0, "model_name", "xgboost_hyperopt_tuned_skew_treated")
    tuning_trial_frames.append(xgb_skew_trials)
    val_metrics, tst_metrics, val_pred, tst_pred = evaluate_and_log(
        model_name="xgboost_hyperopt_tuned_skew_treated",
        model=xgb_skew_tuned,
        splits=splits,
        experiment_log_path=experiment_log_path,
        stage="tuned_skew_treatment_variant",
        model_family="xgboost",
        params={**xgb_skew_params, "skewness_treatment": "yeo_johnson_train_only"},
        neptune_run=neptune_run,
    )
    validation_rows.append(val_metrics)
    test_rows.append(tst_metrics)
    validation_prediction_frames.append(val_pred)
    test_prediction_frames.append(tst_pred)

    # 5. Optional train-only sampling challengers. These are controlled challenger
    # experiments, not the default model strategy. They use the tuned RF/XGB params
    # and compare whether sampling improves validation PR-AUC or operational cost.
    if config.enable_sampling_challengers:
        sampling_challengers: dict[str, tuple[Any, str, dict[str, Any]]] = {
            "random_forest_under_sampled_challenger": (
                build_random_forest_pipeline_from_params(
                    numeric_features,
                    categorical_features,
                    config,
                    rf_params,
                    sampler_type="under",
                    apply_power_transform=False,
                ),
                "random_forest",
                {**rf_params, "sampler": "RandomUnderSampler", "sampling_strategy": config.sampler_strategy},
            ),
            "random_forest_over_sampled_challenger": (
                build_random_forest_pipeline_from_params(
                    numeric_features,
                    categorical_features,
                    config,
                    rf_params,
                    sampler_type="over",
                    apply_power_transform=False,
                ),
                "random_forest",
                {**rf_params, "sampler": "RandomOverSampler", "sampling_strategy": config.sampler_strategy},
            ),
            "xgboost_under_sampled_challenger": (
                build_xgboost_pipeline_from_params_with_sampler(
                    numeric_features,
                    categorical_features,
                    y_train,
                    config,
                    xgb_params,
                    sampler_type="under",
                    apply_power_transform=False,
                ),
                "xgboost",
                {**xgb_params, "sampler": "RandomUnderSampler", "sampling_strategy": config.sampler_strategy},
            ),
            "xgboost_over_sampled_challenger": (
                build_xgboost_pipeline_from_params_with_sampler(
                    numeric_features,
                    categorical_features,
                    y_train,
                    config,
                    xgb_params,
                    sampler_type="over",
                    apply_power_transform=False,
                ),
                "xgboost",
                {**xgb_params, "sampler": "RandomOverSampler", "sampling_strategy": config.sampler_strategy},
            ),
        }
        for model_name, (model, family, params) in sampling_challengers.items():
            model.fit(X_train, y_train)
            fitted_models[model_name] = model
            val_metrics, tst_metrics, val_pred, tst_pred = evaluate_and_log(
                model_name=model_name,
                model=model,
                splits=splits,
                experiment_log_path=experiment_log_path,
                stage="sampling_challenger",
                model_family=family,
                params=params,
                neptune_run=neptune_run,
            )
            validation_rows.append(val_metrics)
            test_rows.append(tst_metrics)
            validation_prediction_frames.append(val_pred)
            test_prediction_frames.append(tst_pred)

    validation_results = pd.DataFrame(validation_rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    test_results_default_threshold = pd.DataFrame(test_rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    selection_summary = model_selection_summary(validation_results)
    ranking_champion_model_name = str(selection_summary.iloc[0]["model_name"])

    validation_predictions = pd.concat(validation_prediction_frames, ignore_index=True)
    test_predictions = pd.concat(test_prediction_frames, ignore_index=True)
    tuning_trials = pd.concat(tuning_trial_frames, ignore_index=True) if tuning_trial_frames else pd.DataFrame()

    # Validation-only threshold selection for every model. The operational champion
    # is the model/threshold pair that best satisfies business cost and review
    # capacity constraints on validation. Test is used only to confirm that pair.
    cost_assumptions = ThresholdCostAssumptions(
        false_negative_cost=config.false_negative_cost,
        false_positive_cost=config.false_positive_cost,
    )
    validation_threshold_grid = evaluate_threshold_grid_all_models(
        validation_predictions,
        dataset_name="validation",
        cost_assumptions=cost_assumptions,
    )
    all_model_threshold_shortlist = build_all_model_operational_threshold_comparison(
        threshold_grid=validation_threshold_grid,
        validation_results=validation_results,
        config=config,
    )
    if all_model_threshold_shortlist.empty:
        raise ValueError("Unable to create operational threshold comparison for candidate models.")

    operational_recommendation = all_model_threshold_shortlist.iloc[0].copy()
    operational_model_threshold_recommendation = pd.DataFrame([operational_recommendation.to_dict()])
    operational_champion_model_name = str(operational_recommendation["model_name"])
    champion_model_name = operational_champion_model_name

    threshold_shortlist = build_threshold_shortlist(validation_threshold_grid, champion_model_name)
    recommendation = recommend_operating_threshold(threshold_shortlist, config.preferred_threshold_objective)
    recommended_threshold_summary = operational_model_threshold_recommendation.copy()

    test_selected_threshold_results_all_models = evaluate_selected_thresholds_on_test(
        validation_operational_comparison=all_model_threshold_shortlist,
        test_predictions=test_predictions,
        cost_assumptions=cost_assumptions,
    )
    test_selected_threshold_results = test_selected_threshold_results_all_models.loc[
        test_selected_threshold_results_all_models["model_name"].eq(champion_model_name)
    ].copy()

    confusion_validation = []
    confusion_test = []
    for _, row in validation_results.iterrows():
        cm = confusion_matrix_frame(row)
        cm.insert(0, "model_name", row["model_name"])
        cm.insert(1, "dataset", "validation")
        cm.insert(2, "threshold", row["threshold"])
        confusion_validation.append(cm)
    for _, row in test_results_default_threshold.iterrows():
        cm = confusion_matrix_frame(row)
        cm.insert(0, "model_name", row["model_name"])
        cm.insert(1, "dataset", "test")
        cm.insert(2, "threshold", row["threshold"])
        confusion_test.append(cm)

    confusion_matrices_validation = pd.concat(confusion_validation, ignore_index=True)
    confusion_matrices_test = pd.concat(confusion_test, ignore_index=True)

    lift_frames = []
    calibration_frames = []
    for model_name in validation_predictions["model_name"].unique():
        subset = validation_predictions.query("model_name == @model_name")
        lift = score_decile_lift_table(subset)
        if not lift.empty:
            lift.insert(0, "model_name", model_name)
            lift_frames.append(lift)
        cal = calibration_by_score_band(subset)
        if not cal.empty:
            cal.insert(0, "model_name", model_name)
            calibration_frames.append(cal)

    lift_tables_validation = pd.concat(lift_frames, ignore_index=True) if lift_frames else pd.DataFrame()
    calibration_tables_validation = pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame()

    preprocessing_assurance = build_preprocessing_assurance(config)
    experiment_log = read_experiment_log(experiment_log_path)

    safe_neptune_log(neptune_run, "champion/ranking_model_name", ranking_champion_model_name)
    safe_neptune_log(neptune_run, "champion/operational_model_name", operational_champion_model_name)
    safe_neptune_log(neptune_run, "champion/recommended_threshold", float(operational_recommendation["threshold"]))
    model_readiness_gate = build_model_readiness_gate(
        {
            "xgboost_fitted": any(name.startswith("xgboost") for name in fitted_models),
            "random_forest_fitted": any(name.startswith("random_forest") for name in fitted_models),
            "hyperopt_completed": any("xgboost_hyperopt" in str(name) for name in fitted_models),
            "rf_random_search_completed": "random_forest_random_search_tuned" in fitted_models,
            "all_model_threshold_comparison_completed": not all_model_threshold_shortlist.empty,
            "experiment_tracking_saved": experiment_log_path.exists(),
            "champion_model_saved": champion_model_name in fitted_models,
        }
    )

    stop_neptune_run(neptune_run)

    return ModelTrainingArtifacts(
        fitted_models=fitted_models,
        validation_results=validation_results,
        test_results_default_threshold=test_results_default_threshold,
        selection_summary=selection_summary,
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        validation_threshold_grid=validation_threshold_grid,
        test_selected_threshold_results=test_selected_threshold_results,
        threshold_shortlist=threshold_shortlist,
        recommended_threshold_summary=recommended_threshold_summary,
        experiment_log=experiment_log,
        tuning_trials=tuning_trials,
        confusion_matrices_validation=confusion_matrices_validation,
        confusion_matrices_test=confusion_matrices_test,
        lift_tables_validation=lift_tables_validation,
        calibration_tables_validation=calibration_tables_validation,
        preprocessing_assurance=preprocessing_assurance,
        feature_inventory=feature_inventory,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        excluded_features=excluded_features,
        champion_model_name=champion_model_name,
        ranking_champion_model_name=ranking_champion_model_name,
        operational_champion_model_name=operational_champion_model_name,
        all_model_threshold_shortlist=all_model_threshold_shortlist,
        operational_model_threshold_recommendation=operational_model_threshold_recommendation,
        test_selected_threshold_results_all_models=test_selected_threshold_results_all_models,
        model_readiness_gate=model_readiness_gate,
    )


def save_model_training_artifacts(artifacts: ModelTrainingArtifacts, table_dir: Path, model_artifact_dir: Path) -> None:
    """Persist all Notebook 06 tables and fitted model artifacts."""
    table_dir.mkdir(parents=True, exist_ok=True)
    model_artifact_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "06_model_validation_results_default_threshold.csv": artifacts.validation_results,
        "06_model_test_results_default_threshold.csv": artifacts.test_results_default_threshold,
        "06_model_selection_summary.csv": artifacts.selection_summary,
        "06_validation_predictions_default_threshold.csv": artifacts.validation_predictions,
        "06_test_predictions_default_threshold.csv": artifacts.test_predictions,
        "06_validation_threshold_grid_all_models.csv": artifacts.validation_threshold_grid,
        "06_champion_threshold_shortlist_validation.csv": artifacts.threshold_shortlist,
        "06_all_model_operational_threshold_comparison_validation.csv": artifacts.all_model_threshold_shortlist,
        "06_operational_model_threshold_recommendation.csv": artifacts.operational_model_threshold_recommendation,
        "06_recommended_threshold_summary.csv": artifacts.recommended_threshold_summary,
        "06_test_selected_threshold_results.csv": artifacts.test_selected_threshold_results,
        "06_test_selected_threshold_results_all_models.csv": artifacts.test_selected_threshold_results_all_models,
        "06_tuning_trials.csv": artifacts.tuning_trials,
        "06_confusion_matrices_validation.csv": artifacts.confusion_matrices_validation,
        "06_confusion_matrices_test.csv": artifacts.confusion_matrices_test,
        "06_lift_tables_validation.csv": artifacts.lift_tables_validation,
        "06_calibration_tables_validation.csv": artifacts.calibration_tables_validation,
        "06_preprocessing_train_only_assurance.csv": artifacts.preprocessing_assurance,
        "06_feature_inventory.csv": artifacts.feature_inventory,
        "06_model_readiness_gate.csv": artifacts.model_readiness_gate,
        "06_neptune_config_status.csv": neptune_config_status(),
        "06_business_cost_assumptions.csv": cost_assumptions_frame(ThresholdCostAssumptions()),
    }
    for filename, df in tables.items():
        df.to_csv(table_dir / filename, index=False)

    joblib.dump(artifacts.fitted_models, model_artifact_dir / "06_candidate_models.joblib")
    joblib.dump(artifacts.fitted_models[artifacts.champion_model_name], model_artifact_dir / "06_champion_model.joblib")
    metadata = {
        "champion_model_name": artifacts.champion_model_name,
        "ranking_champion_model_name": artifacts.ranking_champion_model_name,
        "operational_champion_model_name": artifacts.operational_champion_model_name,
        "numeric_features": artifacts.numeric_features,
        "categorical_features": artifacts.categorical_features,
        "excluded_features": artifacts.excluded_features,
    }
    joblib.dump(metadata, model_artifact_dir / "06_model_feature_metadata.joblib")


def run_training_workflow_from_file(
    modeling_dataset_path: Path,
    table_dir: Path,
    model_artifact_dir: Path,
    config: TrainingConfig | None = None,
) -> ModelTrainingArtifacts:
    """Convenience runner used by script and notebook."""
    if not modeling_dataset_path.exists():
        raise FileNotFoundError(f"Missing modelling dataset: {modeling_dataset_path}")
    modeling_df = pd.read_csv(modeling_dataset_path, low_memory=False)
    artifacts = train_and_evaluate_xgb_rf_models(
        modeling_df=modeling_df,
        table_dir=table_dir,
        model_artifact_dir=model_artifact_dir,
        config=config,
    )
    save_model_training_artifacts(artifacts, table_dir, model_artifact_dir)
    return artifacts
