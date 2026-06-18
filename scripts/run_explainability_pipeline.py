from __future__ import annotations
import os

# Inherited by joblib/loky child processes, so resource_tracker cleanup
# warnings are not printed after the pipeline finishes.
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# Keep joblib temporary files inside the project when possible.
# This reduces Windows temp-folder cleanup warnings from loky/resource_tracker.
try:
    from pathlib import Path as _QuietPath
    _quiet_project_root = _QuietPath(__file__).resolve().parents[1]
    _quiet_joblib_tmp = _quiet_project_root / ".joblib_tmp"
    _quiet_joblib_tmp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(_quiet_joblib_tmp))
except Exception:
    pass

# Deepchecks/permutation importance can spawn workers that trigger noisy
# joblib cleanup warnings on Windows. Keeping this conservative is cleaner
# for portfolio/demo runs.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")


import logging
import warnings


# Global warning suppression for expected third-party cleanup noise.
# Exceptions and tracebacks are still shown.
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

def _suppress_warning_display(*args, **kwargs):
    return None

warnings.showwarning = _suppress_warning_display
logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.ERROR)

# Suppress warning-level log messages globally while preserving ERROR/CRITICAL.
logging.disable(logging.WARNING)

# Explicitly silence known noisy libraries.
for _logger_name in [
    "deepchecks",
    "deepchecks.core",
    "deepchecks.tabular",
    "deepchecks.utils",
    "joblib",
    "joblib.externals.loky",
    "joblib.externals.loky.backend.resource_tracker",
    "loky",
]:
    _logger = logging.getLogger(_logger_name)
    _logger.setLevel(logging.ERROR)
    _logger.disabled = True
    _logger.propagate = False

# Suppress Deepchecks logging messages such as permutation-importance warnings.
for logger_name in [
    "deepchecks",
    "deepchecks.core",
    "deepchecks.tabular",
    "deepchecks.utils",
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
    logging.getLogger(logger_name).propagate = False

# Suppress joblib/loky temporary-file cleanup warnings after Deepchecks finishes.
warnings.filterwarnings(
    "ignore",
    message=r".*resource_tracker.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r".*joblib_memmapping_folder.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r".*Cannot use model's built-in feature importance.*",
    category=UserWarning,
)

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from credit_risk.explainability.anchors_analysis import build_anchor_like_rules
from credit_risk.explainability.counterfactuals import best_counterfactual_per_account, generate_counterfactual_scenarios
from credit_risk.explainability.deepchecks_analysis import (
    fallback_model_weakness_diagnostics,
    model_enhancement_recommendations,
    run_optional_deepchecks_model_evaluation,
)
from credit_risk.explainability.shap_analysis import (
    business_regulator_summary_model_insights,
    build_explainability_readiness_gate,
    classification_metrics_at_threshold,
    compute_tree_shap_values,
    confusion_matrix_summary,
    ensure_expected_model_features,
    feature_columns_from_metadata,
    get_model_split,
    individual_top_contributions,
    load_explainability_artifacts,
    plot_global_shap_bar,
    plot_numeric_shap_dependence,
    plot_shap_beeswarm,
    predict_scores,
    select_explanation_sample,
    stakeholder_metric_impact_summary,
    summarize_global_shap,
    summarize_grouped_shap,
    summarize_probability_deciles,
    summarize_shap_by_segment,
)

TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
HTML_DIR = PROJECT_ROOT / "reports" / "html"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _write_manifest(rows: list[dict[str, Any]]) -> pd.DataFrame:
    manifest = pd.DataFrame(rows)
    manifest.to_csv(TABLE_DIR / "08_xai_output_manifest.csv", index=False)
    return manifest



def _cleanup_joblib_temp_folder() -> None:
    """Best-effort cleanup of local joblib temp folder after Deepchecks."""
    try:
        import shutil
        joblib_tmp = Path(os.getenv("JOBLIB_TEMP_FOLDER", PROJECT_ROOT / ".joblib_tmp"))
        if joblib_tmp.exists():
            shutil.rmtree(joblib_tmp, ignore_errors=True)
            joblib_tmp.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    max_shap_rows = _env_int("XAI_SHAP_SAMPLE_ROWS", 1500)
    false_negative_cost = _env_float("XAI_FALSE_NEGATIVE_COST", 5000.0)
    false_positive_cost = _env_float("XAI_FALSE_POSITIVE_COST", 500.0)

    artifacts = load_explainability_artifacts(project_root=PROJECT_ROOT)
    feature_cols = feature_columns_from_metadata(artifacts.metadata, artifacts.modeling_df)
    repaired_modeling_df, feature_repair_audit = ensure_expected_model_features(artifacts.modeling_df, feature_cols)
    feature_repair_audit.to_csv(TABLE_DIR / "08_xai_feature_alignment_audit.csv", index=False)

    X_train, y_train, _ = get_model_split(repaired_modeling_df, artifacts.metadata, split_name="train")
    X_test, y_test, identity_test = get_model_split(repaired_modeling_df, artifacts.metadata, split_name="test")
    scores = pd.Series(predict_scores(artifacts.champion_model, X_test), index=X_test.index, name="predicted_default_probability")

    metrics = classification_metrics_at_threshold(
        y_test,
        scores,
        threshold=artifacts.recommended_threshold,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    metrics_df = pd.DataFrame([{ "model_name": artifacts.champion_model_name, "dataset": "test", **metrics }])
    confusion_df = confusion_matrix_summary(metrics)
    stakeholder_df = stakeholder_metric_impact_summary(metrics)
    decile_df = summarize_probability_deciles(y_test, scores.to_numpy(), artifacts.recommended_threshold)

    metrics_df.to_csv(TABLE_DIR / "08_xai_classification_metrics_at_operating_threshold.csv", index=False)
    confusion_df.to_csv(TABLE_DIR / "08_xai_confusion_matrix_interpretation.csv", index=False)
    stakeholder_df.to_csv(TABLE_DIR / "08_xai_stakeholder_metric_impact_summary.csv", index=False)
    decile_df.to_csv(TABLE_DIR / "08_xai_probability_decile_profile.csv", index=False)

    segment_columns = [
        col for col in [
            "loan_category",
            "employment_type",
            "home",
            "loan_to_income_band",
            "interest_rate_band",
            "amount_band",
            "income_band",
            "tenure_band",
        ] if col in X_test.columns
    ]

    deepchecks_status = run_optional_deepchecks_model_evaluation(
        model=artifacts.champion_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        output_html_path=HTML_DIR / "08_deepchecks_model_evaluation.html",
    )
    weakness_df = fallback_model_weakness_diagnostics(
        X=X_test,
        y=y_test,
        scores=scores,
        threshold=artifacts.recommended_threshold,
        segment_columns=segment_columns,
    )
    recommendations_df = model_enhancement_recommendations(metrics, weakness_df)
    deepchecks_status.to_csv(TABLE_DIR / "08_deepchecks_execution_status.csv", index=False)
    weakness_df.to_csv(TABLE_DIR / "08_deepchecks_fallback_model_weakness_diagnostics.csv", index=False)
    recommendations_df.to_csv(TABLE_DIR / "08_model_enhancement_recommendations.csv", index=False)

    sample_idx = select_explanation_sample(
        X_test,
        y_test,
        scores.to_numpy(),
        threshold=artifacts.recommended_threshold,
        max_rows=max_shap_rows,
        high_risk_rows=250,
        near_threshold_rows=250,
        random_state=42,
    )
    X_sample = X_test.loc[sample_idx].copy()
    shap_df, X_transformed = compute_tree_shap_values(artifacts.champion_model, X_sample)
    raw_feature_names = feature_cols

    global_importance = summarize_global_shap(shap_df, raw_feature_names=raw_feature_names)
    grouped_importance = summarize_grouped_shap(global_importance)
    regulator_summary = business_regulator_summary_model_insights(grouped_importance, top_n=15)
    subsegment_shap = summarize_shap_by_segment(
        X_raw=X_test.loc[shap_df.index],
        shap_df=shap_df,
        grouped_importance=grouped_importance,
        segment_columns=segment_columns,
        raw_feature_names=raw_feature_names,
        top_n_features=8,
        min_segment_rows=100,
    )

    global_importance.to_csv(TABLE_DIR / "08_xai_global_shap_importance_transformed.csv", index=False)
    grouped_importance.to_csv(TABLE_DIR / "08_xai_grouped_shap_importance.csv", index=False)
    regulator_summary.to_csv(TABLE_DIR / "08_xai_business_regulator_summary_model_insights.csv", index=False)
    subsegment_shap.to_csv(TABLE_DIR / "08_xai_shap_feature_dependence_subsegment_insights.csv", index=False)
    shap_df.to_csv(TABLE_DIR / "08_xai_shap_values_sample.csv", index=True)

    high_risk_idx = scores.sort_values(ascending=False).head(20).index.tolist()
    near_threshold_idx = (scores - artifacts.recommended_threshold).abs().sort_values().head(20).index.tolist()
    local_candidate_idx = pd.Index(high_risk_idx + near_threshold_idx).drop_duplicates().tolist()

    missing_local = [idx for idx in local_candidate_idx if idx not in shap_df.index]
    if missing_local:
        local_shap, _ = compute_tree_shap_values(artifacts.champion_model, X_test.loc[missing_local])
        shap_df = pd.concat([shap_df, local_shap], axis=0)

    individual_df = individual_top_contributions(
        X_raw=X_test,
        identity=identity_test,
        shap_df=shap_df,
        scores=scores,
        candidate_indices=local_candidate_idx,
        threshold=artifacts.recommended_threshold,
        raw_feature_names=raw_feature_names,
        top_n=6,
    )
    individual_df.to_csv(TABLE_DIR / "08_xai_individual_prediction_shap_explanations.csv", index=False)

    anchor_candidate_idx = high_risk_idx[:15]
    anchor_rules = build_anchor_like_rules(
        X_reference=X_test,
        y_reference=y_test,
        scores_reference=scores,
        X_explain=X_test,
        shap_explain=shap_df,
        candidate_indices=anchor_candidate_idx,
        threshold=artifacts.recommended_threshold,
        raw_feature_names=raw_feature_names,
        max_conditions=3,
    )
    anchor_rules.to_csv(TABLE_DIR / "08_xai_anchor_style_rules.csv", index=False)

    cf_candidate_idx = pd.Index(high_risk_idx[:5] + near_threshold_idx[:10]).drop_duplicates().tolist()
    counterfactuals = generate_counterfactual_scenarios(
        pipeline=artifacts.champion_model,
        X_candidates=X_test.loc[cf_candidate_idx],
        baseline_scores=scores,
        threshold=artifacts.recommended_threshold,
        reference_df=X_train,
    )
    best_cfs = best_counterfactual_per_account(counterfactuals)
    counterfactuals.to_csv(TABLE_DIR / "08_xai_counterfactual_scenarios.csv", index=False)
    best_cfs.to_csv(TABLE_DIR / "08_xai_best_counterfactual_per_account.csv", index=False)

    plot_global_shap_bar(grouped_importance.rename(columns={"raw_feature": "transformed_feature"}), FIGURE_DIR / "08_xai_global_grouped_shap_top_features.png", top_n=20)
    plot_global_shap_bar(global_importance, FIGURE_DIR / "08_xai_global_transformed_shap_top_features.png", top_n=20)
    common_beeswarm_idx = shap_df.index.intersection(X_transformed.index)
    plot_shap_beeswarm(shap_df.loc[common_beeswarm_idx], X_transformed.loc[common_beeswarm_idx], FIGURE_DIR / "08_xai_shap_summary_beeswarm.png", max_display=20)

    for raw_feature in ["interest_rate", "loan_to_income_ratio", "total_income_pa", "amount", "delinq_2yrs", "number_of_loans"]:
        plot_numeric_shap_dependence(
            shap_df=shap_df,
            X_raw=X_test,
            raw_feature=raw_feature,
            output_path=FIGURE_DIR / f"08_xai_shap_dependence_{raw_feature}.png",
            raw_feature_names=raw_feature_names,
        )

    readiness_rows = [
        {"check": "Champion model loaded", "status": "pass", "detail": str(artifacts.model_path)},
        {"check": "Recommended threshold loaded", "status": "pass", "detail": f"{artifacts.recommended_threshold:.4f}"},
        {"check": "Classification metrics exported", "status": "pass", "detail": "08_xai_classification_metrics_at_operating_threshold.csv"},
        {"check": "Deepchecks/fallback diagnostics exported", "status": "pass", "detail": "08_deepchecks_execution_status.csv"},
        {"check": "SHAP global and local explanations exported", "status": "pass", "detail": f"{len(shap_df):,} explained rows"},
        {"check": "Anchor-style explanations exported", "status": "pass" if not anchor_rules.empty else "warning", "detail": f"{len(anchor_rules):,} rows"},
        {"check": "Counterfactual diagnostics exported", "status": "pass" if not counterfactuals.empty else "warning", "detail": f"{len(counterfactuals):,} scenario rows"},
        {"check": "Regulatory/stakeholder interpretation exported", "status": "pass", "detail": "08_xai_business_regulator_summary_model_insights.csv"},
    ]
    readiness = build_explainability_readiness_gate(readiness_rows)
    readiness.to_csv(TABLE_DIR / "08_xai_readiness_gate.csv", index=False)

    manifest_rows = []
    for folder in [TABLE_DIR, FIGURE_DIR, HTML_DIR]:
        for path in sorted(folder.glob("08_*")):
            manifest_rows.append({"artifact_type": folder.name, "path": str(path.relative_to(PROJECT_ROOT)), "exists": path.exists()})
    manifest = _write_manifest(manifest_rows)

    _cleanup_joblib_temp_folder()

    print("Notebook 08 explainability pipeline completed.")
    print(f"Champion model: {artifacts.champion_model_name}")
    print(f"Operating threshold: {artifacts.recommended_threshold:.4f}")
    print(f"Test recall: {metrics['recall']:.4f}")
    print(f"Test precision: {metrics['precision']:.4f}")
    print(f"Test F1: {metrics['f1']:.4f}")
    print(f"SHAP sample rows: {len(shap_df):,}")
    print("Top grouped SHAP drivers:")
    print(grouped_importance.head(10).to_string(index=False))
    print(f"Saved {len(manifest):,} Notebook 08 artifacts.")


if __name__ == "__main__":
    main()
