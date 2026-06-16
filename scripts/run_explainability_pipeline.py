from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from credit_risk.explainability.anchors_analysis import build_anchor_like_rules
from credit_risk.explainability.counterfactuals import best_counterfactual_per_account, generate_counterfactual_scenarios
from credit_risk.explainability.shap_analysis import (
    compute_tree_shap_values,
    get_model_split,
    individual_top_contributions,
    load_explainability_artifacts,
    plot_global_shap_bar,
    plot_numeric_shap_dependence,
    predict_scores,
    select_explanation_sample,
    summarize_global_shap,
    summarize_grouped_shap,
    summarize_probability_deciles,
)

TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
MODEL_DIR = PROJECT_ROOT / "reports" / "model_artifacts"
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    artifacts = load_explainability_artifacts(
        model_path=MODEL_DIR / "champion_model.joblib",
        metadata_path=MODEL_DIR / "model_feature_metadata.joblib",
        modeling_dataset_path=DATA_DIR / "credit_risk_modeling_dataset.csv",
        recommended_threshold_path=TABLE_DIR / "recommended_threshold_summary.csv",
    )

    X_test, y_test, identity_test = get_model_split(artifacts.modeling_df, artifacts.metadata, split_name="test")
    scores = predict_scores(artifacts.champion_model, X_test)
    score_series = pd.Series(scores, index=X_test.index, name="predicted_default_probability")

    sample_indices = select_explanation_sample(X_test, y_test, scores, max_rows=1500, high_risk_rows=250, random_state=42)
    X_sample = X_test.loc[sample_indices].copy()
    shap_df, _ = compute_tree_shap_values(artifacts.champion_model, X_sample)

    global_importance = summarize_global_shap(shap_df)
    grouped_importance = summarize_grouped_shap(global_importance)
    deciles = summarize_probability_deciles(y_test, scores, artifacts.recommended_threshold)

    high_risk_indices = score_series.sort_values(ascending=False).head(20).index.tolist()
    closest_above_indices = score_series.loc[score_series.ge(artifacts.recommended_threshold)].sort_values().head(20).index.tolist()
    local_candidate_indices = pd.Index(high_risk_indices + closest_above_indices).drop_duplicates().tolist()

    missing_local = [idx for idx in local_candidate_indices if idx not in shap_df.index]
    if missing_local:
        local_shap, _ = compute_tree_shap_values(artifacts.champion_model, X_test.loc[missing_local])
        shap_df = pd.concat([shap_df, local_shap], axis=0)
        X_sample = pd.concat([X_sample, X_test.loc[missing_local]], axis=0)

    individual = individual_top_contributions(
        X_raw=X_test,
        identity=identity_test,
        shap_df=shap_df,
        scores=score_series,
        candidate_indices=local_candidate_indices,
        threshold=artifacts.recommended_threshold,
        top_n=5,
    )

    anchor_rules = build_anchor_like_rules(
        X_reference=X_test,
        y_reference=y_test,
        scores_reference=score_series,
        X_explain=X_test,
        shap_explain=shap_df,
        candidate_indices=high_risk_indices[:10],
        threshold=artifacts.recommended_threshold,
        max_conditions=3,
    )

    cf_candidate_indices = pd.Index(high_risk_indices[:5] + closest_above_indices[:10]).drop_duplicates().tolist()
    counterfactuals = generate_counterfactual_scenarios(
        pipeline=artifacts.champion_model,
        X_candidates=X_test.loc[cf_candidate_indices],
        baseline_scores=score_series,
        threshold=artifacts.recommended_threshold,
        reference_df=X_test,
    )
    best_cfs = best_counterfactual_per_account(counterfactuals)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    global_importance.to_csv(TABLE_DIR / "xai_global_shap_importance.csv", index=False)
    grouped_importance.to_csv(TABLE_DIR / "xai_grouped_shap_importance.csv", index=False)
    deciles.to_csv(TABLE_DIR / "xai_probability_decile_profile.csv", index=False)
    individual.to_csv(TABLE_DIR / "xai_individual_local_explanations.csv", index=False)
    anchor_rules.to_csv(TABLE_DIR / "xai_anchor_like_rules.csv", index=False)
    counterfactuals.to_csv(TABLE_DIR / "xai_counterfactual_scenarios.csv", index=False)
    best_cfs.to_csv(TABLE_DIR / "xai_best_counterfactual_per_account.csv", index=False)

    plot_global_shap_bar(global_importance, FIGURE_DIR / "xai_global_shap_top_features.png", top_n=20)
    plot_numeric_shap_dependence(shap_df, X_test.loc[shap_df.index], "interest_rate", FIGURE_DIR / "xai_shap_dependence_interest_rate.png")
    plot_numeric_shap_dependence(shap_df, X_test.loc[shap_df.index], "loan_to_income_ratio", FIGURE_DIR / "xai_shap_dependence_loan_to_income_ratio.png")
    plot_numeric_shap_dependence(shap_df, X_test.loc[shap_df.index], "total_income_pa", FIGURE_DIR / "xai_shap_dependence_total_income_pa.png")

    print("Explainability pipeline completed")
    print(f"Champion model: {artifacts.champion_model_name}")
    print(f"Operating threshold: {artifacts.recommended_threshold:.4f}")
    print(f"Explanation sample rows: {len(X_sample):,}")
    print("Top grouped SHAP features:")
    print(grouped_importance.head(10).to_string(index=False))
    if not best_cfs.empty:
        print("\nBest counterfactual examples:")
        print(best_cfs.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
