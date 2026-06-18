
from __future__ import annotations

"""Professional threshold selection and business-cost utilities.

Design principles:
- Thresholds are selected on validation predictions only.
- The test set is used only once for confirmation of the selected operating rule.
- Model ranking and operating decisions are separated.
- Business cost assumptions are scenario assumptions, not accounting estimates.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from credit_risk.models.evaluate import (
        DEFAULT_FALSE_NEGATIVE_COST,
        DEFAULT_FALSE_POSITIVE_COST,
        classification_metrics_at_threshold,
    )
except Exception:  # pragma: no cover - standalone fallback for notebook/script portability
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    DEFAULT_FALSE_NEGATIVE_COST = 5_000
    DEFAULT_FALSE_POSITIVE_COST = 500

    def classification_metrics_at_threshold(
        y_true,
        y_score,
        threshold: float = 0.50,
        false_negative_cost: float = DEFAULT_FALSE_NEGATIVE_COST,
        false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST,
    ) -> dict:
        y_true_arr = np.asarray(y_true).astype(int)
        y_score_arr = np.asarray(y_score).astype(float)
        y_pred = (y_score_arr >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
        return {
            "threshold": float(threshold),
            "roc_auc": float(roc_auc_score(y_true_arr, y_score_arr)),
            "pr_auc": float(average_precision_score(y_true_arr, y_score_arr)),
            "brier_score": float(brier_score_loss(y_true_arr, y_score_arr)),
            "accuracy": float(accuracy_score(y_true_arr, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
            "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
            "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true_arr, y_pred)),
            "review_rate": float(y_pred.mean()),
            "business_cost": float(fn * false_negative_cost + fp * false_positive_cost),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "default_count": int(y_true_arr.sum()),
            "non_default_count": int((1 - y_true_arr).sum()),
        }

TARGET_COLUMN = "defaulter"
MODEL_COLUMN = "model_name"
PROBABILITY_COLUMN = "predicted_default_probability"


@dataclass(frozen=True)
class ThresholdCostAssumptions:
    """Business-cost scenario used to compare operating thresholds."""

    false_negative_cost: float = DEFAULT_FALSE_NEGATIVE_COST
    false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST

    @property
    def cost_ratio_threshold(self) -> float:
        denominator = self.false_negative_cost + self.false_positive_cost
        if denominator <= 0:
            return 0.50
        return self.false_positive_cost / denominator


def read_first_existing_csv(paths: Sequence[Path], required: bool = True) -> pd.DataFrame:
    """Read the first existing CSV from a list of compatible possible paths."""
    for path in paths:
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    if required:
        raise FileNotFoundError("None of these expected files exist: " + ", ".join(str(p) for p in paths))
    return pd.DataFrame()


def build_threshold_grid(start: float = 0.01, stop: float = 0.99, step: float = 0.005) -> np.ndarray:
    """Create a stable threshold grid for threshold and business-cost analysis."""
    if not (0 < start < 1 and 0 < stop < 1):
        raise ValueError("start and stop must be between 0 and 1.")
    if start >= stop:
        raise ValueError("start must be lower than stop.")
    if step <= 0:
        raise ValueError("step must be positive.")
    return np.round(np.arange(start, stop + step / 2, step), 3)


def validate_prediction_frame(predictions: pd.DataFrame) -> None:
    required_columns = {TARGET_COLUMN, MODEL_COLUMN, PROBABILITY_COLUMN}
    missing = required_columns - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing required columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("Prediction frame is empty.")
    if not predictions[PROBABILITY_COLUMN].between(0, 1).all():
        raise ValueError("Predicted probabilities must be between 0 and 1.")


def evaluate_threshold_grid_for_model(
    predictions: pd.DataFrame,
    model_name: str,
    thresholds: Iterable[float] | None = None,
    dataset_name: str | None = None,
    cost_assumptions: ThresholdCostAssumptions | None = None,
) -> pd.DataFrame:
    """Evaluate one model over a grid of thresholds."""
    validate_prediction_frame(predictions)
    thresholds = list(build_threshold_grid() if thresholds is None else thresholds)
    cost_assumptions = cost_assumptions or ThresholdCostAssumptions()

    model_predictions = predictions.loc[predictions[MODEL_COLUMN].eq(model_name)].copy()
    if model_predictions.empty:
        raise ValueError(f"No predictions found for model_name={model_name!r}.")

    rows: list[dict] = []
    for threshold in thresholds:
        metrics = classification_metrics_at_threshold(
            model_predictions[TARGET_COLUMN].astype(int),
            model_predictions[PROBABILITY_COLUMN].astype(float),
            threshold=float(threshold),
            false_negative_cost=cost_assumptions.false_negative_cost,
            false_positive_cost=cost_assumptions.false_positive_cost,
        )
        rows.append({"model_name": model_name, "dataset": dataset_name or "unknown", **metrics})
    return pd.DataFrame(rows)


def evaluate_threshold_grid_all_models(
    predictions: pd.DataFrame,
    thresholds: Iterable[float] | None = None,
    dataset_name: str | None = None,
    cost_assumptions: ThresholdCostAssumptions | None = None,
) -> pd.DataFrame:
    """Evaluate every available model over the same threshold grid."""
    validate_prediction_frame(predictions)
    frames = []
    for model_name in sorted(predictions[MODEL_COLUMN].dropna().unique()):
        frames.append(
            evaluate_threshold_grid_for_model(
                predictions=predictions,
                model_name=str(model_name),
                thresholds=thresholds,
                dataset_name=dataset_name,
                cost_assumptions=cost_assumptions,
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def apply_selected_threshold_to_predictions(
    predictions: pd.DataFrame,
    model_name: str,
    selected_threshold: float,
    dataset_name: str,
    cost_assumptions: ThresholdCostAssumptions | None = None,
) -> pd.DataFrame:
    """Evaluate one already-selected operating threshold on a prediction dataset."""
    cost_assumptions = cost_assumptions or ThresholdCostAssumptions()
    return evaluate_threshold_grid_for_model(
        predictions=predictions,
        model_name=model_name,
        thresholds=[float(selected_threshold)],
        dataset_name=dataset_name,
        cost_assumptions=cost_assumptions,
    )

def cost_assumptions_frame(cost_assumptions: ThresholdCostAssumptions) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "assumption": "false_negative_cost",
                "value": cost_assumptions.false_negative_cost,
                "description": "Illustrative cost of missing a borrower who later defaults.",
            },
            {
                "assumption": "false_positive_cost",
                "value": cost_assumptions.false_positive_cost,
                "description": "Illustrative operational and customer-friction cost of reviewing a non-default account.",
            },
            {
                "assumption": "calibrated_probability_cost_ratio_threshold",
                "value": cost_assumptions.cost_ratio_threshold,
                "description": "Theoretical cutoff if model probabilities were perfectly calibrated.",
            },
        ]
    )


def _attach_default_threshold_metrics(candidate: pd.Series, validation_results: pd.DataFrame | None) -> pd.Series:
    if validation_results is None or validation_results.empty or "model_name" not in validation_results.columns:
        return candidate
    match = validation_results.loc[validation_results["model_name"].eq(candidate["model_name"])]
    if match.empty:
        return candidate
    default_row = match.iloc[0]
    for col in ["pr_auc", "roc_auc", "brier_score", "recall", "precision", "review_rate", "business_cost"]:
        if col in default_row.index:
            candidate[f"validation_{col}_at_default_threshold"] = default_row[col]
    return candidate


def build_all_model_operational_threshold_comparison(
    threshold_grid: pd.DataFrame,
    validation_results: pd.DataFrame | None = None,
    review_rate_cap: float = 0.30,
    min_recall: float = 0.60,
    objective_name: str = "minimum_cost_review_rate_le_30pct",
) -> pd.DataFrame:
    """Select the best operating threshold for every model under one business policy.

    The primary policy minimizes estimated business cost subject to a manual-review
    capacity cap and a minimum recall floor. If a model has no threshold meeting the
    recall floor, the function relaxes the recall floor but clearly marks that row.
    """
    required = {"model_name", "threshold", "business_cost", "review_rate", "recall", "precision", "pr_auc", "roc_auc"}
    missing = required - set(threshold_grid.columns)
    if missing:
        raise ValueError(f"Threshold grid is missing required columns: {sorted(missing)}")

    rows = []
    for model_name, model_grid in threshold_grid.groupby("model_name"):
        working = model_grid.loc[model_grid["review_rate"].le(review_rate_cap)].copy()
        selection_basis = f"minimum business cost under review_rate <= {review_rate_cap:.0%}; min_recall >= {min_recall:.0%}"
        if working.empty:
            working = model_grid.copy()
            selection_basis = "fallback: no threshold met review-rate cap; selected minimum business cost"
        recall_working = working.loc[working["recall"].ge(min_recall)].copy()
        if recall_working.empty:
            recall_working = working.copy()
            selection_basis += " | recall floor relaxed because no feasible threshold met it"
        best = recall_working.sort_values(
            ["business_cost", "review_rate", "false_negative", "false_positive"],
            ascending=[True, True, True, True],
        ).iloc[0].copy()
        best["objective"] = objective_name
        best["selection_basis"] = selection_basis
        best = _attach_default_threshold_metrics(best, validation_results)
        rows.append(best)

    output = pd.DataFrame(rows)
    output = output.sort_values(
        ["business_cost", "recall", "precision", "review_rate"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    output.insert(0, "operational_rank", range(1, len(output) + 1))
    return output


def recommend_operational_model_threshold(all_model_operational_comparison: pd.DataFrame) -> pd.DataFrame:
    """Return the top operating model-threshold pair as a one-row DataFrame."""
    if all_model_operational_comparison.empty:
        raise ValueError("Operational comparison is empty.")
    return all_model_operational_comparison.sort_values("operational_rank").head(1).reset_index(drop=True)


def apply_recommendation_to_threshold_grid(
    recommendation: pd.DataFrame | pd.Series,
    threshold_grid: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Apply a validation-selected model and threshold to another threshold grid."""
    row = recommendation.iloc[0] if isinstance(recommendation, pd.DataFrame) else recommendation
    model_name = str(row["model_name"])
    threshold = float(row["threshold"])
    matched = threshold_grid.loc[
        threshold_grid["model_name"].eq(model_name) & threshold_grid["threshold"].round(6).eq(round(threshold, 6))
    ].copy()
    if matched.empty:
        raise ValueError(f"No threshold-grid row found for model={model_name!r}, threshold={threshold}.")
    matched.insert(0, "validation_objective", row.get("objective", "validation_selected_threshold"))
    matched.insert(0, "validation_operational_rank", row.get("operational_rank", 1))
    matched["dataset"] = dataset_name
    return matched.reset_index(drop=True)


def build_policy_option_table(
    threshold_grid: pd.DataFrame,
    validation_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare common risk-policy alternatives for stakeholder discussion."""
    policies = [
        ("capacity_constrained", 0.25, 0.55, "Conservative review workload; may miss more defaults."),
        ("balanced_operating_policy", 0.30, 0.60, "Balanced risk capture and operational capacity."),
        ("risk_capture_policy", 0.35, 0.65, "More default capture with higher review workload."),
        ("high_recall_policy", 0.40, 0.70, "Higher risk capture; greater customer/operations friction."),
    ]
    frames = []
    for policy_name, cap, recall_floor, description in policies:
        comp = build_all_model_operational_threshold_comparison(
            threshold_grid=threshold_grid,
            validation_results=validation_results,
            review_rate_cap=cap,
            min_recall=recall_floor,
            objective_name=policy_name,
        )
        top = comp.sort_values("operational_rank").head(1).copy()
        top["policy_description"] = description
        top["review_rate_cap"] = cap
        top["min_recall_floor"] = recall_floor
        frames.append(top)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_cost_sensitivity_scenarios() -> list[dict]:
    """Create business-cost scenarios for robustness testing."""
    scenarios = []
    for fn_cost in [2_500, 5_000, 7_500, 10_000]:
        for fp_cost in [250, 500, 1_000]:
            scenarios.append(
                {
                    "scenario_name": f"fn_{fn_cost}_fp_{fp_cost}",
                    "false_negative_cost": float(fn_cost),
                    "false_positive_cost": float(fp_cost),
                }
            )
    return scenarios


def run_business_cost_sensitivity_analysis(
    validation_predictions: pd.DataFrame,
    validation_results: pd.DataFrame | None,
    thresholds: Iterable[float],
    review_rate_cap: float = 0.30,
    min_recall: float = 0.60,
    scenarios: list[dict] | None = None,
) -> pd.DataFrame:
    """Re-select the operating champion under alternative cost assumptions."""
    scenarios = scenarios or build_cost_sensitivity_scenarios()
    outputs = []
    for scenario in scenarios:
        assumptions = ThresholdCostAssumptions(
            false_negative_cost=scenario["false_negative_cost"],
            false_positive_cost=scenario["false_positive_cost"],
        )
        scenario_grid = evaluate_threshold_grid_all_models(
            validation_predictions,
            thresholds=thresholds,
            dataset_name="validation",
            cost_assumptions=assumptions,
        )
        comp = build_all_model_operational_threshold_comparison(
            threshold_grid=scenario_grid,
            validation_results=validation_results,
            review_rate_cap=review_rate_cap,
            min_recall=min_recall,
            objective_name="sensitivity_min_cost_review_cap",
        )
        top = recommend_operational_model_threshold(comp)
        top["scenario_name"] = scenario["scenario_name"]
        top["false_negative_cost"] = scenario["false_negative_cost"]
        top["false_positive_cost"] = scenario["false_positive_cost"]
        outputs.append(top)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def make_confusion_matrix_long(row: pd.Series | pd.DataFrame, dataset_name: str | None = None) -> pd.DataFrame:
    """Convert one threshold result row into a readable confusion-matrix table."""
    r = row.iloc[0] if isinstance(row, pd.DataFrame) else row
    dataset = dataset_name or r.get("dataset", "unknown")
    model_name = r.get("model_name", "unknown")
    threshold = r.get("threshold", np.nan)
    return pd.DataFrame(
        [
            {"dataset": dataset, "model_name": model_name, "threshold": threshold, "actual": "non_default", "predicted": "non_default", "count": int(r.get("true_negative", 0)), "cell": "true_negative"},
            {"dataset": dataset, "model_name": model_name, "threshold": threshold, "actual": "non_default", "predicted": "default_review", "count": int(r.get("false_positive", 0)), "cell": "false_positive"},
            {"dataset": dataset, "model_name": model_name, "threshold": threshold, "actual": "default", "predicted": "non_default", "count": int(r.get("false_negative", 0)), "cell": "false_negative"},
            {"dataset": dataset, "model_name": model_name, "threshold": threshold, "actual": "default", "predicted": "default_review", "count": int(r.get("true_positive", 0)), "cell": "true_positive"},
        ]
    )


def build_metric_stakeholder_impact_summary(selected_row: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Create a plain-language summary of how metrics affect stakeholders."""
    r = selected_row.iloc[0] if isinstance(selected_row, pd.DataFrame) else selected_row
    return pd.DataFrame(
        [
            {
                "metric": "Recall",
                "selected_value": r.get("recall", np.nan),
                "business_meaning": "Share of actual defaulters captured for review.",
                "stakeholder_impact": "Higher recall helps credit risk teams reduce missed defaults, but usually increases review workload.",
            },
            {
                "metric": "Precision",
                "selected_value": r.get("precision", np.nan),
                "business_meaning": "Share of reviewed borrowers who actually default.",
                "stakeholder_impact": "Higher precision reduces unnecessary reviews and customer friction.",
            },
            {
                "metric": "False negatives",
                "selected_value": r.get("false_negative", np.nan),
                "business_meaning": "Defaulters not sent for review.",
                "stakeholder_impact": "Creates credit-loss exposure and is usually more costly than a false positive.",
            },
            {
                "metric": "False positives",
                "selected_value": r.get("false_positive", np.nan),
                "business_meaning": "Non-defaulters sent for review.",
                "stakeholder_impact": "Creates operational workload and may create customer friction or fairness concerns.",
            },
            {
                "metric": "Review rate",
                "selected_value": r.get("review_rate", np.nan),
                "business_meaning": "Share of accounts routed to manual/risk review.",
                "stakeholder_impact": "Must fit available credit operations capacity and service-level expectations.",
            },
            {
                "metric": "Business cost",
                "selected_value": r.get("business_cost", np.nan),
                "business_meaning": "Scenario-weighted cost of false negatives and false positives.",
                "stakeholder_impact": "Allows credit risk, operations, and business leaders to compare thresholds using a common decision framework.",
            },
        ]
    )


def build_threshold_policy_decision_table(
    validation_recommendation: pd.DataFrame,
    test_confirmation: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact decision table for the model documentation."""
    val = validation_recommendation.iloc[0]
    test = test_confirmation.iloc[0]
    return pd.DataFrame(
        [
            {"decision_item": "Operating model", "decision": val.get("model_name"), "evidence": "Selected on validation data using the business-cost policy."},
            {"decision_item": "Operating threshold", "decision": float(val.get("threshold")), "evidence": "Validation-selected threshold; test set used only for confirmation."},
            {"decision_item": "Validation recall", "decision": float(val.get("recall")), "evidence": "Risk-capture level at the selected threshold."},
            {"decision_item": "Validation precision", "decision": float(val.get("precision")), "evidence": "Review efficiency at the selected threshold."},
            {"decision_item": "Validation review rate", "decision": float(val.get("review_rate")), "evidence": "Operational workload estimate under the selected policy."},
            {"decision_item": "Test recall", "decision": float(test.get("recall")), "evidence": "Out-of-sample confirmation of risk capture."},
            {"decision_item": "Test precision", "decision": float(test.get("precision")), "evidence": "Out-of-sample confirmation of review efficiency."},
            {"decision_item": "Test business cost", "decision": float(test.get("business_cost")), "evidence": "Out-of-sample confirmation of cost trade-off."},
            {"decision_item": "Proceed to explainability", "decision": "Yes", "evidence": "Selected model and threshold are documented for SHAP, Anchors, and counterfactual analysis."},
        ]
    )


def build_model_handoff_for_explainability(validation_recommendation: pd.DataFrame, test_confirmation: pd.DataFrame) -> pd.DataFrame:
    val = validation_recommendation.iloc[0]
    test = test_confirmation.iloc[0]
    return pd.DataFrame(
        [
            {
                "handoff_item": "operational_model_name",
                "value": val.get("model_name"),
                "description": "Model to explain in Notebook 08.",
            },
            {
                "handoff_item": "operating_threshold",
                "value": val.get("threshold"),
                "description": "Threshold to use when labeling review/non-review decisions.",
            },
            {
                "handoff_item": "test_recall",
                "value": test.get("recall"),
                "description": "Out-of-sample risk capture rate at selected threshold.",
            },
            {
                "handoff_item": "test_precision",
                "value": test.get("precision"),
                "description": "Out-of-sample review efficiency at selected threshold.",
            },
            {
                "handoff_item": "test_review_rate",
                "value": test.get("review_rate"),
                "description": "Out-of-sample share routed to review.",
            },
        ]
    )


def build_threshold_readiness_gate(
    validation_grid: pd.DataFrame,
    test_grid: pd.DataFrame,
    recommendation: pd.DataFrame,
    test_confirmation: pd.DataFrame,
) -> pd.DataFrame:
    checks = [
        ("validation_threshold_grid_created", not validation_grid.empty, "Thresholds evaluated on validation predictions."),
        ("test_threshold_grid_created", not test_grid.empty, "Test threshold grid created for confirmation only."),
        ("operational_recommendation_created", not recommendation.empty, "Operational model-threshold selected from validation."),
        ("test_confirmation_created", not test_confirmation.empty, "Selected threshold confirmed on test set."),
        ("review_rate_documented", "review_rate" in recommendation.columns, "Manual review workload is documented."),
        ("business_cost_documented", "business_cost" in recommendation.columns, "Cost trade-off is documented."),
        ("ready_for_explainability", not recommendation.empty and not test_confirmation.empty, "Notebook 08 can explain the selected operational model."),
    ]
    return pd.DataFrame([{"check": name, "passed": bool(passed), "note": note} for name, passed, note in checks])


def save_threshold_outputs(
    table_dir: Path,
    outputs: dict[str, pd.DataFrame],
) -> None:
    """Persist Notebook 07 outputs using stable 07_ filenames."""
    table_dir.mkdir(parents=True, exist_ok=True)
    for filename, df in outputs.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(table_dir / filename, index=False)


# ============================================================
# Notebook 06/07/08 compatibility helpers
# ============================================================
# These functions keep older Notebook 06 model artifacts compatible with the
# refined Notebook 07 thresholding module. They are intentionally conservative:
# validation data is still used for threshold selection, and test data remains
# confirmation-only.

def build_threshold_shortlist(
    threshold_grid: pd.DataFrame,
    model_name: str,
    review_rate_caps: Iterable[float] = (0.25, 0.30, 0.35, 0.40),
    recall_floors: Iterable[float] = (0.60, 0.70, 0.75),
) -> pd.DataFrame:
    """Create a governance-ready shortlist of threshold candidates for one model.

    This preserves the Notebook 06 API expected by saved model artifacts while
    using the refined Notebook 07 business-cost framework.
    """
    if threshold_grid.empty:
        raise ValueError("Threshold grid is empty.")
    if MODEL_COLUMN not in threshold_grid.columns:
        raise ValueError(f"Threshold grid must contain {MODEL_COLUMN!r}.")

    model_grid = threshold_grid.loc[threshold_grid[MODEL_COLUMN].eq(model_name)].copy()
    if model_grid.empty:
        raise ValueError(f"No threshold grid rows found for model_name={model_name!r}.")

    def _pick(objective: str, working: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> pd.DataFrame:
        if working.empty:
            return pd.DataFrame()
        row = working.sort_values(sort_cols, ascending=ascending).head(1).copy()
        row.insert(0, "objective", objective)
        return row

    rows: list[pd.DataFrame] = []

    rows.append(
        _pick(
            "default_threshold_0_50",
            model_grid.loc[model_grid["threshold"].round(6).eq(0.50)],
            ["threshold"],
            [True],
        )
    )
    rows.append(_pick("minimum_business_cost", model_grid, ["business_cost", "review_rate"], [True, True]))

    if "f1" in model_grid.columns:
        rows.append(_pick("maximum_f1", model_grid, ["f1", "business_cost"], [False, True]))
    if "mcc" in model_grid.columns:
        rows.append(_pick("maximum_mcc", model_grid, ["mcc", "business_cost"], [False, True]))
    if "balanced_accuracy" in model_grid.columns:
        rows.append(_pick("maximum_balanced_accuracy", model_grid, ["balanced_accuracy", "business_cost"], [False, True]))

    for cap in review_rate_caps:
        rows.append(
            _pick(
                f"minimum_cost_review_rate_le_{int(cap * 100)}pct",
                model_grid.loc[model_grid["review_rate"].le(cap)],
                ["business_cost", "review_rate"],
                [True, True],
            )
        )

    for floor in recall_floors:
        rows.append(
            _pick(
                f"minimum_cost_recall_ge_{int(floor * 100)}pct",
                model_grid.loc[model_grid["recall"].ge(floor)],
                ["business_cost", "review_rate"],
                [True, True],
            )
        )

    shortlist = pd.concat([row for row in rows if isinstance(row, pd.DataFrame) and not row.empty], ignore_index=True)
    if shortlist.empty:
        return shortlist
    return shortlist.drop_duplicates(subset=["objective", "threshold"]).reset_index(drop=True)


def recommend_operating_threshold(
    shortlist: pd.DataFrame,
    preferred_objective: str = "minimum_cost_review_rate_le_30pct",
) -> pd.Series:
    """Return the recommended threshold row from a one-model threshold shortlist.

    This is the Notebook 06-compatible API. Notebook 07 uses
    recommend_operational_model_threshold for all-model comparison.
    """
    if shortlist.empty:
        raise ValueError("Threshold shortlist is empty.")
    preferred = shortlist.loc[shortlist["objective"].eq(preferred_objective)]
    if not preferred.empty:
        return preferred.iloc[0]
    minimum_cost = shortlist.loc[shortlist["objective"].eq("minimum_business_cost")]
    if not minimum_cost.empty:
        return minimum_cost.sort_values(["business_cost", "review_rate"], ascending=[True, True]).iloc[0]
    return shortlist.sort_values(["business_cost", "review_rate"], ascending=[True, True]).iloc[0]


def apply_thresholds_to_dataset(
    threshold_shortlist: pd.DataFrame,
    threshold_grid: pd.DataFrame,
    model_name: str,
    dataset_name: str,
) -> pd.DataFrame:
    """Apply validation-selected shortlist thresholds to another dataset threshold grid."""
    if threshold_shortlist.empty:
        raise ValueError("Threshold shortlist is empty.")
    model_grid = threshold_grid.loc[threshold_grid[MODEL_COLUMN].eq(model_name)].copy()
    if model_grid.empty:
        raise ValueError(f"No threshold grid rows found for model_name={model_name!r}.")
    selected = threshold_shortlist[["objective", "threshold"]].drop_duplicates().copy()
    output = selected.merge(model_grid, on="threshold", how="left", suffixes=("", "_metric"))
    output["model_name"] = model_name
    output["dataset"] = dataset_name
    return output


# Flexible wrapper so both older calls (validation_threshold_grid=...) and newer
# calls (threshold_grid=...) work safely.
_build_all_model_operational_threshold_comparison_impl = build_all_model_operational_threshold_comparison

def build_all_model_operational_threshold_comparison(
    threshold_grid: pd.DataFrame | None = None,
    validation_threshold_grid: pd.DataFrame | None = None,
    validation_results: pd.DataFrame | None = None,
    review_rate_cap: float = 0.30,
    min_recall: float = 0.60,
    objective_name: str = "minimum_cost_review_rate_le_30pct",
    config=None,
) -> pd.DataFrame:
    """Select the best operating threshold for every model under one business policy.

    Accepts both threshold_grid and validation_threshold_grid for compatibility.
    """
    grid = threshold_grid if threshold_grid is not None else validation_threshold_grid
    if grid is None:
        raise ValueError("A threshold_grid or validation_threshold_grid is required.")

    if config is not None:
        if isinstance(config, dict):
            review_rate_cap = float(config.get("review_rate_cap", review_rate_cap))
            min_recall = float(config.get("min_recall", min_recall))
        else:
            review_rate_cap = float(getattr(config, "review_rate_cap", review_rate_cap))
            min_recall = float(getattr(config, "min_recall", min_recall))

    return _build_all_model_operational_threshold_comparison_impl(
        threshold_grid=grid,
        validation_results=validation_results,
        review_rate_cap=review_rate_cap,
        min_recall=min_recall,
        objective_name=objective_name,
    )

