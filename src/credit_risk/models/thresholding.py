from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from credit_risk.models.evaluate import (
    DEFAULT_FALSE_NEGATIVE_COST,
    DEFAULT_FALSE_POSITIVE_COST,
    classification_metrics_at_threshold,
)

TARGET_COLUMN = "defaulter"
MODEL_COLUMN = "model_name"
PROBABILITY_COLUMN = "predicted_default_probability"


@dataclass(frozen=True)
class ThresholdCostAssumptions:
    """Simple business-cost assumptions for threshold comparison.

    The values are intentionally illustrative, not accounting guidance. They are used
    only to compare threshold trade-offs consistently across candidate models.
    """

    false_negative_cost: float = DEFAULT_FALSE_NEGATIVE_COST
    false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST

    @property
    def cost_ratio_threshold(self) -> float:
        """The theoretical probability cutoff when probabilities are calibrated."""
        denominator = self.false_negative_cost + self.false_positive_cost
        if denominator <= 0:
            return 0.50
        return self.false_positive_cost / denominator


def build_threshold_grid(
    start: float = 0.01,
    stop: float = 0.99,
    step: float = 0.005,
) -> np.ndarray:
    """Create a stable threshold grid for business-threshold analysis."""
    if not (0 < start < 1 and 0 < stop < 1):
        raise ValueError("start and stop must be between 0 and 1.")
    if start >= stop:
        raise ValueError("start must be lower than stop.")
    if step <= 0:
        raise ValueError("step must be positive.")
    return np.round(np.arange(start, stop + step / 2, step), 3)


def validate_prediction_frame(predictions: pd.DataFrame) -> None:
    """Validate the prediction frame expected from Notebook 06."""
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
    """Evaluate one model over a grid of operating thresholds."""
    validate_prediction_frame(predictions)
    thresholds = list(build_threshold_grid() if thresholds is None else thresholds)
    cost_assumptions = cost_assumptions or ThresholdCostAssumptions()

    model_predictions = predictions.loc[predictions[MODEL_COLUMN].eq(model_name)].copy()
    if model_predictions.empty:
        raise ValueError(f"No predictions found for model_name={model_name!r}.")

    rows: list[dict[str, float | int | str]] = []
    for threshold in thresholds:
        metrics = classification_metrics_at_threshold(
            model_predictions[TARGET_COLUMN].astype(int),
            model_predictions[PROBABILITY_COLUMN].astype(float),
            threshold=float(threshold),
            false_negative_cost=cost_assumptions.false_negative_cost,
            false_positive_cost=cost_assumptions.false_positive_cost,
        )
        rows.append(
            {
                "model_name": model_name,
                "dataset": dataset_name or str(model_predictions.get("split", pd.Series(["unknown"])).iloc[0]),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def evaluate_threshold_grid_all_models(
    predictions: pd.DataFrame,
    thresholds: Iterable[float] | None = None,
    dataset_name: str | None = None,
    cost_assumptions: ThresholdCostAssumptions | None = None,
) -> pd.DataFrame:
    """Evaluate all models in a prediction frame over the same threshold grid."""
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
    return pd.concat(frames, ignore_index=True)


def _pick_row(
    grid: pd.DataFrame,
    objective_name: str,
    sort_column: str,
    ascending: bool,
    filter_mask: pd.Series | None = None,
) -> pd.DataFrame:
    working = grid.loc[filter_mask].copy() if filter_mask is not None else grid.copy()
    if working.empty:
        return pd.DataFrame()
    row = working.sort_values(
        [sort_column, "business_cost", "review_rate"],
        ascending=[ascending, True, True],
    ).head(1).copy()
    row.insert(0, "objective", objective_name)
    return row


def build_threshold_shortlist(
    threshold_grid: pd.DataFrame,
    model_name: str,
    review_rate_caps: Iterable[float] = (0.25, 0.30, 0.35, 0.40),
    recall_floors: Iterable[float] = (0.60, 0.70, 0.75),
) -> pd.DataFrame:
    """Create a shortlist of threshold candidates for governance review."""
    model_grid = threshold_grid.loc[threshold_grid[MODEL_COLUMN].eq(model_name)].copy()
    if model_grid.empty:
        raise ValueError(f"No threshold grid rows found for model_name={model_name!r}.")

    rows = [
        _pick_row(model_grid, "default_threshold_0_50", "threshold", True, model_grid["threshold"].eq(0.50)),
        _pick_row(model_grid, "minimum_business_cost", "business_cost", True),
        _pick_row(model_grid, "maximum_f1", "f1", False),
        _pick_row(model_grid, "maximum_mcc", "mcc", False),
        _pick_row(model_grid, "maximum_balanced_accuracy", "balanced_accuracy", False),
    ]

    for cap in review_rate_caps:
        rows.append(
            _pick_row(
                model_grid,
                f"minimum_cost_review_rate_le_{int(cap * 100)}pct",
                "business_cost",
                True,
                model_grid["review_rate"].le(cap),
            )
        )

    for floor in recall_floors:
        rows.append(
            _pick_row(
                model_grid,
                f"minimum_cost_recall_ge_{int(floor * 100)}pct",
                "business_cost",
                True,
                model_grid["recall"].ge(floor),
            )
        )

    shortlist = pd.concat([row for row in rows if not row.empty], ignore_index=True)
    shortlist = shortlist.drop_duplicates(subset=["objective", "threshold"]).reset_index(drop=True)
    return shortlist


def apply_thresholds_to_dataset(
    threshold_shortlist: pd.DataFrame,
    threshold_grid: pd.DataFrame,
    model_name: str,
    dataset_name: str,
) -> pd.DataFrame:
    """Apply validation-selected thresholds to another dataset's threshold grid."""
    model_grid = threshold_grid.loc[threshold_grid[MODEL_COLUMN].eq(model_name)].copy()
    selected_thresholds = threshold_shortlist[["objective", "threshold"]].drop_duplicates()
    output = selected_thresholds.merge(model_grid, on="threshold", how="left", suffixes=("", "_metric"))
    output["model_name"] = model_name
    output["dataset"] = dataset_name
    return output


def recommend_operating_threshold(
    shortlist: pd.DataFrame,
    preferred_objective: str = "minimum_cost_review_rate_le_30pct",
) -> pd.Series:
    """Pick a recommended operating threshold from the shortlist.

    The default recommendation uses a 30% review-rate cap to reflect limited manual
    credit-review capacity. If that row is unavailable, the minimum-cost row is used.
    """
    if shortlist.empty:
        raise ValueError("Threshold shortlist is empty.")
    preferred = shortlist.loc[shortlist["objective"].eq(preferred_objective)]
    if not preferred.empty:
        return preferred.iloc[0]
    fallback = shortlist.loc[shortlist["objective"].eq("minimum_business_cost")]
    if not fallback.empty:
        return fallback.iloc[0]
    return shortlist.sort_values("business_cost", ascending=True).iloc[0]


def cost_assumptions_frame(cost_assumptions: ThresholdCostAssumptions) -> pd.DataFrame:
    """Return cost assumptions as an auditable table."""
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
                "description": "Illustrative operational/customer-friction cost of reviewing a non-default account.",
            },
            {
                "assumption": "calibrated_probability_cost_ratio_threshold",
                "value": cost_assumptions.cost_ratio_threshold,
                "description": "Theoretical cutoff if model probabilities were perfectly calibrated.",
            },
        ]
    )


def save_threshold_outputs(
    table_dir,
    validation_grid: pd.DataFrame,
    test_grid: pd.DataFrame,
    validation_shortlist: pd.DataFrame,
    test_shortlist: pd.DataFrame,
    recommendation: pd.Series,
    cost_assumptions: ThresholdCostAssumptions,
) -> None:
    """Persist threshold-selection outputs."""
    table_dir.mkdir(parents=True, exist_ok=True)
    validation_grid.to_csv(table_dir / "threshold_grid_validation_all_models.csv", index=False)
    test_grid.to_csv(table_dir / "threshold_grid_test_all_models.csv", index=False)
    validation_shortlist.to_csv(table_dir / "champion_threshold_shortlist_validation.csv", index=False)
    test_shortlist.to_csv(table_dir / "champion_threshold_shortlist_test.csv", index=False)
    pd.DataFrame([recommendation.to_dict()]).to_csv(table_dir / "recommended_threshold_summary.csv", index=False)
    cost_assumptions_frame(cost_assumptions).to_csv(table_dir / "business_cost_assumptions.csv", index=False)
