# Model Monitoring Plan

## Monitoring objective

Ensure the credit-risk model remains stable, explainable, and operationally useful after deployment or future data refreshes.

## KPI snapshot

| kpi                    | value                                                                                                        | interpretation                                               |
|:-----------------------|:-------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------|
| Operational model      | xgboost_weighted_baseline                                                                                    | Model-threshold pair selected by validation business policy. |
| Portfolio default rate | 9.04%                                                                                                        | Base rate for interpreting precision and review workload.    |
| Threshold objective    | minimum_cost_review_rate_le_30pct                                                                            | Business rule used for threshold selection.                  |
| Operating threshold    | 0.560                                                                                                        | Probability cutoff for manual-review flag.                   |
| Test recall            | 62.21%                                                                                                       | Share of defaults captured at selected threshold.            |
| Test precision         | 19.09%                                                                                                       | Share of reviewed accounts that defaulted.                   |
| Test review rate       | 29.46%                                                                                                       | Operational workload from the selected threshold.            |
| Test business cost     | $5,848,500                                                                                                   | Scenario cost from false positives and false negatives.      |
| Top SHAP drivers       | interest_rate, amount_missing_flag, amount_missing_raw_flag, broad_data_quality_issue_count, total_income_pa | Main explanation drivers to monitor for drift.               |

## Risk limits and escalation triggers

| metric                         | baseline                         | warning_limit   | breach_limit   | frequency           | action                                                           |
|:-------------------------------|:---------------------------------|:----------------|:---------------|:--------------------|:-----------------------------------------------------------------|
| Score PSI                      | Training/test score distribution | > 0.10          | > 0.25         | Monthly             | Investigate population shift, recalibration, or retraining need. |
| Top SHAP driver PSI            | Top XAI feature distributions    | > 0.10          | > 0.25         | Monthly             | Review feature drift and data source changes.                    |
| Review rate                    | 29.46%                           | > 34.46%        | > 39.46%       | Weekly/monthly      | Review threshold capacity and staffing impact.                   |
| Recall on matured labels       | 62.21%                           | < 57.21%        | < 52.21%       | After labels mature | Assess missed-default concentration and refresh need.            |
| Precision on reviewed accounts | 19.09%                           | < 16.09%        | < 14.09%       | After labels mature | Review false-positive burden and threshold.                      |
| Critical feature missingness   | Notebook 02/03 profile           | +25% relative   | +50% relative  | Each refresh        | Open data-quality incident and assess model use pause.           |

## Suggested cadence

- **Each data refresh:** schema checks, duplicate-key checks, missingness checks, and leakage-policy checks.
- **Monthly:** score distribution, review rate, feature drift, data-quality drift, top-SHAP-driver drift.
- **After labels mature:** realized default rate, recall, precision, false negatives, and false positives.
- **Quarterly or material-change event:** threshold review, challenger review, governance sign-off.

## Escalation actions

If a breach occurs, pause automated refreshes if necessary, document the issue, identify root cause, quantify borrower/business impact, and decide whether remediation, recalibration, threshold adjustment, or retraining is required.
