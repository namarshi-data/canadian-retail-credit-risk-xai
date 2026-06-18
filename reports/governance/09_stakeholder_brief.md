# Stakeholder Brief - Retail Credit Default-Risk Model

## What the model does

The model ranks borrowers by estimated default risk so that a credit-risk team can prioritize manual review and portfolio monitoring.

## Recommended operating point

- **Operational model:** xgboost_weighted_baseline
- **Operating threshold:** 0.560
- **Test recall:** 62.21%
- **Test precision:** 19.09%
- **Test review rate:** 29.46%

## Business interpretation

At the selected threshold, the model captures a meaningful share of future defaults while keeping the review population close to the operational cap used in this project.

## Main model drivers

| raw_feature                    | feature_label                  |   mean_abs_shap |   mean_shap |   transformed_feature_count |   positive_contribution_share | governance_note                                                                             | governance_action                                                        |
|:-------------------------------|:-------------------------------|----------------:|------------:|----------------------------:|------------------------------:|:--------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------|
| interest_rate                  | Interest Rate                  |        0.628578 |  0.00155322 |                           1 |                      0.562667 | Pricing/risk signal: explain carefully because pricing can reflect prior underwriting risk. | Monitor as top driver; include in periodic drift and reason-code review. |
| amount_missing_flag            | Amount Missing Flag            |        0.192372 |  0.0609848  |                           1 |                      0.298    | Data-quality signal: monitor for drift and avoid treating this as borrower behaviour alone. | Monitor as top driver; include in periodic drift and reason-code review. |
| amount_missing_raw_flag        | Amount Missing Raw Flag        |        0.181842 |  0.05832    |                           1 |                      0.298    | Data-quality signal: monitor for drift and avoid treating this as borrower behaviour alone. | Monitor as top driver; include in periodic drift and reason-code review. |
| broad_data_quality_issue_count | Broad Data Quality Issue Count |        0.158858 | -0.0328993  |                           1 |                      0.156    | Data-quality signal: monitor for drift and avoid treating this as borrower behaviour alone. | Monitor as top driver; include in periodic drift and reason-code review. |
| total_income_pa                | Total Income Pa                |        0.14197  | -0.0201614  |                           1 |                      0.520667 | Affordability/exposure signal: suitable for portfolio risk interpretation.                  | Monitor as top driver; include in periodic drift and reason-code review. |

## How this should be used

Use the score to support analyst review, portfolio segmentation, and monitoring. Do not use it as a standalone automated lending decision without additional validation, fairness testing, compliance review, and production monitoring.
