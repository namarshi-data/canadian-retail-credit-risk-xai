# Model Validation Summary

## Executive summary

This model is positioned as a **manual-review prioritization and portfolio-monitoring tool**. It is not approved as a standalone automated credit decision engine.

| operational_model         |   modeling_rows |   portfolio_default_rate |   operating_threshold | threshold_objective               |   test_recall |   test_precision |   test_review_rate |   test_business_cost | primary_governance_decision                                                                         |
|:--------------------------|----------------:|-------------------------:|----------------------:|:----------------------------------|--------------:|-----------------:|-------------------:|---------------------:|:----------------------------------------------------------------------------------------------------|
| xgboost_weighted_baseline |          134417 |                0.0904127 |                  0.56 | minimum_cost_review_rate_le_30pct |      0.622052 |         0.190877 |           0.294649 |           5.8485e+06 | Use as decision-support/manual-review prioritization model, not as automated credit-decline engine. |

## Performance evidence

| evaluation_view                         | model_name                | dataset    |   threshold |   roc_auc |   pr_auc |   brier_score |   recall |   precision |       f1 |   balanced_accuracy |      mcc |   review_rate |   business_cost |   false_negative |   false_positive |   true_positive |   true_negative | selected_operating_threshold   |
|:----------------------------------------|:--------------------------|:-----------|------------:|----------:|---------:|--------------:|---------:|------------:|---------:|--------------------:|---------:|--------------:|----------------:|-----------------:|-----------------:|----------------:|----------------:|:-------------------------------|
| validation_default_0_50                 | xgboost_weighted_baseline | validation |        0.5  |  0.751186 | 0.226251 |      0.202026 | 0.715853 |    0.170923 | 0.275957 |            0.685353 | 0.219168 |      0.378664 |      5.755e+06  |              518 |             6330 |            1305 |           12010 | False                          |
| test_default_0_50                       | xgboost_weighted_baseline | test       |        0.5  |  0.747839 | 0.214653 |      0.201375 | 0.72079  |    0.173534 | 0.279723 |            0.689784 | 0.224775 |      0.375539 |      5.674e+06  |              509 |             6258 |            1314 |           12082 | False                          |
| validation_selected_operating_threshold | xgboost_weighted_baseline | validation |        0.56 |  0.751186 | 0.226251 |      0.202026 | 0.625891 |    0.190484 | 0.292077 |            0.680748 | 0.226857 |      0.297079 |      5.8345e+06 |              682 |             4849 |            1141 |           13491 | True                           |
| test_selected_operating_threshold       | xgboost_weighted_baseline | test       |        0.56 |  0.747839 | 0.214653 |      0.201375 | 0.622052 |    0.190877 | 0.292117 |            0.679973 | 0.226423 |      0.294649 |      5.8485e+06 |              689 |             4807 |            1134 |           13533 | True                           |

## Feature governance

| reason                                                                                  |   feature_count |
|:----------------------------------------------------------------------------------------|----------------:|
| Deterministic, row-level, explainable feature not marked as leakage or high-risk proxy. |              37 |
| Repayment-derived/post-origination information may leak outcome timing.                 |               7 |
| Sensitive or high-risk proxy field retained for fairness/governance monitoring.         |               4 |
| Potential socioeconomic/operational proxy; can be used only with documented monitoring. |               3 |
| High-cardinality/encrypted field is not explainable enough for first baseline model.    |               2 |
| Identifier/audit grain; useful for traceability but not predictive modelling.           |               2 |
| May be valid only if known at prediction time; do not include in conservative baseline. |               2 |
| Target variable; never used as a predictor.                                             |               1 |

## XAI governance summary

| raw_feature                    | feature_label                  |   mean_abs_shap |    mean_shap |   transformed_feature_count |   positive_contribution_share | governance_note                                                                             | governance_action                                                        |
|:-------------------------------|:-------------------------------|----------------:|-------------:|----------------------------:|------------------------------:|:--------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------|
| interest_rate                  | Interest Rate                  |       0.628578  |  0.00155322  |                           1 |                      0.562667 | Pricing/risk signal: explain carefully because pricing can reflect prior underwriting risk. | Monitor as top driver; include in periodic drift and reason-code review. |
| amount_missing_flag            | Amount Missing Flag            |       0.192372  |  0.0609848   |                           1 |                      0.298    | Data-quality signal: monitor for drift and avoid treating this as borrower behaviour alone. | Monitor as top driver; include in periodic drift and reason-code review. |
| amount_missing_raw_flag        | Amount Missing Raw Flag        |       0.181842  |  0.05832     |                           1 |                      0.298    | Data-quality signal: monitor for drift and avoid treating this as borrower behaviour alone. | Monitor as top driver; include in periodic drift and reason-code review. |
| broad_data_quality_issue_count | Broad Data Quality Issue Count |       0.158858  | -0.0328993   |                           1 |                      0.156    | Data-quality signal: monitor for drift and avoid treating this as borrower behaviour alone. | Monitor as top driver; include in periodic drift and reason-code review. |
| total_income_pa                | Total Income Pa                |       0.14197   | -0.0201614   |                           1 |                      0.520667 | Affordability/exposure signal: suitable for portfolio risk interpretation.                  | Monitor as top driver; include in periodic drift and reason-code review. |
| core_data_quality_issue_count  | Core Data Quality Issue Count  |       0.10624   | -0.077969    |                           1 |                      0.689333 | Data-quality signal: monitor for drift and avoid treating this as borrower behaviour alone. | Monitor as top driver; include in periodic drift and reason-code review. |
| amount                         | Amount                         |       0.0630213 | -0.0214208   |                           1 |                      0.525333 | Affordability/exposure signal: suitable for portfolio risk interpretation.                  | Monitor as top driver; include in periodic drift and reason-code review. |
| income_to_loan_buffer          | Income To Loan Buffer          |       0.0626191 |  0.0406544   |                           1 |                      0.799333 | Affordability/exposure signal: suitable for portfolio risk interpretation.                  | Monitor as top driver; include in periodic drift and reason-code review. |
| high_interest_flag             | High Interest Flag             |       0.0619946 |  0.0118545   |                           1 |                      0.322    | Pricing/risk signal: explain carefully because pricing can reflect prior underwriting risk. | Monitor as top driver; include in periodic drift and reason-code review. |
| tenure_years                   | Tenure Years                   |       0.0618124 | -0.0115693   |                           1 |                      0.636667 | Segment/product signal: monitor stability and business reasonableness.                      | Monitor as top driver; include in periodic drift and reason-code review. |
| loan_to_income_missing_flag    | Loan To Income Missing Flag    |       0.0616362 |  0.010457    |                           1 |                      0.298    | Data-quality signal: monitor for drift and avoid treating this as borrower behaviour alone. | Monitor as top driver; include in periodic drift and reason-code review. |
| loan_category                  | Loan Category                  |       0.0567772 | -0.000154568 |                           1 |                      0.567333 | Segment/product signal: monitor stability and business reasonableness.                      | Monitor as top driver; include in periodic drift and reason-code review. |

## Validation decision

The model is acceptable for portfolio analytics and decision-support demonstration purposes, subject to the documented monitoring plan and limitations. Before production use, it would require independent model validation, data lineage review, calibration review, fairness testing, privacy/legal review, and user-acceptance testing with credit-risk stakeholders.
