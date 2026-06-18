# Model Card - Canadian Retail Credit Risk XAI

## Model overview

- **Business purpose:** Early-warning retail credit default-risk ranking and manual-review prioritization.
- **Operational model:** `xgboost_weighted_baseline`
- **Operating threshold:** `0.560`
- **Threshold objective:** `minimum_cost_review_rate_le_30pct`
- **Target:** borrower default indicator.
- **Intended use:** decision support for portfolio monitoring and credit-risk review.
- **Out-of-scope use:** automated credit decline, pricing decisioning, adverse-action communication, or production use without independent validation, legal/privacy review, and fairness assessment.

## Model inventory

| item                   | value                                                                                 |
|:-----------------------|:--------------------------------------------------------------------------------------|
| business_use           | Early-warning retail credit default-risk ranking and manual-review prioritization     |
| model_scope            | Portfolio monitoring and decision-support; not an automated credit-decline engine     |
| target                 | Defaulter indicator                                                                   |
| operational_model      | xgboost_weighted_baseline                                                             |
| operating_threshold    | 0.56                                                                                  |
| threshold_objective    | minimum_cost_review_rate_le_30pct                                                     |
| modeling_rows          | 134417                                                                                |
| portfolio_default_rate | 0.0904126710163149                                                                    |
| feature_count          | 40                                                                                    |
| sensitive_proxy_use    | Excluded from baseline model; available only for permitted governance review          |
| leakage_control        | Repayment-derived variables excluded from modelling features                          |
| explainability_assets  | SHAP, anchor-style rules, counterfactual diagnostics, Deepchecks/fallback diagnostics |

## Validation and test performance

| evaluation_view                         | model_name                | dataset    |   threshold |   roc_auc |   pr_auc |   brier_score |   recall |   precision |       f1 |   balanced_accuracy |      mcc |   review_rate |   business_cost |   false_negative |   false_positive |   true_positive |   true_negative | selected_operating_threshold   |
|:----------------------------------------|:--------------------------|:-----------|------------:|----------:|---------:|--------------:|---------:|------------:|---------:|--------------------:|---------:|--------------:|----------------:|-----------------:|-----------------:|----------------:|----------------:|:-------------------------------|
| validation_default_0_50                 | xgboost_weighted_baseline | validation |        0.5  |  0.751186 | 0.226251 |      0.202026 | 0.715853 |    0.170923 | 0.275957 |            0.685353 | 0.219168 |      0.378664 |      5.755e+06  |              518 |             6330 |            1305 |           12010 | False                          |
| test_default_0_50                       | xgboost_weighted_baseline | test       |        0.5  |  0.747839 | 0.214653 |      0.201375 | 0.72079  |    0.173534 | 0.279723 |            0.689784 | 0.224775 |      0.375539 |      5.674e+06  |              509 |             6258 |            1314 |           12082 | False                          |
| validation_selected_operating_threshold | xgboost_weighted_baseline | validation |        0.56 |  0.751186 | 0.226251 |      0.202026 | 0.625891 |    0.190484 | 0.292077 |            0.680748 | 0.226857 |      0.297079 |      5.8345e+06 |              682 |             4849 |            1141 |           13491 | True                           |
| test_selected_operating_threshold       | xgboost_weighted_baseline | test       |        0.56 |  0.747839 | 0.214653 |      0.201375 | 0.622052 |    0.190877 | 0.292117 |            0.679973 | 0.226423 |      0.294649 |      5.8485e+06 |              689 |             4807 |            1134 |           13533 | True                           |

## Top explainability drivers and governance notes

| raw_feature                    | feature_label                  |   mean_abs_shap |   mean_shap |   transformed_feature_count |   positive_contribution_share | governance_note                                                                             | governance_action                                                        |
|:-------------------------------|:-------------------------------|----------------:|------------:|----------------------------:|------------------------------:|:--------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------|
| interest_rate                  | Interest Rate                  |       0.628578  |  0.00155322 |                           1 |                      0.562667 | Pricing/risk signal: explain carefully because pricing can reflect prior underwriting risk. | Monitor as top driver; include in periodic drift and reason-code review. |
| amount_missing_flag            | Amount Missing Flag            |       0.192372  |  0.0609848  |                           1 |                      0.298    | Data-quality signal: monitor for drift and avoid treating this as borrower behaviour alone. | Monitor as top driver; include in periodic drift and reason-code review. |
| amount_missing_raw_flag        | Amount Missing Raw Flag        |       0.181842  |  0.05832    |                           1 |                      0.298    | Data-quality signal: monitor for drift and avoid treating this as borrower behaviour alone. | Monitor as top driver; include in periodic drift and reason-code review. |
| broad_data_quality_issue_count | Broad Data Quality Issue Count |       0.158858  | -0.0328993  |                           1 |                      0.156    | Data-quality signal: monitor for drift and avoid treating this as borrower behaviour alone. | Monitor as top driver; include in periodic drift and reason-code review. |
| total_income_pa                | Total Income Pa                |       0.14197   | -0.0201614  |                           1 |                      0.520667 | Affordability/exposure signal: suitable for portfolio risk interpretation.                  | Monitor as top driver; include in periodic drift and reason-code review. |
| core_data_quality_issue_count  | Core Data Quality Issue Count  |       0.10624   | -0.077969   |                           1 |                      0.689333 | Data-quality signal: monitor for drift and avoid treating this as borrower behaviour alone. | Monitor as top driver; include in periodic drift and reason-code review. |
| amount                         | Amount                         |       0.0630213 | -0.0214208  |                           1 |                      0.525333 | Affordability/exposure signal: suitable for portfolio risk interpretation.                  | Monitor as top driver; include in periodic drift and reason-code review. |
| income_to_loan_buffer          | Income To Loan Buffer          |       0.0626191 |  0.0406544  |                           1 |                      0.799333 | Affordability/exposure signal: suitable for portfolio risk interpretation.                  | Monitor as top driver; include in periodic drift and reason-code review. |
| high_interest_flag             | High Interest Flag             |       0.0619946 |  0.0118545  |                           1 |                      0.322    | Pricing/risk signal: explain carefully because pricing can reflect prior underwriting risk. | Monitor as top driver; include in periodic drift and reason-code review. |
| tenure_years                   | Tenure Years                   |       0.0618124 | -0.0115693  |                           1 |                      0.636667 | Segment/product signal: monitor stability and business reasonableness.                      | Monitor as top driver; include in periodic drift and reason-code review. |

## Key controls

| control_area               | risk                                                            | control                                                                                      | evidence                                    | owner                       | frequency                   |
|:---------------------------|:----------------------------------------------------------------|:---------------------------------------------------------------------------------------------|:--------------------------------------------|:----------------------------|:----------------------------|
| Data ingestion             | Many-to-many joins or duplicate keys alter target rate.         | Use stable borrower-record key and duplicate checks.                                         | Notebook 01/05 record-key and split outputs | Risk analytics              | Each refresh                |
| Data quality               | Missing or inconsistent inputs distort model scores.            | Track missingness, range, and logical consistency; open data-quality issues for breaches.    | Notebook 02/03/08 diagnostics               | Data owner                  | Each refresh/monthly        |
| Leakage prevention         | Repayment-derived variables leak future outcome information.    | Exclude repayment-derived variables from modelling features and document exceptions.         | Feature policy                              | Model developer             | Before release              |
| Sensitive/proxy governance | Sensitive or proxy variables may cause unfair outcomes.         | Exclude direct sensitive fields from baseline model; use only for approved audit review.     | Feature policy and fairness/proxy review    | Model governance/compliance | Before release and annually |
| Model performance          | Ranking performance deteriorates after portfolio changes.       | Monitor PR-AUC, ROC-AUC, recall, precision, F1, Brier score when labels mature.              | Notebook 06/07 outputs                      | Model owner                 | Monthly/quarterly           |
| Threshold governance       | Threshold creates too many reviews or too many missed defaults. | Use validation-selected threshold with review-rate cap and test confirmation.                | Notebook 07 outputs                         | Credit strategy             | Quarterly/material change   |
| Explainability             | Model decisions cannot be understood by stakeholders.           | Maintain SHAP, local reasons, anchor-style rules, and counterfactual diagnostics.            | Notebook 08 outputs                         | Risk analytics              | Each release                |
| Monitoring                 | Score, feature, or data-quality drift changes model behaviour.  | Track score distribution, top-feature drift, review rate, realized default outcomes.         | Notebook 09 monitoring plan                 | Model monitoring team       | Monthly                     |
| Change management          | Uncontrolled model/file changes break reproducibility.          | Version model artifact, training code, features, threshold, and governance outputs together. | Git + artifact manifest                     | Model owner                 | Each release                |

## Monitoring limits

| metric                         | baseline                         | warning_limit   | breach_limit   | frequency           | action                                                           |
|:-------------------------------|:---------------------------------|:----------------|:---------------|:--------------------|:-----------------------------------------------------------------|
| Score PSI                      | Training/test score distribution | > 0.10          | > 0.25         | Monthly             | Investigate population shift, recalibration, or retraining need. |
| Top SHAP driver PSI            | Top XAI feature distributions    | > 0.10          | > 0.25         | Monthly             | Review feature drift and data source changes.                    |
| Review rate                    | 29.46%                           | > 34.46%        | > 39.46%       | Weekly/monthly      | Review threshold capacity and staffing impact.                   |
| Recall on matured labels       | 62.21%                           | < 57.21%        | < 52.21%       | After labels mature | Assess missed-default concentration and refresh need.            |
| Precision on reviewed accounts | 19.09%                           | < 16.09%        | < 14.09%       | After labels mature | Review false-positive burden and threshold.                      |
| Critical feature missingness   | Notebook 02/03 profile           | +25% relative   | +50% relative  | Each refresh        | Open data-quality incident and assess model use pause.           |

## Limitations

- This is a portfolio project built on available/synthetic project data and must not be treated as production banking advice.
- Counterfactuals are diagnostic scenario analysis, not customer instructions.
- Threshold cost assumptions are illustrative scenario assumptions, not accounting estimates.
