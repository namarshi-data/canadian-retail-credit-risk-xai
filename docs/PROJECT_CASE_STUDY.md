# Project Case Study

## Canadian Retail Credit Risk Analytics

### One-Line Summary

Built an end-to-end credit risk analytics workflow that ranks borrowers by default risk, selects an operational review threshold, explains model decisions, and documents governance controls for a Canadian retail lending portfolio.

---

## 1. Business Problem

Retail lenders need to detect elevated default risk early enough to support portfolio monitoring and manual review. A useful risk model must be accurate enough to rank borrowers, but it must also be explainable, auditable, operationally feasible, and governed.

This project answers five business questions:

1. Which borrower segments have elevated observed default risk?
2. Which data-quality issues affect risk monitoring?
3. Which machine-learning model ranks default risk best under class imbalance?
4. Which score threshold fits review capacity and risk appetite?
5. What governance documentation is needed before responsible use?

---

## 2. Data Challenge

The raw data was supplied across multiple Excel sheets. The most important data engineering issue was **record-grain integrity**. Some borrower IDs appeared more than once across source sheets, so a naive merge on borrower ID alone could inflate the dataset and distort portfolio metrics.

The project uses a safer grain-validation approach to preserve borrower records and avoid many-to-many merge inflation.

---

## 3. Data Quality and Leakage Controls

The workflow documents:

- Missingness by feature and target
- Duplicate-record review
- Logical consistency checks
- Missingness flags
- Data-quality issue counts
- Leakage-prone repayment-derived variables
- Sensitive/proxy-sensitive variables excluded from the baseline model

Repayment-derived variables are kept for monitoring analysis where appropriate, but excluded from predictive modelling when they could leak target information.

---

## 4. Portfolio Monitoring

Portfolio analysis reviews risk across borrower and loan dimensions such as:

- Loan category
- Employment type
- Interest-rate bands
- Income and affordability profile
- Loan-to-income bands
- Data-quality/missingness flags

This gives the project a finance-analytics foundation before modelling begins.

---

## 5. Model Development

Candidate models include:

- Logistic Regression baseline
- Random Forest baseline and challenger
- XGBoost weighted baseline and challenger
- Optional train-only resampling challengers

Model selection prioritizes imbalanced-classification and credit-risk metrics:

- ROC-AUC
- PR-AUC
- Recall
- Precision
- F1
- MCC
- Brier score
- Review rate
- Illustrative business cost

Accuracy is not treated as the main metric because the default rate is approximately 9.04%.

---

## 6. Threshold Strategy

The final operational threshold is selected on validation data, then confirmed on the test set.

| Decision item | Value |
|---|---:|
| Champion operating model | `xgboost_weighted_baseline` |
| Selected threshold | 0.560 |
| Threshold objective | `minimum_cost_review_rate_le_30pct` |
| Test recall | 62.21% |
| Test precision | 19.09% |
| Test review rate | 29.46% |
| Test business cost | $5.85M |

The selected threshold supports operational review capacity by keeping the review rate below 30% while still capturing approximately 62.21% of defaults in the held-out test set.

---

## 7. Explainability

The explainability layer includes:

- Global SHAP drivers
- Grouped SHAP feature importance
- Dependence-style plots
- Individual borrower-level reason summaries
- Anchor-style high-risk rules
- Counterfactual sensitivity scenarios
- Deepchecks diagnostics

Top SHAP drivers include `interest_rate, amount_missing_flag, amount_missing_raw_flag, broad_data_quality_issue_count, total_income_pa, core_data_quality_issue_count`.

Counterfactuals are documented as diagnostic model-sensitivity scenarios, not customer instructions.

---

## 8. Governance Outputs

Notebook 09 creates:

- Model card
- Validation summary
- Stakeholder brief
- Monitoring plan
- Model control register
- Risk-limit register
- Monitoring KPI snapshot
- Governance summary

Governance decision: the model is suitable for portfolio analytics and manual-review prioritization demonstration, subject to validation, fairness testing, legal/privacy review, and production monitoring before any real deployment.

---

## 9. What This Demonstrates to Employers

This project demonstrates the ability to:

- Translate a credit-risk problem into an analytical workflow
- Build reproducible Python pipelines
- Handle data-quality and leakage issues
- Evaluate imbalanced models appropriately
- Convert model scores into operational decisions
- Explain model outputs to stakeholders
- Produce model governance documentation
- Think like a finance/risk analytics professional rather than only a coder

---

## 10. Limitations

- The dataset is for portfolio demonstration, not production banking.
- Business-cost assumptions are illustrative.
- Counterfactuals are diagnostic only.
- Production use would require independent model validation, fairness testing, calibration review, privacy/legal review, monitoring automation, and governance approval.
