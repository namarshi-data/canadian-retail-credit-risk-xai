# Interview Talking Points

## 30-Second Project Pitch

I built an end-to-end Canadian retail credit risk analytics project that predicts borrower default risk and supports portfolio monitoring, explainability, and model governance. The workflow starts with multi-sheet Excel ingestion and record-grain validation, then moves through data-quality review, leakage-controlled feature engineering, Logistic Regression, Random Forest, and XGBoost modelling, validation-based threshold selection under a manual-review cap, SHAP explainability, counterfactual diagnostics, and governance documentation including a model card, validation summary, control register, and monitoring plan.

---

## 60-Second Project Pitch

This project simulates how a Canadian retail lender could prioritize borrowers for credit-risk review. I first validated the data grain across multiple Excel sheets because duplicate borrower IDs can create many-to-many merge inflation if merged carelessly. After fixing the merge logic, I performed data-quality checks, created missingness flags, and analyzed risk by loan category, employment profile, interest-rate band, income, and loan-to-income profile.

For modelling, I excluded repayment-derived leakage variables and sensitive/proxy-sensitive variables from the baseline model. I trained Logistic Regression, Random Forest, and XGBoost models and selected an XGBoost weighted baseline as the operating champion. Instead of relying on the default 0.50 cutoff, I selected an operating threshold of **0.560** using validation data under a 30% manual-review cap. On the test set, the model captured **62.21%** of defaults with **19.09%** precision and a **29.46%** review rate. I completed the project with SHAP explainability, local explanations, counterfactual diagnostics, a model card, validation summary, control register, risk limits, and monitoring plan.

---

## STAR Story: Data Quality Issue

**Situation:** The project used multiple Excel sheets containing borrower, loan, employment, and credit-behaviour information.

**Task:** I needed to create a reliable modelling dataset without inflating rows or distorting default rates.

**Action:** I reviewed duplicate borrower IDs and validated the record grain before merging. Instead of merging only on borrower ID, I preserved record sequencing to avoid many-to-many merge inflation.

**Result:** The final modelling dataset preserved the intended population and protected downstream portfolio KPIs, model metrics, and governance conclusions.

---

## Common Interview Questions

### Why did you frame this as early-warning risk review instead of automated credit approval?

Because credit-risk models require strong validation, fairness review, governance, monitoring, and policy approval before they can support automated credit decisions. This project uses behavioural and portfolio-level information, so positioning it as a review-prioritization and portfolio-monitoring tool is more realistic and responsible.

### What was the most important data issue?

Record-grain integrity. Duplicate borrower IDs across source sheets meant that a naive merge could inflate row counts. I solved this by validating the grain and preserving record sequencing before merging.

### How did you handle class imbalance?

The default rate was about 9.04%, so accuracy was not the primary metric. I used class-weighted models and evaluated ROC-AUC, PR-AUC, recall, precision, F1, MCC, review rate, and business cost. I also selected the threshold using validation data rather than relying on 0.50.

### Why did you exclude repayment-derived variables?

Repayment-derived variables may contain information that occurs after or near the target event, creating leakage. I retained those variables for monitoring where appropriate but excluded them from the predictive model feature set.

### Why keep missingness flags?

Missingness was not purely random. Missing loan amount and data-quality flags appeared to carry risk information. I kept missingness indicators but documented that these are operational/data-quality signals and should not be interpreted as borrower behaviour alone.

### Why was XGBoost selected?

XGBoost provided strong ranking performance under class imbalance and worked well with SHAP for explainability. The operating champion was selected using validation results and an operational threshold policy rather than accuracy alone.

### Why not optimize for 95% accuracy?

In an imbalanced default dataset, a model can achieve high accuracy by predicting most borrowers as non-default. Credit-risk teams care more about ranking risky borrowers, capturing defaults, managing false positives, and keeping review workload feasible.

### What does the threshold mean?

The threshold of **0.560** is the probability cutoff used to send borrowers to manual review. It was selected on validation data using a business-cost and review-cap policy, then confirmed on test data.

### How did you explain the model?

I used SHAP for global and local explanations, dependence-style plots for key numerical drivers, anchor-style rules for business-readable high-risk patterns, and counterfactual diagnostics to understand score sensitivity.

### What are the main governance controls?

The project documents record-grain validation, data-quality checks, leakage prevention, sensitive/proxy-feature exclusion, validation-based threshold selection, test confirmation, explainability review, control registers, monitoring KPIs, and risk limits.

### How would you improve it in a real bank?

I would add time-based validation, SQL/Power BI monitoring dashboards, production data lineage, fairness testing, reject inference if applicable, calibration monitoring, champion/challenger tracking, independent model validation, and workflow integration with credit-risk operations.

---

## Resume Bullets

- Built an end-to-end Canadian retail credit risk analytics project in Python covering data ingestion, data-quality profiling, portfolio monitoring, feature engineering, model training, threshold selection, explainability, and governance documentation.
- Trained and evaluated Logistic Regression, Random Forest, and XGBoost default-risk models on 134,417 borrower records using ROC-AUC, PR-AUC, recall, precision, F1, MCC, review rate, and illustrative business-cost metrics.
- Selected a validation-based operating threshold of 0.560, capturing 62.21% of defaults on the test set while limiting the manual-review population to 29.46%.
- Implemented leakage controls by excluding repayment-derived variables from modelling and retaining sensitive/proxy-sensitive fields only for governance review.
- Produced SHAP explainability, local borrower-level explanations, counterfactual diagnostics, model card, validation summary, control register, risk-limit register, and monitoring plan.

---

## Interview Closer

The main value of this project is that it connects technical modelling with how credit-risk teams actually use analytics: monitoring portfolio risk, prioritizing reviews, explaining drivers, documenting controls, and preparing for ongoing monitoring. I intentionally included threshold strategy, explainability, and governance instead of stopping at model accuracy.
