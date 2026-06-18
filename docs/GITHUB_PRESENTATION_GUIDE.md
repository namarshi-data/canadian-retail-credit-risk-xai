# GitHub Presentation Guide

## Repository Name

```text
canadian-retail-credit-risk-xai
```

## Repository Description

```text
End-to-end Canadian retail credit risk analytics project covering default prediction, portfolio monitoring, explainable AI, threshold optimization, and model governance.
```

## Suggested GitHub Topics

```text
credit-risk
risk-analytics
machine-learning
xgboost
random-forest
explainable-ai
shap
model-governance
model-risk-management
portfolio-monitoring
python
banking-analytics
financial-analytics
```

---

## README Layout to Prioritize

The README should allow a recruiter or hiring manager to understand the project in under 60 seconds.

Recommended top-of-page order:

1. Project title and badges
2. Executive summary
3. Portfolio positioning for Canadian finance roles
4. Key results table
5. Business impact
6. Workflow diagram
7. Notebook workflow table
8. Example visuals
9. Governance outputs
10. Limitations and intended use

---

## Key Results to Pin

Use these final results consistently:

| Item | Value |
|---|---:|
| Records | 134,417 |
| Observed default rate | 9.04% |
| Champion operating model | `xgboost_weighted_baseline` |
| Operating threshold | 0.560 |
| Test ROC-AUC | 0.7478 |
| Test PR-AUC | 0.2147 |
| Test recall | 62.21% |
| Test precision | 19.09% |
| Test review rate | 29.46% |
| Test business cost | $5.85M |

---

## Recommended README Figures

Based on the current `reports/figures/` screenshot, use these paths:

```markdown
![Portfolio target distribution](reports/figures/portfolio_target_distribution.png)
![Default rate by loan category](reports/figures/default_rate_by_loan_category.png)
![Default rate by interest-rate quantile](reports/figures/default_rate_by_interest_rate_quantile.png)
![Global SHAP drivers](reports/figures/08_xai_global_grouped_shap_top_features.png)
![SHAP summary beeswarm](reports/figures/08_xai_shap_summary_beeswarm.png)
```

Use three visuals in the main README to avoid clutter. Put extra visuals in notebooks or governance/report folders.

---

## Governance File Naming Recommendation

Your screenshot shows both older unprefixed governance markdown files and newer `09_`-prefixed files. For final GitHub presentation, prefer the final `09_` versions:

```text
reports/governance/09_model_card.md
reports/governance/09_model_validation_summary.md
reports/governance/09_model_monitoring_plan.md
reports/governance/09_stakeholder_brief.md
```

Archive or remove older duplicates if they contain stale threshold or model values.

---

## What to Commit

Commit:

- `README.md`
- `LICENSE`
- `.gitignore`
- `requirements.txt`
- `pyproject.toml`
- `config/`
- `src/credit_risk/`
- `scripts/`
- `notebooks/`
- `docs/`
- Safe aggregate tables under `reports/tables/`
- Safe figures under `reports/figures/`
- Governance markdown files under `reports/governance/`

Do not commit:

- Raw Excel workbook
- Interim or processed datasets
- Row-level predictions
- Row-level SHAP values
- Counterfactual rows if borrower-level
- Model binaries such as `.joblib` or `.pkl`
- `.env`, Neptune tokens, or local credentials
- `.venv/`

---

## Suggested Final Commit

Run `git status` first. Then stage safe files only.

```bash
git add README.md LICENSE .gitignore requirements.txt pyproject.toml config/ src/ scripts/ notebooks/ docs/ reports/figures/ reports/governance/
git status
git commit -m "Complete Canadian retail credit risk XAI portfolio project"
```

If unsafe files are staged, unstage them:

```bash
git restore --staged data/raw data/interim data/processed reports/model_artifacts
git restore --staged reports/tables/*predictions*.csv reports/tables/*shap_values*.csv
```

---

## Final Quality Checklist

- [ ] README renders cleanly on GitHub
- [ ] Repository About section is filled out
- [ ] GitHub topics are added
- [ ] Raw data is not committed
- [ ] Model artifacts are not committed
- [ ] No `.env` or credentials are committed
- [ ] Final metrics are consistent across README and docs
- [ ] Notebook 07 shows threshold `0.560`
- [ ] Notebook 08 shows champion `xgboost_weighted_baseline`
- [ ] Notebook 09 governance outputs use latest metrics
- [ ] Figures render correctly
- [ ] Project is pinned on GitHub profile
