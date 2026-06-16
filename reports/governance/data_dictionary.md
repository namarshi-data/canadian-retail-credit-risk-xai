# Data Dictionary - Initial Schema Review

This dictionary is based on the source workbook structure reviewed during project setup.

## loan_information

| Column | Meaning | Initial role |
|---|---|---|
| User_id | Borrower/customer identifier | Join key; not a model feature |
| Loan Category | Loan product/category | Categorical feature |
| Amount | Loan amount | Numeric feature |
| Interest Rate | Contract interest rate | Numeric feature |
| Tenure(years) | Loan term in years | Numeric feature |

## Employment

| Column | Meaning | Initial role |
|---|---|---|
| User id | Borrower/customer identifier | Join key; not a model feature |
| Employmet type | Employment type; typo retained in raw source | Categorical feature after renaming |
| Tier of Employment | Employment tier | Ordinal/categorical feature |
| Industry | Industry field; appears encoded/anonymized | High-cardinality categorical feature |
| Role | Role field; appears encoded/anonymized | High-cardinality categorical feature |
| Work Experience | Work experience band | Ordinal feature |
| Total Income(PA) | Annual income | Numeric feature |

## Personal_information

| Column | Meaning | Initial role |
|---|---|---|
| User id | Borrower/customer identifier | Join key; not a model feature |
| Gender | Gender | Fairness/protected-attribute review; avoid using for final decision model unless justified |
| Married | Marital status | Categorical feature; fairness review |
| Dependents | Number of dependents | Numeric/ordinal feature |
| Home | Housing ownership status | Categorical feature |
| Pincode | Location proxy | Candidate for Canadian postal-code synthetic remapping or exclusion depending on governance decision |
| Social Profile | Social profile flag | Categorical feature; fairness/proxy review |
| Is_verified | Verification status | Categorical feature |

## Other_information

| Column | Meaning | Initial role |
|---|---|---|
| User_id | Borrower/customer identifier | Join key; not a model feature |
| Delinq_2yrs | Delinquencies in prior 2 years | Credit behaviour feature |
| Total Payement | Total payment; typo retained in raw source | Candidate repayment feature; leakage review required |
| Received Principal | Principal repaid | Candidate repayment feature; leakage review required |
| Interest Received | Interest received | Candidate repayment feature; leakage review required |
| Number of loans | Existing loan count | Credit exposure feature |
| Defaulter | Default indicator | Target variable |
