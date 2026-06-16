from credit_risk.data.ingestion import standardize_column_name


def test_standardize_known_raw_columns():
    assert standardize_column_name("Employmet type") == "employment_type"
    assert standardize_column_name("Total Payement ") == "total_payment"
    assert standardize_column_name("Tenure(years)") == "tenure_years"
