from pathlib import Path

PROJECT_ROOT = Path(**file**).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
MODEL_ARTIFACTS_DIR = REPORTS_DIR / "model_artifacts"
GOVERNANCE_DIR = REPORTS_DIR / "governance"
HTML_REPORTS_DIR = REPORTS_DIR / "html"