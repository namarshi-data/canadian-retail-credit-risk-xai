from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repository paths. This assumes the standard project structure:
# project_root/src/credit_risk/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
SAMPLE_DIR = DATA_DIR / "sample"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
MODEL_ARTIFACT_DIR = REPORT_DIR / "model_artifacts"
GOVERNANCE_DIR = REPORT_DIR / "governance"


def ensure_project_directories() -> None:
    """Create standard project directories if they do not exist."""
    for path in [
        CONFIG_DIR,
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        EXTERNAL_DIR,
        SAMPLE_DIR,
        REPORT_DIR,
        FIGURE_DIR,
        TABLE_DIR,
        MODEL_ARTIFACT_DIR,
        GOVERNANCE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
