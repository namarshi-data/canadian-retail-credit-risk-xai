from __future__ import annotations

"""Project configuration and path constants.

Expected location:
    src/credit_risk/config.py

This module centralizes project paths and lightweight YAML loading. It keeps
backward-compatible aliases used across the notebooks, scripts, and src modules.
"""

import os
from pathlib import Path
from typing import Any

import yaml


def find_project_root(start_path: Path | None = None) -> Path:
    """Find the repository root.

    Priority:
    1. PROJECT_ROOT environment variable, if defined.
    2. Walk upward from this file until common project markers are found.
    3. Fallback to the expected src/credit_risk/config.py layout.
    """
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    start = (start_path or Path(__file__)).resolve()
    current = start if start.is_dir() else start.parent

    project_markers = {
        "pyproject.toml",
        "requirements.txt",
        "README.md",
        ".git",
    }

    for candidate in [current, *current.parents]:
        if any((candidate / marker).exists() for marker in project_markers):
            return candidate

    # Expected fallback:
    # src/credit_risk/config.py -> project root is parents[2]
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = find_project_root()

# ---------------------------------------------------------------------
# Top-level project folders
# ---------------------------------------------------------------------

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_DIR = REPORTS_DIR  # Backward-compatible alias
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"
LOGS_DIR = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------
# Data folders
# ---------------------------------------------------------------------

RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
SAMPLE_DIR = DATA_DIR / "sample"

# Alternative aliases used in some notebooks/modules.
RAW_DATA_DIR = RAW_DIR
INTERIM_DATA_DIR = INTERIM_DIR
PROCESSED_DATA_DIR = PROCESSED_DIR
EXTERNAL_DATA_DIR = EXTERNAL_DIR
SAMPLE_DATA_DIR = SAMPLE_DIR

# ---------------------------------------------------------------------
# Report/output folders
# ---------------------------------------------------------------------

FIGURE_DIR = REPORTS_DIR / "figures"
TABLE_DIR = REPORTS_DIR / "tables"
MODEL_ARTIFACT_DIR = REPORTS_DIR / "model_artifacts"
MODEL_ARTIFACTS_DIR = MODEL_ARTIFACT_DIR
GOVERNANCE_DIR = REPORTS_DIR / "governance"
HTML_REPORTS_DIR = REPORTS_DIR / "html"

# Alternative aliases used in newer modules.
FIGURES_DIR = FIGURE_DIR
TABLES_DIR = TABLE_DIR
MODEL_DIR = MODEL_ARTIFACT_DIR
HTML_DIR = HTML_REPORTS_DIR

# ---------------------------------------------------------------------
# Common config files
# ---------------------------------------------------------------------

CONFIG_PATH = CONFIG_DIR / "config.yaml"
MODEL_CONFIG_PATH = CONFIG_DIR / "model_config.yaml"

# ---------------------------------------------------------------------
# Common data/model artifact paths
# ---------------------------------------------------------------------

RAW_WORKBOOK_PATH = RAW_DIR / "Credit_Risk_Dataset.xlsx"
INTERIM_MERGED_DATASET_PATH = INTERIM_DIR / "credit_risk_merged_interim.csv"
CLEANED_DATASET_PATH = PROCESSED_DIR / "credit_risk_cleaned.csv"
CLEANED_PARQUET_PATH = PROCESSED_DIR / "credit_risk_cleaned.parquet"
MODELING_DATASET_PATH = PROCESSED_DIR / "credit_risk_modeling_dataset.csv"

PREPROCESSING_COLUMN_GROUPS_PATH = TABLE_DIR / "05_preprocessing_column_groups.json"

CHAMPION_MODEL_PATH = MODEL_ARTIFACT_DIR / "champion_model.joblib"
CHAMPION_FEATURE_METADATA_PATH = MODEL_ARTIFACT_DIR / "model_feature_metadata.joblib"

NOTEBOOK06_CHAMPION_MODEL_PATH = MODEL_ARTIFACT_DIR / "06_champion_model.joblib"
NOTEBOOK06_FEATURE_METADATA_PATH = MODEL_ARTIFACT_DIR / "06_model_feature_metadata.joblib"

PROJECT_DIRECTORIES = [
    CONFIG_DIR,
    DATA_DIR,
    RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    EXTERNAL_DIR,
    SAMPLE_DIR,
    REPORTS_DIR,
    FIGURE_DIR,
    TABLE_DIR,
    MODEL_ARTIFACT_DIR,
    GOVERNANCE_DIR,
    HTML_REPORTS_DIR,
    NOTEBOOKS_DIR,
    SCRIPTS_DIR,
    SRC_DIR,
    DOCS_DIR,
    TESTS_DIR,
    LOGS_DIR,
]


def ensure_project_directories() -> None:
    """Create standard project directories if they do not exist."""
    for path in PROJECT_DIRECTORIES:
        path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Empty YAML files return an empty dictionary. Missing files raise a clear
    FileNotFoundError so pipeline errors are easy to diagnose.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

    return loaded or {}


def load_project_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the main project config YAML."""
    return load_yaml(path or CONFIG_PATH)


def load_model_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the model config YAML."""
    return load_yaml(path or MODEL_CONFIG_PATH)


def project_path(*parts: str | os.PathLike[str]) -> Path:
    """Build an absolute path under the project root."""
    return PROJECT_ROOT.joinpath(*map(Path, parts))


def data_path(*parts: str | os.PathLike[str]) -> Path:
    """Build an absolute path under data/."""
    return DATA_DIR.joinpath(*map(Path, parts))


def raw_data_path(*parts: str | os.PathLike[str]) -> Path:
    """Build an absolute path under data/raw/."""
    return RAW_DIR.joinpath(*map(Path, parts))


def interim_data_path(*parts: str | os.PathLike[str]) -> Path:
    """Build an absolute path under data/interim/."""
    return INTERIM_DIR.joinpath(*map(Path, parts))


def processed_data_path(*parts: str | os.PathLike[str]) -> Path:
    """Build an absolute path under data/processed/."""
    return PROCESSED_DIR.joinpath(*map(Path, parts))


def report_path(*parts: str | os.PathLike[str]) -> Path:
    """Build an absolute path under reports/."""
    return REPORTS_DIR.joinpath(*map(Path, parts))


def table_path(file_name: str) -> Path:
    """Build an absolute path under reports/tables/."""
    return TABLE_DIR / file_name


def figure_path(file_name: str) -> Path:
    """Build an absolute path under reports/figures/."""
    return FIGURE_DIR / file_name


def model_artifact_path(file_name: str) -> Path:
    """Build an absolute path under reports/model_artifacts/."""
    return MODEL_ARTIFACT_DIR / file_name


def governance_path(file_name: str) -> Path:
    """Build an absolute path under reports/governance/."""
    return GOVERNANCE_DIR / file_name


def html_report_path(file_name: str) -> Path:
    """Build an absolute path under reports/html/."""
    return HTML_REPORTS_DIR / file_name


__all__ = [
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "DATA_DIR",
    "REPORTS_DIR",
    "REPORT_DIR",
    "NOTEBOOKS_DIR",
    "SCRIPTS_DIR",
    "SRC_DIR",
    "DOCS_DIR",
    "TESTS_DIR",
    "LOGS_DIR",
    "RAW_DIR",
    "INTERIM_DIR",
    "PROCESSED_DIR",
    "EXTERNAL_DIR",
    "SAMPLE_DIR",
    "RAW_DATA_DIR",
    "INTERIM_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "EXTERNAL_DATA_DIR",
    "SAMPLE_DATA_DIR",
    "FIGURE_DIR",
    "TABLE_DIR",
    "MODEL_ARTIFACT_DIR",
    "MODEL_ARTIFACTS_DIR",
    "GOVERNANCE_DIR",
    "HTML_REPORTS_DIR",
    "FIGURES_DIR",
    "TABLES_DIR",
    "MODEL_DIR",
    "HTML_DIR",
    "CONFIG_PATH",
    "MODEL_CONFIG_PATH",
    "RAW_WORKBOOK_PATH",
    "INTERIM_MERGED_DATASET_PATH",
    "CLEANED_DATASET_PATH",
    "CLEANED_PARQUET_PATH",
    "MODELING_DATASET_PATH",
    "PREPROCESSING_COLUMN_GROUPS_PATH",
    "CHAMPION_MODEL_PATH",
    "CHAMPION_FEATURE_METADATA_PATH",
    "NOTEBOOK06_CHAMPION_MODEL_PATH",
    "NOTEBOOK06_FEATURE_METADATA_PATH",
    "PROJECT_DIRECTORIES",
    "find_project_root",
    "ensure_project_directories",
    "load_yaml",
    "load_project_config",
    "load_model_config",
    "project_path",
    "data_path",
    "raw_data_path",
    "interim_data_path",
    "processed_data_path",
    "report_path",
    "table_path",
    "figure_path",
    "model_artifact_path",
    "governance_path",
    "html_report_path",
]
