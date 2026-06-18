from __future__ import annotations

"""Experiment tracking helpers for Notebook 06.

Local CSV tracking is always available. Neptune tracking is optional and is enabled
only when ENABLE_NEPTUNE=1 and the required Neptune environment variables exist.
No API token is ever hard-coded.
"""

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def neptune_config_status() -> pd.DataFrame:
    """Return a safe status table. Does not reveal the API token."""
    token = os.getenv("NEPTUNE_API_TOKEN")
    project = os.getenv("NEPTUNE_PROJECT")
    enabled = os.getenv("ENABLE_NEPTUNE", "0").strip() == "1"

    return pd.DataFrame(
        [
            {"setting": "ENABLE_NEPTUNE", "status": str(enabled), "required_for_neptune": True},
            {"setting": "NEPTUNE_PROJECT", "status": "set" if project else "missing", "required_for_neptune": True},
            {"setting": "NEPTUNE_API_TOKEN", "status": "set_masked" if token else "missing", "required_for_neptune": True},
            {"setting": "NEPTUNE_MODE", "status": os.getenv("NEPTUNE_MODE", "async"), "required_for_neptune": False},
        ]
    )


def neptune_enabled() -> bool:
    """True only when Neptune tracking is explicitly enabled and configured."""
    enabled = os.getenv("ENABLE_NEPTUNE", "0").strip() == "1"
    return bool(enabled and os.getenv("NEPTUNE_PROJECT") and os.getenv("NEPTUNE_API_TOKEN"))


def start_neptune_run(run_name: str, tags: list[str] | None = None):
    """Start a Neptune run, or return None if disabled."""
    if not neptune_enabled():
        return None

    try:
        import neptune
    except ImportError as exc:
        raise ImportError("Neptune is enabled but not installed. Run: pip install -U neptune") from exc

    mode = os.getenv("NEPTUNE_MODE", "async")
    run = neptune.init_run(
        project=os.getenv("NEPTUNE_PROJECT"),
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
        name=run_name,
        tags=tags or [],
        mode=mode,
    )
    return run


def safe_neptune_log(run, path: str, value: Any) -> None:
    """Best-effort Neptune logging."""
    if run is None:
        return
    try:
        run[path] = value
    except Exception:
        try:
            run[path].append(value)
        except Exception:
            pass


def stop_neptune_run(run) -> None:
    """Stop Neptune run safely."""
    if run is None:
        return
    try:
        run.stop()
    except Exception:
        pass


def flatten_params(params: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested params for CSV logging."""
    out: dict[str, Any] = {}
    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_params(value, full_key))
        else:
            out[full_key] = value
    return out


def _csv_safe_value(value: Any) -> Any:
    """Convert complex values to CSV-safe scalars."""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, default=str, sort_keys=True)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def read_experiment_log(path: Path) -> pd.DataFrame:
    """Read the local experiment log defensively.

    Older runs may have created malformed CSV rows because different experiments
    logged different parameter columns while appending without re-aligning the
    header. If a malformed file is found, it is preserved with a .corrupt suffix
    and a clean empty log is returned so the training workflow can finish.
    """
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        corrupt_path = path.with_suffix(path.suffix + ".corrupt")
        counter = 1
        while corrupt_path.exists():
            corrupt_path = path.with_suffix(path.suffix + f".corrupt_{counter}")
            counter += 1
        path.rename(corrupt_path)
        return pd.DataFrame()


def append_experiment_log(row: dict[str, Any], path: Path) -> None:
    """Append one experiment row to a local CSV log safely.

    This function rewrites the full log each time instead of appending raw rows.
    That prevents ParserError issues when different model families log different
    hyperparameter columns. Complex values are JSON-encoded and all fields are
    quoted for CSV safety.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_row = {key: _csv_safe_value(value) for key, value in row.items()}
    new_df = pd.DataFrame([safe_row])
    existing_df = read_experiment_log(path)

    combined = pd.concat([existing_df, new_df], ignore_index=True, sort=False)

    # Keep important columns first and then preserve all experiment-specific columns.
    base_cols = [
        "experiment_id",
        "timestamp_utc",
        "model_name",
        "model_family",
        "stage",
        "notes",
    ]
    ordered_cols = [col for col in base_cols if col in combined.columns]
    ordered_cols += [col for col in combined.columns if col not in ordered_cols]
    combined = combined[ordered_cols]

    combined.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )


def make_experiment_row(
    experiment_id: str,
    model_name: str,
    model_family: str,
    stage: str,
    params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Create one local experiment-tracking row."""
    row: dict[str, Any] = {
        "experiment_id": experiment_id,
        "timestamp_utc": now_utc_iso(),
        "model_name": model_name,
        "model_family": model_family,
        "stage": stage,
        "notes": notes,
    }
    if params:
        for key, value in flatten_params(params).items():
            row[f"param_{key}"] = value
    if metrics:
        for key, value in metrics.items():
            row[f"metric_{key}"] = value
    return row
