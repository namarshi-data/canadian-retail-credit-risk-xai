from __future__ import annotations

"""Run Notebook 09 model-governance, monitoring, and reporting pipeline."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from credit_risk.config import GOVERNANCE_DIR, PROCESSED_DIR, TABLE_DIR, ensure_project_directories
except Exception:  # pragma: no cover
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
    GOVERNANCE_DIR = PROJECT_ROOT / "reports" / "governance"

    def ensure_project_directories() -> None:
        for path in [PROCESSED_DIR, TABLE_DIR, GOVERNANCE_DIR]:
            path.mkdir(parents=True, exist_ok=True)

from credit_risk.governance.model_governance import (
    build_governance_readiness_gate,
    build_governance_summary,
    build_monitoring_kpi_snapshot,
    load_governance_inputs,
    save_governance_outputs,
)


def main() -> None:
    ensure_project_directories()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    GOVERNANCE_DIR.mkdir(parents=True, exist_ok=True)

    inputs = load_governance_inputs(processed_dir=PROCESSED_DIR, table_dir=TABLE_DIR)
    saved = save_governance_outputs(
        inputs,
        table_dir=TABLE_DIR,
        governance_dir=GOVERNANCE_DIR,
        project_root=PROJECT_ROOT,
    )

    governance_summary = build_governance_summary(inputs)
    kpi_snapshot = build_monitoring_kpi_snapshot(inputs)
    readiness_gate = build_governance_readiness_gate(inputs)

    print("Notebook 09 governance pipeline completed.")

    print("\nGovernance summary:")
    print(governance_summary.to_string(index=False))

    print("\nMonitoring KPI snapshot:")
    print(kpi_snapshot.to_string(index=False))

    print("\nReadiness gate:")
    print(readiness_gate.to_string(index=False))

    print("\nSaved outputs:")
    for name, path in saved.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
