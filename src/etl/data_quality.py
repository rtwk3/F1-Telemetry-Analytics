"""
Data quality layer for F1 Telemetry Analytics Platform.

This module runs quality assurance checks on the combined raw dataset
(e.g., null values, duplicate laps, out-of-bound lap times) and exports
a data quality report in JSON format to the processed data directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.utils.config import (
    DATA_RAW,
    DATA_PROCESSED,
    VALID_TIRE_COMPOUNDS,
)

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def check_null_percentages(df: pd.DataFrame) -> dict[str, float]:
    """Calculate the percentage of null values in each column."""
    null_pct = df.isnull().mean() * 100
    return {col: float(val) for col, val in null_pct.items()}


def check_duplicate_laps(df: pd.DataFrame) -> int:
    """Count duplicate laps by driver, race, and lap number."""
    # Group by key fields and check for duplicates, ignoring NaNs in keys
    subset = ["race_name", "Driver", "LapNumber"]
    # Drop rows where any key is missing to check duplicate records properly
    clean_keys = df.dropna(subset=subset)
    duplicates = clean_keys.duplicated(subset=subset, keep=False)
    return int(duplicates.sum())


def check_invalid_lap_times(df: pd.DataFrame) -> int:
    """Count invalid lap times (<= 0s or > 300s)."""
    if "LapTime" not in df.columns:
        return 0
    # Convert timedelta to float seconds for check
    lap_seconds = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
    invalid_mask = (lap_seconds <= 0) | (lap_seconds > 300)
    # NaNs are not considered invalid but missing, which is captured in null checks
    return int(invalid_mask.sum())


def check_invalid_compounds(df: pd.DataFrame) -> list[str]:
    """Identify compound values not defined in VALID_TIRE_COMPOUNDS."""
    if "Compound" not in df.columns:
        return []
    unique_compounds = df["Compound"].dropna().unique()
    invalid = [
        str(comp)
        for comp in unique_compounds
        if str(comp).upper() not in VALID_TIRE_COMPOUNDS
    ]
    return invalid


def run_data_quality_pipeline() -> None:
    """Run data quality analysis and write JSON report."""
    input_path = DATA_RAW / "all_races_combined.parquet"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw combined race file not found: {input_path}"
        )

    logger.info("Loading raw combined dataset for quality checks...")
    df = pd.read_parquet(input_path)

    logger.info("Checking null percentages...")
    null_pct = check_null_percentages(df)

    logger.info("Checking for duplicate laps...")
    num_duplicates = check_duplicate_laps(df)

    logger.info("Checking for invalid lap times...")
    num_invalid_laps = check_invalid_lap_times(df)

    logger.info("Checking for invalid tire compounds...")
    invalid_compounds = check_invalid_compounds(df)

    # Compile the final report
    report = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicate_laps": num_duplicates,
        "invalid_lap_times": num_invalid_laps,
        "invalid_tire_compounds": invalid_compounds,
        "null_percentages": null_pct,
    }

    # Ensure output directory exists
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED / "data_quality_report.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    logger.info("Data quality report written to %s", output_path)
    logger.info("--- Data Quality Summary ---")
    logger.info("Total Rows        : %d", report["row_count"])
    logger.info("Duplicate Laps    : %d", report["duplicate_laps"])
    logger.info("Invalid Laps (>300s or <=0s): %d", report["invalid_lap_times"])
    logger.info("Invalid Compounds : %s", report["invalid_tire_compounds"])


if __name__ == "__main__":
    run_data_quality_pipeline()
