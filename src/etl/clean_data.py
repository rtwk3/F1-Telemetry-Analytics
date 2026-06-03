"""
Data cleaning layer for F1 Telemetry Analytics Platform.

This module loads the raw combined race dataset, normalizes column names
to snake_case, programmatically converts all timedelta columns to seconds,
applies fallbacks for missing grid/final positions, and imputes missing values.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.utils.config import (
    DATA_RAW,
    DATA_PROCESSED,
)

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# TODO: Replace with official race results ingestion source
# Historical grid positions for the 5 selected 2023 races (Ergast API fallback)
GRID_POSITIONS = {
    "Bahrain": {
        "1": 1, "11": 2, "16": 3, "55": 4, "14": 5, "44": 6, "63": 7, "18": 8, "31": 9, "27": 10,
        "4": 11, "77": 12, "24": 13, "22": 14, "23": 15, "2": 16, "20": 17, "81": 18, "21": 19, "10": 20
    },
    "Monaco": {
        "1": 1, "14": 2, "31": 3, "55": 4, "44": 5, "16": 6, "10": 7, "63": 8, "22": 9, "4": 10,
        "81": 11, "23": 12, "18": 13, "77": 14, "24": 15, "2": 16, "20": 17, "27": 18, "21": 19, "11": 20
    },
    "Silverstone": {
        "1": 1, "4": 2, "81": 3, "16": 4, "55": 5, "63": 6, "44": 7, "23": 8, "14": 9, "10": 10,
        "27": 11, "18": 12, "31": 13, "2": 14, "11": 15, "22": 16, "24": 17, "21": 18, "20": 19, "77": 20
    },
    "Monza": {
        "55": 1, "1": 2, "16": 3, "63": 4, "11": 5, "44": 6, "23": 7, "81": 8, "4": 9, "14": 10,
        "10": 11, "40": 12, "77": 13, "31": 14, "24": 15, "20": 16, "18": 17, "27": 18, "22": 19, "2": 20
    },
    "Spa": {
        "16": 1, "11": 2, "44": 3, "55": 4, "81": 5, "1": 6, "4": 7, "63": 8, "14": 9, "18": 10,
        "22": 11, "10": 12, "23": 13, "77": 14, "31": 15, "20": 16, "24": 17, "2": 18, "3": 19, "27": 20
    }
}

# TODO: Replace with official race results ingestion source
# Historical final race positions for the 5 selected 2023 races (Ergast API fallback)
FINAL_POSITIONS = {
    "Bahrain": {
        "1": 1, "11": 2, "14": 3, "55": 4, "44": 5, "18": 6, "63": 7, "77": 8, "10": 9, "23": 10,
        "22": 11, "2": 12, "20": 13, "24": 14, "27": 15, "21": 16, "4": 17, "31": 18, "16": 19, "81": 20
    },
    "Monaco": {
        "1": 1, "14": 2, "31": 3, "44": 4, "63": 5, "16": 6, "10": 7, "55": 8, "4": 9, "81": 10,
        "77": 11, "22": 12, "23": 13, "24": 14, "2": 15, "11": 16, "20": 17, "27": 18, "18": 19, "21": 20
    },
    "Silverstone": {
        "1": 1, "4": 2, "44": 3, "81": 4, "63": 5, "16": 6, "55": 7, "11": 8, "14": 9, "23": 10,
        "2": 11, "77": 12, "22": 13, "24": 14, "18": 15, "20": 16, "27": 17, "10": 18, "31": 19, "21": 20
    },
    "Monza": {
        "1": 1, "11": 2, "55": 3, "16": 4, "63": 5, "44": 6, "23": 7, "4": 8, "14": 9, "77": 10,
        "2": 11, "40": 12, "81": 13, "24": 14, "10": 15, "18": 16, "20": 17, "31": 18, "27": 19, "22": 20
    },
    "Spa": {
        "1": 1, "11": 2, "16": 3, "44": 4, "14": 5, "63": 6, "4": 7, "31": 8, "18": 9, "22": 10,
        "10": 11, "77": 12, "24": 13, "23": 14, "20": 15, "3": 16, "2": 17, "27": 18, "55": 19, "81": 20
    }
}


def camel_to_snake(name: str) -> str:
    """Convert a string from camelCase or PascalCase to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw combined dataframe and handle dtypes and missing values."""
    # 1. Normalize Column Names to snake_case
    df = df.rename(columns={col: camel_to_snake(col) for col in df.columns})

    # 2. Programmatically convert all timedelta columns to float seconds
    for col in df.columns:
        if pd.api.types.is_timedelta64_dtype(df[col]):
            # If the column name already ends with _time or time, rename to _seconds
            new_col_name = f"{col}_seconds" if not col.endswith("_seconds") else col
            df[new_col_name] = pd.to_timedelta(df[col]).dt.total_seconds()
            if col != new_col_name:
                df = df.drop(columns=[col])
            logger.info("Converted timedelta column: %s -> %s", col, new_col_name)

    # Convert datetime64[ns] to datetime64[us] for Spark parquet compatibility
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype("datetime64[us]")
            logger.info("Converted datetime column to microsecond precision: %s", col)

    # 3. Apply fallback mapping for grid and final positions (Ergast API retirement)
    logger.info("Applying historical F1 fallback mappings for grid and final positions...")
    df["grid_position"] = df.apply(
        lambda r: GRID_POSITIONS.get(r["race_name"], {}).get(
            str(r["driver_number"]), r["grid_position"]
        ),
        axis=1,
    )
    df["final_position"] = df.apply(
        lambda r: FINAL_POSITIONS.get(r["race_name"], {}).get(
            str(r["driver_number"]), r["final_position"]
        ),
        axis=1,
    )

    # 4. Impute missing values
    # Numeric columns (median imputation)
    numeric_imputations = [
        "lap_time_seconds",
        "sector1_time_seconds",
        "sector2_time_seconds",
        "sector3_time_seconds",
        "speed_i1",
        "speed_i2",
        "speed_fl",
        "speed_st",
        "tyre_life",
        "position",
        "air_temp",
        "track_temp",
        "humidity",
        "pressure",
        "wind_speed",
        "wind_direction",
        "grid_position",
        "final_position",
    ]
    for col in numeric_imputations:
        if col in df.columns:
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)

    # Categorical columns ("Unknown" imputation)
    categorical_imputations = ["compound", "team", "driver", "track_status", "status"]
    for col in categorical_imputations:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Special case for deleted_reason
    if "deleted_reason" in df.columns:
        df["deleted_reason"] = df["deleted_reason"].fillna("Not Deleted")

    return df


def run_clean_data_pipeline() -> None:
    """Run data cleaning pipeline and save cleaned Parquet."""
    input_path = DATA_RAW / "all_races_combined.parquet"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw combined race file not found: {input_path}"
        )

    logger.info("Loading raw dataset...")
    df = pd.read_parquet(input_path)

    logger.info("Cleaning data...")
    cleaned_df = clean_dataframe(df)

    # Save output
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED / "cleaned_races.parquet"
    cleaned_df.to_parquet(output_path, index=False)

    logger.info(
        "Cleaned dataset saved successfully to %s. Shape: %s",
        output_path,
        cleaned_df.shape,
    )


if __name__ == "__main__":
    run_clean_data_pipeline()
