"""
Data transformation layer for F1 Telemetry Analytics Platform.

This module loads the cleaned race dataset and performs analytical calculations:
1. Sector deltas (driver's sector time vs. fastest sector time of the lap).
2. Race pace index (rolling 5-lap average pace).
3. Circuit performance score (normalized driver vs. race average pace).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.config import (
    DATA_PROCESSED,
    ROLLING_PACE_WINDOW,
)

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def calculate_sector_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the difference between driver sector times and the fastest sector times."""
    # Find the fastest sector times for each lap in each race
    fastest_s1 = df.groupby(["race_name", "lap_number"])["sector1_time_seconds"].transform("min")
    fastest_s2 = df.groupby(["race_name", "lap_number"])["sector2_time_seconds"].transform("min")
    fastest_s3 = df.groupby(["race_name", "lap_number"])["sector3_time_seconds"].transform("min")

    df["sector1_delta"] = df["sector1_time_seconds"] - fastest_s1
    df["sector2_delta"] = df["sector2_time_seconds"] - fastest_s2
    df["sector3_delta"] = df["sector3_time_seconds"] - fastest_s3

    return df


def calculate_race_pace_index(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rolling average and delta of lap times per driver in each race."""
    # Ensure correct sorting order for rolling calculation
    df = df.sort_values(by=["race_name", "driver", "lap_number"]).copy()

    # Compute rolling average within each race/driver group
    rolling_pace = (
        df.groupby(["race_name", "driver"])["lap_time_seconds"]
        .rolling(window=ROLLING_PACE_WINDOW, min_periods=1)
        .mean()
    )

    # Align the index of the rolling result with the sorted dataframe
    df["race_pace_index"] = rolling_pace.reset_index(level=[0, 1], drop=True)

    # Calculate race pace delta (driver's lap time vs their own rolling pace average)
    df["race_pace_delta"] = df["lap_time_seconds"] - df["race_pace_index"]

    return df


def calculate_circuit_performance_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate a score comparing driver's average lap time to overall race average.

    Score interpretation:
    - score > 100 -> driver is faster than the race average.
    - score = 100 -> driver is exactly at the race average.
    - score < 100 -> driver is slower than the race average.
    """
    # Race average lap time
    race_avg = df.groupby("race_name")["lap_time_seconds"].transform("mean")

    # Driver average lap time per race
    driver_avg = df.groupby(["race_name", "driver"])["lap_time_seconds"].transform("mean")

    # Higher score = faster driver. Ratio is inverted because lower lap times are faster.
    df["circuit_performance_score"] = (race_avg / driver_avg) * 100

    return df


def run_transform_data_pipeline() -> None:
    """Run data transformation pipeline and save transformed Parquet."""
    input_path = DATA_PROCESSED / "cleaned_races.parquet"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {input_path}"
        )

    logger.info("Loading cleaned dataset...")
    df = pd.read_parquet(input_path)

    logger.info("Calculating sector deltas...")
    df = calculate_sector_deltas(df)

    logger.info("Calculating rolling race pace index...")
    df = calculate_race_pace_index(df)

    logger.info("Calculating circuit performance scores...")
    df = calculate_circuit_performance_score(df)

    # Save output
    output_path = DATA_PROCESSED / "transformed_races.parquet"
    df.to_parquet(output_path, index=False)

    logger.info(
        "Transformed dataset saved successfully to %s. Shape: %s",
        output_path,
        df.shape,
    )


if __name__ == "__main__":
    run_transform_data_pipeline()
