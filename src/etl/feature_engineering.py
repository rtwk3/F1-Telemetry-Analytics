"""
Feature engineering layer for F1 Telemetry Analytics Platform.

This module loads the transformed dataset and engineers key features:
1. Position change (grid_position - final_position).
2. Stint length (max tyre_life per driver per stint).
3. Tire degradation rate (coefficient of tyre_life in stint-level multi-regression).

It exports a final combined parquet file and individual per-race files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from src.utils.config import (
    DATA_PROCESSED,
    get_processed_race_path,
)
from src.utils.race_selector import get_selected_races

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def calculate_position_change(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the change from starting grid position to final position."""
    df["position_change"] = df["grid_position"] - df["final_position"]
    return df


def calculate_stint_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the total length (maximum tyre life) of each stint."""
    stint_lengths = (
        df.groupby(["race_name", "driver", "stint"])["tyre_life"]
        .max()
        .reset_index(name="stint_length")
    )
    df = df.merge(stint_lengths, on=["race_name", "driver", "stint"], how="left")
    return df


def compute_stint_degradation_slope(group: pd.DataFrame) -> float:
    """Fit multiple linear regression to find temperature-controlled tyre wear slope."""
    # We require at least 4 laps to fit 3 coefficients (tyre_life, air_temp, track_temp) + intercept
    if len(group) < 4 or group["tyre_life"].nunique() < 2:
        return 0.0

    X = group[["tyre_life", "air_temp", "track_temp"]].fillna(0)
    y = group["lap_time_seconds"].fillna(0)

    try:
        model = LinearRegression()
        model.fit(X, y)
        return float(model.coef_[0])  # Coefficient of tyre_life
    except Exception:
        return 0.0


def calculate_tire_degradation_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the tire degradation rate for each driver stint."""
    # Group by stint and apply regression slope calculation
    degrad_rates = (
        df.groupby(["race_name", "driver", "stint"])
        .apply(compute_stint_degradation_slope, include_groups=False)
        .reset_index(name="tire_degradation_rate")
    )

    df = df.merge(degrad_rates, on=["race_name", "driver", "stint"], how="left")
    return df


def run_feature_engineering_pipeline(races: list[tuple[int, str, str]] | None = None) -> None:
    """Run feature engineering pipeline and export processed Parquet files."""
    input_path = DATA_PROCESSED / "transformed_races.parquet"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Transformed dataset not found: {input_path}"
        )

    logger.info("Loading transformed dataset...")
    df = pd.read_parquet(input_path)

    logger.info("Calculating position changes...")
    df = calculate_position_change(df)

    logger.info("Calculating stint lengths...")
    df = calculate_stint_lengths(df)

    logger.info("Calculating tire degradation rates...")
    df = calculate_tire_degradation_rates(df)

    # Save output (combined dataset)
    output_path = DATA_PROCESSED / "engineered_races.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(
        "Combined engineered dataset saved to %s. Shape: %s",
        output_path,
        df.shape,
    )

    # Save output (individual per-race files)
    logger.info("Saving individual processed race files...")
    if races is None:
        races = get_selected_races()

    for year, race_name, _ in races:
        race_df = df[df["race_name"] == race_name]
        if not race_df.empty:
            race_output_path = get_processed_race_path(year, race_name)
            race_df.to_parquet(race_output_path, index=False)
            logger.info("Saved race file -> %s", race_output_path.name)


if __name__ == "__main__":
    run_feature_engineering_pipeline()
