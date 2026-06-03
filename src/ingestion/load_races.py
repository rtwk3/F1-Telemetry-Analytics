"""
Load raw Formula 1 race datasets from Parquet files.

This module reads all race files defined in config.RACES,
combines them into a single DataFrame, performs basic validation,
and optionally saves the merged dataset for downstream ETL.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.config import (
    DATA_RAW,
    get_race_path,
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


def load_race_file(
    year: int,
    race_name: str,
) -> pd.DataFrame:
    """
    Load a single race parquet file.

    Args:
        year: F1 season year.
        race_name: Race name.

    Returns:
        Race dataframe.
    """

    file_path = get_race_path(
        year=year,
        race_name=race_name,
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Race file not found: {file_path}"
        )

    logger.info("Loading %s", file_path.name)

    return pd.read_parquet(file_path)


def load_all_races(races: list[tuple[int, str, str]] | None = None) -> pd.DataFrame:
    """
    Load all configured races and concatenate.

    Returns:
        Combined race dataframe.
    """

    dataframes: list[pd.DataFrame] = []

    if races is None:
        races = get_selected_races()

    for year, race_name, _ in races:
        try:
            df = load_race_file(
                year=year,
                race_name=race_name,
            )

            dataframes.append(df)

        except Exception as exc:
            logger.error(
                "Failed loading %s %s: %s",
                race_name,
                year,
                exc,
            )

    if not dataframes:
        raise ValueError(
            "No race datasets were loaded."
        )

    combined_df = pd.concat(
        dataframes,
        ignore_index=True,
    )

    logger.info(
        "Combined dataset shape: %s",
        combined_df.shape,
    )

    return combined_df


def validate_dataset(
    df: pd.DataFrame,
) -> None:
    """
    Validate required columns exist.

    Args:
        df: Combined race dataframe.
    """

    required_columns = [
        "Driver",
        "LapNumber",
        "LapTime",
        "Compound",
        "TyreLife",
        "Position",
        "race_name",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    logger.info(
        "Validation successful."
    )


def save_combined_dataset(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save merged dataframe.

    Args:
        df: Combined dataframe.
        output_path: Output parquet path.
    """

    df.to_parquet(
        output_path,
        index=False,
    )

    logger.info(
        "Saved merged dataset -> %s",
        output_path,
    )


def main() -> None:
    """
    Load, validate, and save combined dataset.
    """

    logger.info(
        "Loading all raw race datasets..."
    )

    combined_df = load_all_races()

    validate_dataset(combined_df)

    output_path = (
        DATA_RAW / "all_races_combined.parquet"
    )

    save_combined_dataset(
        combined_df,
        output_path,
    )

    logger.info(
        "Total rows: %s",
        len(combined_df),
    )

    logger.info(
        "Pipeline completed successfully."
    )


if __name__ == "__main__":
    main()