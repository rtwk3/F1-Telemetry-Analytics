"""
Extract Formula 1 race lap data from FastF1 and save as Parquet.

This module downloads race data defined in config.RACES and stores
raw lap-level datasets in data/raw for downstream ETL processing.
"""

from __future__ import annotations

import logging
from pathlib import Path

# pyrefly: ignore [missing-import]
import fastf1
import pandas as pd

from src.utils.config import (
    CACHE_DIR,
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


def initialize_cache() -> None:
    """Initialize FastF1 cache."""
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def extract_race(
    year: int,
    race_name: str,
    session_type: str,
) -> pd.DataFrame:
    """Download a race session and return lap data."""

    logger.info(
        "Loading race: %s %s (%s)",
        race_name,
        year,
        session_type,
    )

    session = fastf1.get_session(
        year,
        race_name,
        session_type,
    )

    session.load(laps=True, telemetry=False, weather=True, messages=False)

    laps = session.laps.copy().reset_index(drop=True)
    weather_df = session.laps.get_weather_data().reset_index(drop=True)
    weather_df = weather_df.drop(columns=["Time"], errors="ignore")
    laps = pd.concat([laps, weather_df], axis=1)

    # Merge session results for grid and final positions
    if hasattr(session, "results") and session.results is not None and not session.results.empty:
        results_df = session.results[["DriverNumber", "GridPosition", "Position", "Status"]].copy()
        results_df = results_df.rename(columns={
            "GridPosition": "grid_position",
            "Position": "final_position",
            "Status": "status",
        })

        # Fallback reconstruction for missing positions due to Ergast retirement
        if results_df["final_position"].isnull().all() and not laps.empty:
            last_laps = laps.sort_values("LapNumber").groupby("DriverNumber")["Position"].last()
            results_df["final_position"] = results_df["DriverNumber"].map(last_laps)

        if results_df["grid_position"].isnull().all() and not laps.empty:
            first_laps = laps[laps["LapNumber"] == 1].set_index("DriverNumber")["Position"]
            results_df["grid_position"] = results_df["DriverNumber"].map(first_laps)
            results_df["grid_position"] = results_df["grid_position"].fillna(results_df["final_position"])

        laps = laps.merge(results_df, on="DriverNumber", how="left")

    laps["race_name"] = race_name
    laps["season"] = year
    laps["session_type"] = session_type

    return laps


def save_race(
    df: pd.DataFrame,
    year: int,
    race_name: str,
) -> Path:
    """Save race dataframe as parquet."""

    output_path = get_race_path(
        year=year,
        race_name=race_name,
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    logger.info(
        "Saved %s rows -> %s",
        len(df),
        output_path.name,
    )

    return output_path


def process_race(
    year: int,
    race_name: str,
    session_type: str,
) -> None:
    """Extract and save one race."""

    try:
        race_df = extract_race(
            year=year,
            race_name=race_name,
            session_type=session_type,
        )

        save_race(
            df=race_df,
            year=year,
            race_name=race_name,
        )

    except Exception as exc:
        logger.exception(
            "Failed processing %s %s: %s",
            race_name,
            year,
            exc,
        )


def main() -> None:
    """Run extraction pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract Formula 1 race lap data dynamically.")
    parser.add_argument("--season", type=int, choices=[2022, 2023, 2024, 2025], help="F1 season year")
    parser.add_argument("--mode", type=str, choices=["First N races", "Specific races", "Entire season"], help="Ingestion mode")
    parser.add_argument("--selection", type=str, help="Selection: number of races (for 'First N races') or comma-separated list of race names (for 'Specific races')")

    args = parser.parse_args()

    initialize_cache()

    DATA_RAW.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("Starting FastF1 extraction")

    # Determine races to extract
    if args.season is not None and args.mode is not None:
        selection_val = None
        if args.mode == "First N races":
            selection_val = int(args.selection) if args.selection else 5
        elif args.mode == "Specific races":
            selection_val = [s.strip() for s in args.selection.split(",")] if args.selection else []
        races_to_extract = get_selected_races(args.season, args.mode, selection_val)
    else:
        # Fall back to load_saved_races() or config.RACES
        races_to_extract = get_selected_races()

    logger.info("Races to extract: %s", races_to_extract)

    for year, race_name, session_type in races_to_extract:
        process_race(
            year=year,
            race_name=race_name,
            session_type=session_type,
        )

    logger.info("Extraction completed")


if __name__ == "__main__":
    main()