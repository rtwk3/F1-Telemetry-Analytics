"""
Utility functions for dynamic race selection using FastF1 event schedules.

This module provides support for querying F1 event schedules dynamically,
saving/loading selections to/from a local config file, and resolving
race list queries based on ingestion modes.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.config import BASE_DIR, CACHE_DIR, RACES

CONFIG_DIR: Path = BASE_DIR / "config"
DYNAMIC_CONFIG_PATH: Path = CONFIG_DIR / "selected_races.json"


def save_selected_races(races: list[tuple[int, str, str]]) -> None:
    """
    Save the selected races list to a JSON file for ETL pipeline consumption.

    The file is saved inside the root config/ directory to separate
    user selections from telemetry data.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(DYNAMIC_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(races, f, indent=4)


def load_saved_races() -> list[tuple[int, str, str]] | None:
    """
    Load the saved races list from JSON if it exists.

    Returns:
        List of (year, race_name, session_type) tuples, or None if the file doesn't exist.
    """
    if DYNAMIC_CONFIG_PATH.exists():
        try:
            with open(DYNAMIC_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [(item[0], item[1], item[2]) for item in data]
        except Exception:
            return None
    return None


def get_selected_races(
    year: int | None = None,
    mode: str | None = None,
    selection: int | list[str] | None = None,
) -> list[tuple[int, str, str]]:
    """
    Get a list of (year, race_name, "R") tuples based on the selection criteria.
    If no arguments are provided, it will check for a saved JSON file,
    and fallback to the original 5 races (config.RACES) if not found.

    Args:
        year: F1 season year (2022, 2023, 2024, 2025).
        mode: Ingestion mode ('First N races', 'Specific races', 'Entire season').
        selection: The value associated with the mode (int for 'First N races',
                   list of strings for 'Specific races', or None for 'Entire season').

    Returns:
        A list of (year, race_name, "R") tuples.
    """
    if year is None and mode is None and selection is None:
        saved = load_saved_races()
        if saved is not None:
            return saved
        return RACES

    import fastf1

    # Enable cache if not already enabled
    try:
        fastf1.Cache.enable_cache(str(CACHE_DIR))
    except Exception:
        pass

    # Retrieve event schedule and filter for actual races (RoundNumber > 0)
    schedule = fastf1.get_event_schedule(year)
    races_df = schedule[schedule["RoundNumber"] > 0]

    if mode == "First N races":
        n = int(selection) if selection is not None else 5
        selected_df = races_df.head(n)
        races = [(year, str(row["EventName"]), "R") for _, row in selected_df.iterrows()]
    elif mode == "Specific races":
        selected_names = selection if selection is not None else []
        selected_df = races_df[races_df["EventName"].isin(selected_names)]
        races = [(year, str(row["EventName"]), "R") for _, row in selected_df.iterrows()]
    elif mode == "Entire season":
        races = [(year, str(row["EventName"]), "R") for _, row in races_df.iterrows()]
    else:
        races = RACES

    return races
