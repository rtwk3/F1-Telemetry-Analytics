"""
Unit tests for the ETL pipeline layers.

This module contains unit tests for clean_dataframe and transform_data functions
using mock Pandas dataframes to verify columns, types, and logic correctness.
"""

from __future__ import annotations

import unittest
import pandas as pd
import numpy as np

from src.etl.clean_data import clean_dataframe
from src.etl.transform_data import calculate_sector_deltas, calculate_circuit_performance_score


class TestETLPipeline(unittest.TestCase):
    """Test suite for data cleaning and transformation helper methods."""

    def setUp(self) -> None:
        """Set up mock raw lap data."""
        self.mock_raw_df = pd.DataFrame({
            "Time": pd.to_timedelta(["0 days 01:02:00", "0 days 01:03:30"]),
            "Driver": ["VER", "PER"],
            "DriverNumber": ["1", "11"],
            "LapTime": pd.to_timedelta(["0 days 00:01:30", "0 days 00:01:32"]),
            "LapNumber": [1.0, 1.0],
            "Sector1Time": pd.to_timedelta(["0 days 00:00:30", "0 days 00:00:31"]),
            "Sector2Time": pd.to_timedelta(["0 days 00:00:30", "0 days 00:00:31"]),
            "Sector3Time": pd.to_timedelta(["0 days 00:00:30", "0 days 00:00:30"]),
            "Compound": ["SOFT", "SOFT"],
            "TyreLife": [1.0, 1.0],
            "Team": ["Red Bull Racing", "Red Bull Racing"],
            "Position": [1.0, 2.0],
            "race_name": ["Bahrain", "Bahrain"],
            "AirTemp": [25.0, 25.0],
            "TrackTemp": [35.0, 35.0],
            "Rainfall": [False, False],
            "grid_position": [np.nan, np.nan],
            "final_position": [np.nan, np.nan],
        })

    def test_clean_dataframe_converts_timedeltas(self) -> None:
        """Verify timedelta columns are converted to float seconds with appropriate suffixes."""
        cleaned = clean_dataframe(self.mock_raw_df.copy())
        
        # Verify columns renamed to snake_case
        self.assertIn("lap_time_seconds", cleaned.columns)
        self.assertIn("time_seconds", cleaned.columns)
        
        # Verify correct conversion values
        self.assertEqual(cleaned["lap_time_seconds"].iloc[0], 90.0)
        self.assertEqual(cleaned["sector1_time_seconds"].iloc[0], 30.0)
        self.assertEqual(cleaned["sector1_time_seconds"].iloc[1], 31.0)

    def test_clean_dataframe_applies_fallback_positions(self) -> None:
        """Verify fallback grid/final positions are successfully mapped for Bahrain."""
        cleaned = clean_dataframe(self.mock_raw_df.copy())
        
        # VER (1) and PER (11) fallbacks for Bahrain
        self.assertEqual(cleaned["grid_position"].iloc[0], 1)
        self.assertEqual(cleaned["final_position"].iloc[0], 1)
        self.assertEqual(cleaned["grid_position"].iloc[1], 2)
        self.assertEqual(cleaned["final_position"].iloc[1], 2)

    def test_calculate_sector_deltas(self) -> None:
        """Verify sector deltas are correctly calculated relative to the fastest lap sector."""
        # Setup mock cleaned df
        df = pd.DataFrame({
            "race_name": ["Bahrain", "Bahrain"],
            "lap_number": [1.0, 1.0],
            "sector1_time_seconds": [30.0, 31.5],
            "sector2_time_seconds": [30.0, 30.0],
            "sector3_time_seconds": [30.0, 29.0]
        })
        
        transformed = calculate_sector_deltas(df)
        
        # Delta 1: VER = 0.0 (fastest is 30.0), PER = 1.5 (31.5 - 30.0)
        self.assertEqual(transformed["sector1_delta"].iloc[0], 0.0)
        self.assertEqual(transformed["sector1_delta"].iloc[1], 1.5)
        
        # Delta 3: VER = 1.0 (30.0 - 29.0), PER = 0.0 (fastest is 29.0)
        self.assertEqual(transformed["sector3_delta"].iloc[0], 1.0)
        self.assertEqual(transformed["sector3_delta"].iloc[1], 0.0)

    def test_calculate_circuit_performance_score(self) -> None:
        """Verify circuit performance score yields correct ratio against race average."""
        df = pd.DataFrame({
            "race_name": ["Bahrain", "Bahrain"],
            "driver": ["VER", "PER"],
            "lap_time_seconds": [90.0, 110.0]  # Average is 100.0
        })
        
        transformed = calculate_circuit_performance_score(df)
        
        # VER avg = 90. Race avg = 100. Score = 100 * (100 / 90) = 111.11
        # PER avg = 110. Race avg = 100. Score = 100 * (100 / 110) = 90.91
        self.assertAlmostEqual(transformed["circuit_performance_score"].iloc[0], 111.11, places=2)
        self.assertAlmostEqual(transformed["circuit_performance_score"].iloc[1], 90.91, places=2)


class TestRaceSelector(unittest.TestCase):
    """Test suite for the dynamic race selector utilities."""

    def test_get_selected_races_default(self) -> None:
        """Verify get_selected_races defaults to config.RACES when no params/config files are found."""
        import os
        from src.utils.race_selector import get_selected_races, DYNAMIC_CONFIG_PATH

        # Temporarily rename/remove dynamic config if it exists
        existed = DYNAMIC_CONFIG_PATH.exists()
        backup_path = DYNAMIC_CONFIG_PATH.with_suffix(".json.bak")
        if existed:
            if backup_path.exists():
                os.remove(backup_path)
            os.rename(DYNAMIC_CONFIG_PATH, backup_path)

        try:
            races = get_selected_races()
            from src.utils.config import RACES
            self.assertEqual(races, RACES)
        finally:
            # Restore backup
            if existed:
                if DYNAMIC_CONFIG_PATH.exists():
                    os.remove(DYNAMIC_CONFIG_PATH)
                os.rename(backup_path, DYNAMIC_CONFIG_PATH)

    def test_save_and_load_selected_races(self) -> None:
        """Verify save_selected_races and load_saved_races works correctly."""
        import os
        from src.utils.race_selector import get_selected_races, save_selected_races, load_saved_races, DYNAMIC_CONFIG_PATH

        # Backup existing
        existed = DYNAMIC_CONFIG_PATH.exists()
        backup_path = DYNAMIC_CONFIG_PATH.with_suffix(".json.bak")
        if existed:
            if backup_path.exists():
                os.remove(backup_path)
            os.rename(DYNAMIC_CONFIG_PATH, backup_path)

        try:
            test_races = [(2023, "Test GP", "R")]
            save_selected_races(test_races)
            loaded = load_saved_races()
            self.assertEqual(loaded, test_races)

            resolved = get_selected_races()
            self.assertEqual(resolved, test_races)
        finally:
            # Clean up test file
            if DYNAMIC_CONFIG_PATH.exists():
                os.remove(DYNAMIC_CONFIG_PATH)
            # Restore backup
            if existed:
                os.rename(backup_path, DYNAMIC_CONFIG_PATH)


if __name__ == "__main__":
    unittest.main()
