"""
Unit tests for the Spark analytics modules.

This module contains unit tests for compute_driver_analytics and compute_tire_analytics
using a local SparkSession and mock Spark DataFrames to verify logic correctness.
"""

from __future__ import annotations

import unittest
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from src.spark.driver_analytics import compute_driver_analytics
from src.spark.tire_analytics import compute_tire_analytics
from src.spark.pitstop_analytics import compute_pitstop_analytics


class TestSparkAnalytics(unittest.TestCase):
    """Test suite for PySpark analytics transformations."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize a local lightweight Spark session for testing."""
        cls.spark = (
            SparkSession.builder
            .appName("UnitTestSpark")
            .master("local[1]")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Stop the local Spark session."""
        cls.spark.stop()

    def test_compute_driver_analytics(self) -> None:
        """Verify driver analytics aggregates averages, fastest lap, and consistency index."""
        schema = StructType([
            StructField("season", DoubleType(), True),
            StructField("race_name", StringType(), True),
            StructField("driver", StringType(), True),
            StructField("lap_time_seconds", DoubleType(), True),
            StructField("grid_position", DoubleType(), True),
            StructField("final_position", DoubleType(), True)
        ])
        
        # VER has 3 laps: 90.0, 90.0, 90.0 (stddev = 0, consistency_index = 0)
        # HAM has 2 laps: 91.0, 93.0 (stddev = sqrt(2) = 1.414, consistency_index = 100/1.414 = 70.71)
        data = [
            (2023.0, "Bahrain", "VER", 90.0, 1.0, 1.0),
            (2023.0, "Bahrain", "VER", 90.0, 1.0, 1.0),
            (2023.0, "Bahrain", "VER", 90.0, 1.0, 1.0),
            (2023.0, "Bahrain", "HAM", 91.0, 3.0, 2.0),
            (2023.0, "Bahrain", "HAM", 93.0, 3.0, 2.0)
        ]
        
        df = self.spark.createDataFrame(data, schema)
        result = compute_driver_analytics(df).collect()
        
        # Convert list of rows to a dictionary
        res_dict = {row["driver"]: row for row in result}
        
        # Check VER
        self.assertEqual(res_dict["VER"]["avg_lap_time_seconds"], 90.0)
        self.assertEqual(res_dict["VER"]["fastest_lap"], 90.0)
        self.assertEqual(res_dict["VER"]["consistency_index"], 0.0)
        self.assertEqual(res_dict["VER"]["position_change"], 0.0)
        
        # Check HAM
        self.assertEqual(res_dict["HAM"]["avg_lap_time_seconds"], 92.0)
        self.assertEqual(res_dict["HAM"]["fastest_lap"], 91.0)
        self.assertAlmostEqual(res_dict["HAM"]["consistency_index"], 70.71, places=2)
        self.assertEqual(res_dict["HAM"]["position_change"], 1.0)

    def test_compute_tire_analytics(self) -> None:
        """Verify tire analytics calculates stint statistics and compound usage percent."""
        schema = StructType([
            StructField("season", DoubleType(), True),
            StructField("race_name", StringType(), True),
            StructField("driver", StringType(), True),
            StructField("compound", StringType(), True),
            StructField("lap_time_seconds", DoubleType(), True),
            StructField("tire_degradation_rate", DoubleType(), True),
            StructField("stint_length", DoubleType(), True)
        ])
        
        # Total laps in race = 4 (3 SOFT, 1 HARD)
        # SOFT usage = 3/4 = 75.0%, HARD usage = 1/4 = 25.0%
        data = [
            (2023.0, "Bahrain", "VER", "SOFT", 90.0, 0.05, 10.0),
            (2023.0, "Bahrain", "VER", "SOFT", 91.0, 0.05, 10.0),
            (2023.0, "Bahrain", "HAM", "SOFT", 92.0, 0.06, 12.0),
            (2023.0, "Bahrain", "HAM", "HARD", 94.0, 0.02, 15.0)
        ]
        
        df = self.spark.createDataFrame(data, schema)
        result = compute_tire_analytics(df).collect()
        
        # Filter and check usage percentages
        for row in result:
            if row["compound"] == "SOFT":
                self.assertEqual(row["compound_usage_pct"], 75.0)
            elif row["compound"] == "HARD":
                self.assertEqual(row["compound_usage_pct"], 25.0)

    def test_compute_pitstop_analytics(self) -> None:
        """Verify pitstop analytics calculates pit counts, strategy, and undercut rate."""
        schema = StructType([
            StructField("season", DoubleType(), True),
            StructField("race_name", StringType(), True),
            StructField("driver", StringType(), True),
            StructField("lap_number", DoubleType(), True),
            StructField("position", DoubleType(), True),
            StructField("pit_in_time_seconds", DoubleType(), True),
            StructField("final_position", DoubleType(), True)
        ])
        
        data = [
            (2023.0, "Bahrain", "VER", 1.0, 1.0, None, 2.0),
            (2023.0, "Bahrain", "VER", 2.0, 1.0, 25.0, 2.0),
            (2023.0, "Bahrain", "VER", 3.0, 3.0, None, 2.0),
            (2023.0, "Bahrain", "VER", 4.0, 2.0, None, 2.0),
            (2023.0, "Bahrain", "HAM", 1.0, 4.0, None, 3.0),
            (2023.0, "Bahrain", "HAM", 2.0, 4.0, 24.5, 3.0),
            (2023.0, "Bahrain", "HAM", 3.0, 5.0, None, 3.0),
            (2023.0, "Bahrain", "HAM", 4.0, 3.0, None, 3.0)
        ]
        
        df = self.spark.createDataFrame(data, schema)
        result = compute_pitstop_analytics(df).collect()
        
        res_dict = {row["driver"]: row for row in result}
        
        # Check VER
        self.assertEqual(res_dict["VER"]["pit_count"], 1)
        self.assertEqual(res_dict["VER"]["avg_pit_window"], 2.0)
        self.assertEqual(res_dict["VER"]["undercut_success_rate"], 0.0)
        self.assertEqual(res_dict["VER"]["pit_stop_efficiency_score"], -1.0)
        self.assertEqual(res_dict["VER"]["strategy_type"], "1-stop")
        
        # Check HAM
        self.assertEqual(res_dict["HAM"]["pit_count"], 1)
        self.assertEqual(res_dict["HAM"]["avg_pit_window"], 2.0)
        self.assertEqual(res_dict["HAM"]["undercut_success_rate"], 1.0)
        self.assertEqual(res_dict["HAM"]["pit_stop_efficiency_score"], 1.0)
        self.assertEqual(res_dict["HAM"]["strategy_type"], "1-stop")


if __name__ == "__main__":
    unittest.main()
