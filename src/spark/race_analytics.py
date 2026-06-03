"""
Race analytics layer using PySpark.

This module calculates race-level and driver-level benchmarks:
1. Fastest sector times (sector1, sector2, sector3) per race.
2. Average race pace (overall average lap time).
3. Estimated positions gained per driver and per race (excluding pit entry/exit).
4. Clean race pace ranking per driver per race (excluding pit laps).

Outputs are written as a partitioned Parquet dataset in the gold layer (warehouse).
"""

from __future__ import annotations

import logging
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import avg, col, lag, min, rank, sum

from src.spark.spark_session import get_spark_session
from src.utils.config import DATA_PROCESSED, DATA_WAREHOUSE

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def compute_race_analytics(df: DataFrame) -> DataFrame:
    """Calculate sector benchmarks, race-level averages, positions gained, and pace ranks."""
    # Window specification for driver's lap history to compute position changes
    driver_window = Window.partitionBy("season", "race_name", "driver").orderBy("lap_number")

    # Get the position from the previous lap
    df_laps = df.withColumn("prev_position", lag("position", 1).over(driver_window))

    # Calculate on-track positions gained per driver per lap
    # Exclude lap 1 (start grid differences) and pit entry/exit laps
    laps_gained = df_laps.filter(
        (col("lap_number") > 1)
        & (col("prev_position").isNotNull())
        & (col("position") < col("prev_position"))
        & (col("pit_in_time_seconds").isNull())
        & (col("pit_out_time_seconds").isNull())
    ).withColumn("gain", col("prev_position") - col("position"))

    # Sum positions gained per driver per race
    driver_gains = laps_gained.groupBy("season", "race_name", "driver").agg(
        sum("gain").alias("driver_positions_gained")
    )

    # Sum total positions gained per race (overall race metric)
    race_gains = driver_gains.groupBy("season", "race_name").agg(
        sum("driver_positions_gained").alias("estimated_positions_gained")
    )

    # Calculate race-level sector benchmarks and overall average pace
    race_benchmarks = df.groupBy("season", "race_name").agg(
        min("sector1_time_seconds").alias("fastest_sector1_time"),
        min("sector2_time_seconds").alias("fastest_sector2_time"),
        min("sector3_time_seconds").alias("fastest_sector3_time"),
        avg("lap_time_seconds").alias("avg_race_pace"),
    )

    # Calculate clean driver pace (average lap time excluding pit laps)
    clean_laps = df.filter(
        (col("pit_in_time_seconds").isNull())
        & (col("pit_out_time_seconds").isNull())
    )
    driver_clean_pace = clean_laps.groupBy("season", "race_name", "driver").agg(
        avg("lap_time_seconds").alias("driver_avg_clean_lap_time")
    )

    # Rank drivers within each race by their clean average lap time (1 is fastest)
    race_window = Window.partitionBy("season", "race_name").orderBy("driver_avg_clean_lap_time")
    driver_ranked = driver_clean_pace.withColumn(
        "race_pace_ranking", rank().over(race_window)
    )

    # Join driver-level metrics (pace ranks, positions gained)
    result = driver_ranked.join(driver_gains, on=["season", "race_name", "driver"], how="left").fillna(0)

    # Join race-level metrics (benchmarks, total positions gained)
    result = result.join(race_benchmarks, on=["season", "race_name"], how="left")
    result = result.join(race_gains, on=["season", "race_name"], how="left").fillna(0)

    return result


def run_race_analytics_pipeline() -> None:
    """Run race analytics Spark job and save partitioned Parquet output."""
    spark = get_spark_session()

    input_path = DATA_PROCESSED / "engineered_races.parquet"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Engineered race dataset not found: {input_path}"
        )

    logger.info("Loading engineered race dataset...")
    df = spark.read.parquet(str(input_path))

    logger.info("Computing race analytics...")
    analytics_df = compute_race_analytics(df)

    output_path = DATA_WAREHOUSE / "race_analytics"
    logger.info("Saving race analytics Parquet dataset to %s...", output_path)

    (
        analytics_df
        .repartition("season", "race_name")
        .write
        .mode("overwrite")
        .partitionBy("season", "race_name")
        .parquet(str(output_path))
    )

    logger.info("Race analytics completed successfully.")
    spark.stop()


if __name__ == "__main__":
    run_race_analytics_pipeline()
