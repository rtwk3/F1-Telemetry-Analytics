"""
Driver analytics layer using PySpark.

This module reads the engineered race dataset, performs driver-level aggregation
to calculate average lap times, fastest lap, consistency index, and position changes,
and exports the result as a partitioned Parquet dataset.
"""

from __future__ import annotations

import logging
from pyspark.sql import DataFrame
from pyspark.sql.functions import avg, col, first, min, stddev, when, round

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


def compute_driver_analytics(df: DataFrame) -> DataFrame:
    """Aggregate lap-level data to calculate driver performance metrics."""
    # Group by season, race and driver, aggregating lap-level variables
    driver_agg = df.groupBy("season", "race_name", "driver").agg(
        avg("lap_time_seconds").alias("avg_lap_time_seconds"),
        min("lap_time_seconds").alias("fastest_lap"),
        stddev("lap_time_seconds").alias("lap_time_std"),
        first("grid_position").alias("grid_position"),
        first("final_position").alias("final_position"),
    )

    # Calculate derived metrics: consistency_index and position_change
    # Consistency index is 100 / stddev (higher means more consistent lap times).
    # Handle stddev being 0 or null (e.g. if driver has only 1 lap).
    driver_analytics_df = (
        driver_agg
        .withColumn(
            "consistency_index",
            when(col("lap_time_std").isNull() | (col("lap_time_std") == 0), 0.0)
            .otherwise(round(100.0 / col("lap_time_std"), 2)),
        )
        .withColumn(
            "position_change",
            col("grid_position") - col("final_position"),
        )
    )

    return driver_analytics_df


def run_driver_analytics_pipeline() -> None:
    """Run driver analytics job and write partitioned Parquet output."""
    spark = get_spark_session()

    input_path = DATA_PROCESSED / "engineered_races.parquet"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Engineered race dataset not found: {input_path}"
        )

    logger.info("Loading engineered race dataset...")
    df = spark.read.parquet(str(input_path))

    logger.info("Computing driver analytics...")
    analytics_df = compute_driver_analytics(df)

    # Repartition by partition key to ensure clean partition files
    output_path = DATA_WAREHOUSE / "driver_analytics"
    logger.info("Saving driver analytics Parquet dataset to %s...", output_path)

    (
        analytics_df
        .repartition("season", "race_name")
        .write
        .mode("overwrite")
        .partitionBy("season", "race_name")
        .parquet(str(output_path))
    )

    logger.info("Driver analytics completed successfully.")
    spark.stop()


if __name__ == "__main__":
    run_driver_analytics_pipeline()
