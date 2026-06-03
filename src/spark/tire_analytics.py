"""
Tire analytics layer using PySpark.

This module aggregates lap-level data by compound to calculate:
1. Average lap times per compound.
2. Average tire degradation rates per driver, stint, and compound.
3. Maximum stint lengths per driver/compound.
4. Compound usage percentage per race.

Outputs are written as a partitioned Parquet dataset in the gold layer (warehouse).
"""

from __future__ import annotations

import logging
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import avg, col, max, sum

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


def compute_tire_analytics(df: DataFrame) -> DataFrame:
    """Aggregate tire-related metrics for F1 race strategy analysis."""
    # 1. Base aggregations per season, race, driver, and compound
    # Average lap time, average degradation rate, and max stint length
    base_agg = df.groupBy("season", "race_name", "driver", "compound").agg(
        avg("lap_time_seconds").alias("avg_lap_time"),
        avg("tire_degradation_rate").alias("tire_degradation_rate"),
        max("stint_length").alias("max_stint_length"),
    )

    # 2. Compound usage percentage per season and race
    laps_per_comp = df.groupBy("season", "race_name", "compound").count()
    window_spec = Window.partitionBy("season", "race_name")
    
    usage_pct = (
        laps_per_comp
        .withColumn("total_laps", sum("count").over(window_spec))
        .withColumn("compound_usage_pct", (col("count") / col("total_laps")) * 100.0)
        .drop("count", "total_laps")
    )

    # 3. Join the compound usage percentage back to the driver-level metrics
    tire_analytics_df = base_agg.join(usage_pct, on=["season", "race_name", "compound"], how="left")

    return tire_analytics_df


def run_tire_analytics_pipeline() -> None:
    """Run tire analytics Spark job and save partitioned Parquet output."""
    spark = get_spark_session()

    input_path = DATA_PROCESSED / "engineered_races.parquet"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Engineered race dataset not found: {input_path}"
        )

    logger.info("Loading engineered race dataset...")
    df = spark.read.parquet(str(input_path))

    logger.info("Computing tire analytics...")
    analytics_df = compute_tire_analytics(df)

    output_path = DATA_WAREHOUSE / "tire_analytics"
    logger.info("Saving tire analytics Parquet dataset to %s...", output_path)

    (
        analytics_df
        .repartition("season", "race_name")
        .write
        .mode("overwrite")
        .partitionBy("season", "race_name")
        .parquet(str(output_path))
    )

    logger.info("Tire analytics completed successfully.")
    spark.stop()


if __name__ == "__main__":
    run_tire_analytics_pipeline()
