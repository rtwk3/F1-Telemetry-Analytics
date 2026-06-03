"""
Pit stop analytics layer using PySpark.

This module calculates:
1. Pit counts per driver per race.
2. Average pit windows (lap number).
3. Undercut success rate (accounting for finishing status and DNFs).
4. Strategy type ("1-stop", "2-stop", etc.).
5. Pit stop efficiency score (positions gained/lost after stops).

Outputs are written as a partitioned Parquet dataset in the gold layer (warehouse).
"""

from __future__ import annotations

import logging
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import avg, col, coalesce, count, lag, lead, sum, when

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


def compute_pitstop_analytics(df: DataFrame) -> DataFrame:
    """Calculate pit stop metrics, strategy classifications, and efficiency scores."""
    # Window specification to track driver's track position over laps
    window_spec = Window.partitionBy("season", "race_name", "driver").orderBy("lap_number")

    # Get position before the stop (1 lap before) and after the stop (10 laps after or final_position)
    df_with_positions = (
        df
        .withColumn("pos_before_stop", lag("position", 1).over(window_spec))
        .withColumn("pos_after_stop", coalesce(lead("position", 10).over(window_spec), col("final_position")))
    )

    # Filter to laps where a pit entry occurred
    pit_laps = df_with_positions.filter(col("pit_in_time_seconds").isNotNull())

    # Undercut success: driver gained position (lower number) after the stop
    undercut_success_col = when(
        col("pos_before_stop").isNotNull() & col("pos_after_stop").isNotNull() & (col("pos_after_stop") < col("pos_before_stop")),
        1
    ).otherwise(0)

    # Position gain/loss: positive means positions gained (e.g. P5 -> P4)
    position_gain_col = when(
        col("pos_before_stop").isNotNull() & col("pos_after_stop").isNotNull(),
        col("pos_before_stop") - col("pos_after_stop")
    ).otherwise(None)

    # Apply calculations on pit entry laps
    pit_metrics = (
        pit_laps
        .withColumn("undercut_success_flag", undercut_success_col)
        .withColumn("position_gain", position_gain_col)
    )

    # Aggregate by season, race and driver
    pit_agg = pit_metrics.groupBy("season", "race_name", "driver").agg(
        count("lap_number").alias("pit_count"),
        avg("lap_number").alias("avg_pit_window"),
        avg("undercut_success_flag").alias("undercut_success_rate"),
        avg("position_gain").alias("pit_stop_efficiency_score")
    )

    # Final mapping to strategy types and filling nulls for efficiency/undercuts
    pit_analytics_df = (
        pit_agg
        .withColumn(
            "strategy_type",
            when(col("pit_count") == 1, "1-stop")
            .when(col("pit_count") == 2, "2-stop")
            .when(col("pit_count") == 3, "3-stop")
            .when(col("pit_count") > 3, "3+ stops")
            .otherwise("0-stop")
        )
        .withColumn("pit_stop_efficiency_score", col("pit_stop_efficiency_score").cast("double"))
        .fillna({"pit_stop_efficiency_score": 0.0, "undercut_success_rate": 0.0})
    )

    return pit_analytics_df


def run_pitstop_analytics_pipeline() -> None:
    """Run pit stop analytics Spark job and save partitioned Parquet output."""
    spark = get_spark_session()

    input_path = DATA_PROCESSED / "engineered_races.parquet"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Engineered race dataset not found: {input_path}"
        )

    logger.info("Loading engineered race dataset...")
    df = spark.read.parquet(str(input_path))

    logger.info("Computing pit stop analytics...")
    analytics_df = compute_pitstop_analytics(df)

    output_path = DATA_WAREHOUSE / "pitstop_analytics"
    logger.info("Saving pit stop analytics Parquet dataset to %s...", output_path)

    (
        analytics_df
        .repartition("season", "race_name")
        .write
        .mode("overwrite")
        .partitionBy("season", "race_name")
        .parquet(str(output_path))
    )

    logger.info("Pit stop analytics completed successfully.")
    spark.stop()


if __name__ == "__main__":
    run_pitstop_analytics_pipeline()
