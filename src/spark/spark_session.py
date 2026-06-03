"""
Spark Session initializer for the F1 Telemetry Analytics Platform.

This module provides a helper function to create or retrieve a SparkSession
with Hive support and local[*] execution configuration.
"""

from __future__ import annotations

import logging
from pyspark.sql import SparkSession

from src.utils.config import (
    HIVE_WAREHOUSE_DIR,
    SPARK_APP_NAME,
    SPARK_MASTER,
)

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def get_spark_session() -> SparkSession:
    """Create or get the active SparkSession with Hive metastore support."""
    # Ensure warehouse directory is resolved absolutely
    warehouse_path = HIVE_WAREHOUSE_DIR.resolve()
    warehouse_uri = warehouse_path.as_uri()

    # Configure Derby metastore connection URL to build database within warehouse folder
    # This prevents the Derby metastore_db and derby.log files from cluttering the root directory.
    derby_url = f"jdbc:derby:{warehouse_path.as_posix()}/metastore_db;create=true"

    logger.info("Initializing Spark Session: '%s'", SPARK_APP_NAME)
    logger.info("Master: %s", SPARK_MASTER)
    logger.info("Warehouse Directory: %s", warehouse_uri)
    logger.info("Derby JDBC URL: %s", derby_url)

    spark = (
        SparkSession.builder
        .appName(SPARK_APP_NAME)
        .master(SPARK_MASTER)
        .config("spark.sql.warehouse.dir", warehouse_uri)
        .config("spark.sql.catalogImplementation", "hive")
        .config("spark.driver.extraJavaOptions", f"-Dderby.system.home={warehouse_path.as_posix()}")
        .config("spark.hadoop.javax.jdo.option.ConnectionURL", derby_url)
        .enableHiveSupport()
        .getOrCreate()
    )

    # Verify startup and log available databases in metastore
    try:
        logger.info("Active Hive databases in metastore:")
        spark.sql("SHOW DATABASES").show(truncate=False)
    except Exception as exc:
        logger.warning("Failed to show databases: %s", exc)

    return spark


if __name__ == "__main__":
    # Test initialization
    session = get_spark_session()
    print(f"Spark version successfully loaded: {session.version}")
    session.stop()
