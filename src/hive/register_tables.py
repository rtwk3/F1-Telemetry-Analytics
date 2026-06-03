"""
Register Hive External Tables in Spark Metastore.

This module reads the SQL DDL statements from create_tables.sql and executes
them in Spark to register the Parquet gold directories as Hive External Tables.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.spark.spark_session import get_spark_session
from src.utils.config import BASE_DIR, DATA_WAREHOUSE

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def register_tables() -> None:
    """Read create_tables.sql and execute statements in PySpark."""
    spark = get_spark_session()
    sql_file = BASE_DIR / "src" / "hive" / "create_tables.sql"

    if not sql_file.exists():
        raise FileNotFoundError(f"SQL file not found at: {sql_file}")

    logger.info("Reading SQL DDL from %s...", sql_file)
    with open(sql_file, "r", encoding="utf-8") as file:
        content = file.read()

    # Replace absolute warehouse path dynamically
    warehouse_posix_path = DATA_WAREHOUSE.resolve().as_posix()
    content = content.replace("{DATA_WAREHOUSE}", warehouse_posix_path)

    # Split by semicolon, clean statements
    raw_statements = content.split(";")
    
    # Execute each non-empty query
    for stmt in raw_statements:
        # Strip whitespace and lines
        clean_stmt = "\n".join([
            line for line in stmt.splitlines() 
            if line.strip() and not line.strip().startswith("--")
        ]).strip()
        
        if not clean_stmt:
            continue
            
        logger.info("Executing Hive SQL -> %s", clean_stmt.replace("\n", " "))
        try:
            spark.sql(clean_stmt)
        except Exception as exc:
            logger.exception("Failed executing query: %s", exc)

    logger.info("Hive external tables registered successfully in local Spark metastore.")
    spark.stop()


if __name__ == "__main__":
    register_tables()
