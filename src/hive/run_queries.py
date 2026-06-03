"""
Execute Hive Analytical Queries on the F1 Telemetry Analytics Platform.

This module reads the SQL queries defined in analytics_queries.sql,
executes them in the Spark Hive metastore, and prints the resulting tables.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.spark.spark_session import get_spark_session
from src.utils.config import BASE_DIR

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def run_analytical_queries() -> None:
    """Read analytics_queries.sql and execute statements in PySpark SQL."""
    spark = get_spark_session()
    sql_file = BASE_DIR / "src" / "hive" / "analytics_queries.sql"

    if not sql_file.exists():
        raise FileNotFoundError(f"SQL file not found at: {sql_file}")

    logger.info("Reading analytical queries from %s...", sql_file)
    with open(sql_file, "r", encoding="utf-8") as file:
        content = file.read()

    # Split by semicolon, clean statements
    raw_statements = content.split(";")

    for idx, stmt in enumerate(raw_statements, 1):
        # Strip whitespace and lines, remove comments
        lines = stmt.splitlines()
        clean_lines = []
        comment_header = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("--"):
                comment_header.append(stripped.replace("--", "").strip())
            else:
                clean_lines.append(line)

        clean_stmt = "\n".join(clean_lines).strip()

        if not clean_stmt:
            continue

        header_text = " | ".join(comment_header) if comment_header else f"Query #{idx}"
        logger.info("=" * 80)
        logger.info("Running: %s", header_text)
        logger.info("=" * 80)
        logger.info("Executing SQL:\n%s\n", clean_stmt)

        try:
            df = spark.sql(clean_stmt)
            # Show up to 20 rows of results without truncation
            df.show(20, truncate=False)
        except Exception as exc:
            logger.exception("Failed executing query: %s", exc)

    logger.info("All analytical queries completed.")
    spark.stop()


if __name__ == "__main__":
    run_analytical_queries()
