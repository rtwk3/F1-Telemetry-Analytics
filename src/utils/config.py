"""
Central configuration module for the F1 Telemetry Analytics Platform.

This module defines:
- Project root and data directories
- FastF1 cache location
- Race configurations
- Spark settings
- Hive warehouse settings

All project files should import paths and constants from here
instead of hardcoding values.
"""

from pathlib import Path

# ============================================================
# Project Root
# ============================================================

# Assumes this file is located at:
# src/utils/config.py
BASE_DIR: Path = Path(__file__).resolve().parents[2]

# ============================================================
# Data Directories
# ============================================================

DATA_DIR: Path = BASE_DIR / "data"

DATA_RAW: Path = DATA_DIR / "raw"
DATA_PROCESSED: Path = DATA_DIR / "processed"
DATA_WAREHOUSE: Path = DATA_DIR / "warehouse"

CACHE_DIR: Path = BASE_DIR / "cache"

# ============================================================
# Additional Project Directories
# ============================================================

SRC_DIR: Path = BASE_DIR / "src"
DASHBOARD_DIR: Path = BASE_DIR / "dashboards"
TESTS_DIR: Path = BASE_DIR / "tests"
DOCS_DIR: Path = BASE_DIR / "docs"
MODELS_DIR: Path = BASE_DIR / "models"

# ============================================================
# Spark Configuration
# ============================================================

SPARK_APP_NAME: str = "F1TelemetryAnalytics"
SPARK_MASTER: str = "local[*]"

# Embedded Hive warehouse location
HIVE_WAREHOUSE_DIR: Path = DATA_WAREHOUSE / "hive"

# ============================================================
# Race Configuration
# ============================================================

RACES: list[tuple[int, str, str]] = [
    (2023, "Bahrain", "R"),
    (2023, "Monaco", "R"),
    (2023, "Monza", "R"),
    (2023, "Silverstone", "R"),
    (2023, "Spa", "R"),
]

# ============================================================
# File Naming
# ============================================================

RAW_FILE_EXTENSION: str = ".parquet"
PROCESSED_FILE_EXTENSION: str = ".parquet"

# ============================================================
# ETL Configuration
# ============================================================

UNKNOWN_CATEGORY_VALUE: str = "Unknown"

NUMERIC_IMPUTATION_METHOD: str = "median"
CATEGORICAL_IMPUTATION_VALUE: str = "Unknown"

# ============================================================
# Analytics Constants
# ============================================================

ROLLING_PACE_WINDOW: int = 5

VALID_TIRE_COMPOUNDS: list[str] = [
    "SOFT",
    "MEDIUM",
    "HARD",
    "INTERMEDIATE",
    "WET",
    "UNKNOWN",
]

# ============================================================
# ML Configuration
# ============================================================

TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42

MODEL_FILENAME: str = "tire_degradation_model.pkl"

# ============================================================
# Create Required Directories
# ============================================================

DIRECTORIES_TO_CREATE: list[Path] = [
    DATA_RAW,
    DATA_PROCESSED,
    DATA_WAREHOUSE,
    CACHE_DIR,
    MODELS_DIR,
    DOCS_DIR,
]

for directory in DIRECTORIES_TO_CREATE:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# Helper Functions
# ============================================================

def get_race_filename(year: int, race_name: str) -> str:
    """
    Return standardized race filename.

    Example:
        Bahrain_2023.parquet
    """
    race_slug = race_name.lower().replace(" ", "_")
    return f"{race_slug}_{year}{RAW_FILE_EXTENSION}"


def get_processed_race_filename(year: int, race_name: str) -> str:
    """
    Return processed race filename.

    Example:
        processed_bahrain_2023.parquet
    """
    race_slug = race_name.lower().replace(" ", "_")
    return f"processed_{race_slug}_{year}{PROCESSED_FILE_EXTENSION}"


def get_race_path(year: int, race_name: str) -> Path:
    """
    Return raw race file path.
    """
    return DATA_RAW / get_race_filename(year, race_name)


def get_processed_race_path(year: int, race_name: str) -> Path:
    """
    Return processed race file path.
    """
    return DATA_PROCESSED / get_processed_race_filename(year, race_name)


# ============================================================
# Debug Output
# ============================================================

if __name__ == "__main__":
    print("=== F1 Telemetry Analytics Configuration ===")
    print(f"BASE_DIR           : {BASE_DIR}")
    print(f"DATA_RAW           : {DATA_RAW}")
    print(f"DATA_PROCESSED     : {DATA_PROCESSED}")
    print(f"DATA_WAREHOUSE     : {DATA_WAREHOUSE}")
    print(f"CACHE_DIR          : {CACHE_DIR}")
    print(f"HIVE_WAREHOUSE_DIR : {HIVE_WAREHOUSE_DIR}")
    print(f"SPARK_MASTER       : {SPARK_MASTER}")
    print(f"RACES              : {len(RACES)} configured")