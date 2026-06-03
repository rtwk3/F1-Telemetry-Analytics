-- DDL Script to create External Hive Tables for F1 Telemetry Analytics
-- Stored as Parquet and Partitioned by season and race_name

CREATE DATABASE IF NOT EXISTS f1_analytics;
USE f1_analytics;

-- 1. Driver Analytics External Table
DROP TABLE IF EXISTS driver_analytics;
CREATE EXTERNAL TABLE IF NOT EXISTS driver_analytics (
    driver STRING,
    avg_lap_time_seconds DOUBLE,
    fastest_lap DOUBLE,
    lap_time_std DOUBLE,
    grid_position DOUBLE,
    final_position DOUBLE,
    consistency_index DOUBLE,
    position_change DOUBLE
)
PARTITIONED BY (season STRING, race_name STRING)
STORED AS PARQUET
LOCATION '{DATA_WAREHOUSE}/driver_analytics';

-- 2. Tire Analytics External Table
DROP TABLE IF EXISTS tire_analytics;
CREATE EXTERNAL TABLE IF NOT EXISTS tire_analytics (
    compound STRING,
    driver STRING,
    avg_lap_time DOUBLE,
    tire_degradation_rate DOUBLE,
    max_stint_length DOUBLE,
    compound_usage_pct DOUBLE
)
PARTITIONED BY (season STRING, race_name STRING)
STORED AS PARQUET
LOCATION '{DATA_WAREHOUSE}/tire_analytics';

-- 3. Pit Stop Analytics External Table
DROP TABLE IF EXISTS pitstop_analytics;
CREATE EXTERNAL TABLE IF NOT EXISTS pitstop_analytics (
    driver STRING,
    pit_count BIGINT,
    avg_pit_window DOUBLE,
    undercut_success_rate DOUBLE,
    pit_stop_efficiency_score DOUBLE,
    strategy_type STRING
)
PARTITIONED BY (season STRING, race_name STRING)
STORED AS PARQUET
LOCATION '{DATA_WAREHOUSE}/pitstop_analytics';

-- 4. Race Analytics External Table
DROP TABLE IF EXISTS race_analytics;
CREATE EXTERNAL TABLE IF NOT EXISTS race_analytics (
    driver STRING,
    driver_avg_clean_lap_time DOUBLE,
    race_pace_ranking INT,
    driver_positions_gained DOUBLE,
    fastest_sector1_time DOUBLE,
    fastest_sector2_time DOUBLE,
    fastest_sector3_time DOUBLE,
    avg_race_pace DOUBLE,
    estimated_positions_gained DOUBLE
)
PARTITIONED BY (season STRING, race_name STRING)
STORED AS PARQUET
LOCATION '{DATA_WAREHOUSE}/race_analytics';

-- Repair partitioning tables to discover existing partitioned folders
MSCK REPAIR TABLE driver_analytics;
MSCK REPAIR TABLE tire_analytics;
MSCK REPAIR TABLE pitstop_analytics;
MSCK REPAIR TABLE race_analytics;
