-- ============================================================
-- SQL Queries for F1 Telemetry Analytics (Hive/Spark SQL)
-- 
-- This script contains production-style analytical queries to extract
-- insights from the f1_analytics database external tables:
-- - driver_analytics
-- - tire_analytics
-- - pitstop_analytics
-- - race_analytics
-- ============================================================

USE f1_analytics;

-- ------------------------------------------------------------
-- 1. Top 5 Most Consistent Drivers per Race
-- ------------------------------------------------------------
-- Ranks drivers based on the lap time standard deviation (consistency index).
-- Higher consistency index indicates more stable lap-by-lap pace.
SELECT 
    race_name,
    driver,
    lap_time_std,
    consistency_index,
    DENSE_RANK() OVER (PARTITION BY race_name ORDER BY consistency_index DESC) as consistency_rank
FROM 
    driver_analytics
ORDER BY 
    race_name, 
    consistency_rank;


-- ------------------------------------------------------------
-- 2. Drivers with the Most Track Positions Gained
-- ------------------------------------------------------------
-- Identifies the "Overtaker of the Day" by comparing starting grid
-- positions against final classifications.
SELECT 
    race_name,
    driver,
    grid_position,
    final_position,
    position_change
FROM 
    driver_analytics
WHERE 
    position_change > 0
ORDER BY 
    race_name, 
    position_change DESC;


-- ------------------------------------------------------------
-- 3. Tire Compound Performance & Life Analysis
-- ------------------------------------------------------------
-- Computes the average lap time, tire degradation slope (s/lap),
-- and maximum stint longevity for each compound across all races.
SELECT 
    compound,
    ROUND(AVG(avg_lap_time), 3) as global_avg_lap_time_seconds,
    ROUND(AVG(tire_degradation_rate), 4) as avg_degradation_rate_seconds_per_lap,
    MAX(max_stint_length) as max_observed_stint_laps,
    ROUND(AVG(compound_usage_pct), 2) as avg_compound_usage_pct
FROM 
    tire_analytics
GROUP BY 
    compound
ORDER BY 
    global_avg_lap_time_seconds ASC;


-- ------------------------------------------------------------
-- 4. Pit Stop Strategy Efficiency & Undercut Success
-- ------------------------------------------------------------
-- Analyzes strategy effectiveness by checking the percentage of stops
-- where a driver successfully gained positions after pitting (undercut).
SELECT 
    race_name,
    driver,
    pit_count,
    avg_pit_window,
    strategy_type,
    ROUND(undercut_success_rate * 100.0, 1) as undercut_success_percentage,
    pit_stop_efficiency_score
FROM 
    pitstop_analytics
ORDER BY 
    race_name, 
    pit_stop_efficiency_score DESC;


-- ------------------------------------------------------------
-- 5. Race Pace Ranking vs. Actual Final Classification
-- ------------------------------------------------------------
-- Compares a driver's clean average pace ranking (excluding pit lane time)
-- against their actual final position to see where strategy/traffic made a difference.
SELECT 
    r.race_name,
    r.driver,
    r.race_pace_ranking,
    r.driver_avg_clean_lap_time,
    d.final_position,
    (d.final_position - r.race_pace_ranking) as pace_to_result_delta
FROM 
    race_analytics r
JOIN 
    driver_analytics d ON r.race_name = d.race_name AND r.driver = d.driver
ORDER BY 
    r.race_name, 
    d.final_position;


-- ------------------------------------------------------------
-- 6. Sector Dominance and Circuit Benchmarks
-- ------------------------------------------------------------
-- Identifies the fastest theoretical sector combination per race
-- and compares it with the average race pace of the field.
SELECT 
    race_name,
    ROUND(MIN(fastest_sector1_time), 3) as theoretical_fastest_s1,
    ROUND(MIN(fastest_sector2_time), 3) as theoretical_fastest_s2,
    ROUND(MIN(fastest_sector3_time), 3) as theoretical_fastest_s3,
    ROUND(MIN(fastest_sector1_time + fastest_sector2_time + fastest_sector3_time), 3) as theoretical_lap_record,
    ROUND(AVG(avg_race_pace), 3) as overall_average_race_pace,
    SUM(estimated_positions_gained) as total_on_track_overtakes
FROM 
    race_analytics
GROUP BY 
    race_name
ORDER BY 
    race_name;
