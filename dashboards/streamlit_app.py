"""
F1 Telemetry Analytics Streamlit Dashboard.

This module provides an interactive multi-page Streamlit dashboard to visualize
driver performance, tire degradation, pit stop strategies, race benchmarks,
and machine learning lap time predictions.
"""

from __future__ import annotations

import logging
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Add project root to sys.path to enable imports of the src package
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils.config import (
    DATA_PROCESSED,
    DATA_WAREHOUSE,
    MODELS_DIR,
    RACES,
    CACHE_DIR,
)
from src.utils.race_selector import (
    get_selected_races,
    save_selected_races,
)

# ------------------------------------------------------------------
# Setup and Style
# ------------------------------------------------------------------



LOGO_PATH = Path(__file__).parent / "assets" / "image.png"

st.set_page_config(
    page_title="F1 Telemetry Analytics Platform",
    page_icon=str(LOGO_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_svg(filename: str) -> str:
    """Load SVG content from the assets folder."""
    path = Path(__file__).resolve().parent / "assets" / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    return ""


def render_header(title: str, icon_svg: str) -> None:
    """Render a header with an inline SVG icon for offline-compatible styling."""
    html_content = (
        f"<div style='display: flex; align-items: center; gap: 12px; margin-bottom: 25px; margin-top: 10px;'>"
        f"{icon_svg}"
        f"<h1 style='margin: 0; font-size: 32px; font-family: \"Source Sans Pro\", sans-serif; font-weight: 700; line-height: 1.2;'>{title}</h1>"
        f"</div>"
    )
    st.markdown(html_content, unsafe_allow_html=True)


def render_sidebar_title() -> None:
    """Render the sidebar title with an F1 speedometer icon."""
    logo_svg = load_svg("logo.svg")
    html_content = (
        f"<div style='display: flex; align-items: center; gap: 10px; margin-top: -15px; margin-bottom: 20px;'>"
        f"{logo_svg}"
        f"<span style='font-size: 22px; font-weight: bold; color: #FFFFFF; font-family: \"Source Sans Pro\", sans-serif;'>F1 Analytics</span>"
        f"</div>"
    )
    st.sidebar.markdown(html_content, unsafe_allow_html=True)

# F1 Team Color Palette for charts
TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "Ferrari": "#F91536",
    "Mercedes": "#6CD3BF",
    "Aston Martin": "#358C75",
    "McLaren": "#F58020",
    "Alpine": "#2293D1",
    "Williams": "#37BEDD",
    "Haas F1 Team": "#B6BABD",
    "Alfa Romeo": "#C62C3E",
    "AlphaTauri": "#5E8FAA",
    "Unknown": "#B6BABD",
}


# ------------------------------------------------------------------
# Data Loading Helpers (with caching & automatic file-watch invalidation)
# ------------------------------------------------------------------

def get_mtime(path: Path) -> float:
    """Get the maximum modification time of a file or directory's contents."""
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_mtime
    # For directories (like partitioned parquets), get the max mtime of any parquet file inside
    parquet_files = list(path.glob("**/*.parquet"))
    if not parquet_files:
        return path.stat().st_mtime
    return max(f.stat().st_mtime for f in parquet_files)


@st.cache_data
def load_engineered_laps(mtime: float) -> pd.DataFrame:
    """Load the detailed lap-level engineered dataset from the silver layer."""
    path = DATA_PROCESSED / "engineered_races.parquet"
    if not path.exists():
        st.error(f"Silver dataset not found: {path}. Run ETL pipeline first.")
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_driver_analytics(mtime: float) -> pd.DataFrame:
    """Load driver-level aggregated analytics from the gold layer."""
    path = DATA_WAREHOUSE / "driver_analytics"
    if not path.exists():
        st.error(f"Driver analytics dataset not found: {path}. Run Spark job first.")
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_tire_analytics(mtime: float) -> pd.DataFrame:
    """Load tire-level aggregated analytics from the gold layer."""
    path = DATA_WAREHOUSE / "tire_analytics"
    if not path.exists():
        st.error(f"Tire analytics dataset not found: {path}. Run Spark job first.")
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_pitstop_analytics(mtime: float) -> pd.DataFrame:
    """Load pit stop strategy analytics from the gold layer."""
    path = DATA_WAREHOUSE / "pitstop_analytics"
    if not path.exists():
        st.error(f"Pit stop analytics dataset not found: {path}. Run Spark job first.")
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_race_analytics(mtime: float) -> pd.DataFrame:
    """Load race-level and pace rank benchmarks from the gold layer."""
    path = DATA_WAREHOUSE / "race_analytics"
    if not path.exists():
        st.error(f"Race analytics dataset not found: {path}. Run Spark job first.")
        return pd.DataFrame()
    return pd.read_parquet(path)


# ------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------

def render_race_overview(
    laps_df: pd.DataFrame,
    driver_df: pd.DataFrame,
    race_df: pd.DataFrame,
) -> None:
    """Render the Race Overview dashboard page."""
    render_header("Race Overview & Benchmarks", load_svg("flag.svg"))

    # Race selection with season disambiguation
    race_options = laps_df[["season", "race_name"]].drop_duplicates().sort_values(by=["season", "race_name"])
    race_list = [f"{int(row['season'])} - {row['race_name']}" for _, row in race_options.iterrows()]
    selected_race = st.selectbox("Select Race", race_list)

    # Split selection back into season and race_name
    selected_season = int(selected_race.split(" - ", 1)[0])
    selected_race_name = selected_race.split(" - ", 1)[1]

    # Filter data for selected race
    race_laps = laps_df[(laps_df["race_name"] == selected_race_name) & (laps_df["season"] == selected_season)]
    race_drivers = driver_df[(driver_df["race_name"] == selected_race_name) & (driver_df["season"] == selected_season)]
    race_bench = race_df[(race_df["race_name"] == selected_race_name) & (race_df["season"] == selected_season)]

    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    # Find Winner (final_position = 1)
    winner_row = race_drivers[race_drivers["final_position"] == 1]
    winner_name = winner_row["driver"].iloc[0] if not winner_row.empty else "N/A"
    winner_team = race_laps[race_laps["driver"] == winner_name]["team"].iloc[0] if not winner_row.empty else "N/A"

    # Find Fastest Lap
    fastest_row = race_drivers.loc[race_drivers["fastest_lap"].idxmin()] if not race_drivers.empty else None
    fastest_driver = fastest_row["driver"] if fastest_row is not None else "N/A"
    fastest_time = fastest_row["fastest_lap"] if fastest_row is not None else 0.0

    total_laps = int(race_laps["lap_number"].max())
    total_gains = int(race_bench["estimated_positions_gained"].iloc[0]) if not race_bench.empty else 0

    with col1:
        st.metric("Race Winner", f"{winner_name} ({winner_team})")
    with col2:
        st.metric("Fastest Lap", f"{fastest_time:.3f}s", f"by {fastest_driver}")
    with col3:
        st.metric("Total Laps", f"{total_laps}")
    with col4:
        st.metric("Positions Gained on Track", f"{total_gains}")

    st.markdown("---")

    # Race lap chart (positions over laps)
    st.subheader("Driver Track Position over Laps")
    
    # Sort for chart
    race_laps_sorted = race_laps.sort_values(by=["driver", "lap_number"])
    
    fig = px.line(
        race_laps_sorted,
        x="lap_number",
        y="position",
        color="driver",
        title="Position Chart",
        labels={"lap_number": "Lap Number", "position": "Position on Track", "driver": "Driver"},
        color_discrete_map={d: TEAM_COLORS.get(t, "#FFFFFF") for d, t in zip(race_laps_sorted["driver"], race_laps_sorted["team"])}
    )
    fig.update_yaxes(autorange="reversed", tick0=1, dtick=1)  # P1 at top
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


def render_driver_analysis(
    laps_df: pd.DataFrame,
    driver_df: pd.DataFrame,
) -> None:
    """Render the Driver Analysis page."""
    render_header("Driver Performance & Consistency", load_svg("driver.svg"))

    unique_drivers = sorted(laps_df["driver"].unique())
    selected_drivers = st.multiselect("Select Drivers for Analysis", unique_drivers, default=unique_drivers[:3])

    if not selected_drivers:
        st.warning("Please select at least one driver.")
        return

    # Filter data
    filtered_laps = laps_df[laps_df["driver"].isin(selected_drivers)].copy()
    filtered_laps["race_display"] = filtered_laps["season"].astype(str) + " - " + filtered_laps["race_name"].astype(str)
    filtered_drivers = driver_df[driver_df["driver"].isin(selected_drivers)].copy()
    filtered_drivers["race_display"] = filtered_drivers["season"].astype(str) + " - " + filtered_drivers["race_name"].astype(str)

    # 1. Lap times comparison chart
    st.subheader("Lap Times over Laps")
    fig_lap_times = px.line(
        filtered_laps.sort_values(by=["driver", "lap_number"]),
        x="lap_number",
        y="lap_time_seconds",
        color="driver",
        facet_col="race_display",
        facet_col_wrap=3,
        labels={"lap_number": "Lap Number", "lap_time_seconds": "Lap Time (seconds)", "driver": "Driver", "race_display": "Race"},
        title="Lap Time Pace Charts by Race"
    )
    st.plotly_chart(fig_lap_times, use_container_width=True)

    # 2. Consistency vs Position Changes Columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Consistency Index (Higher = More Consistent)")
        # Calculate consistency index
        fig_consistency = px.bar(
            filtered_drivers,
            x="driver",
            y="consistency_index",
            color="race_display",
            barmode="group",
            labels={"consistency_index": "Consistency Score (100 / stddev)", "driver": "Driver", "race_display": "Race"},
            title="Consistency Index per Driver per Race"
        )
        st.plotly_chart(fig_consistency, use_container_width=True)

    with col2:
        st.subheader("Position Change (Starting Grid vs Final Classification)")
        fig_position = px.bar(
            filtered_drivers,
            x="driver",
            y="position_change",
            color="race_display",
            barmode="group",
            labels={"position_change": "Grid Position - Final Position", "driver": "Driver", "race_display": "Race"},
            title="Positions Gained/Lost (Positive is gained)"
        )
        st.plotly_chart(fig_position, use_container_width=True)


def render_tire_analysis(
    laps_df: pd.DataFrame,
    tire_df: pd.DataFrame,
) -> None:
    """Render the Tire Analysis page."""
    render_header("Tire Degradation & Stint Analysis", load_svg("tire.svg"))

    race_options = laps_df[["season", "race_name"]].drop_duplicates().sort_values(by=["season", "race_name"])
    race_list = [f"{int(row['season'])} - {row['race_name']}" for _, row in race_options.iterrows()]
    selected_race = st.selectbox("Select Race", race_list, key="tire_race_select")

    # Split selection back into season and race_name
    selected_season = int(selected_race.split(" - ", 1)[0])
    selected_race_name = selected_race.split(" - ", 1)[1]

    # Filter data for selected race
    race_laps = laps_df[(laps_df["race_name"] == selected_race_name) & (laps_df["season"] == selected_season)]
    race_tires = tire_df[(tire_df["race_name"] == selected_race_name) & (tire_df["season"] == selected_season)]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tire Degradation Curves")
        compounds = sorted(race_laps["compound"].unique())
        selected_compound = st.selectbox("Select Compound", compounds)
        
        comp_laps = race_laps[race_laps["compound"] == selected_compound]
        
        fig_deg = px.scatter(
            comp_laps,
            x="tyre_life",
            y="lap_time_seconds",
            color="driver",
            trendline="ols",
            labels={"tyre_life": "Tyre Life (Laps Run)", "lap_time_seconds": "Lap Time (Seconds)", "driver": "Driver"},
            title=f"Degradation Slope on {selected_compound} Tires"
        )
        st.plotly_chart(fig_deg, use_container_width=True)

    with col2:
        st.subheader("Compound Usage Percentage")
        # Unique rows per race/compound for usage chart
        usage_pct = race_tires[["compound", "compound_usage_pct"]].drop_duplicates()
        
        fig_pie = px.pie(
            usage_pct,
            values="compound_usage_pct",
            names="compound",
            color="compound",
            color_discrete_map={
                "SOFT": "#F91536", "MEDIUM": "#FFE500", "HARD": "#FFFFFF",
                "INTERMEDIATE": "#47B03C", "WET": "#1C76C4", "UNKNOWN": "#B6BABD"
            },
            title="Compound Stint Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("Maximum Stint Length and Average Degradation Rate")
    
    col3, col4 = st.columns(2)
    with col3:
        fig_stint = px.bar(
            race_tires,
            x="driver",
            y="max_stint_length",
            color="compound",
            barmode="group",
            labels={"max_stint_length": "Maximum Tyre Life reached", "driver": "Driver"},
            title="Maximum Stint Length by Driver and Compound"
        )
        st.plotly_chart(fig_stint, use_container_width=True)
        
    with col4:
        fig_deg_rate = px.bar(
            race_tires,
            x="driver",
            y="tire_degradation_rate",
            color="compound",
            barmode="group",
            labels={"tire_degradation_rate": "Degradation Rate (Seconds per Lap)", "driver": "Driver"},
            title="Average Tire Degradation Rate per Driver"
        )
        st.plotly_chart(fig_deg_rate, use_container_width=True)


def render_pitstop_analysis(
    laps_df: pd.DataFrame,
    pit_df: pd.DataFrame,
) -> None:
    """Render the Pit Stop Analysis page."""
    render_header("Pit Stop Strategy & Undercut Success", load_svg("pitstop.svg"))

    race_options = laps_df[["season", "race_name"]].drop_duplicates().sort_values(by=["season", "race_name"])
    race_list = [f"{int(row['season'])} - {row['race_name']}" for _, row in race_options.iterrows()]
    selected_race = st.selectbox("Select Race", race_list, key="pit_race_select")

    # Split selection back into season and race_name
    selected_season = int(selected_race.split(" - ", 1)[0])
    selected_race_name = selected_race.split(" - ", 1)[1]

    # Filter data for selected race
    race_laps = laps_df[(laps_df["race_name"] == selected_race_name) & (laps_df["season"] == selected_season)]
    race_pits = pit_df[(pit_df["race_name"] == selected_race_name) & (pit_df["season"] == selected_season)]

    if race_pits.empty:
        st.warning("No pit stop analytics data available for the selected race.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Strategy Classification Distribution")
        # Counts of strategy types
        strat_df = race_pits.groupby("strategy_type").size().reset_index(name="driver_count")
        
        fig_strat = px.bar(
            strat_df,
            x="strategy_type",
            y="driver_count",
            labels={"strategy_type": "Strategy Type", "driver_count": "Driver Count"},
            title="Pit Stop Strategies Adopted"
        )
        st.plotly_chart(fig_strat, use_container_width=True)

    with col2:
        # Pit timing scatter plot
        st.subheader("Pit Stop Timing (Lap Numbers)")
        pit_laps = race_laps[race_laps["pit_in_time_seconds"].notnull()]
        
        if pit_laps.empty:
            st.info("No pit entry laps recorded for this race.")
        else:
            fig_timing = px.scatter(
                pit_laps,
                x="lap_number",
                y="driver",
                color="compound",
                labels={"lap_number": "Lap of Pit Stop", "driver": "Driver", "compound": "Tire Compound Exiting"},
                title="Pit Entry Lap Window by Driver"
            )
            st.plotly_chart(fig_timing, use_container_width=True)

    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Pit Stop Efficiency Score (Positions Gained after stop)")
        # Sorted by efficiency
        fig_eff = px.bar(
            race_pits.sort_values(by="pit_stop_efficiency_score", ascending=False),
            x="driver",
            y="pit_stop_efficiency_score",
            labels={"pit_stop_efficiency_score": "Net Positions Gained (over 2 laps)", "driver": "Driver"},
            title="Strategy / Undercut Efficiency Score"
        )
        st.plotly_chart(fig_eff, use_container_width=True)

    with col4:
        st.subheader("Undercut Success Rate")
        avg_undercut = float(race_pits["undercut_success_rate"].mean() * 100.0)
        
        if avg_undercut == 0.0:
            st.info("No successful undercuts detected in selected races.")
            st.metric(
                label="Average Undercut Success Rate", 
                value="0.0%",
                help="Percentage of pit stops that resulted in a net track position gain 2 laps after pit exit."
            )
        else:
            st.metric(
                label="Average Undercut Success Rate", 
                value=f"{avg_undercut:.1f}%",
                help="Percentage of pit stops that resulted in a net track position gain 2 laps after pit exit."
            )
        
        # Breakdown by driver
        fig_undercut = px.bar(
            race_pits,
            x="driver",
            y="undercut_success_rate",
            labels={"undercut_success_rate": "Success Rate Ratio", "driver": "Driver"},
            title="Undercut Success Ratio per Driver"
        )
        st.plotly_chart(fig_undercut, use_container_width=True)


def render_ml_predictions(laps_df: pd.DataFrame) -> None:
    """Render the Machine Learning Predictions and inputs page."""
    render_header("Machine Learning Lap Time Predictor", load_svg("ml.svg"))

    model_path = MODELS_DIR / "tire_degradation_model.pkl"
    model_loaded = False
    model = None

    if model_path.exists():
        try:
            model = joblib.load(model_path)
            model_loaded = True
        except Exception as exc:
            st.error(f"Failed to load model from {model_path}: {exc}")
    else:
        st.warning(
            "XGBoost Model file not found. Please train the model by running:\n"
            "```powershell\n"
            "python -m src.ml.tire_prediction\n"
            "```\n"
            "Displaying baseline heuristic predictions for demonstration purposes."
        )

    # 1. Inputs Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predictive Features")
        
        # User sliders/inputs
        tyre_life = st.slider("Tyre Life (Laps Run)", 1, 50, 5)
        lap_number = st.slider("Lap Number in Race", 1, 70, 10)
        air_temp = st.slider("Air Temperature (°C)", 10.0, 45.0, 25.0)
        track_temp = st.slider("Track Temperature (°C)", 10.0, 60.0, 35.0)
        
        compounds = sorted(laps_df["compound"].unique())
        compound = st.selectbox("Tire Compound", compounds, index=compounds.index("MEDIUM") if "MEDIUM" in compounds else 0)
        
        drivers = sorted(laps_df["driver"].unique())
        driver = st.selectbox("Driver Name", drivers)
        
        races = sorted(laps_df["race_name"].unique())
        race = st.selectbox("Race / Circuit", races)
        
        rainfall = st.checkbox("Rainfall Active")

    with col2:
        st.subheader("Predicted Output")

        # 2. Prediction Calculation
        if model_loaded:
            # Create features DataFrame matching column names in training script
            input_df = pd.DataFrame([{
                "TyreLife": float(tyre_life),
                "Compound": str(compound),
                "LapNumber": float(lap_number),
                "Circuit": str(race),
                "Driver": str(driver),
                "AirTemp": float(air_temp),
                "TrackTemp": float(track_temp),
                "Rainfall": bool(rainfall)
            }])
            
            try:
                pred_sec = float(model.predict(input_df)[0])
                st.success(f"### Predicted Lap Time: {pred_sec:.3f} seconds")
            except Exception as exc:
                st.error(f"Prediction error: {exc}")
                pred_sec = 90.0  # fallback
        else:
            # Baseline motorsport heuristic prediction
            base_time = 85.0
            deg_factor = 0.08 if compound == "SOFT" else (0.05 if compound == "MEDIUM" else 0.03)
            weather_factor = 0.02 * (track_temp - 30.0)
            wet_factor = 10.0 if rainfall else 0.0
            
            pred_sec = base_time + (tyre_life * deg_factor) + (lap_number * -0.01) + weather_factor + wet_factor
            st.info(f"### [Heuristic Mode] Predicted Lap Time: {pred_sec:.3f} seconds")

        st.markdown("---")
        st.subheader("Feature Importance")

        # Display mock or real feature importance
        if model_loaded and hasattr(model, "named_steps") and "regressor" in model.named_steps:
            try:
                regressor = model.named_steps["regressor"]
                importances = regressor.feature_importances_
                
                # Retrieve engineered features list
                feature_names = ["TyreLife", "Compound", "LapNumber", "Circuit", "Driver", "AirTemp", "TrackTemp", "Rainfall"]
                imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
                
                fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title="XGBoost Feature Importance")
                st.plotly_chart(fig_imp, use_container_width=True)
            except Exception as exc:
                st.error(f"Failed to display model importance: {exc}")
        else:
            # Display heuristic feature importance
            mock_imp = pd.DataFrame({
                "Feature": ["TyreLife", "Compound", "LapNumber", "Circuit", "Driver", "AirTemp", "TrackTemp", "Rainfall"],
                "Importance": [0.35, 0.20, 0.08, 0.05, 0.02, 0.05, 0.10, 0.15]
            }).sort_values(by="Importance", ascending=False)
            
            fig_imp = px.bar(mock_imp, x="Importance", y="Feature", orientation="h", title="XGBoost Feature Importance (Heuristic)")
            st.plotly_chart(fig_imp, use_container_width=True)


@st.cache_data
def get_cached_schedule(year: int) -> pd.DataFrame:
    """Retrieve event schedule using FastF1 and cache the results."""
    import fastf1
    try:
        fastf1.Cache.enable_cache(str(CACHE_DIR))
    except Exception:
        pass
    schedule = fastf1.get_event_schedule(year)
    return schedule[schedule["RoundNumber"] > 0]


def render_ingestion_settings() -> list[tuple[int, str, str]]:
    """Render the Data Ingestion Settings sidebar section and return selected races."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Ingestion Settings")

    # Initialize session state variables if they don't exist
    if "ingestion_season" not in st.session_state:
        st.session_state["ingestion_season"] = 2023
    if "ingestion_mode" not in st.session_state:
        st.session_state["ingestion_mode"] = "Specific races"
    if "ingestion_n_races" not in st.session_state:
        st.session_state["ingestion_n_races"] = 5
    if "ingestion_specific_races" not in st.session_state:
        # Default options for 2023 season
        st.session_state["ingestion_specific_races"] = [
            "Bahrain Grand Prix",
            "Monaco Grand Prix",
            "Italian Grand Prix",
            "British Grand Prix",
            "Belgian Grand Prix",
        ]

    # Season Selection
    season = st.sidebar.selectbox(
        "Select Season",
        options=[2022, 2023, 2024, 2025],
        index=[2022, 2023, 2024, 2025].index(st.session_state["ingestion_season"]),
        key="ingestion_season"
    )

    # Ingestion Mode Selection
    mode = st.sidebar.selectbox(
        "Ingestion Mode",
        options=["First N races", "Specific races", "Entire season"],
        index=["First N races", "Specific races", "Entire season"].index(st.session_state["ingestion_mode"]),
        key="ingestion_mode"
    )

    # Load schedule dynamically
    available_races = []
    try:
        schedule_df = get_cached_schedule(season)
        available_races = schedule_df["EventName"].tolist()
    except Exception as e:
        st.sidebar.error(f"Failed to fetch F1 schedule: {e}")

    # Mode specific widget
    selection = None
    if mode == "First N races":
        max_races = len(available_races) if available_races else 24
        n_races = st.sidebar.slider(
            "Number of Races",
            min_value=1,
            max_value=max_races,
            value=min(st.session_state["ingestion_n_races"], max_races),
            key="ingestion_n_races"
        )
        selection = n_races
    elif mode == "Specific races":
        default_sel = [r for r in st.session_state["ingestion_specific_races"] if r in available_races]
        # Fallback to map original 5 races if year is 2023 and no matches found
        if not default_sel and season == 2023:
            original_names = ["Bahrain Grand Prix", "Monaco Grand Prix", "Italian Grand Prix", "British Grand Prix", "Belgian Grand Prix"]
            default_sel = [r for r in original_names if r in available_races]

        selected_races = st.sidebar.multiselect(
            "Select Specific Races",
            options=available_races,
            default=default_sel,
            key="ingestion_specific_races"
        )
        selection = selected_races
    else:
        selection = None

    # Resolve list of races
    resolved_races = get_selected_races(season, mode, selection)

    # Save to config file
    try:
        save_selected_races(resolved_races)
    except Exception as e:
        st.sidebar.error(f"Error saving ingestion config: {e}")

    # Display dataset estimation metrics
    num_races = len(resolved_races)
    estimated_laps = num_races * 1200 # 60 laps * 20 drivers

    st.sidebar.markdown("### Estimated Dataset Size")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Selected Races", f"{num_races}")
    col2.metric("Est. Laps (All Drivers)", f"{estimated_laps:,}")

    # Add size warnings
    if num_races > 20:
        st.sidebar.error("Telemetry extraction may require several GB of storage.")
    elif num_races > 15:
        st.sidebar.warning("Large dataset selected. Extraction may take 15-30 minutes.")

    # Show CLI command helper
    st.sidebar.markdown("### Run Ingestion Pipeline")
    st.sidebar.caption("Run the command below in your shell to ingest the selected dataset:")
    st.sidebar.code("python -m src.ingestion.extract_fastf1\npython -m src.ingestion.load_races", language="powershell")

    # Reset button to return to the original 5-race configuration
    if st.sidebar.button("Reset to Default", help="Clear custom configurations and return to the original 5 races"):
        import os
        from src.utils.race_selector import DYNAMIC_CONFIG_PATH
        if DYNAMIC_CONFIG_PATH.exists():
            try:
                os.remove(DYNAMIC_CONFIG_PATH)
            except Exception:
                pass
        # Clear session state keys to force fallback defaults
        for key in ["ingestion_season", "ingestion_mode", "ingestion_n_races", "ingestion_specific_races"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    return resolved_races


# ------------------------------------------------------------------
# Pipeline Control Center Constants and Helpers
# ------------------------------------------------------------------

PIPELINE_CONFIG = {
    "Extract Data": {
        "modules": [
            "src.ingestion.extract_fastf1",
            "src.ingestion.load_races"
        ],
        "est_time": "~2-10 min",
    },
    "Run ETL": {
        "modules": [
            "src.etl.data_quality",
            "src.etl.clean_data",
            "src.etl.transform_data",
            "src.etl.feature_engineering"
        ],
        "est_time": "~10 sec",
    },
    "Run Analytics": {
        "modules": [
            "src.spark.driver_analytics",
            "src.spark.tire_analytics",
            "src.spark.pitstop_analytics",
            "src.spark.race_analytics"
        ],
        "est_time": "~20 sec",
    },
    "Register Hive Tables": {
        "modules": [
            "src.hive.register_tables"
        ],
        "est_time": "~5 sec",
    },
    "Train ML Model": {
        "modules": [
            "src.ml.tire_prediction"
        ],
        "est_time": "~30 sec",
    },
    "Run Full Pipeline": {
        "modules": [
            "src.ingestion.extract_fastf1",
            "src.ingestion.load_races",
            "src.etl.data_quality",
            "src.etl.clean_data",
            "src.etl.transform_data",
            "src.etl.feature_engineering",
            "src.spark.driver_analytics",
            "src.spark.tire_analytics",
            "src.spark.pitstop_analytics",
            "src.spark.race_analytics",
            "src.hive.register_tables",
            "src.ml.tire_prediction"
        ],
        "est_time": "~5-15 min",
    }
}


def run_command(cmd: list[str], cwd: Path, log_area) -> tuple[int, str]:
    """Execute a command, displaying log outputs in real-time inside the log_area."""
    if cmd[0] == "python":
        cmd[0] = sys.executable

    logs = []
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            logs.append(line)
            # Accumulate logs in session state
            current_logs = st.session_state.get("pipeline_logs", "") + line
            st.session_state["pipeline_logs"] = current_logs
            # Update the log area in real-time, showing last 30 lines
            log_area.code(current_logs[-30:], language="text")

    return_code = process.wait()
    return return_code, "".join(logs)


def run_pipeline_step(
    step_name: str,
    modules: list[str],
    progress_bar,
    status_text,
    log_area
) -> bool:
    """Run pipeline step by executing the list of python modules sequentially."""
    total_steps = len(modules)
    success = True

    for idx, module in enumerate(modules):
        step_num = idx + 1
        progress_bar.progress(idx / total_steps)
        status_text.text(f"Step {step_num}/{total_steps}: Running {module}...")

        cmd = ["python", "-m", module]
        return_code, logs = run_command(cmd, project_root, log_area)
        if return_code != 0:
            success = False
            status_text.text(f"Step {step_num}/{total_steps}: Failed running {module}")
            break

    if success:
        progress_bar.progress(1.0)
        status_text.text(f"Completed: {step_name} successfully!")
    return success


def get_dataset_status() -> dict[str, bool]:
    """Check availability of datasets and model files."""
    from src.utils.config import DATA_RAW, DATA_PROCESSED, DATA_WAREHOUSE, MODELS_DIR

    raw_file = DATA_RAW / "all_races_combined.parquet"
    cleaned_file = DATA_PROCESSED / "cleaned_races.parquet"
    transformed_file = DATA_PROCESSED / "transformed_races.parquet"
    engineered_file = DATA_PROCESSED / "engineered_races.parquet"
    driver_analytics = DATA_WAREHOUSE / "driver_analytics"
    model_file = MODELS_DIR / "tire_degradation_model.pkl"

    return {
        "Raw Data": raw_file.exists(),
        "Cleaned Data": cleaned_file.exists(),
        "Transformed Data": transformed_file.exists(),
        "Engineered Data": engineered_file.exists(),
        "Analytics Warehouse": driver_analytics.exists(),
        "ML Model": model_file.exists()
    }


def render_pipeline_controls() -> None:
    """Render the Pipeline Control section in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Pipeline Control")

    # Initialize session states
    if "pipeline_running" not in st.session_state:
        st.session_state["pipeline_running"] = False
    if "pipeline_history" not in st.session_state:
        st.session_state["pipeline_history"] = []
    if "pipeline_status" not in st.session_state:
        st.session_state["pipeline_status"] = "Ready"  # Ready, Running, Failed
    if "confirm_full_pipeline" not in st.session_state:
        st.session_state["confirm_full_pipeline"] = False
    if "pipeline_logs" not in st.session_state:
        st.session_state["pipeline_logs"] = ""
    if "pipeline_durations" not in st.session_state:
        st.session_state["pipeline_durations"] = {}

    status = get_dataset_status()
    raw_exists = status["Raw Data"]
    engineered_exists = status["Engineered Data"]
    analytics_exists = status["Analytics Warehouse"]

    running = st.session_state["pipeline_running"]

    # Display Current status
    status_icon_map = {
        "Ready": "🟢 Ready",
        "Running": "🟡 Running",
        "Failed": "🔴 Failed"
    }
    current_status_label = status_icon_map.get(st.session_state["pipeline_status"], "🟢 Ready")
    st.sidebar.markdown(f"**Pipeline Status**: {current_status_label}")

    # Checkbox for smart skipping completed steps
    skip_completed = st.sidebar.checkbox(
        "Skip completed steps",
        value=False,
        help="If checked, already compiled datasets or model runs will be skipped during execution."
    )

    # Helper function to trigger step
    def trigger_step(step_name: str):
        st.session_state["pipeline_running"] = True
        st.session_state["pipeline_status"] = "Running"
        st.session_state["running_step_name"] = step_name
        st.rerun()

    # Buttons with runtimes
    if not st.session_state["confirm_full_pipeline"]:
        st.sidebar.button(
            "📥 Extract Data (est. ~2-10 min)",
            disabled=running,
            key="btn_extract",
            on_click=trigger_step,
            args=("Extract Data",)
        )

        st.sidebar.button(
            "🧹 Run ETL (est. ~10 sec)",
            disabled=running or not raw_exists,
            key="btn_etl",
            on_click=trigger_step,
            args=("Run ETL",)
        )

        st.sidebar.button(
            "⚡ Run Analytics (est. ~20 sec)",
            disabled=running or not engineered_exists,
            key="btn_analytics",
            on_click=trigger_step,
            args=("Run Analytics",)
        )

        st.sidebar.button(
            "🗄️ Register Hive Tables (est. ~5 sec)",
            disabled=running or not analytics_exists,
            key="btn_hive",
            on_click=trigger_step,
            args=("Register Hive Tables",)
        )

        st.sidebar.button(
            "🤖 Train ML Model (est. ~30 sec)",
            disabled=running or not engineered_exists,
            key="btn_ml",
            on_click=trigger_step,
            args=("Train ML Model",)
        )

        st.sidebar.button(
            "🚀 Run Full Pipeline (est. ~5-15 min)",
            disabled=running,
            key="btn_full_pipeline"
        )

        # Handle clicking on Run Full Pipeline
        if st.session_state.get("btn_full_pipeline"):
            st.session_state["confirm_full_pipeline"] = True
            st.rerun()
    else:
        # Show confirmation warning in the sidebar
        st.sidebar.warning("⚠️ **Warning**: This operation may take several minutes and download large amounts of telemetry data. Continue?")
        col_yes, col_no = st.sidebar.columns(2)
        if col_yes.button("Yes, Continue"):
            st.session_state["confirm_full_pipeline"] = False
            trigger_step("Run Full Pipeline")
        if col_no.button("Cancel"):
            st.session_state["confirm_full_pipeline"] = False
            st.rerun()

    # If a step is running, render execution progress and logs inside the sidebar
    if running and "running_step_name" in st.session_state:
        step_name = st.session_state["running_step_name"]
        st.sidebar.markdown(f"### ⚙️ Executing: {step_name}")

        progress_bar = st.sidebar.progress(0.0)
        status_text = st.sidebar.empty()

        log_expander = st.sidebar.expander("Pipeline Logs", expanded=True)
        with log_expander:
            log_area = st.empty()

        # Run the step
        start_time = time.time()
        st.session_state["pipeline_logs"] = ""  # Reset logs for this run

        # Check if the selected races match the last executed configuration
        last_config_path = project_root / "config" / "last_pipeline_config.json"
        current_races = get_selected_races()
        last_races = None
        if last_config_path.exists():
            try:
                import json
                with open(last_config_path, "r", encoding="utf-8") as f:
                    last_races = [tuple(item) for item in json.load(f)]
            except Exception:
                pass

        current_races_tuples = [tuple(item) for item in current_races]
        config_changed = (last_races != current_races_tuples)

        effective_skip_completed = skip_completed
        if config_changed and skip_completed:
            st.session_state["pipeline_logs"] += "⚠️ Selected races changed since last run. Forcing re-execution of all steps.\n"
            effective_skip_completed = False

        if step_name == "Run Full Pipeline":
            sub_steps = ["Extract Data", "Run ETL", "Run Analytics", "Register Hive Tables", "Train ML Model"]
        else:
            sub_steps = [step_name]

        total_sub_steps = len(sub_steps)
        success = True

        for step_idx, sub_step in enumerate(sub_steps):
            # Calculate dynamic status
            current_status = get_dataset_status()

            # Check for skip
            skip = False
            if effective_skip_completed:
                if sub_step == "Extract Data" and current_status["Raw Data"]:
                    skip = True
                elif sub_step == "Run ETL" and current_status["Engineered Data"]:
                    skip = True
                elif sub_step == "Run Analytics" and current_status["Analytics Warehouse"]:
                    skip = True
                elif sub_step == "Register Hive Tables" and current_status["Analytics Warehouse"]:
                    skip = True
                elif sub_step == "Train ML Model" and current_status["ML Model"]:
                    skip = True

            # Progress bar update
            progress_bar.progress(step_idx / total_sub_steps)

            if skip:
                skip_msg = f"⏭️ {sub_step} - Skipped (Outputs already exist)\n"
                st.session_state["pipeline_logs"] += skip_msg
                log_area.code(st.session_state["pipeline_logs"][-30:], language="text")
                status_text.text(f"Step {step_idx+1}/{total_sub_steps}: {sub_step} (Skipped)")
                time.sleep(0.5)  # small pause for visual feedback
            else:
                run_msg = f"🤖 {sub_step} - Running...\n"
                st.session_state["pipeline_logs"] += run_msg
                log_area.code(st.session_state["pipeline_logs"][-30:], language="text")
                status_text.text(f"Step {step_idx+1}/{total_sub_steps}: Running {sub_step}...")

                # Run all modules for this sub-step
                sub_success = True
                modules = PIPELINE_CONFIG[sub_step]["modules"]
                sub_start = time.time()
                for mod_idx, module in enumerate(modules):
                    cmd = ["python", "-m", module]
                    return_code, cmd_logs = run_command(cmd, project_root, log_area)
                    if return_code != 0:
                        sub_success = False
                        break

                if not sub_success:
                    success = False
                    st.session_state["pipeline_logs"] += f"❌ {sub_step} - Failed\n"
                    log_area.code(st.session_state["pipeline_logs"][-30:], language="text")
                    break
                else:
                    sub_duration = time.time() - sub_start
                    st.session_state["pipeline_durations"][sub_step] = sub_duration
                    st.session_state["pipeline_logs"] += f"✅ {sub_step} - Completed in {sub_duration:.1f}s\n"
                    log_area.code(st.session_state["pipeline_logs"][-30:], language="text")

        # Finished pipeline execution
        duration_sec = time.time() - start_time
        duration_str = f"{duration_sec:.1f}s" if duration_sec < 60 else f"{int(duration_sec // 60)}m {int(duration_sec % 60)}s"

        st.session_state["pipeline_running"] = False
        st.session_state["pipeline_status"] = "Ready" if success else "Failed"

        # Save config if successful
        if success:
            try:
                import json
                last_config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(last_config_path, "w", encoding="utf-8") as f:
                    json.dump(current_races, f, indent=4)
            except Exception as exc:
                st.session_state["pipeline_logs"] += f"⚠️ Failed to save last pipeline config: {exc}\n"

        # Append history
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state["pipeline_history"].append({
            "step": step_name,
            "status": "Success" if success else "Failed",
            "duration": duration_str,
            "timestamp": timestamp_str
        })

        if success:
            st.sidebar.success(f"Success: {step_name} completed in {duration_str}")
        else:
            st.sidebar.error(f"Failed: {step_name} encountered an error.")

        # Remove active running state
        del st.session_state["running_step_name"]
        st.rerun()

    # Render Logs Download button
    if st.session_state.get("pipeline_logs"):
        st.sidebar.download_button(
            label="📥 Download Logs",
            data=st.session_state["pipeline_logs"],
            file_name="pipeline_logs.txt",
            mime="text/plain",
            help="Download stdout/stderr logs of the last run."
        )

    # Render History expander
    if st.session_state["pipeline_history"]:
        history_expander = st.sidebar.expander("Show History", expanded=False)
        with history_expander:
            for item in reversed(st.session_state["pipeline_history"]):
                status_color = "🟢" if item["status"] == "Success" else "🔴"
                st.markdown(
                    f"{status_color} **{item['step']}** - {item['status']}\n"
                    f"⏱️ {item['duration']} | 📅 {item['timestamp']}"
                )

    # Render Pipeline Statistics and Runtime trend
    history = st.session_state["pipeline_history"]
    if history:
        st.sidebar.markdown("---")
        with st.sidebar.expander("📊 Pipeline Statistics", expanded=False):
            total_runs = len(history)
            success_runs = sum(1 for item in history if item["status"] == "Success")
            failed_runs = total_runs - success_runs
            success_rate = (success_runs / total_runs) * 100

            st.write(f"**Total Runs**: {total_runs}")
            st.write(f"**Success**: {success_runs} | **Failed**: {failed_runs}")
            st.write(f"**Success Rate**: {success_rate:.1f}%")

            # Last updated
            last_success = [item for item in history if item["status"] == "Success"]
            if last_success:
                st.write(f"**Last Success**: {last_success[-1]['timestamp']}")

            durations = st.session_state.get("pipeline_durations", {})
            if durations:
                st.markdown("### Runtime Trend")
                for step, dur in durations.items():
                    dur_str = f"{dur:.1f}s" if dur < 60 else f"{int(dur // 60)}m {int(dur % 60)}s"
                    st.caption(f"- **{step}**: {dur_str}")

                # Create a small native bar chart
                dur_df = pd.DataFrame([
                    {"Step": k, "Duration (s)": v}
                    for k, v in durations.items()
                ])
                st.bar_chart(dur_df.set_index("Step"))

    # Render Dataset Status
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Dataset Status")
    for label, available in status.items():
        icon = "✅ Available" if available else "❌ Missing"
        st.sidebar.markdown(f"**{label}**: {icon}")


# ------------------------------------------------------------------
# Main Orchestration
# ------------------------------------------------------------------

def main() -> None:
    """Orchestrate multi-page visualization app."""
    render_sidebar_title()
    st.sidebar.markdown("Production-Style Strategy & Performance Platform")

    # Check if data files exist before loading to avoid noisy st.error alerts
    path_eng = DATA_PROCESSED / "engineered_races.parquet"
    path_driver = DATA_WAREHOUSE / "driver_analytics"
    path_tire = DATA_WAREHOUSE / "tire_analytics"
    path_pit = DATA_WAREHOUSE / "pitstop_analytics"
    path_race = DATA_WAREHOUSE / "race_analytics"

    data_missing = (
        not path_eng.exists()
        or not path_driver.exists()
        or not path_tire.exists()
        or not path_pit.exists()
        or not path_race.exists()
    )

    # Render page routing navigation if data is ready
    if not data_missing:
        page = st.sidebar.radio(
            "Navigate Platform",
            [
                "Race Overview",
                "Driver Analysis",
                "Tire Analysis",
                "Pit Stop Analysis",
                "ML Predictions",
            ],
        )
    else:
        st.sidebar.info("Dashboard navigation will appear after ingesting telemetry data.")
        st.warning("**Platform Data Status**: Datasets are currently missing or incomplete. Use the **Pipeline Control** center in the sidebar to extract telemetry, run the ETL pipeline, compute Spark analytics, register Hive tables, and train the model.")

    # Render ingestion settings in the sidebar
    render_ingestion_settings()

    # Render pipeline controls in the sidebar
    render_pipeline_controls()

    if data_missing:
        # Footer in sidebar
        st.sidebar.markdown("---")
        st.sidebar.caption("Built with Python • PySpark • Hive • Streamlit")
        return

    # Load datasets (using reactive file mtimes to bypass Streamlit stale caching)
    laps_df = load_engineered_laps(get_mtime(path_eng))
    driver_df = load_driver_analytics(get_mtime(path_driver))
    tire_df = load_tire_analytics(get_mtime(path_tire))
    pit_df = load_pitstop_analytics(get_mtime(path_pit))
    race_df = load_race_analytics(get_mtime(path_race))

    if laps_df.empty or driver_df.empty or tire_df.empty or pit_df.empty or race_df.empty:
        st.warning("Data loading returned empty datasets. Please ensure all pipeline runs completed successfully.")
        return

    # Page execution
    if page == "Race Overview":
        render_race_overview(laps_df, driver_df, race_df)
    elif page == "Driver Analysis":
        render_driver_analysis(laps_df, driver_df)
    elif page == "Tire Analysis":
        render_tire_analysis(laps_df, tire_df)
    elif page == "Pit Stop Analysis":
        render_pitstop_analysis(laps_df, pit_df)
    elif page == "ML Predictions":
        render_ml_predictions(laps_df)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Python • PySpark • Hive • Streamlit")


if __name__ == "__main__":
    main()
