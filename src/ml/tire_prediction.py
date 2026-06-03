"""
Machine learning model training layer for the F1 Telemetry Analytics Platform.

This module fits an XGBoost Regressor model to predict lap_time_seconds based
on tire life, compound, lap number, weather conditions, and driver.
It uses an sklearn Pipeline with a ColumnTransformer (OrdinalEncoder for
categoricals and passthrough for numerics) and exports the trained model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

from src.utils.config import (
    DATA_PROCESSED,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def train_model() -> None:
    """Load engineered races dataset, preprocess features, fit XGBoost, and export."""
    input_path = DATA_PROCESSED / "engineered_races.parquet"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Engineered race dataset not found: {input_path}. Please run ETL first."
        )

    logger.info("Loading engineered race dataset...")
    df = pd.read_parquet(input_path)

    # 1. Feature Engineering & Column Mapping
    # Standardize columns to expected capitalized features for dashboard compatibility
    X = pd.DataFrame({
        "TyreLife": df["tyre_life"],
        "Compound": df["compound"],
        "LapNumber": df["lap_number"],
        "Circuit": df["race_name"],
        "Driver": df["driver"],
        "AirTemp": df["air_temp"],
        "TrackTemp": df["track_temp"],
        "Rainfall": df["rainfall"]
    })
    y = df["lap_time_seconds"]

    # 2. Split dataset into train and test sets
    logger.info("Splitting dataset into train and test sets (holdout: %s)...", TEST_SIZE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 3. Design preprocessing and model pipeline
    categorical_features = ["Compound", "Circuit", "Driver"]
    numerical_features = ["TyreLife", "LapNumber", "AirTemp", "TrackTemp", "Rainfall"]

    # Robust OrdinalEncoder to handle unseen categories at inference time
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat", 
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), 
                categorical_features
            ),
            ("num", "passthrough", numerical_features)
        ]
    )

    # Build the full modeling pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor", 
                XGBRegressor(
                    n_estimators=120,
                    learning_rate=0.08,
                    max_depth=5,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        ]
    )

    # 4. Train model
    logger.info("Fitting XGBoost regressor model pipeline...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate model
    logger.info("Evaluating model predictions on test holdout...")
    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info("--- Model Evaluation Metrics ---")
    logger.info("RMSE (Root Mean Squared Error): %.3f seconds", rmse)
    logger.info("MAE (Mean Absolute Error)     : %.3f seconds", mae)
    logger.info("R² (Coefficient of Determination): %.4f", r2)

    # 6. Save model pkl
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / "tire_degradation_model.pkl"
    logger.info("Exporting trained model pipeline to %s...", output_path)
    
    joblib.dump(pipeline, output_path)
    logger.info("Model pipeline successfully saved.")


if __name__ == "__main__":
    train_model()
