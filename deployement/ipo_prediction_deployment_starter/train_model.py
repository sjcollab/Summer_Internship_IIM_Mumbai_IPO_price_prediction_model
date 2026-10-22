"""
Train and save the IPO one-month price model.

Put your cleaned dataset at data/ipo_data.csv (or set IPO_DATA_PATH).
Expected target: Price_after_1m, stored as actual rupee price / 1000.
The app reverses this final scaling and displays rupees.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("IPO_DATA_PATH", ROOT / "data" / "ipo_data.csv"))
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

TARGET = "Price_after_1m"
DROP_COLUMNS = {
    TARGET, "Price_after_3m", "Price_after_6m",
    # identifiers and post-listing fields are not suitable model inputs
    "Issuer Name", "Issuer Ticker", "Ticker", "yahoo_ticker",
    "Listing Date (DD - MM - YYYY)", "Listing Date",
}

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place your cleaned CSV at data/ipo_data.csv or set IPO_DATA_PATH."
        )

    # Load CSV; if your source is Excel, export your cleaned sheet as CSV.
    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    if TARGET not in df.columns:
        raise ValueError(f"Target '{TARGET}' not found. Available columns: {df.columns.tolist()}")

    # Keep rows with a known target; coerce candidate features to numeric.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    if len(df) < 10:
        raise ValueError(f"Only {len(df)} rows have a valid target; more data is needed.")

    # This project assumes Price_after_1m is actual INR price divided by 1000.
    # Do not divide it by 1000 again here.
    candidate_cols = [c for c in df.columns if c not in DROP_COLUMNS]
    X_raw = df[candidate_cols].apply(pd.to_numeric, errors="coerce")
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)

    # Remove columns that contain no usable numeric values.
    X_raw = X_raw.dropna(axis=1, how="all")
    if X_raw.shape[1] == 0:
        raise ValueError("No numeric feature columns remain after cleaning.")

    y = df[TARGET].astype(float)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )

    # Fit imputing values, feature selection, and scalers on training data only.
    fill_values = X_train_raw.median(numeric_only=True)
    X_train_raw = X_train_raw.fillna(fill_values)
    X_test_raw = X_test_raw.fillna(fill_values)

    train_frame = X_train_raw.copy()
    train_frame[TARGET] = y_train.values
    corrs = train_frame.corr(numeric_only=True)[TARGET].drop(labels=[TARGET], errors="ignore")
    selected = corrs[corrs.abs() >= 0.05].index.tolist()

    # Correlation threshold can remove everything on small/constant datasets.
    if not selected:
        selected = X_train_raw.columns.tolist()
    if not selected:
        raise ValueError("No features available for training.")

    X_train = X_train_raw[selected].to_numpy(dtype=float)
    X_test = X_test_raw[selected].to_numpy(dtype=float)

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.to_numpy().reshape(-1, 1)).ravel()

    model = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.2),
        Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    model.fit(
        X_train_scaled, y_train_scaled,
        validation_split=0.2,
        epochs=500,
        batch_size=8,
        callbacks=[early_stop],
        verbose=1
    )

    pred_scaled = model.predict(X_test_scaled, verbose=0).reshape(-1, 1)
    pred_target_units = y_scaler.inverse_transform(pred_scaled).ravel()
    actual_target_units = y_test.to_numpy()

    # Dataset target units are INR / 1000, so convert both to INR for reporting.
    pred_inr = pred_target_units * 1000.0
    actual_inr = actual_target_units * 1000.0
    metrics = {
        "test_rows": int(len(y_test)),
        "features": selected,
        "mae_inr": float(mean_absolute_error(actual_inr, pred_inr)),
        "rmse_inr": float(np.sqrt(mean_squared_error(actual_inr, pred_inr))),
        "r2": float(r2_score(actual_inr, pred_inr)) if len(y_test) >= 2 else None,
        "target_definition": "Price_after_1m is rupee price divided by 1000"
    }

    model.save(ARTIFACTS / "ipo_model.keras")
    joblib.dump(x_scaler, ARTIFACTS / "x_scaler.joblib")
    joblib.dump(y_scaler, ARTIFACTS / "y_scaler.joblib")
    joblib.dump(fill_values.to_dict(), ARTIFACTS / "fill_values.joblib")
    (ARTIFACTS / "feature_columns.json").write_text(json.dumps(selected, indent=2))
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"actual_price_inr": actual_inr, "predicted_price_inr": pred_inr}).to_csv(
        ARTIFACTS / "test_predictions.csv", index=False
    )

    print("\nTraining complete.")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved artifacts to: {ARTIFACTS}")

if __name__ == "__main__":
    main()
