"""
preprocess.py
─────────────
Cleans and validates the hourly ad metrics dataset, handles missing timestamps,
removes outlier noise before anomaly detection, and outputs a clean series
ready for ARIMA modelling.

Usage:
    python src/preprocess.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

INPUT_FILE = Path(cfg["data"]["processed_file"])
PROCESSED_DIR = Path(cfg["data"]["processed_dir"])
CLEAN_FILE = PROCESSED_DIR / "ad_metrics_clean.csv"

METRICS = cfg["metrics"]


def load_data() -> pd.DataFrame:
    print(f"[preprocess] Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"[preprocess] Shape: {df.shape}")
    return df


def fill_missing_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure continuous hourly time index — fill gaps with forward-fill then
    interpolate so ARIMA sees a gapless series.
    """
    print("[preprocess] Checking for missing timestamps...")
    full_range = pd.date_range(
        start=df["datetime"].min(),
        end=df["datetime"].max(),
        freq="H"
    )
    df = df.set_index("datetime").reindex(full_range)
    df.index.name = "datetime"

    gaps = df[METRICS].isna().sum()
    if gaps.any().any() if isinstance(gaps.any(), pd.Series) else gaps.any():
        print(f"[preprocess]  → Gaps found:\n{gaps[gaps > 0]}")
        df[METRICS] = df[METRICS].interpolate(method="time")
        print("[preprocess]  → Interpolated.")
    else:
        print("[preprocess]  → No gaps found.")

    # Forward/backward fill segment columns
    seg_cols = [c for c in df.columns if c not in METRICS]
    df[seg_cols] = df[seg_cols].fillna(method="ffill").fillna(method="bfill")

    df = df.reset_index()
    return df


def validate_metric_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clamp metrics to physically valid ranges.
    fill_rate and ctr must be [0,1]; impressions and ecpm must be > 0.
    """
    print("[preprocess] Validating metric ranges...")
    df["fill_rate"] = df["fill_rate"].clip(0.0, 1.0)
    df["ctr"] = df["ctr"].clip(0.0, 1.0)
    df["ecpm"] = df["ecpm"].clip(lower=0.01)
    df["impressions"] = df["impressions"].clip(lower=0)
    df["arpdau"] = df["arpdau"].clip(lower=0.0)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features useful for contextualising anomalies in reports."""
    df["hour_of_day"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["date"] = df["datetime"].dt.date
    return df


def compute_rolling_stats(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """
    Pre-compute rolling mean and std for each metric.
    Window = 24h (1 day) by default.
    These are used in reporting to give context around flagged anomalies.
    """
    print(f"[preprocess] Computing {window}h rolling stats...")
    for m in METRICS:
        df[f"{m}_roll_mean"] = df[m].rolling(window=window, min_periods=1).mean()
        df[f"{m}_roll_std"] = df[m].rolling(window=window, min_periods=1).std().fillna(0.001)
    return df


def summary_stats(df: pd.DataFrame):
    print("\n[preprocess] ── Metric Summary ──────────────────────────")
    print(df[METRICS].describe().round(4).to_string())
    print("─────────────────────────────────────────────────────────\n")


def main():
    df = load_data()
    df = fill_missing_timestamps(df)
    df = validate_metric_ranges(df)
    df = add_time_features(df)
    df = compute_rolling_stats(df)
    summary_stats(df)

    df.to_csv(CLEAN_FILE, index=False)
    print(f"[preprocess] ✓ Clean data saved to {CLEAN_FILE}")
    print(f"[preprocess]   Shape: {df.shape}")
    print("\n[preprocess] Done. Run next: python src/arima_detector.py")


if __name__ == "__main__":
    main()
