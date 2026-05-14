"""
fetch_data.py
─────────────
Downloads the Avazu CTR dataset from Kaggle and engineers synthetic
ad-monetization metrics (eCPM, fill_rate, ARPDAU) on top of it.

Usage:
    python src/fetch_data.py

Requirements:
    - kaggle CLI configured: place ~/.kaggle/kaggle.json with your API key
      (get it from https://www.kaggle.com/settings → API → Create New Token)
"""

import os
import gzip
import shutil
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

import yaml

# ── Config ────────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

RAW_DIR = Path(cfg["data"]["raw_dir"])
PROCESSED_DIR = Path(cfg["data"]["processed_dir"])
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_FILE = Path(cfg["data"]["processed_file"])
SAMPLE_ROWS = 2_000_000   # use 2M rows from the 40M train set — enough for time series


# ── Step 1: Download via Kaggle CLI ───────────────────────────────────────────
def download_dataset():
    out_path = RAW_DIR / "train.gz"
    if out_path.exists():
        print(f"[fetch] Raw file already exists at {out_path}, skipping download.")
        return out_path

    print("[fetch] Downloading Avazu CTR dataset from Kaggle...")
    print("        Make sure ~/.kaggle/kaggle.json is in place.\n")

    cmd = [
        "kaggle", "competitions", "download",
        "-c", "avazu-ctr-prediction",
        "-f", "train.gz",
        "-p", str(RAW_DIR),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] Kaggle download failed:")
        print(result.stderr)
        print("\nManual download instructions:")
        print("  1. Go to: https://www.kaggle.com/c/avazu-ctr-prediction/data")
        print("  2. Download 'train.gz'")
        print(f"  3. Place it in: {RAW_DIR}/")
        raise SystemExit(1)

    print(f"[fetch] Downloaded to {out_path}")
    return out_path


# ── Step 2: Load + sample ─────────────────────────────────────────────────────
def load_raw(gz_path: Path, nrows: int = SAMPLE_ROWS) -> pd.DataFrame:
    print(f"[fetch] Reading {nrows:,} rows from {gz_path} ...")
    df = pd.read_csv(gz_path, nrows=nrows)
    print(f"[fetch] Loaded shape: {df.shape}")
    return df


# ── Step 3: Parse timestamp ───────────────────────────────────────────────────
def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    # Avazu 'hour' column format: YYMMDDHH  e.g. 14101523 = 2014-10-15 23:00
    df["hour_str"] = df["hour"].astype(str)
    df["datetime"] = pd.to_datetime(df["hour_str"], format="%y%m%d%H")
    df["date"] = df["datetime"].dt.date
    df["hour_of_day"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    return df


# ── Step 4: Aggregate to hourly ad metrics ────────────────────────────────────
def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    print("[fetch] Aggregating to hourly ad metrics...")

    agg = df.groupby("datetime").agg(
        clicks=("click", "sum"),
        requests=("click", "count"),
    ).reset_index()

    # Derived metrics
    agg["impressions"] = agg["requests"]   # treat each row as an ad request/impression
    agg["ctr"] = agg["clicks"] / agg["impressions"].clip(lower=1)

    # Synthetic eCPM: base CPM floor + CTR lift + noise
    # Real eCPM = (clicks * CPC * 1000) / impressions — we simulate CPC ~$0.30
    np.random.seed(42)
    base_cpm = 2.5   # $2.50 floor eCPM
    cpc = 0.30
    noise = np.random.normal(0, 0.15, len(agg))
    agg["ecpm"] = (base_cpm + agg["ctr"] * cpc * 1000 + noise).clip(lower=0.1)

    # Synthetic fill rate: 85–98% baseline with some drops
    fill_noise = np.random.normal(0, 0.025, len(agg))
    agg["fill_rate"] = (0.92 + fill_noise).clip(0.0, 1.0)

    # Synthetic ARPDAU: eCPM × impressions / estimated DAU
    estimated_dau = agg["impressions"] / 8   # ~8 ad impressions per user per hour
    agg["arpdau"] = (agg["ecpm"] * agg["impressions"]) / (1000 * estimated_dau.clip(lower=1))

    # Inject realistic anomalies for detection to find
    agg = inject_anomalies(agg)

    agg = agg.sort_values("datetime").reset_index(drop=True)
    return agg


# ── Step 5: Inject synthetic anomalies ────────────────────────────────────────
def inject_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject known anomaly windows so the detector has something to catch.
    In a real system these would be actual revenue events.
    """
    print("[fetch] Injecting synthetic anomaly windows...")
    df = df.copy()
    n = len(df)

    # Anomaly 1: eCPM crash (demand-side issue) — 6 hour window
    idx1 = int(n * 0.25)
    df.loc[idx1:idx1+5, "ecpm"] *= 0.35

    # Anomaly 2: Fill rate drop (supply issue) — 4 hour window
    idx2 = int(n * 0.50)
    df.loc[idx2:idx2+3, "fill_rate"] *= 0.40

    # Anomaly 3: Impression spike (bot traffic / misconfig) — 3 hour window
    idx3 = int(n * 0.72)
    df.loc[idx3:idx3+2, "impressions"] *= 4.5

    # Anomaly 4: CTR spike (click fraud) — 2 hour window
    idx4 = int(n * 0.85)
    df.loc[idx4:idx4+1, "ctr"] *= 3.2
    df.loc[idx4:idx4+1, "ecpm"] *= 2.8

    print(f"[fetch]  → Injected 4 anomaly windows at indices {idx1}, {idx2}, {idx3}, {idx4}")
    return df


# ── Step 6: Add segment columns ───────────────────────────────────────────────
def add_segment_columns(raw_df: pd.DataFrame, hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge dominant segment labels (geo proxy, format proxy) back onto hourly data.
    We take the modal value per hour from the raw data.
    """
    print("[fetch] Adding segment columns (geo, format, demand)...")

    seg = raw_df.groupby("datetime").agg(
        geo_proxy=("site_domain", lambda x: x.mode()[0] if len(x) > 0 else "unknown"),
        format_proxy=("banner_pos", lambda x: x.mode()[0] if len(x) > 0 else 0),
        demand_proxy=("app_category", lambda x: x.mode()[0] if len(x) > 0 else "unknown"),
    ).reset_index()

    hourly_df = hourly_df.merge(seg, on="datetime", how="left")
    return hourly_df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    gz_path = download_dataset()
    raw_df = load_raw(gz_path)
    raw_df = parse_timestamp(raw_df)

    hourly_df = aggregate_hourly(raw_df)
    hourly_df = add_segment_columns(raw_df, hourly_df)

    # Save processed file
    hourly_df.to_csv(PROCESSED_FILE, index=False)
    print(f"\n[fetch] ✓ Processed data saved to {PROCESSED_FILE}")
    print(f"[fetch]   Shape: {hourly_df.shape}")
    print(f"[fetch]   Columns: {list(hourly_df.columns)}")
    print(f"[fetch]   Date range: {hourly_df['datetime'].min()} → {hourly_df['datetime'].max()}")
    print("\n[fetch] Done. Run next: python src/preprocess.py")


if __name__ == "__main__":
    main()
