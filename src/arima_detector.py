"""
arima_detector.py
─────────────────
Core anomaly detection engine.

For each metric in [ecpm, fill_rate, ctr, impressions, arpdau]:
  1. Fit ARIMA(p,d,q) on a rolling training window
  2. Forecast next step with confidence interval
  3. Flag points outside CI as ARIMA anomalies
  4. Additionally flag points with |z-score| > threshold
  5. Output a combined anomaly log CSV

Usage:
    python src/arima_detector.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
import warnings
import yaml

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

CLEAN_FILE = Path(cfg["data"]["processed_dir"]) / "ad_metrics_clean.csv"
OUTPUT_DIR = Path(cfg["reporting"]["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = cfg["metrics"]
ARIMA_ORDER = tuple(cfg["anomaly_detection"]["arima_order"])
TRAIN_WINDOW = cfg["anomaly_detection"]["train_window_days"] * 24   # hours
FORECAST_H = cfg["anomaly_detection"]["forecast_horizon_hours"]
Z_THRESH = cfg["anomaly_detection"]["zscore_threshold"]
CI = cfg["anomaly_detection"]["confidence_interval"]
ROLL_WINDOW = cfg["anomaly_detection"]["rolling_window_days"] * 24

SEVERITY = cfg["reporting"]["severity_levels"]


# ── Z-Score Anomaly Detection ─────────────────────────────────────────────────

def zscore_flag(series: pd.Series, window: int = ROLL_WINDOW, threshold: float = Z_THRESH) -> pd.Series:
    """
    Rolling z-score anomaly flag.
    Returns a Series of z-scores; |z| > threshold → anomaly.
    """
    roll_mean = series.rolling(window=window, min_periods=1).mean()
    roll_std = series.rolling(window=window, min_periods=1).std().fillna(0.001)
    z = (series - roll_mean) / roll_std
    return z


# ── ARIMA One-Step Forecast ───────────────────────────────────────────────────

def arima_forecast_flag(series: pd.Series, train_window: int = TRAIN_WINDOW) -> pd.DataFrame:
    """
    Walk-forward ARIMA: for each point after the initial training window,
    fit ARIMA on the previous `train_window` observations and forecast 1 step.
    Flag if actual value falls outside the forecast confidence interval.

    Returns DataFrame with columns:
        forecast, lower_ci, upper_ci, arima_anomaly
    """
    n = len(series)
    forecasts = [np.nan] * n
    lower_ci = [np.nan] * n
    upper_ci = [np.nan] * n

    print(f"    Running walk-forward ARIMA on {n} points (train_window={train_window}h) ...")

    step = max(1, n // 20)   # progress every 5%
    for i in range(train_window, n):
        if i % step == 0:
            pct = (i - train_window) / (n - train_window) * 100
            print(f"      {pct:.0f}% ...", end="\r")

        train = series.iloc[i - train_window: i].values
        try:
            model = ARIMA(train, order=ARIMA_ORDER)
            fit = model.fit()
            fc = fit.get_forecast(steps=1)
            forecasts[i] = fc.predicted_mean[0]
            ci_df = fc.conf_int(alpha=1 - CI)
            lower_ci[i] = ci_df.iloc[0, 0]
            upper_ci[i] = ci_df.iloc[0, 1]
        except Exception:
            # If ARIMA fails (e.g. not enough data variation), fall back to last value
            forecasts[i] = series.iloc[i - 1]
            std_est = series.iloc[i - train_window: i].std()
            lower_ci[i] = forecasts[i] - 2 * std_est
            upper_ci[i] = forecasts[i] + 2 * std_est

    print(f"      100% ✓              ")

    result = pd.DataFrame({
        "forecast": forecasts,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
    })
    result["arima_anomaly"] = (
        (series.values < result["lower_ci"]) |
        (series.values > result["upper_ci"])
    ).astype(int)
    result["arima_anomaly"] = result["arima_anomaly"].fillna(0).astype(int)

    return result


# ── Severity Classification ───────────────────────────────────────────────────

def classify_severity(z: float) -> str:
    az = abs(z)
    if az >= SEVERITY["high"]:
        return "HIGH"
    elif az >= SEVERITY["medium"]:
        return "MEDIUM"
    elif az >= SEVERITY["low"]:
        return "LOW"
    return "NONE"


# ── Main Detection Loop ───────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    anomaly_records = []

    for metric in METRICS:
        print(f"\n[detector] ── Processing metric: {metric} ──────────────")
        series = df[metric].copy()

        # Z-score
        z_scores = zscore_flag(series)

        # ARIMA
        arima_results = arima_forecast_flag(series)

        # Combine
        is_anomaly = (z_scores.abs() > Z_THRESH) | (arima_results["arima_anomaly"] == 1)
        anomaly_idx = df.index[is_anomaly]

        for idx in anomaly_idx:
            z = z_scores.iloc[idx]
            severity = classify_severity(z)
            anomaly_records.append({
                "datetime": df["datetime"].iloc[idx],
                "metric": metric,
                "observed_value": round(series.iloc[idx], 5),
                "forecast_value": round(arima_results["forecast"].iloc[idx], 5) if not np.isnan(arima_results["forecast"].iloc[idx]) else None,
                "lower_ci": round(arima_results["lower_ci"].iloc[idx], 5) if not np.isnan(arima_results["lower_ci"].iloc[idx]) else None,
                "upper_ci": round(arima_results["upper_ci"].iloc[idx], 5) if not np.isnan(arima_results["upper_ci"].iloc[idx]) else None,
                "z_score": round(z, 4),
                "arima_flag": int(arima_results["arima_anomaly"].iloc[idx]),
                "zscore_flag": int(abs(z) > Z_THRESH),
                "severity": severity,
                "geo_proxy": df["geo_proxy"].iloc[idx] if "geo_proxy" in df.columns else "unknown",
                "format_proxy": df["format_proxy"].iloc[idx] if "format_proxy" in df.columns else "unknown",
                "demand_proxy": df["demand_proxy"].iloc[idx] if "demand_proxy" in df.columns else "unknown",
                "is_weekend": df["is_weekend"].iloc[idx] if "is_weekend" in df.columns else 0,
            })

        n_flagged = is_anomaly.sum()
        print(f"[detector]   → Flagged {n_flagged} anomalies ({n_flagged/len(df)*100:.1f}% of timepoints)")

        # Store per-metric ARIMA columns for report generator
        df[f"{metric}_forecast"] = arima_results["forecast"].values
        df[f"{metric}_lower_ci"] = arima_results["lower_ci"].values
        df[f"{metric}_upper_ci"] = arima_results["upper_ci"].values
        df[f"{metric}_zscore"] = z_scores.values
        df[f"{metric}_anomaly"] = is_anomaly.astype(int).values

    return df, pd.DataFrame(anomaly_records)


def main():
    print("[detector] Loading clean data...")
    df = pd.read_csv(CLEAN_FILE, parse_dates=["datetime"])
    print(f"[detector] Shape: {df.shape}, Date range: {df['datetime'].min()} → {df['datetime'].max()}")

    df, anomaly_log = detect_anomalies(df)

    # Save enriched df (with ARIMA outputs per metric)
    enriched_path = Path(cfg["data"]["processed_dir"]) / "ad_metrics_enriched.csv"
    df.to_csv(enriched_path, index=False)
    print(f"\n[detector] ✓ Enriched data saved to {enriched_path}")

    # Save anomaly log
    anomaly_log = anomaly_log.sort_values(["datetime", "metric"]).reset_index(drop=True)
    log_path = OUTPUT_DIR / "anomaly_log.csv"
    anomaly_log.to_csv(log_path, index=False)
    print(f"[detector] ✓ Anomaly log saved to {log_path}")
    print(f"[detector]   Total anomalies detected: {len(anomaly_log)}")

    sev_counts = anomaly_log["severity"].value_counts()
    print(f"\n[detector] Severity breakdown:\n{sev_counts.to_string()}")
    print("\n[detector] Done. Run next: python src/segment_analysis.py")


if __name__ == "__main__":
    main()
