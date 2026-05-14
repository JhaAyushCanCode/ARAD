"""
report_generator.py
───────────────────
Generates the daily annotated anomaly report:
  - One subplot per metric showing:
      • Actual observed values
      • ARIMA forecast + confidence band
      • Flagged anomaly points (coloured by severity)
      • Rolling mean trend line
  - Saves PNG report and CSV summary

Usage:
    python src/report_generator.py [--date YYYY-MM-DD]
    (defaults to the last date in the dataset)
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import warnings

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

ENRICHED_FILE = Path(cfg["data"]["processed_dir"]) / "ad_metrics_enriched.csv"
ANOMALY_LOG = Path(cfg["reporting"]["output_dir"]) / "anomaly_log.csv"
OUTPUT_DIR = Path(cfg["reporting"]["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = cfg["metrics"]

METRIC_LABELS = {
    "ecpm": "eCPM ($)",
    "fill_rate": "Fill Rate",
    "ctr": "CTR",
    "impressions": "Impressions",
    "arpdau": "ARPDAU ($)",
}

SEVERITY_COLORS = {
    "HIGH": "#d62728",
    "MEDIUM": "#ff7f0e",
    "LOW": "#1f77b4",
}


def load_data(report_date: str):
    print(f"[report] Loading enriched data for report date: {report_date} ...")
    df = pd.read_csv(ENRICHED_FILE, parse_dates=["datetime"])
    anomaly_log = pd.read_csv(ANOMALY_LOG, parse_dates=["datetime"])

    # Show last 7 days leading up to report date
    end_dt = pd.Timestamp(report_date) + timedelta(hours=23)
    start_dt = end_dt - timedelta(days=7)
    df = df[(df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)].copy()

    anomaly_log = anomaly_log[
        (anomaly_log["datetime"] >= start_dt) &
        (anomaly_log["datetime"] <= end_dt)
    ].copy()

    print(f"[report]   Window: {start_dt} → {end_dt}")
    print(f"[report]   Data points: {len(df)}, Anomalies in window: {len(anomaly_log)}")
    return df, anomaly_log


def generate_report_figure(df: pd.DataFrame, anomaly_log: pd.DataFrame, report_date: str):
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(16, 4 * n_metrics), sharex=False)
    fig.suptitle(
        f"Ad Revenue Anomaly Detection Report  |  {report_date}",
        fontsize=15, fontweight="bold", y=1.005
    )

    for ax, metric in zip(axes, METRICS):
        label = METRIC_LABELS.get(metric, metric.upper())
        series = df[metric]
        times = df["datetime"]

        # ── 1. Plot actual values ─────────────────────────────────────────────
        ax.plot(times, series, color="#555555", linewidth=1.0, label="Observed", zorder=2)

        # ── 2. Rolling mean ───────────────────────────────────────────────────
        roll_col = f"{metric}_roll_mean"
        if roll_col in df.columns:
            ax.plot(times, df[roll_col], color="#aaaaaa", linewidth=0.8,
                    linestyle="--", label="Rolling Mean", zorder=2)

        # ── 3. ARIMA forecast + CI band ───────────────────────────────────────
        fc_col = f"{metric}_forecast"
        lo_col = f"{metric}_lower_ci"
        hi_col = f"{metric}_upper_ci"
        if fc_col in df.columns:
            mask = df[fc_col].notna()
            ax.plot(times[mask], df.loc[mask, fc_col],
                    color="#2196F3", linewidth=1.0, linestyle=":", label="ARIMA Forecast", zorder=3)
            ax.fill_between(
                times[mask],
                df.loc[mask, lo_col],
                df.loc[mask, hi_col],
                color="#2196F3", alpha=0.10, label="95% CI"
            )

        # ── 4. Anomaly scatter ────────────────────────────────────────────────
        metric_anomalies = anomaly_log[anomaly_log["metric"] == metric]
        for sev, color in SEVERITY_COLORS.items():
            sub = metric_anomalies[metric_anomalies["severity"] == sev]
            if not sub.empty:
                # Match anomaly timestamps back to series
                anom_vals = pd.merge(
                    sub[["datetime"]], df[["datetime", metric]],
                    on="datetime", how="left"
                )
                ax.scatter(
                    anom_vals["datetime"], anom_vals[metric],
                    color=color, s=50, zorder=5, label=f"{sev} Anomaly",
                    edgecolors="white", linewidths=0.5
                )
                # Vertical dashed line for each anomaly
                for dt in anom_vals["datetime"]:
                    ax.axvline(dt, color=color, alpha=0.15, linewidth=0.8, zorder=1)

        # ── 5. Format axis ────────────────────────────────────────────────────
        ax.set_ylabel(label, fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %Hh"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.setp(ax.get_xticklabels(), rotation=30, fontsize=7)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlim(times.min(), times.max())
        ax.grid(axis="y", alpha=0.3)
        ax.grid(axis="x", alpha=0.15)

        n_anom = len(metric_anomalies)
        ax.set_title(
            f"{label}  —  {n_anom} anomalies flagged",
            fontsize=10, loc="left", fontweight="bold"
        )

        # Compact legend
        handles, labels_leg = ax.get_legend_handles_labels()
        seen = {}
        unique = [(h, l) for h, l in zip(handles, labels_leg) if not seen.setdefault(l, False) and not seen.update({l: True})]
        ax.legend(*zip(*unique), fontsize=7, loc="upper right", framealpha=0.7)

    plt.tight_layout()
    return fig


def generate_daily_summary(anomaly_log: pd.DataFrame, report_date: str) -> pd.DataFrame:
    if anomaly_log.empty:
        return pd.DataFrame()

    summary = (
        anomaly_log.groupby(["metric", "severity"])
        .agg(
            count=("datetime", "count"),
            mean_z=("z_score", lambda x: round(x.abs().mean(), 3)),
            max_z=("z_score", lambda x: round(x.abs().max(), 3)),
            first_seen=("datetime", "min"),
            last_seen=("datetime", "max"),
        )
        .reset_index()
        .sort_values(["count", "max_z"], ascending=False)
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None,
                        help="Report date YYYY-MM-DD (defaults to last date in data)")
    args = parser.parse_args()

    # Determine report date
    if args.date:
        report_date = args.date
    else:
        df_tmp = pd.read_csv(ENRICHED_FILE, parse_dates=["datetime"])
        report_date = str(df_tmp["datetime"].max().date())

    print(f"[report] Generating anomaly report for: {report_date}")

    df, anomaly_log = load_data(report_date)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = generate_report_figure(df, anomaly_log, report_date)
    fig_path = OUTPUT_DIR / f"anomaly_report_{report_date}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[report] ✓ Report figure saved to {fig_path}")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary = generate_daily_summary(anomaly_log, report_date)
    csv_path = OUTPUT_DIR / f"anomaly_summary_{report_date}.csv"
    summary.to_csv(csv_path, index=False)
    print(f"[report] ✓ Summary CSV saved to {csv_path}")

    print("\n[report] ── Anomaly Summary ──────────────────────────────")
    if not summary.empty:
        print(summary.to_string(index=False))
    else:
        print("No anomalies in the reporting window.")
    print("\n[report] Pipeline complete ✓")


if __name__ == "__main__":
    main()
