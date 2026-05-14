"""
segment_analysis.py
───────────────────
Breaks down detected anomalies by:
  - Geo proxy (site_domain)
  - Ad format proxy (banner_pos)
  - Demand source proxy (app_category)
  - Hour of day / day of week patterns

Outputs:
  - outputs/reports/segment_summary.csv
  - outputs/reports/segment_heatmap.png

Usage:
    python src/segment_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
import yaml
import warnings

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

ANOMALY_LOG = Path(cfg["reporting"]["output_dir"]) / "anomaly_log.csv"
OUTPUT_DIR = Path(cfg["reporting"]["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = cfg["metrics"]

sns.set_theme(style="whitegrid", palette="muted")
PALETTE = {"HIGH": "#d62728", "MEDIUM": "#ff7f0e", "LOW": "#1f77b4"}


def load_anomalies() -> pd.DataFrame:
    df = pd.read_csv(ANOMALY_LOG, parse_dates=["datetime"])
    df["hour_of_day"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.day_name()
    df["date"] = df["datetime"].dt.date
    return df


# ── 1. Top anomalous segments ─────────────────────────────────────────────────

def top_segments(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["metric", "geo_proxy", "format_proxy", "demand_proxy"])
        .agg(
            anomaly_count=("datetime", "count"),
            mean_zscore=("z_score", lambda x: x.abs().mean()),
            high_severity=("severity", lambda x: (x == "HIGH").sum()),
        )
        .reset_index()
        .sort_values(["anomaly_count", "mean_zscore"], ascending=False)
    )
    return summary


# ── 2. Hourly anomaly heatmap ─────────────────────────────────────────────────

def plot_hourly_heatmap(df: pd.DataFrame):
    pivot = (
        df.groupby(["day_of_week", "hour_of_day"])
        .size()
        .reset_index(name="anomaly_count")
    )
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot["day_of_week"] = pd.Categorical(pivot["day_of_week"], categories=day_order, ordered=True)
    pivot = pivot.sort_values("day_of_week")
    heatmap_data = pivot.pivot(index="day_of_week", columns="hour_of_day", values="anomaly_count").fillna(0)

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(
        heatmap_data, ax=ax, cmap="YlOrRd",
        linewidths=0.3, linecolor="#dddddd",
        annot=True, fmt=".0f", annot_kws={"size": 7},
        cbar_kws={"label": "Anomaly Count"},
    )
    ax.set_title("Anomaly Frequency by Day of Week & Hour", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Hour of Day (UTC)", fontsize=10)
    ax.set_ylabel("")
    plt.tight_layout()
    path = OUTPUT_DIR / "segment_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[segment]  → Saved hourly heatmap to {path}")


# ── 3. Metric-level anomaly bar chart ────────────────────────────────────────

def plot_metric_bar(df: pd.DataFrame):
    sev_order = ["LOW", "MEDIUM", "HIGH"]
    counts = (
        df.groupby(["metric", "severity"])
        .size()
        .reset_index(name="count")
    )
    counts["severity"] = pd.Categorical(counts["severity"], categories=sev_order, ordered=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    for sev, color in PALETTE.items():
        sub = counts[counts["severity"] == sev]
        ax.bar(sub["metric"], sub["count"], label=sev, color=color, alpha=0.85)

    ax.set_title("Anomaly Count by Metric and Severity", fontsize=13, fontweight="bold")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Anomaly Count")
    ax.legend(title="Severity")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    path = OUTPUT_DIR / "metric_severity_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[segment]  → Saved metric severity bar chart to {path}")


# ── 4. Geo / format breakdown ─────────────────────────────────────────────────

def plot_geo_breakdown(df: pd.DataFrame):
    top_geo = (
        df.groupby("geo_proxy")
        .agg(count=("datetime", "count"), mean_z=("z_score", lambda x: x.abs().mean()))
        .nlargest(10, "count")
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Geo
    axes[0].barh(top_geo["geo_proxy"], top_geo["count"], color="#4c72b0", alpha=0.85)
    axes[0].set_title("Top 10 Anomalous Geo Proxies", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Anomaly Count")
    axes[0].invert_yaxis()

    # Format proxy
    fmt = (
        df.groupby("format_proxy")
        .agg(count=("datetime", "count"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    fmt["format_proxy"] = "Banner Pos " + fmt["format_proxy"].astype(str)
    axes[1].bar(fmt["format_proxy"], fmt["count"], color="#dd8452", alpha=0.85)
    axes[1].set_title("Anomalies by Ad Format (Banner Position)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Ad Format Proxy")
    axes[1].set_ylabel("Anomaly Count")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = OUTPUT_DIR / "geo_format_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[segment]  → Saved geo/format breakdown to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("[segment] Loading anomaly log...")
    df = load_anomalies()
    print(f"[segment] {len(df)} total anomaly records across {df['metric'].nunique()} metrics.")

    seg_summary = top_segments(df)
    seg_path = OUTPUT_DIR / "segment_summary.csv"
    seg_summary.to_csv(seg_path, index=False)
    print(f"[segment] ✓ Segment summary saved to {seg_path}")

    print("[segment] Generating visualisations...")
    plot_hourly_heatmap(df)
    plot_metric_bar(df)
    plot_geo_breakdown(df)

    print("\n[segment] ── Top Anomalous Segments ───────────────────────")
    print(seg_summary.head(10).to_string(index=False))
    print("\n[segment] Done. Run next: python src/report_generator.py")


if __name__ == "__main__":
    main()
