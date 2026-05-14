# Ad Revenue Anomaly Detector

An end-to-end anomaly detection pipeline for ad monetization metrics — built on top of real ad impression data to flag revenue dips, fill-rate drops, and eCPM anomalies using ARIMA-based forecasting and z-score outlier detection.

---

##  Problem Statement

Ad monetization platforms generate continuous streams of impression, click, fill-rate, and revenue data. Sudden drops in eCPM, fill rate, or ARPDAU can indicate demand-side issues, geo-level underperformance, tracking failures, or product regressions — and they need to be caught fast.

This project builds an automated pipeline that:
- Ingests and preprocesses ad metrics time-series data
- Detects anomalies using ARIMA forecasting + z-score flagging
- Segments anomalies by geo, ad format, and demand source
- Outputs daily annotated reports with visualisations

---

##  Project Structure

```
ad_revenue_anomaly_detector/
│
├── data/
│   ├── raw/                    # Raw downloaded dataset (not committed)
│   └── processed/              # Cleaned, feature-engineered data
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_arima_modeling.ipynb
│   └── 04_anomaly_report.ipynb
│
├── src/
│   ├── fetch_data.py           # Kaggle dataset download + synthetic eCPM generation
│   ├── preprocess.py           # Cleaning, aggregation, feature engineering
│   ├── arima_detector.py       # ARIMA model + z-score anomaly flagging
│   ├── segment_analysis.py     # Geo-level, format-level breakdown
│   └── report_generator.py     # Automated daily report with annotated plots
│
├── outputs/
│   └── reports/                # Generated anomaly reports (PNG + CSV)
│
├── requirements.txt
├── config.yaml
└── README.md
```

---

##  Dataset

**Source:** [Avazu Click-Through Rate Prediction — Kaggle](https://www.kaggle.com/c/avazu-ctr-prediction/data)

This dataset contains 10 days of ad click-through data with fields including:
- `hour` — timestamp (YYMMDDHH format)
- `click` — 0/1 click indicator
- `banner_pos` — ad placement position
- `site_category`, `app_category` — content category
- `device_type`, `device_conn_type` — device signals

On top of this, the pipeline **synthetically derives** ad monetization metrics that mirror real adtech platforms:
- `impressions` — aggregated per hour/day
- `eCPM` — estimated from click rate + CPM floor model
- `fill_rate` — simulated from request volume
- `ARPDAU` — derived from eCPM × impressions / active users
- `CTR` — clicks / impressions

**To fetch data:**
```bash
# Install Kaggle CLI first
pip install kaggle
# Place your kaggle.json in ~/.kaggle/
python src/fetch_data.py
```

---

##  Setup

```bash
git clone https://github.com/YOUR_USERNAME/ad-revenue-anomaly-detector.git
cd ad_revenue_anomaly_detector
pip install -r requirements.txt
```

---

##  Run Pipeline

```bash
# Step 1: Download + prepare data
python src/fetch_data.py

# Step 2: Preprocess + feature engineering
python src/preprocess.py

# Step 3: Run ARIMA anomaly detection
python src/arima_detector.py

# Step 4: Segment analysis by geo / format
python src/segment_analysis.py

# Step 5: Generate daily report
python src/report_generator.py
```

Or run the notebooks in order for an interactive walkthrough.

---

##  Methodology

### ARIMA Forecasting
- Fit ARIMA(p,d,q) model on rolling 7-day training window per metric
- Forecast next 24h of expected values with 95% confidence intervals
- Points outside the CI are candidates for anomaly

### Z-Score Flagging
- Compute rolling mean and std over a 7-day window
- Z-score = (observed - rolling_mean) / rolling_std
- Flag as anomaly if |z| > 2.5 (configurable in `config.yaml`)

### Segmentation
- Anomalies are broken down by:
  - **Geo proxy** (device country / site domain cluster)
  - **Ad format** (banner_pos as proxy for format type)
  - **Demand source** (app vs site category)

---

##  Output Example

Each run generates:
- `outputs/reports/anomaly_report_YYYY-MM-DD.png` — annotated time series with flagged anomalies
- `outputs/reports/anomaly_summary_YYYY-MM-DD.csv` — tabular anomaly log with metric, timestamp, z-score, severity

---

##  Tech Stack

| Layer | Tools |
|---|---|
| Data ingestion | `kaggle`, `pandas` |
| Preprocessing | `pandas`, `numpy` |
| Modelling | `statsmodels` (ARIMA), `scipy` |
| Visualisation | `matplotlib`, `seaborn` |
| Reporting | `matplotlib` (annotated PNG), `pandas` (CSV) |

---

##  Key Metrics Monitored

| Metric | Description |
|---|---|
| **eCPM** | Revenue per 1000 impressions |
| **Fill Rate** | % of ad requests successfully filled |
| **CTR** | Click-through rate |
| **ARPDAU** | Avg revenue per daily active user |
| **Impressions** | Total ad impressions per time unit |

---

##  Interview Notes / Findings

- ARIMA(2,1,2) was found to be optimal for eCPM time series via AIC minimisation
- Z-score threshold of 2.5 gave best precision-recall tradeoff on held-out anomaly windows
- Fill rate drops often preceded eCPM drops by 2–4 hours — useful as an early warning signal
- Banner position 0 (interstitial) showed highest anomaly frequency on weekends

---

##  License

MIT
