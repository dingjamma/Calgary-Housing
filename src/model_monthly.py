"""
Monthly XGBoost model — predicts next month's CREB benchmark price change.

Uses CREB monthly data (2025-2026 currently, grows as Wayback scraper adds 2019-2024).
Features: lagged monthly economic indicators + price momentum.
Target: month-over-month % change in Total Residential benchmark price.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)


def build_monthly_dataset() -> pd.DataFrame:
    """Join CREB monthly with economic indicators and engineer lag features."""
    # Load CREB — citywide Total Residential benchmark
    creb = pd.read_csv("data/raw/creb_housing_prices.csv", parse_dates=["date"])
    tr = creb[creb["district"] == "Calgary"].copy()

    # Also load historical if it exists
    import os
    if os.path.exists("data/raw/creb_housing_historical.csv"):
        hist = pd.read_csv("data/raw/creb_housing_historical.csv", parse_dates=["date"])
        hist_tr = hist[hist["district"] == "Calgary"].copy()
        tr = pd.concat([hist_tr, tr], ignore_index=True)
        print(f"  Loaded historical CREB: {hist['date'].nunique()} months")

    # Keep only Total Residential for the benchmark model
    tr = tr[tr["property_type"] == "Total Residential"].sort_values("date").reset_index(drop=True)
    print(f"  Total Residential months available: {len(tr)} ({tr['date'].iloc[0].date()} to {tr['date'].iloc[-1].date()})")

    # Load daily indicators → resample to monthly (last trading day of month)
    daily = pd.read_csv("data/raw/daily_indicators.csv", parse_dates=["date"])
    monthly_econ = daily.set_index("date").resample("MS").last().reset_index()

    # Merge
    df = tr.merge(monthly_econ, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    # --- Feature engineering ---
    # MoM % change in benchmark (target)
    df["benchmark_mom_pct"] = df["benchmark_price"].pct_change() * 100

    # Lag features (use prior month's data to predict this month)
    for lag in [1, 2, 3]:
        df[f"oil_lag{lag}"] = df["oil_price_usd"].shift(lag)
        df[f"rate_lag{lag}"] = df["overnight_rate"].shift(lag)
        df[f"bond_lag{lag}"] = df["ca_5yr_bond_yield"].shift(lag)
        df[f"cadusd_lag{lag}"] = df["cadusd_rate"].shift(lag)
        df[f"natgas_lag{lag}"] = df["natgas_price"].shift(lag)
        df[f"etf_lag{lag}"] = df["alberta_etf"].shift(lag)

    # Rolling windows on oil (key driver)
    df["oil_roll3"] = df["oil_price_usd"].shift(1).rolling(3).mean()
    df["oil_roll6"] = df["oil_price_usd"].shift(1).rolling(6).mean()
    df["oil_momentum"] = df["oil_lag1"] - df["oil_lag3"]   # 3-month oil trend

    # Price momentum
    df["price_lag1"] = df["benchmark_price"].shift(1)
    df["price_mom_lag1"] = df["benchmark_mom_pct"].shift(1)
    df["price_mom_lag2"] = df["benchmark_mom_pct"].shift(2)

    # Month of year (seasonality)
    df["month"] = df["date"].dt.month

    return df


FEATURE_COLS = [
    "oil_lag1", "oil_lag2", "oil_lag3",
    "oil_roll3", "oil_roll6", "oil_momentum",
    "rate_lag1", "rate_lag2",
    "bond_lag1", "bond_lag2",
    "cadusd_lag1", "cadusd_lag2",
    "natgas_lag1",
    "etf_lag1",
    "price_lag1", "price_mom_lag1", "price_mom_lag2",
    "month",
]
TARGET = "benchmark_mom_pct"


def train_monthly_model() -> tuple:
    print("Building monthly model...")
    df = build_monthly_dataset()

    clean = df.dropna(subset=FEATURE_COLS + [TARGET]).copy()
    print(f"  Usable rows after lag engineering: {len(clean)}")

    if len(clean) < 6:
        print("  Not enough data for time-series CV — need more months from Wayback scraper.")
        print("  Training on all available data and predicting next month...")
        model = xgb.XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.1,
                                  reg_alpha=1.0, reg_lambda=2.0, random_state=42, verbosity=0)
        X = clean[FEATURE_COLS].values
        y = clean[TARGET].values
        model.fit(X, y)
        mae, r2 = None, None
    else:
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=min(5, len(clean) - 3))
        preds_all = np.zeros(len(clean))
        actual_all = clean[TARGET].values

        for train_idx, test_idx in tscv.split(clean):
            X_tr = clean.iloc[train_idx][FEATURE_COLS].values
            y_tr = actual_all[train_idx]
            X_te = clean.iloc[test_idx][FEATURE_COLS].values

            m = xgb.XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.1,
                                  reg_alpha=1.0, reg_lambda=2.0, random_state=42, verbosity=0)
            m.fit(X_tr, y_tr)
            preds_all[test_idx] = m.predict(X_te)

        # Only evaluate test folds (not the initial train-only rows)
        first_test = list(tscv.split(clean))[0][1][0]
        eval_mask = np.zeros(len(clean), dtype=bool)
        eval_mask[first_test:] = True
        mae = mean_absolute_error(actual_all[eval_mask], preds_all[eval_mask])
        r2 = r2_score(actual_all[eval_mask], preds_all[eval_mask])
        print(f"\n  Time-series CV MAE:  {mae:.2f} ppt")
        print(f"  Time-series CV R2:   {r2:.3f}")

        # Final model on all data
        model = xgb.XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.1,
                                  reg_alpha=1.0, reg_lambda=2.0, random_state=42, verbosity=0)
        model.fit(clean[FEATURE_COLS].values, actual_all)

    # --- Predict next month (March 2026) ---
    last = df.iloc[-1]
    next_features = {col: last[col] for col in FEATURE_COLS}
    # Update oil lag with current $100 spike
    next_features["oil_lag1"] = 100.06
    next_features["oil_momentum"] = 100.06 - (last["oil_lag2"] or last["oil_lag1"])
    next_features["month"] = (last["date"].month % 12) + 1

    X_next = np.array([[next_features[c] for c in FEATURE_COLS]])
    pred_next = model.predict(X_next)[0]
    last_price = last["benchmark_price"]
    pred_price = last_price * (1 + pred_next / 100)

    print(f"\n=== NEXT MONTH PREDICTION (March/April 2026) ===")
    print(f"  Last benchmark (Feb 2026):     ${last_price:,.0f}")
    print(f"  Predicted MoM change:          {pred_next:+.2f}%")
    print(f"  Predicted next benchmark:      ${pred_price:,.0f}")
    print(f"  (Oil at $100 — highest since 2014 spike)")

    # --- Persist metrics ---
    import json
    from pathlib import Path
    from datetime import datetime

    residuals = clean[TARGET].values - model.predict(clean[FEATURE_COLS].values)
    std_residual = float(np.std(residuals))
    metrics = {
        "monthly": {
            "mae": float(mae) if mae is not None else None,
            "r2": float(r2) if r2 is not None else None,
            "n_samples": int(len(clean)),
            "cv_method": "TimeSeriesSplit",
            "prediction_next_mom_pct": float(pred_next),
            "prediction_next_price": float(pred_price),
            "ci_95_low_pct": float(pred_next - 1.96 * std_residual),
            "ci_95_high_pct": float(pred_next + 1.96 * std_residual),
            "feature_importance": dict(zip(FEATURE_COLS, model.feature_importances_.tolist())),
            "updated": datetime.now().isoformat(),
        }
    }
    metrics_path = Path("data/processed/model_metrics.json")
    existing = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    existing.update(metrics)
    metrics_path.write_text(json.dumps(existing, indent=2))
    print(f"\n  95% CI: [{pred_next - 1.96 * std_residual:+.2f}%, {pred_next + 1.96 * std_residual:+.2f}%]")
    print(f"  Metrics saved to {metrics_path}")

    return model, clean, df


if __name__ == "__main__":
    model, clean, df = train_monthly_model()
    print("\nMonthly model done. Run again after Wayback scraper adds 2019-2024 data.")
