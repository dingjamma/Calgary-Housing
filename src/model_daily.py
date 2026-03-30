"""
Daily housing pressure model — real-time indicator of Calgary housing market direction.

Strategy:
- Features: rolling windows of daily oil, CAD/USD, nat gas, Alberta ETF, bond yields
- Target: next month's CREB benchmark MoM % change (labeled back onto daily data)
- Output: a daily "Housing Pressure Score" — positive = upward pressure on prices

This gives a live signal that updates every trading day, not just when CREB publishes.
The current $100 oil spike can be quantified in real time.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)


def build_daily_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each day, compute rolling window features.
    Label each day with the CREB benchmark change for the following month.
    """
    import os

    daily = pd.read_csv("data/raw/daily_indicators.csv", parse_dates=["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    # --- Rolling features (all computed from past data only) ---
    for col, alias in [("oil_price_usd", "oil"), ("cadusd_rate", "cadusd"),
                       ("natgas_price", "natgas"), ("alberta_etf", "etf"),
                       ("ca_5yr_bond_yield", "bond"), ("overnight_rate", "rate")]:
        daily[f"{alias}_7d"]  = daily[col].rolling(7,  min_periods=1).mean()
        daily[f"{alias}_30d"] = daily[col].rolling(30, min_periods=1).mean()
        daily[f"{alias}_90d"] = daily[col].rolling(90, min_periods=1).mean()
        daily[f"{alias}_mom"] = daily[col] / daily[col].shift(30) - 1  # 30-day momentum

    # Day of week, month (seasonality signals)
    daily["dow"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month

    # --- Load CREB monthly to create labels ---
    creb = pd.read_csv("data/raw/creb_housing_prices.csv", parse_dates=["date"])
    tr = creb[(creb["district"] == "Calgary") & (creb["property_type"] == "Total Residential")].copy()

    if os.path.exists("data/raw/creb_housing_historical.csv"):
        hist = pd.read_csv("data/raw/creb_housing_historical.csv", parse_dates=["date"])
        hist_tr = hist[(hist["district"] == "Calgary") & (hist["property_type"] == "Total Residential")].copy()
        tr = pd.concat([hist_tr, tr]).sort_values("date").reset_index(drop=True)

    tr["benchmark_mom_pct"] = tr["benchmark_price"].pct_change() * 100
    tr = tr[["date", "benchmark_price", "benchmark_mom_pct"]].dropna()

    # Label each day with the CREB result for the month that starts after that day's month
    # i.e., days in January get labeled with February's MoM change
    daily["month_start"] = daily["date"].dt.to_period("M").dt.to_timestamp()

    # Shift CREB by 1 month to create forward labels
    tr_shifted = tr.copy()
    tr_shifted["label_month"] = tr_shifted["date"] - pd.DateOffset(months=1)
    tr_shifted["label_month"] = tr_shifted["label_month"].dt.to_period("M").dt.to_timestamp()

    label_map = tr_shifted.set_index("label_month")["benchmark_mom_pct"].to_dict()
    daily["target_next_month_mom"] = daily["month_start"].map(label_map)

    print(f"  Daily rows: {len(daily)}")
    print(f"  Labeled rows (have CREB target): {daily['target_next_month_mom'].notna().sum()}")

    return daily, tr


FEATURE_COLS = [
    "oil_7d", "oil_30d", "oil_90d", "oil_mom",
    "cadusd_7d", "cadusd_30d", "cadusd_mom",
    "natgas_7d", "natgas_30d",
    "etf_7d", "etf_30d", "etf_mom",
    "bond_30d", "bond_mom",
    "rate_30d",
    "month", "dow",
]
TARGET = "target_next_month_mom"


def train_daily_model() -> tuple:
    print("Building daily model...")
    daily, creb_monthly = build_daily_dataset()

    labeled = daily.dropna(subset=FEATURE_COLS + [TARGET]).copy()
    print(f"  Usable labeled rows: {len(labeled)}")

    X = labeled[FEATURE_COLS].values
    y = labeled[TARGET].values
    dates = labeled["date"].values

    if len(labeled) < 30:
        print("  Not enough labeled data yet — need Wayback CREB 2019-2024.")
        print("  Fitting on available data for live scoring...")
        model = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, reg_alpha=1.0, reg_lambda=2.0,
                                  random_state=42, verbosity=0)
        model.fit(X, y)
        mae, r2 = None, None
    else:
        tscv = TimeSeriesSplit(n_splits=5)
        preds_cv = np.zeros(len(labeled))

        for train_idx, test_idx in tscv.split(X):
            m = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, reg_alpha=1.0, reg_lambda=2.0,
                                  random_state=42, verbosity=0)
            m.fit(X[train_idx], y[train_idx])
            preds_cv[test_idx] = m.predict(X[test_idx])

        first_test = list(tscv.split(X))[0][1][0]
        eval_mask = np.zeros(len(labeled), dtype=bool)
        eval_mask[first_test:] = True
        mae = mean_absolute_error(y[eval_mask], preds_cv[eval_mask])
        r2 = r2_score(y[eval_mask], preds_cv[eval_mask])
        print(f"\n  Time-series CV MAE: {mae:.2f} ppt")
        print(f"  Time-series CV R2:  {r2:.3f}")

        model = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, reg_alpha=1.0, reg_lambda=2.0,
                                  random_state=42, verbosity=0)
        model.fit(X, y)

    # --- Score ALL daily data (including unlabeled recent days) ---
    scoreable = daily.dropna(subset=FEATURE_COLS).copy()
    scoreable["housing_pressure_score"] = model.predict(scoreable[FEATURE_COLS].values)
    scoreable.to_csv("data/processed/daily_housing_pressure.csv", index=False)
    print(f"\nSaved daily pressure scores to data/processed/daily_housing_pressure.csv")

    # Print recent scores
    recent = scoreable[scoreable["date"] >= "2026-01-01"][["date", "oil_price_usd", "housing_pressure_score"]]
    print(f"\n=== DAILY HOUSING PRESSURE SCORE (Jan-Mar 2026) ===")
    print(f"  (positive = upward pressure on next month's benchmark)")
    print(recent.tail(20).to_string(index=False))

    # Today's score
    today = scoreable.iloc[-1]
    print(f"\n>>> TODAY ({today['date'].strftime('%Y-%m-%d')}): score = {today['housing_pressure_score']:+.2f}%")
    print(f"    Oil = ${today['oil_price_usd']:.2f}/barrel")

    # --- Persist metrics ---
    import json
    from pathlib import Path
    from datetime import datetime

    residuals = y - model.predict(X)
    std_residual = float(np.std(residuals))
    today_score = float(today['housing_pressure_score'])
    metrics = {
        "daily": {
            "mae": float(mae) if mae is not None else None,
            "r2": float(r2) if r2 is not None else None,
            "n_samples": int(len(labeled)),
            "cv_method": "TimeSeriesSplit",
            "today_score_pct": today_score,
            "today_oil_usd": float(today['oil_price_usd']),
            "ci_95_low_pct": float(today_score - 1.96 * std_residual),
            "ci_95_high_pct": float(today_score + 1.96 * std_residual),
            "feature_importance": dict(zip(FEATURE_COLS, model.feature_importances_.tolist())),
            "updated": datetime.now().isoformat(),
        }
    }
    metrics_path = Path("data/processed/model_metrics.json")
    existing = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    existing.update(metrics)
    metrics_path.write_text(json.dumps(existing, indent=2))
    print(f"  95% CI: [{today_score - 1.96 * std_residual:+.2f}%, {today_score + 1.96 * std_residual:+.2f}%]")
    print(f"  Metrics saved to {metrics_path}")

    return model, scoreable, labeled


def plot_daily_results(scoreable, creb_monthly):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=False)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    # --- Top: daily oil price with housing pressure score ---
    ax1 = axes[0]
    recent = scoreable[scoreable["date"] >= "2025-01-01"].copy()

    color_oil = "#e67e22"
    ax1.fill_between(recent["date"], recent["oil_price_usd"], alpha=0.3, color=color_oil)
    ax1.plot(recent["date"], recent["oil_price_usd"], color=color_oil, linewidth=1.5, label="WTI Oil (daily)")
    ax1.set_ylabel("WTI Oil USD/barrel", color=color_oil)
    ax1.tick_params(axis="y", labelcolor=color_oil)
    ax1.axhline(100, color="darkred", linestyle=":", linewidth=1.5, alpha=0.8)
    ax1.annotate("$100 — today", xy=(recent["date"].iloc[-1], 100), fontsize=9,
                 color="darkred", xytext=(-120, 8), textcoords="offset points")

    ax1b = ax1.twinx()
    score_color = "#2980b9"
    ax1b.plot(recent["date"], recent["housing_pressure_score"], color=score_color,
              linewidth=2, alpha=0.85, label="Housing Pressure Score")
    ax1b.axhline(0, color="gray", linewidth=0.8)
    ax1b.fill_between(recent["date"],
                      recent["housing_pressure_score"].clip(lower=0), 0,
                      alpha=0.2, color="green")
    ax1b.fill_between(recent["date"],
                      recent["housing_pressure_score"].clip(upper=0), 0,
                      alpha=0.2, color="red")
    ax1b.set_ylabel("Predicted Next-Month MoM % Change", color=score_color)
    ax1b.tick_params(axis="y", labelcolor=score_color)

    ax1.set_title("Daily Housing Pressure Score vs WTI Oil (2025–2026)", fontsize=13)
    lines = [plt.Line2D([0], [0], color=color_oil, linewidth=2),
             plt.Line2D([0], [0], color=score_color, linewidth=2)]
    ax1.legend(lines, ["WTI Oil", "Housing Pressure Score"], loc="upper left", fontsize=9)

    # --- Bottom: 30-day rolling score vs actual CREB MoM ---
    ax2 = axes[1]
    monthly_score = scoreable.set_index("date")["housing_pressure_score"].resample("MS").mean().reset_index()
    monthly_score.columns = ["date", "avg_score"]

    ax2.bar(monthly_score["date"], monthly_score["avg_score"],
            width=20, color=["#2ecc71" if v > 0 else "#e74c3c" for v in monthly_score["avg_score"]],
            alpha=0.7, label="Avg monthly pressure score")

    if len(creb_monthly) > 1:
        ax2.plot(creb_monthly["date"], creb_monthly["benchmark_mom_pct"],
                 "ko-", linewidth=2, markersize=6, label="Actual CREB MoM %", zorder=5)

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Monthly Avg Housing Pressure Score vs Actual CREB Change", fontsize=13)
    ax2.set_ylabel("MoM % Change")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("notebooks/chart6_daily_pressure.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart saved to notebooks/chart6_daily_pressure.png")


if __name__ == "__main__":
    model, scoreable, labeled = train_daily_model()
    creb = pd.read_csv("data/raw/creb_housing_prices.csv", parse_dates=["date"])
    creb_tr = creb[(creb["district"] == "Calgary") & (creb["property_type"] == "Total Residential")].copy()
    creb_tr["benchmark_mom_pct"] = creb_tr["benchmark_price"].pct_change() * 100
    plot_daily_results(scoreable, creb_tr)
