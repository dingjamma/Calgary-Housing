"""
XGBoost model to predict Calgary annual housing price change.

Features (all lagged 1 year so we're predicting forward):
- Oil price avg, Dec snapshot
- Overnight rate avg, Dec snapshot
- 5yr bond yield avg
- CAD/USD rate
- Natural gas price
- Alberta Energy ETF
- Prior year price level (autoregressive)
- Prior year YoY change

Target: next year's YoY % price change
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
from sklearn.model_selection import LeaveOneOut
import warnings
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

FEATURE_COLS = [
    "oil_price_avg",
    "oil_price_dec",
    "overnight_rate_avg",
    "overnight_rate_dec",
    "bond_yield_avg",
    "cadusd_avg",
    "natgas_avg",
    "alberta_etf_avg",
    "avg_assessed_value",   # price level
    "price_yoy_pct",        # momentum
]

TARGET = "next_year_yoy_pct"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """All features are already lagged correctly in annual_merged.csv:
    roll_year N uses reference_year N-1 economic indicators,
    and next_year_yoy_pct is the year N+1 target.
    Drop rows missing target or key features."""
    clean = df.dropna(subset=[TARGET] + FEATURE_COLS).copy()
    return clean


def train_and_evaluate() -> tuple:
    df = pd.read_csv("data/processed/annual_merged.csv")
    data = build_features(df)

    print(f"Dataset: {len(data)} usable years ({data['roll_year'].min()}–{data['roll_year'].max()})")

    X = data[FEATURE_COLS].values
    y = data[TARGET].values
    years = data["roll_year"].values

    # --- Leave-One-Out cross-validation (small dataset, LOO is appropriate) ---
    loo = LeaveOneOut()
    preds = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X):
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            verbosity=0,
        )
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])

    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"\nLOO Cross-Validation Results:")
    print(f"  MAE:  {mae:.2f} percentage points")
    print(f"  R2:   {r2:.3f}")

    # --- Final model trained on all data ---
    final_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        verbosity=0,
    )
    final_model.fit(X, y)

    # --- Predict 2026 ---
    # Use 2025 roll_year row: its economic features = 2024 actuals
    # But we want to predict using current 2025/2026 oil spike
    last_row = df[df["roll_year"] == 2025].iloc[0].copy()

    # Override oil with current spike (March 2026 avg so far ~$87)
    oil_daily = pd.read_csv("data/raw/oil_prices_daily.csv", parse_dates=["date"])
    oil_2025 = oil_daily[oil_daily["date"].dt.year == 2025]["oil_price_usd"].mean()
    oil_2026_ytd = oil_daily[oil_daily["date"].dt.year == 2026]["oil_price_usd"].mean()
    oil_dec_2025 = oil_daily[oil_daily["date"].dt.year == 2025]["oil_price_usd"].iloc[-1] if len(oil_daily[oil_daily["date"].dt.year == 2025]) else last_row["oil_price_dec"]

    # For 2026 prediction: reference year is 2025
    # Use actual 2025 avg oil + current 2026 YTD trend
    pred_features = {
        "oil_price_avg": oil_2025,
        "oil_price_dec": oil_dec_2025,
        "overnight_rate_avg": last_row["overnight_rate_avg"],
        "overnight_rate_dec": last_row["overnight_rate_dec"],
        "bond_yield_avg": last_row["bond_yield_avg"],
        "cadusd_avg": last_row["cadusd_avg"],
        "natgas_avg": last_row["natgas_avg"],
        "alberta_etf_avg": last_row["alberta_etf_avg"],
        "avg_assessed_value": last_row["avg_assessed_value"],
        "price_yoy_pct": last_row["price_yoy_pct"],
    }
    X_2026 = np.array([[pred_features[c] for c in FEATURE_COLS]])
    pred_2026_base = final_model.predict(X_2026)[0]

    # Scenario: oil spike sustains ($90+ for 2026)
    pred_features_spike = pred_features.copy()
    pred_features_spike["oil_price_avg"] = 88.0  # conservative 2026 annual avg given spike
    pred_features_spike["oil_price_dec"] = 100.0
    X_spike = np.array([[pred_features_spike[c] for c in FEATURE_COLS]])
    pred_2026_spike = final_model.predict(X_spike)[0]

    current_price = last_row["avg_assessed_value"]
    print(f"\n=== 2026 PREDICTIONS ===")
    print(f"Current avg assessed value (2025): ${current_price:,.0f}")
    print(f"\nBase scenario (2025 oil avg = ${oil_2025:.0f}/barrel):")
    print(f"  Predicted YoY change: {pred_2026_base:+.1f}%")
    print(f"  Predicted 2026 value: ${current_price * (1 + pred_2026_base/100):,.0f}")
    print(f"\nOil spike scenario (2026 avg ~$88/barrel — current trajectory):")
    print(f"  Predicted YoY change: {pred_2026_spike:+.1f}%")
    print(f"  Predicted 2026 value: ${current_price * (1 + pred_2026_spike/100):,.0f}")

    # --- Persist metrics ---
    import json
    from pathlib import Path
    from datetime import datetime

    residuals = y - preds
    std_residual = float(np.std(residuals))
    metrics = {
        "annual": {
            "mae": float(mae),
            "r2": float(r2),
            "n_samples": int(len(y)),
            "cv_method": "LeaveOneOut",
            "prediction_2026_base_pct": float(pred_2026_base),
            "prediction_2026_spike_pct": float(pred_2026_spike),
            "ci_95_low_pct": float(pred_2026_base - 1.96 * std_residual),
            "ci_95_high_pct": float(pred_2026_base + 1.96 * std_residual),
            "feature_importance": dict(zip(FEATURE_COLS, final_model.feature_importances_.tolist())),
            "updated": datetime.now().isoformat(),
        }
    }
    metrics_path = Path("data/processed/model_metrics.json")
    existing = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    existing.update(metrics)
    metrics_path.write_text(json.dumps(existing, indent=2))
    print(f"\n  95% CI: [{pred_2026_base - 1.96 * std_residual:+.1f}%, {pred_2026_base + 1.96 * std_residual:+.1f}%]")
    print(f"  Metrics saved to {metrics_path}")

    return final_model, data, preds, years, y, pred_2026_base, pred_2026_spike, current_price, oil_2025


def plot_results(final_model, data, preds, years, y, pred_base, pred_spike, current_price, oil_2025):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    # --- Chart 1: Actual vs LOO predicted ---
    ax = axes[0]
    ax.plot(years, y, "b-o", linewidth=2, label="Actual", zorder=5)
    ax.plot(years, preds, "r--s", linewidth=2, label="LOO Predicted", alpha=0.8, zorder=4)
    ax.axhline(0, color="black", linewidth=0.8)

    # 2026 predictions
    next_year = years[-1] + 1
    ax.scatter([next_year], [pred_base], color="green", s=120, zorder=6, label=f"2026 base ({pred_base:+.1f}%)")
    ax.scatter([next_year], [pred_spike], color="orange", marker="*", zorder=6, s=200, label=f"2026 spike ({pred_spike:+.1f}%)")
    ax.set_title("Actual vs Predicted — YoY Housing Price Change", fontsize=12)
    ax.set_ylabel("YoY % Change")
    ax.set_xlabel("Assessment Year")
    ax.legend(fontsize=8)

    # --- Chart 2: Feature importance ---
    ax = axes[1]
    importance = final_model.feature_importances_
    feat_labels = [
        "Oil avg", "Oil Dec", "Rate avg", "Rate Dec",
        "Bond yield", "CAD/USD", "Nat gas", "AB ETF",
        "Price level", "Momentum"
    ]
    idx = np.argsort(importance)
    colors = ["#e74c3c" if "Oil" in feat_labels[i] or "Nat gas" in feat_labels[i]
              else "#3498db" for i in idx]
    ax.barh([feat_labels[i] for i in idx], importance[idx], color=colors)
    ax.set_title("Feature Importance (XGBoost)", fontsize=12)
    ax.set_xlabel("Importance Score")

    # --- Chart 3: 2026 scenario bar ---
    ax = axes[2]
    # Use full df (before dropping rows with missing target) for actuals
    full_df = pd.read_csv("data/processed/annual_merged.csv")
    scenarios = ["2023\nActual", "2024\nActual", "2025\nActual", "2026\nBase\n(oil $75)", "2026\nSpike\n(oil $88)"]
    values = [
        full_df[full_df["roll_year"] == 2023]["price_yoy_pct"].values[0],
        full_df[full_df["roll_year"] == 2024]["price_yoy_pct"].values[0],
        full_df[full_df["roll_year"] == 2025]["price_yoy_pct"].values[0],
        pred_base,
        pred_spike,
    ]
    bar_colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]
    bar_colors[3] = "#27ae60"
    bar_colors[4] = "#f39c12"
    bars = ax.bar(scenarios, values, color=bar_colors, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:+.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Calgary Housing — Recent & Predicted YoY %", fontsize=12)
    ax.set_ylabel("YoY % Change")

    plt.tight_layout()
    plt.savefig("notebooks/chart5_model_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nChart saved to notebooks/chart5_model_results.png")


if __name__ == "__main__":
    final_model, data, preds, years, y, pred_base, pred_spike, current_price, oil_2025 = train_and_evaluate()
    plot_results(final_model, data, preds, years, y, pred_base, pred_spike, current_price, oil_2025)
