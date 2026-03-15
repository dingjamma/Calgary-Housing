"""
Merge all scraped datasets into a single analysis-ready dataframe.

Data sources:
- calgary_assessments.csv    : Annual avg residential assessed value 2005-2025 (primary price series)
- oil_prices.csv             : Monthly WTI oil price (USD/barrel) 2005-2026
- interest_rates.csv         : Monthly Bank of Canada overnight rate 2005-2026
- bond_yields.csv            : Monthly Canadian 5yr bond yield 2005-2026
- economic_indicators.csv    : Monthly CAD/USD, natural gas, Alberta Energy ETF 2005-2026
- creb_housing_prices.csv    : Monthly CREB benchmark prices by type/district 2025-2026

Output:
- data/processed/annual_merged.csv    : Annual panel for modelling (assessment-based)
- data/processed/creb_monthly.csv     : Monthly CREB data with economic features joined
"""

import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)


def load_economic_monthly():
    """Load and merge all monthly economic indicators."""
    oil = pd.read_csv("data/raw/oil_prices.csv", parse_dates=["date"])
    rates = pd.read_csv("data/raw/interest_rates.csv", parse_dates=["date"])
    bonds = pd.read_csv("data/raw/bond_yields.csv", parse_dates=["date"])
    econ = pd.read_csv("data/raw/economic_indicators.csv", parse_dates=["date"])

    # Normalize all to month-start
    for df in [oil, rates, bonds, econ]:
        df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

    merged = oil.merge(rates, on="date", how="outer") \
                .merge(bonds, on="date", how="outer") \
                .merge(econ, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def build_annual_panel():
    """
    Build annual panel: assessment year + annual averages of economic indicators.

    Note on date alignment:
    - Calgary assessments are based on July 1 of the PRIOR year.
    - Roll year 2025 assessment ≈ July 2024 market value.
    - For modelling: use economic indicators from the assessment reference year
      (i.e., for roll_year 2025, use calendar year 2024 indicators).
    """
    print("Building annual panel dataset...")

    assessments = pd.read_csv("data/raw/calgary_assessments.csv", parse_dates=["assessment_date"])
    econ_monthly = load_economic_monthly()

    # Average economic indicators by calendar year
    econ_monthly["year"] = econ_monthly["date"].dt.year
    econ_annual = econ_monthly.groupby("year").agg(
        oil_price_avg=("oil_price_usd", "mean"),
        overnight_rate_avg=("overnight_rate", "mean"),
        bond_yield_avg=("ca_5yr_bond_yield", "mean"),
        cadusd_avg=("cadusd_rate", "mean"),
        natgas_avg=("natgas_price", "mean"),
        alberta_etf_avg=("alberta_energy_etf", "mean"),
        oil_price_dec=("oil_price_usd", "last"),   # December snapshot
        overnight_rate_dec=("overnight_rate", "last"),
    ).reset_index()

    # Align: roll_year 2025 assessment → reference year = 2024
    assessments["reference_year"] = assessments["roll_year"] - 1

    panel = assessments.merge(
        econ_annual,
        left_on="reference_year",
        right_on="year",
        how="left"
    )
    panel = panel.drop(columns=["year"])

    # YoY price change
    panel = panel.sort_values("roll_year").reset_index(drop=True)
    panel["price_yoy_pct"] = panel["avg_assessed_value"].pct_change() * 100

    # Next year's price (prediction target)
    panel["next_year_price"] = panel["avg_assessed_value"].shift(-1)
    panel["next_year_yoy_pct"] = panel["price_yoy_pct"].shift(-1)

    output_file = "data/processed/annual_merged.csv"
    panel.to_csv(output_file, index=False)
    print(f"Saved {len(panel)} annual rows to {output_file}")
    print(panel[["roll_year", "avg_assessed_value", "price_yoy_pct", "oil_price_avg", "overnight_rate_avg"]].to_string(index=False))
    return panel


def build_creb_monthly():
    """
    Merge CREB monthly benchmark prices with monthly economic indicators.
    This gives a high-resolution recent dataset (2025-2026).
    """
    print("\nBuilding CREB monthly dataset...")

    creb = pd.read_csv("data/raw/creb_housing_prices.csv", parse_dates=["date"])
    econ = load_economic_monthly()

    # Filter for Calgary-wide benchmark (aggregate across all districts)
    calgary_creb = creb[creb["district"] == "Calgary"].copy()

    merged = calgary_creb.merge(econ, on="date", how="left")
    merged = merged.sort_values(["date", "property_type"]).reset_index(drop=True)

    output_file = "data/processed/creb_monthly.csv"
    merged.to_csv(output_file, index=False)
    print(f"Saved {len(merged)} CREB monthly rows to {output_file}")
    print(merged[["date", "property_type", "benchmark_price", "oil_price_usd", "overnight_rate"]].head(20).to_string(index=False))
    return merged


if __name__ == "__main__":
    panel = build_annual_panel()
    creb_monthly = build_creb_monthly()
    print("\nMerge complete. Files saved to data/processed/")
