"""
Scrape WTI crude oil price history via yfinance
Saves to data/raw/oil_prices.csv
"""

import yfinance as yf
import pandas as pd

def scrape_oil_prices(start="2005-01-01", end=None):
    print("Fetching WTI crude oil prices (daily, resampled to monthly)...")
    ticker = yf.Ticker("CL=F")
    df = ticker.history(start=start, end=end, interval="1d")
    df = df[["Close"]].rename(columns={"Close": "oil_price_usd"})
    df.index = df.index.tz_localize(None)
    df.index.name = "date"

    # Save daily for reference
    df.reset_index().to_csv("data/raw/oil_prices_daily.csv", index=False)
    print(f"  Latest daily: {df.index[-1].date()} = ${df.iloc[-1, 0]:.2f}")

    # Resample to month-start using last trading day of each month
    monthly = df.resample("MS").last()
    # Include current partial month using the latest available price
    today_month = pd.Timestamp.today().to_period("M").to_timestamp()
    if today_month not in monthly.index:
        monthly.loc[today_month] = df.iloc[-1].values
    monthly = monthly.sort_index().reset_index()

    monthly.to_csv("data/raw/oil_prices.csv", index=False)
    print(f"Saved {len(monthly)} months of oil price data to data/raw/oil_prices.csv")
    return monthly

if __name__ == "__main__":
    df = scrape_oil_prices()
    print(df.tail(10))
