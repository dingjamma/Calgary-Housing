"""
Scrape WTI crude oil price history via yfinance
Saves to data/raw/oil_prices.csv
"""

import yfinance as yf
import pandas as pd

def scrape_oil_prices(start="2005-01-01", end=None):
    print("Fetching WTI crude oil prices...")
    ticker = yf.Ticker("CL=F")
    df = ticker.history(start=start, end=end, interval="1mo")
    df = df[["Close"]].rename(columns={"Close": "oil_price_usd"})
    df.index = df.index.tz_localize(None)
    df.index.name = "date"
    df.to_csv("data/raw/oil_prices.csv")
    print(f"Saved {len(df)} months of oil price data to data/raw/oil_prices.csv")
    return df

if __name__ == "__main__":
    df = scrape_oil_prices()
    print(df.tail(10))
