"""
Pull daily economic indicators for the daily prediction model.
- WTI oil (CL=F) — already done in scrape_oil.py
- CAD/USD (CADUSD=X)
- Natural gas (NG=F)
- Alberta Energy ETF (XEG.TO)
- Canadian 5yr bond yield (Bank of Canada Valet API — daily)
"""

import yfinance as yf
import pandas as pd
import requests


def scrape_daily_yfinance(start="2005-01-01"):
    tickers = {
        "CADUSD=X": "cadusd_rate",
        "NG=F":     "natgas_price",
        "XEG.TO":   "alberta_etf",
    }
    dfs = []
    for ticker, col in tickers.items():
        df = yf.Ticker(ticker).history(start=start, interval="1d")[["Close"]]
        df.index = df.index.tz_localize(None)
        df.index.name = "date"
        df = df.rename(columns={"Close": col})
        dfs.append(df)
        print(f"  {col}: {len(df)} days, latest {df.index[-1].date()} = {df.iloc[-1,0]:.4f}")

    combined = pd.concat(dfs, axis=1).reset_index()
    return combined


def scrape_daily_bond_yields(start="2005-01-01"):
    print("Fetching daily 5yr bond yields from Bank of Canada...")
    url = "https://www.bankofcanada.ca/valet/observations/V122487/json"
    resp = requests.get(url, params={"start_date": start, "order_dir": "asc"}, timeout=30)
    resp.raise_for_status()
    records = [
        {"date": o["d"], "ca_5yr_bond_yield": float(o["V122487"]["v"]) if o["V122487"]["v"] != "" else None}
        for o in resp.json()["observations"]
    ]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  bond yield: {len(df)} days, latest {df['date'].iloc[-1].date()} = {df['ca_5yr_bond_yield'].iloc[-1]:.2f}%")
    return df


def scrape_daily_overnight_rate(start="2005-01-01"):
    print("Fetching daily overnight rate from Bank of Canada...")
    url = "https://www.bankofcanada.ca/valet/observations/V122514/json"
    resp = requests.get(url, params={"start_date": start, "order_dir": "asc"}, timeout=30)
    resp.raise_for_status()
    records = [
        {"date": o["d"], "overnight_rate": float(o["V122514"]["v"]) if o["V122514"]["v"] != "" else None}
        for o in resp.json()["observations"]
    ]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  overnight rate: {len(df)} days, latest {df['date'].iloc[-1].date()} = {df['overnight_rate'].iloc[-1]:.2f}%")
    return df


if __name__ == "__main__":
    print("Fetching daily yfinance indicators...")
    econ = scrape_daily_yfinance()

    bonds = scrape_daily_bond_yields()
    rates = scrape_daily_overnight_rate()

    # Load daily oil (already scraped)
    oil = pd.read_csv("data/raw/oil_prices_daily.csv", parse_dates=["date"])

    # Merge all on date
    merged = oil.merge(bonds, on="date", how="outer") \
                .merge(rates, on="date", how="outer") \
                .merge(econ, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)

    # Forward-fill weekends/holidays (markets closed)
    merged = merged.ffill()

    # Drop pre-2005 and future
    merged = merged[merged["date"] >= "2005-01-01"]

    merged.to_csv("data/raw/daily_indicators.csv", index=False)
    print(f"\nSaved {len(merged)} days to data/raw/daily_indicators.csv")
    print(f"Columns: {list(merged.columns)}")
    print(merged.tail(5).to_string(index=False))
