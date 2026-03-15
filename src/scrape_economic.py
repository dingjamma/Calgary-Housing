"""
Scrape economic indicators relevant to Calgary housing:
- CAD/USD exchange rate (yfinance)
- Alberta unemployment proxy via ETF (yfinance)
- Canadian 5-year bond yield (Bank of Canada Valet API) — key mortgage rate driver
- Natural gas prices (yfinance) — secondary Alberta energy indicator
"""

import yfinance as yf
import pandas as pd
import requests

def scrape_bond_yields(start="2005-01-01"):
    """Canadian 5-year government bond yield — directly drives mortgage rates"""
    print("Fetching Canadian 5-year bond yields (Bank of Canada)...")
    url = "https://www.bankofcanada.ca/valet/observations/V122487/json"
    resp = requests.get(url, params={"start_date": start, "order_dir": "asc"}, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    records = [{"date": o["d"], "ca_5yr_bond_yield": float(o["V122487"]["v"]) if o["V122487"]["v"] != "" else None}
               for o in data["observations"]]

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").resample("MS").last().reset_index()
    df.to_csv("data/raw/bond_yields.csv", index=False)
    print(f"Saved {len(df)} months of bond yield data")
    return df


def scrape_market_indicators(start="2005-01-01"):
    """Pull CAD/USD and natural gas via yfinance"""
    print("Fetching CAD/USD and natural gas prices...")

    tickers = {
        "CADUSD=X": "cadusd_rate",
        "NG=F": "natgas_price",
        "XEG.TO": "alberta_energy_etf"  # iShares S&P/TSX Capped Energy ETF
    }

    dfs = []
    for ticker, col in tickers.items():
        try:
            df = yf.Ticker(ticker).history(start=start, interval="1d")[["Close"]]
            df.index = df.index.tz_localize(None)
            df.index.name = "date"
            df = df.rename(columns={"Close": col})
            # Resample daily → month-start using last trading day of month
            monthly = df.resample("MS").last()
            # Include current partial month
            today_month = pd.Timestamp.today().to_period("M").to_timestamp()
            if today_month not in monthly.index:
                monthly.loc[today_month] = df.iloc[-1].values
            monthly = monthly.sort_index()
            dfs.append(monthly)
            print(f"  {col}: {len(monthly)} months, latest {df.index[-1].date()} = {df.iloc[-1,0]:.4f}")
        except Exception as e:
            print(f"  Failed {ticker}: {e}")

    combined = pd.concat(dfs, axis=1).reset_index()
    combined.to_csv("data/raw/economic_indicators.csv", index=False)
    print(f"Saved economic indicators to data/raw/economic_indicators.csv")
    return combined


if __name__ == "__main__":
    scrape_bond_yields()
    scrape_market_indicators()
    print("\nAll economic indicators saved.")
