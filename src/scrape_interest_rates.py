"""
Scrape Bank of Canada overnight rate and prime rate history
https://www.bankofcanada.ca/rates/interest-rates/canadian-interest-rates/
"""

import pandas as pd
import requests

def scrape_bank_of_canada_rates(start="2005-01-01"):
    print("Fetching Bank of Canada interest rates...")

    # Bank of Canada Valet API — free, no key required
    url = "https://www.bankofcanada.ca/valet/observations/group/bond_yields_all/json"
    params = {
        "start_date": start,
        "order_dir": "asc"
    }

    # Overnight rate
    url_overnight = "https://www.bankofcanada.ca/valet/observations/V122514/json"
    resp = requests.get(url_overnight, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    records = []
    for obs in data["observations"]:
        records.append({
            "date": obs["d"],
            "overnight_rate": float(obs["V122514"]["v"]) if obs["V122514"]["v"] != "" else None
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    # Resample to monthly
    df = df.set_index("date").resample("MS").last().reset_index()
    df.to_csv("data/raw/interest_rates.csv", index=False)
    print(f"Saved {len(df)} months of interest rate data to data/raw/interest_rates.csv")
    return df

if __name__ == "__main__":
    df = scrape_bank_of_canada_rates()
    print(df.tail(10))
