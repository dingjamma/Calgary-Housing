"""
Scrape CMHC (Canada Mortgage and Housing Corporation) housing data for Calgary
Covers: housing starts, completions, absorptions by dwelling type
https://www.cmhc-schl.gc.ca/en/data-and-research
"""

import pandas as pd
import requests

def scrape_cmhc_starts(start_year=2005):
    print("Fetching CMHC housing starts data for Calgary...")

    # CMHC Housing Market Data API
    url = "https://www03.cmhc-schl.gc.ca/hmip-pimh/en/TableMapChart/TableMatchingCriteria"

    params = {
        "GeographyType": "CMA",
        "GeographyId": "825",  # Calgary CMA code
        "CategoryLevel1": "Starting",
        "Frequency": "Monthly",
        "FromYear": str(start_year),
        "FromMonth": "1",
        "ToYear": "2026",
        "ToMonth": "12",
        "SeriesTypeId": "1"
    }

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    resp = requests.get(url, params=params, headers=headers, timeout=30)

    if resp.status_code != 200:
        print(f"CMHC API returned {resp.status_code}, trying alternative source...")
        return scrape_cmhc_alternative()

    data = resp.json()
    df = pd.DataFrame(data)
    df.to_csv("data/raw/cmhc_starts.csv", index=False)
    print(f"Saved CMHC starts data to data/raw/cmhc_starts.csv")
    return df


def scrape_cmhc_alternative():
    """
    Fallback: pull Calgary housing starts from Statistics Canada
    Table 34-10-0143-01 — Canada Mortgage and Housing Corporation, housing starts
    """
    print("Fetching from Statistics Canada...")

    url = "https://www150.statcan.gc.ca/t1/tbl1/en/dtbl/downloadTbl/csvDownload/3410014301-eng.zip"
    resp = requests.get(url, timeout=60)

    import zipfile, io
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    csv_name = [f for f in z.namelist() if f.endswith(".csv") and "MetaData" not in f][0]

    df = pd.read_csv(z.open(csv_name))

    # Filter for Calgary
    df = df[df["GEO"].str.contains("Calgary", na=False)]
    df = df[df["TYPE OF UNIT"].str.contains("Total", na=False)]
    df = df[["REF_DATE", "GEO", "TYPE OF UNIT", "VALUE"]].rename(columns={
        "REF_DATE": "date",
        "VALUE": "housing_starts"
    })

    df.to_csv("data/raw/cmhc_starts.csv", index=False)
    print(f"Saved {len(df)} rows to data/raw/cmhc_starts.csv")
    return df


if __name__ == "__main__":
    df = scrape_cmhc_starts()
    print(df.tail(10))
