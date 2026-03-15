"""
Scrape Calgary residential property assessment data from City of Calgary Open Data Portal.
Uses the Socrata API (data.calgary.ca) to get annual average assessed values 2005-2025.

Dataset: Historical Property Assessments (Parcel) — 4ur7-wsgc
Note: Assessments reflect market value as of July 1 of the prior year.
So 2025 assessment ≈ July 2024 market value (6-month lag).

This serves as the primary proxy for historical Calgary housing prices pre-2025.
"""

import requests
import pandas as pd


SOCRATA_URL = "https://data.calgary.ca/resource/4ur7-wsgc.json"


def scrape_annual_assessed_values():
    """Pull average residential assessed value per year, 2005-2025."""
    print("Fetching Calgary annual assessed values from Open Data Portal...")

    params = {
        "$select": "roll_year, avg(re_assessed_value) as avg_assessed_value, count(*) as property_count",
        "$group": "roll_year",
        "$order": "roll_year ASC",
        "$where": "assessment_class = 'RE' AND re_assessed_value > '50000'",
        "$limit": 50,
    }

    resp = requests.get(SOCRATA_URL, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    df["roll_year"] = df["roll_year"].astype(int)
    df["avg_assessed_value"] = df["avg_assessed_value"].astype(float).round(0)
    df["property_count"] = df["property_count"].astype(int)

    # Calgary assessments are based on July 1 of prior year
    # Roll year 2025 assessment = ~July 2024 market value
    # We'll store with the assessment date (Jan 1 of roll_year) and note the lag
    df["assessment_date"] = pd.to_datetime(df["roll_year"].astype(str) + "-01-01")

    df = df[["roll_year", "assessment_date", "avg_assessed_value", "property_count"]]
    df = df.sort_values("roll_year").reset_index(drop=True)

    output_file = "data/raw/calgary_assessments.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} years of assessment data to {output_file}")
    print(df.to_string(index=False))
    return df


def scrape_by_community(year=None):
    """Pull average assessed values by community for a given year (or all years)."""
    print(f"Fetching Calgary assessments by community{f' for {year}' if year else ''}...")

    params = {
        "$select": "roll_year, comm_code, comm_name, avg(re_assessed_value) as avg_value, count(*) as count",
        "$group": "roll_year, comm_code, comm_name",
        "$order": "roll_year ASC, comm_name ASC",
        "$where": "assessment_class = 'RE' AND re_assessed_value > '50000'",
        "$limit": 50000,
    }
    if year:
        params["$where"] += f" AND roll_year = '{year}'"

    resp = requests.get(SOCRATA_URL, params=params, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    df["roll_year"] = df["roll_year"].astype(int)
    df["avg_value"] = df["avg_value"].astype(float).round(0)
    df["count"] = df["count"].astype(int)

    output_file = f"data/raw/calgary_assessments_by_community{'_' + str(year) if year else ''}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} community-year records to {output_file}")
    return df


if __name__ == "__main__":
    df = scrape_annual_assessed_values()
