"""
Scrape historical CREB PDFs (2019-2024) from the Wayback Machine.
Uses the CDX API to find archived snapshots, then downloads and parses them.
"""

import re
import io
import time
import requests
import pdfplumber
import pandas as pd
from datetime import datetime

# Import parsers from main scraper
from scrape_creb import parse_district_table, parse_citywide_page, PROP_TYPE_ORDER

HEADERS = {"User-Agent": "Mozilla/5.0"}
CDX_API = "http://web.archive.org/cdx/search/cdx"


def find_wayback_snapshot(year, month):
    """Find best Wayback Machine snapshot for a given month's CREB PDF."""
    # Try both URL formats CREB has used
    url_patterns = [
        f"creb.com/Housing_Statistics/documents/{month:02d} {year} Calgary Monthly Stats Package.pdf",
        f"creb.com/Housing_Statistics/documents/{month:02d}_{year}_Calgary Monthly Stats Package.pdf",
        f"creb.com/Housing_Statistics/documents/{month:02d}_{year}_Calgary_Monthly_Stats_Package.pdf",
    ]

    for pattern in url_patterns:
        resp = requests.get(CDX_API, params={
            "url": pattern,
            "output": "json",
            "fl": "timestamp,original,statuscode",
            "filter": "statuscode:200",
            "limit": 5,
            "matchType": "exact"
        }, timeout=15)

        if resp.status_code == 200:
            data = resp.json()
            if len(data) > 1:  # First row is header
                timestamp = data[1][0]
                original = data[1][1]
                wayback_url = f"http://web.archive.org/web/{timestamp}if_/{original}"
                return wayback_url

    return None


def scrape_month_wayback(year, month):
    """Download and parse one monthly PDF from Wayback Machine."""
    url = find_wayback_snapshot(year, month)
    if not url:
        return None

    try:
        resp = requests.get(url, headers=HEADERS, timeout=45)
        if resp.status_code != 200 or len(resp.content) < 10000:
            return None

        # Verify it's a PDF
        if not resp.content.startswith(b'%PDF'):
            return None

        rows = []
        date_str = f"{year}-{month:02d}-01"

        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]

        # District breakdown table
        for text in pages:
            if "TOTAL CITY" not in text or "City Centre" not in text or "$" not in text:
                continue

            splits = re.split(r"(?=TOTAL CITY\s+[\d,]+)", text)
            sections = []
            buffer = ""
            for chunk in splits:
                buffer += chunk
                if re.search(r"^TOTAL CITY\s+[\d,]+", chunk, re.MULTILINE):
                    sections.append(buffer)
                    buffer = ""

            for j, section in enumerate(sections):
                if j >= len(PROP_TYPE_ORDER):
                    break
                prop_type = PROP_TYPE_ORDER[j]
                section_rows = parse_district_table(section, prop_type)
                for row in section_rows:
                    row["date"] = date_str
                    rows.append(row)

        # City-wide stats
        prop_type_map = {
            "RESIDENTIAL": "Total Residential",
            "DETACHED": "Detached",
            "APARTMENT": "Apartment",
            "SEMI": "Semi-Detached",
            " ROW ": "Row",
        }
        for text in pages:
            if "Benchmark Price" not in text:
                continue
            text_upper = text.upper()
            detected_type = None
            for keyword, ptype in prop_type_map.items():
                if keyword in text_upper:
                    detected_type = ptype
                    break
            if not detected_type:
                continue
            city_row = parse_citywide_page(text, detected_type)
            if city_row.get("benchmark_price"):
                city_row["date"] = date_str
                existing = [(r["date"], r["property_type"], r["district"]) for r in rows]
                if (date_str, detected_type, "Calgary") not in existing:
                    rows.append(city_row)

        return rows if rows else None

    except Exception as e:
        print(f"  Error {year}-{month:02d}: {e}")
        return None


def scrape_historical(start_year=2019, end_year=2024, delay=3.0, resume=True):
    """
    Scrape historical data from Wayback Machine.
    - delay: seconds between requests (be polite)
    - resume: skip months already in the output file
    """
    output_file = "data/raw/creb_housing_historical.csv"

    # Load existing data to support resuming
    existing_dates = set()
    if resume and pd.io.common.file_exists(output_file):
        existing = pd.read_csv(output_file)
        existing_dates = set(existing["date"].unique())
        all_rows = existing.to_dict("records")
        print(f"Resuming — {len(existing_dates)} months already scraped")
    else:
        all_rows = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            date_str = f"{year}-{month:02d}-01"

            if date_str in existing_dates:
                print(f"Scraping {year}-{month:02d}... already done, skipping")
                continue

            print(f"Scraping {year}-{month:02d}...", end=" ", flush=True)

            # Retry up to 3 times on connection errors
            rows = None
            for attempt in range(3):
                try:
                    rows = scrape_month_wayback(year, month)
                    break
                except Exception as e:
                    print(f"  attempt {attempt+1} failed: {e}")
                    wait = delay * 3 + random.uniform(5, 20)
                    print(f"  waiting {wait:.0f}s before retry...")
                    time.sleep(wait)

            if rows:
                all_rows.extend(rows)
                print(f"{len(rows)} records")
            else:
                print("skipped")

            # Save progress after every month
            df = pd.DataFrame(all_rows)
            df = df.drop_duplicates(subset=["date", "property_type", "district"])
            df.to_csv(output_file, index=False)

            # Randomized delay — avoids predictable request patterns
            jitter = random.uniform(0, delay * 0.5)
            time.sleep(delay + jitter)

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["date", "property_type", "district"])
    df = df.sort_values(["date", "property_type", "district"]).reset_index(drop=True)
    df.to_csv(output_file, index=False)
    print(f"\nSaved {len(df)} records to {output_file}")
    return df


if __name__ == "__main__":
    import random
    print("=== Running historical scrape 2019-2024 (overnight mode) ===")
    print("Saves progress after each month — safe to interrupt and resume\n")
    # Use longer delay + jitter to avoid Wayback Machine rate limits
    df = scrape_historical(start_year=2019, end_year=2024, delay=15.0, resume=True)
    print(df.tail(10))
