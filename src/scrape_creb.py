"""
Scrape CREB monthly stats PDFs for Calgary housing data.

Extracts from the district summary page (page 4 in most PDFs):
- Sales, New Listings, Inventory, Months of Supply, Benchmark Price, YoY change
- By property type (Detached, Apartment, Semi-Detached, Row) and district

Also extracts city-wide summary by property type from the stats pages.

URL pattern: https://www.creb.com/Housing_Statistics/documents/MM_YYYY_Calgary_Monthly_Stats_Package.pdf
"""

import re
import io
import time
import requests
import pdfplumber
import pandas as pd
from datetime import datetime

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.creb.com/Housing_Statistics/"
}

DISTRICTS = ["City Centre", "North East", "North", "North West", "West", "South", "South East", "East", "TOTAL CITY"]
PROP_TYPE_ORDER = ["Detached", "Apartment", "Semi-Detached", "Row"]


def build_url(year, month):
    return f"https://www.creb.com/Housing_Statistics/documents/{month:02d}_{year}_Calgary_Monthly_Stats_Package.pdf"


def clean_price(val):
    """Clean '$701,500' or '701500' → float"""
    if val is None:
        return None
    val = re.sub(r"[^\d.]", "", str(val))
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def parse_district_table(text, prop_type):
    """
    Parse the district breakdown table for a given property type.
    Format: District | Sales | New Listings | % change | Inventory | Months Supply | Benchmark Price | YoY% | M/M%
    """
    rows = []
    for district in DISTRICTS:
        # Match line starting with district name
        pattern = rf"{re.escape(district)}\s+([\d,]+)\s+([\d,]+)\s+[-\d.]+%\s+([\d,]+)\s+([\d.]+)\s+\$([\d,]+)\s+([-\d.]+)%"
        match = re.search(pattern, text)
        if match:
            rows.append({
                "district": "Calgary" if district == "TOTAL CITY" else district,
                "property_type": prop_type,
                "sales": clean_price(match.group(1)),
                "new_listings": clean_price(match.group(2)),
                "inventory": clean_price(match.group(3)),
                "months_supply": float(match.group(4)),
                "benchmark_price": clean_price(match.group(5)),
                "benchmark_yoy_pct": float(match.group(6)),
            })
    return rows


def parse_citywide_page(text, prop_type):
    """Parse city-wide stats page for a given property type."""
    # Normalize spacing artifacts from PDF
    normalized = re.sub(r"(\d)\s+,\s*(\d)", r"\1,\2", text)
    normalized = re.sub(r"\$\s+", "$", normalized)

    def extract(pattern):
        match = re.search(pattern, normalized)
        if not match:
            return None, None
        return clean_price(match.group(1)), clean_price(match.group(2))

    prev_sales, curr_sales = extract(r"Total Sales\s+([\d,]+)\s+([\d,]+)")
    prev_bench, curr_bench = extract(r"Benchmark Price\s+\$?([\d,]+)\s+\$?([\d,]+)")
    prev_median, curr_median = extract(r"Median Price\s+\$?([\d,]+)\s+\$?([\d,]+)")
    prev_avg, curr_avg = extract(r"Average Price\s+\$?([\d,]+)\s+\$?([\d,]+)")
    prev_inv, curr_inv = extract(r"Inventory\s+([\d,]+)\s+([\d,]+)")
    prev_dom, curr_dom = extract(r"Days on Market\s+([\d,]+)\s+([\d,]+)")
    prev_nl, curr_nl = extract(r"New Listings\s+([\d,]+)\s+([\d,]+)")

    # Sanity check benchmark price (should be 100k–3M)
    def valid_price(p):
        return p if p and 100_000 < p < 3_000_000 else None

    return {
        "district": "Calgary",
        "property_type": prop_type,
        "sales": curr_sales,
        "new_listings": curr_nl,
        "inventory": curr_inv,
        "days_on_market": curr_dom,
        "benchmark_price": valid_price(curr_bench),
        "median_price": valid_price(curr_median),
        "average_price": valid_price(curr_avg),
        "prev_benchmark_price": valid_price(prev_bench),
    }


def scrape_month(year, month):
    """Download and parse one monthly PDF, return list of row dicts."""
    url = build_url(year, month)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return None

        rows = []
        date_str = f"{year}-{month:02d}-01"

        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]

        # --- District breakdown table ---
        # Find pages with district breakdown (has TOTAL CITY + City Centre + $ signs)
        for text in pages:
            if "TOTAL CITY" not in text or "City Centre" not in text or "$" not in text:
                continue

            # Split the page into sections at each "TOTAL CITY" line
            # Each section corresponds to one property type in order: Detached, Apartment, Semi-Detached, Row
            splits = re.split(r"(?=TOTAL CITY\s+[\d,]+)", text)

            # First chunk is the header/intro before any TOTAL CITY, skip it
            # Remaining chunks each end with a TOTAL CITY row
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

        # --- City-wide stats by property type (pages 5-8 typically) ---
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
            # Detect property type from page header
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
                # Only add if not already captured from district table
                existing = [(r["date"], r["property_type"], r["district"]) for r in rows]
                if (date_str, detected_type, "Calgary") not in existing:
                    rows.append(city_row)

        return rows if rows else None

    except Exception as e:
        print(f"  Error {year}-{month:02d}: {e}")
        return None


def scrape_all(start_year=2018, end_year=2026):
    """Scrape all available monthly PDFs."""
    all_rows = []
    now = datetime.now()

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == now.year and month > now.month:
                break

            print(f"Scraping {year}-{month:02d}...", end=" ", flush=True)
            rows = scrape_month(year, month)

            if rows:
                all_rows.extend(rows)
                print(f"{len(rows)} records")
            else:
                print("skipped")

            time.sleep(0.5)

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["date", "property_type", "district"])
    df = df.sort_values(["date", "property_type", "district"]).reset_index(drop=True)
    df.to_csv("data/raw/creb_housing_prices.csv", index=False)
    print(f"\nSaved {len(df)} records to data/raw/creb_housing_prices.csv")
    return df


if __name__ == "__main__":
    # Test single month first
    print("=== Testing Jan 2025 ===")
    rows = scrape_month(2025, 1)
    if rows:
        df = pd.DataFrame(rows)
        print(df[["date", "property_type", "district", "benchmark_price", "sales", "inventory"]].to_string())
        print(f"\nTotal: {len(rows)} rows")
    else:
        print("No data extracted")

    print("\n=== Running full scrape (2018-2026) ===")
    df = scrape_all(start_year=2018)
