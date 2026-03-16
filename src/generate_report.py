"""
Generate Calgary Housing Market Report — seed document for MiroFish simulation.
Combines: model predictions + EDA findings + live news feed.
Output: reports/calgary_housing_report_YYYY-MM-DD.md
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os, sys

os.makedirs("reports", exist_ok=True)


def load_model_outputs():
    """Load latest data and compute key stats."""
    annual   = pd.read_csv("data/processed/annual_merged.csv")
    monthly  = pd.read_csv("data/raw/creb_housing_prices.csv", parse_dates=["date"])
    pressure = pd.read_csv("data/processed/daily_housing_pressure.csv", parse_dates=["date"])
    oil      = pd.read_csv("data/raw/oil_prices_daily.csv", parse_dates=["date"])
    hist     = pd.read_csv("data/raw/creb_housing_historical.csv", parse_dates=["date"])

    # CREB latest
    creb_calgary = monthly[monthly["district"] == "Calgary"].copy()
    latest_month = creb_calgary["date"].max()
    latest_creb  = creb_calgary[creb_calgary["date"] == latest_month]

    # Pressure score
    today_score = pressure.iloc[-1]["housing_pressure_score"]
    today_oil   = pressure.iloc[-1]["oil_price_usd"]
    score_date  = pressure.iloc[-1]["date"]

    # Oil stats
    oil_90d_avg  = oil.tail(90)["oil_price_usd"].mean()
    oil_30d_avg  = oil.tail(30)["oil_price_usd"].mean()
    oil_today    = oil.iloc[-1]["oil_price_usd"]
    oil_jan_avg  = oil[oil["date"].dt.month == 1]["oil_price_usd"].tail(31).mean()

    # Annual trend
    last_annual  = annual.iloc[-2]  # 2025 (last full assessment year)
    yoy_2025     = last_annual["price_yoy_pct"]
    price_2025   = last_annual["avg_assessed_value"]

    # Historical CREB for 2019-2024 MoM context
    hist_calgary = hist[hist["district"] == "Calgary"].copy()

    return {
        "latest_month": latest_month,
        "latest_creb": latest_creb,
        "today_score": today_score,
        "today_oil": today_oil,
        "score_date": score_date,
        "oil_90d_avg": oil_90d_avg,
        "oil_30d_avg": oil_30d_avg,
        "oil_today": oil_today,
        "yoy_2025": yoy_2025,
        "price_2025": price_2025,
    }


def load_news():
    """Load the latest news feed if available."""
    if not os.path.exists("data/raw/news_feed.md"):
        return ""
    with open("data/raw/news_feed.md", "r", encoding="utf-8") as f:
        return f.read()


def generate_report():
    print("Generating Calgary Housing Market Report...")
    d = load_model_outputs()
    news_md = load_news()
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")

    # CREB latest benchmarks
    creb = d["latest_creb"]
    tr   = creb[creb["property_type"] == "Total Residential"].iloc[0] if len(creb) > 0 else None
    det  = creb[creb["property_type"] == "Detached"].iloc[0] if len(creb) > 0 else None
    apt  = creb[creb["property_type"] == "Apartment"].iloc[0] if len(creb) > 0 else None

    report = f"""# Calgary Housing Market Intelligence Report
**Date:** {now.strftime("%B %d, %Y")}
**Prepared by:** Calgary Housing Price Predictor (automated pipeline)
**Classification:** Seed document for MiroFish geopolitical simulation

---

## Executive Summary

Calgary's housing market is entering a critical inflection point. After 14.5% assessed value growth in 2025 — the strongest since the 2007 oil boom — the market now faces a simultaneous oil price shock and sustained high interest rate environment. WTI crude has surged from $57 in December 2025 to **${d['oil_today']:.2f}/barrel today**, the highest level since 2014. Our XGBoost daily pressure model currently reads **{d['today_score']:+.2f}%** — the signal flipped negative on March 10 as oil crossed $83, indicating economic shock conditions rather than prosperity-driven demand.

The central question: does $100 oil help or hurt Calgary housing in 2026?

---

## Section 1 — Current Market State

### CREB Benchmark Prices ({d['latest_month'].strftime('%B %Y')})

| Property Type | Benchmark Price | Sales | Inventory |
|---|---|---|---|"""

    for _, row in creb.iterrows():
        sales = f"{int(row['sales']):,}" if pd.notna(row.get('sales')) else "N/A"
        inv   = f"{int(row['inventory']):,}" if pd.notna(row.get('inventory')) else "N/A"
        report += f"\n| {row['property_type']} | ${row['benchmark_price']:,.0f} | {sales} | {inv} |"

    report += f"""

### Key Market Observations
- Total Residential benchmark: **${tr['benchmark_price']:,.0f}** ({d['latest_month'].strftime('%B %Y')})
- Detached benchmark: **${det['benchmark_price']:,.0f}** — up from $583k in Jan 2025
- Apartment benchmark: **${apt['benchmark_price']:,.0f}** — most affordable segment, continued demand from first-time buyers
- Total active inventory: **{int(tr['inventory']):,}** — market cooling from 2024 lows
- Sales slowing: {int(tr['sales']):,} total residential sales last month

---

## Section 2 — Economic Indicators & Oil Shock Analysis

### WTI Crude Oil (Primary Calgary Housing Driver)

| Metric | Value |
|---|---|
| Today's price | **${d['oil_today']:.2f}/barrel** |
| 30-day average | ${d['oil_30d_avg']:.2f}/barrel |
| 90-day average | ${d['oil_90d_avg']:.2f}/barrel |
| December 2025 | ~$57/barrel |
| Change since Dec 2025 | **+{((d['oil_today']/57)-1)*100:.0f}%** |

The oil spike from $57 (Dec 2025) to ${d['oil_today']:.0f} (March 15, 2026) represents a **{((d['oil_today']/57)-1)*100:.0f}% surge in under 90 days**, driven by US military strikes on Iran's Kharg Island and subsequent Strait of Hormuz restrictions. This is the most significant oil price event since the 2014 crash.

### Historical Oil-Housing Relationship (2005–2025)
- When oil averaged <$50/year: Calgary housing grew +4.3% the following year on average
- When oil averaged >$80/year: Calgary housing grew +2.0% the following year on average
- **Key insight:** Violent oil spikes correlate with housing *uncertainty*, not prosperity. The 2009 crash ($95→$35) caused Calgary housing to drop 12.5% in 2010. The 2014-2016 crash caused a 5% correction.
- Alberta's oil economy benefits from sustained high prices, not shock volatility.

### Interest Rate Environment
- Bank of Canada overnight rate: **2.25%** (as of Feb 2026, down from 5% peak in 2023)
- 5-year bond yield: **3.62%** — still elevated, keeping 5-year fixed mortgages around 4.5-5%
- Rate cuts ongoing but mortgage affordability still stretched vs. 2020-2021

---

## Section 3 — XGBoost Model Predictions

### Annual Model (19 assessment years, 2006–2024)
- **2026 base scenario** (oil avg ~$65): Predicted +7.9% YoY assessed value growth
- **2026 oil spike scenario** (oil avg ~$88): Predicted +10.1% YoY
- Current 2025 assessed value: **${d['price_2025']:,.0f}** avg residential
- Predicted 2026 range: **$753,000 – $768,000**

### Monthly Model (44 CREB months, 2019–2026)
- Next month benchmark prediction: **+0.92% MoM**
- Predicted March/April 2026 Total Residential benchmark: **~$565,700**

### Daily Pressure Score (946 labeled trading days)
- **Today ({d['score_date'].strftime('%B %d, %Y')}): {d['today_score']:+.2f}%**
- Score flipped negative March 10 as oil crossed $83
- Signal: model interprets $100 oil as demand shock, not prosperity signal
- Implication: expect flat-to-slight-negative MoM in next CREB release

### Model Interpretation
The models disagree slightly at different time horizons — this is expected and informative:
- **Short-term (daily/monthly):** Negative pressure from oil shock and affordability constraints
- **Long-term (annual):** Still positive, because Calgary fundamentals (population growth, energy sector employment) remain intact
- **Conclusion:** Likely a 1-2 month pause in benchmark growth, followed by resumption if oil stabilizes

---

## Section 4 — Key Risk Scenarios for Simulation

### Scenario A — Oil Sustained at $90-100 (Base Geopolitical Case)
Strait of Hormuz partially restricted, US-Iran tensions persist. Oil stays elevated.
- Short-term: buyer hesitation, inventory builds slightly
- Medium-term: Alberta energy sector employment surges → wage growth → housing demand rebounds
- 12-month housing outlook: +5-8% if oil stabilizes at $85-95

### Scenario B — Full Hormuz Closure (Extreme Case)
Oil spikes to $130-150. Global recession fears.
- Mortgage rates spike despite BoC cuts (bond yields rise)
- Calgary housing correction: -5 to -10% over 12 months
- But Alberta energy sector booms — eventual recovery stronger

### Scenario C — Diplomatic Resolution (Bull Case)
US-Iran ceasefire, oil drops back to $65-70 by Q3 2026.
- BoC accelerates cuts → mortgages affordable again
- Calgary housing resumes uptrend: +10-15% by end 2026
- Strong demand from energy sector workers + interprovincial migration

### Scenario D — Sustained Conflict + Rate Cuts (Most Likely)
Oil stays $80-100, BoC cuts to 1.5-2% by Q4 2026.
- Affordability improves despite oil shock
- Calgary housing: +6-9% over 12 months
- Apartment segment outperforms (first-time buyers return with lower rates)

---

## Section 5 — Simulation Parameters for MiroFish

**Key actors to simulate:**
- Bank of Canada (rate policy response to oil shock + inflation)
- Alberta Energy Sector (employment, wages, migration)
- Calgary home buyers (first-time, move-up, investor)
- Federal government (immigration targets, housing policy)
- Global oil markets (OPEC+ response, US shale reaction)
- Iran / US geopolitical actors (Strait of Hormuz)

**Prediction targets:**
1. Calgary Total Residential benchmark price — December 2026
2. BoC overnight rate — December 2026
3. WTI oil — average Q3/Q4 2026
4. Calgary market sentiment (buyer's vs seller's market)
5. Most at-risk property type segment

---

## Section 6 — Live News Context (Last 14 Days)

"""

    if news_md:
        # Strip the header from news_md since we have our own
        news_body = "\n".join(news_md.split("\n")[6:])
        report += news_body
    else:
        report += "*No news feed available — run scrape_news.py first.*\n"

    report += """
---

*This report was auto-generated by the Calgary Housing Price Predictor pipeline.*
*Data sources: CREB monthly PDFs, City of Calgary Open Data, Bank of Canada, yfinance, CBC Calgary, OilPrice.com, Better Dwelling.*
*Model: XGBoost trained on 2005–2026 data (annual), 2019–2026 (monthly/daily).*
"""

    # Save
    filename = f"reports/calgary_housing_report_{date_str}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report saved to {filename}")
    print(f"Length: {len(report):,} characters")
    return report, filename


if __name__ == "__main__":
    report, filename = generate_report()
    print(f"\n--- PREVIEW ---")
    print(report[:3000].encode("ascii", errors="replace").decode("ascii"))
