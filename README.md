# Calgary Housing Price Predictor

# Github Action Currently disabled - This repo will work once Github Support answers me

Full automated pipeline: scrape → model → news → report → geopolitical simulation. No manual downloads.

**Live Dashboard:** [calgary-housing.streamlit.app](https://calgary-housing-7phh7dhjyyd2r7mtjymuvm.streamlit.app)

---

## What It Does

Calgary's housing market is uniquely tied to oil. This project builds a complete intelligence pipeline that:

1. Scrapes 21 years of housing, oil, and economic data automatically
2. Trains XGBoost models at three time horizons (annual, monthly, daily)
3. Produces a live **Housing Pressure Score** — currently **+1.66%** at $100/barrel WTI
4. Scrapes live news from CBC Calgary, OilPrice.com, and Better Dwelling
5. Auto-generates a structured market intelligence report
6. Feeds the report into **MiroFish** — a multi-agent geopolitical simulation that predicts December 2026 outcomes across 8 actor types

---

## The Current Scenario (March 2026)

WTI crude surged from **$57 → $100/barrel** in 90 days after US strikes on Iran's Kharg Island. The daily model score flipped negative March 10 as oil crossed $83 (economic shock signal), then recovered to +1.66% with full training data. The central question MiroFish is simulating: **does $100 oil help or hurt Calgary housing in 2026?**

---

## Full Pipeline

```
                          DATA LAYER
┌─────────────────────────────────────────────────────────┐
│  City of Calgary Socrata API  →  annual_merged.csv      │
│  CREB monthly PDF scraper     →  creb_housing_prices    │
│  Wayback Machine (CREB)       →  creb_housing_historical│
│  Bank of Canada Valet API     →  overnight_rate, bonds  │
│  yfinance (daily)             →  WTI oil, CAD/USD, ETF  │
└─────────────────────────────────────────────────────────┘
                            ↓
                        MODEL LAYER
┌─────────────────────────────────────────────────────────┐
│  XGBoost Annual   (19 years)  →  2026 price range       │
│  XGBoost Monthly  (44 months) →  next benchmark         │
│  XGBoost Daily    (946 days)  →  Housing Pressure Score │
└─────────────────────────────────────────────────────────┘
                            ↓
                    INTELLIGENCE LAYER
┌─────────────────────────────────────────────────────────┐
│  scrape_news.py        →  RSS: CBC, OilPrice, Reuters   │
│  generate_report.py    →  structured market report      │
│  run_mirofish_pipeline.py  →  MiroFish API automation   │
│    ↳ upload seed → build Zep graph → generate agents    │
│    ↳ run 40-round Twitter simulation (qwen-plus)        │
│    ↳ report agent synthesizes predictions               │
└─────────────────────────────────────────────────────────┘
                            ↓
                      OUTPUT LAYER
┌─────────────────────────────────────────────────────────┐
│  Streamlit dashboard  →  live pressure score + charts   │
│  reports/             →  dated market intelligence docs │
│  simulations/         →  MiroFish prediction reports    │
└─────────────────────────────────────────────────────────┘
```

---

## Data Sources

| Source | Data | Coverage |
|--------|------|----------|
| City of Calgary Socrata API | Annual avg residential assessed value | 2005–2025 |
| CREB monthly PDF scraper | Benchmark price by property type | 2025–2026 |
| Wayback Machine (CREB archive) | Historical CREB benchmark prices | 2019–2024 |
| Bank of Canada Valet API | Overnight rate, 5yr bond yield | 2005–2026 (daily) |
| yfinance | WTI oil, CAD/USD, nat gas, Alberta ETF | 2005–2026 (daily) |
| CBC Calgary RSS | Housing and energy news | Last 14 days |
| OilPrice.com RSS | WTI, OPEC, Iran/Hormuz news | Last 14 days |
| Better Dwelling RSS | Canadian real estate analysis | Last 14 days |

---

## Models

### Annual XGBoost (19 assessment years, 2006–2024)
- Features: oil avg/Dec, overnight rate avg/Dec, 5yr bond yield, CAD/USD, nat gas, Alberta ETF, YoY change
- 2026 base scenario (oil ~$65): **+7.9%** | oil spike scenario (oil ~$88): **+10.1%**
- Predicted 2026 avg assessed value: **$753k–$768k**

### Monthly XGBoost (44 CREB months, 2019–2026)
- Features: 1/2/3 month lags + rolling 3/6 month averages for all indicators
- Next benchmark prediction: **$565,683** (+0.92% MoM)

### Daily Housing Pressure Score (946 labeled trading days)
- Features: rolling 7d/30d/90d windows, momentum, all daily indicators
- Target: predicts next month's CREB MoM % change
- Today's score: **+1.66%** at $100 WTI

---

## Key Finding: Oil Shocks vs Oil Prosperity

| Scenario | Oil avg | Following year housing |
|----------|---------|----------------------|
| Sustained low (<$50/yr) | — | +4.3% avg |
| Sustained high (>$80/yr) | — | +2.0% avg |
| Violent spike (2009 crash) | $95→$35 | -12.5% |
| 2014–2016 crash | $100→$44 | -5% |

**Alberta's economy benefits from sustained high prices, not shock volatility.** The daily model treats $100 oil reached in 90 days as an uncertainty signal, not a prosperity signal.

---

## MiroFish Simulation (March 15, 2026)

The market report was automatically fed into MiroFish as a seed document. 8 agents simulated on a Twitter-like platform for 40 rounds:

- **Bank of Canada** — rate policy response to oil shock + inflation
- **Alberta Energy Sector** — employment, wages, interprovincial migration
- **Calgary Home Buyers** — first-time, move-up, investor segments
- **Federal Government** — immigration targets, housing supply policy
- **OPEC+** — production response to US shale and Iran disruption
- **US Shale** — production response to $100 oil
- **Iran/IRGC** — Strait of Hormuz escalation decisions
- **Global Media** — CBC Calgary, OilPrice.com, Reuters

See `simulations/2026-03-15-calgary-housing/` for the full prediction report.

---

## Project Structure

```
Calgary-Housing/
├── src/
│   ├── scrape_calgary_assessments.py  ← City of Calgary Socrata API
│   ├── scrape_creb.py                 ← CREB monthly PDFs (current)
│   ├── scrape_creb_historical.py      ← Wayback Machine (2019-2024)
│   ├── scrape_oil.py                  ← WTI via yfinance (daily)
│   ├── scrape_interest_rates.py       ← Bank of Canada overnight rate
│   ├── scrape_economic.py             ← CAD/USD, nat gas, Alberta ETF
│   ├── scrape_daily_indicators.py     ← all daily data combined
│   ├── merge_datasets.py              ← combine all sources
│   ├── model_xgboost.py               ← annual model
│   ├── model_monthly.py               ← monthly CREB model
│   ├── model_daily.py                 ← daily pressure score
│   ├── scrape_news.py                 ← RSS news scraper
│   ├── generate_report.py             ← market intelligence report
│   └── run_mirofish_pipeline.py       ← MiroFish API automation
├── data/
│   ├── raw/                           ← all scraped source data
│   └── processed/                     ← merged feature datasets
├── reports/                           ← dated market intelligence reports
├── simulations/                       ← MiroFish prediction reports
├── notebooks/                         ← EDA charts and analysis
├── dashboard.py                       ← Streamlit live dashboard
└── .streamlit/config.toml             ← dark theme config
```

---

## Running the Full Pipeline

```bash
# 1. Scrape all data
python src/scrape_calgary_assessments.py
python src/scrape_creb.py
python src/scrape_creb_historical.py   # Wayback Machine, takes ~10 min
python src/scrape_oil.py
python src/scrape_interest_rates.py
python src/scrape_economic.py
python src/scrape_daily_indicators.py
python src/merge_datasets.py

# 2. Train models
python src/model_xgboost.py
python src/model_monthly.py
python src/model_daily.py

# 3. Generate intelligence report
python src/scrape_news.py
python src/generate_report.py

# 4. Run MiroFish simulation (requires MiroFish running on localhost:5001)
python src/run_mirofish_pipeline.py

# 5. Launch dashboard
streamlit run dashboard.py
```

---

## Tech Stack

`Python` · `XGBoost` · `pdfplumber` · `yfinance` · `Pandas` · `Streamlit` · `Plotly` · `Bank of Canada API` · `MiroFish` · `Zep GraphRAG` · `OASIS` · `qwen-plus` · `Wayback Machine CDX API`
