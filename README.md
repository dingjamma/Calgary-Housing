# Calgary Housing Price Predictor

Predicts Calgary housing prices by combining traditional real estate signals with oil price data and MiroFish geopolitical risk simulation.

## Why Calgary?
Calgary's housing market is uniquely tied to oil — when oil spikes or crashes, housing follows. This project models that relationship and incorporates forward-looking geopolitical simulation to forecast price movements.

## Pipeline
1. **Data Scraping** — Fully automated, no manual downloads
2. **EDA** — Correlate oil prices, interest rates, and geopolitical events with housing
3. **Model** — XGBoost price predictor
4. **Geopolitical Layer** — MiroFish simulation output → oil price forecast → housing impact

## Data Sources (All Automated)

| Source | Data | Coverage |
|--------|------|----------|
| City of Calgary Open Data Portal | Annual avg residential assessed value | 2005–2025 |
| CREB monthly PDF scraper | Benchmark price by property type & district | 2025–2026 |
| Wayback Machine (CREB archive) | Historical CREB benchmark prices | 2019–2024 |
| Bank of Canada Valet API | Overnight rate, 5yr bond yield | 2005–2026 |
| yfinance | WTI oil, CAD/USD, nat gas, Alberta Energy ETF | 2005–2026 |

## Project Structure
```
Calgary-Housing/
├── data/
│   ├── raw/                          ← scraped source data
│   │   ├── calgary_assessments.csv   ← 21 years of assessed values
│   │   ├── creb_housing_prices.csv   ← monthly CREB benchmarks (2025-2026)
│   │   ├── oil_prices.csv            ← WTI monthly (2005-2026)
│   │   ├── interest_rates.csv        ← BoC overnight rate monthly
│   │   ├── bond_yields.csv           ← 5yr bond yield monthly
│   │   └── economic_indicators.csv   ← CAD/USD, nat gas, Alberta ETF
│   └── processed/
│       ├── annual_merged.csv         ← annual panel: prices + all indicators
│       └── creb_monthly.csv          ← monthly CREB + economic features
├── notebooks/                        ← EDA and modeling
└── src/                              ← scraping pipeline
    ├── scrape_calgary_assessments.py ← City of Calgary Open Data
    ├── scrape_creb.py                ← CREB monthly PDFs (current)
    ├── scrape_creb_historical.py     ← Wayback Machine (2019-2024)
    ├── scrape_oil.py                 ← WTI via yfinance
    ├── scrape_interest_rates.py      ← Bank of Canada overnight rate
    ├── scrape_economic.py            ← CAD/USD, nat gas, Alberta ETF, bond yields
    └── merge_datasets.py             ← combine all sources
```

## Key Insight: Oil-Housing Correlation
Calgary assessments dropped ~4% in 2009 when oil crashed from $95 to $66. They dropped another 12% in 2010 as oil remained depressed. When oil recovered to $97 in 2014, assessments hit their pre-crash highs. The 2015-2017 oil crash ($50 → $44) caused a 5% housing correction. The 2022-2024 oil/rate shock drove a 30%+ surge.

## Geopolitical Layer
MiroFish multi-agent simulation feeds geopolitical risk scenarios (e.g., Strait of Hormuz closure, OPEC+ cuts) → oil price forecast → model predicts housing price impact.
