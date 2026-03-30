"""
Scrape recent news articles relevant to Calgary housing and oil markets.
Sources: CBC Calgary, CREB Newsroom, OilPrice.com, Reuters Energy RSS feeds.
Outputs a clean markdown file of headlines + summaries for MiroFish seed input.
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import re
import os

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
LOOKBACK_DAYS = 14
OUTPUT_FILE = "data/raw/news_feed.md"
OUTPUT_CSV  = "data/raw/news_feed.csv"


def fetch_rss(url, source_name, keywords=None):
    """Fetch and parse an RSS feed, filter by keywords."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        print(f"  [{source_name}] Failed: {e}")
        return []

    cutoff = datetime.now() - timedelta(days=LOOKBACK_DAYS)
    articles = []

    for item in root.iter("item"):
        title   = item.findtext("title", "").strip()
        link    = item.findtext("link", "").strip()
        summary = item.findtext("description", "").strip()
        pub_raw = item.findtext("pubDate", "")

        # Clean HTML from summary
        summary = re.sub(r"<[^>]+>", "", summary).strip()
        summary = summary[:400] + "..." if len(summary) > 400 else summary

        # Parse date
        try:
            pub_date = datetime.strptime(pub_raw[:25].strip(), "%a, %d %b %Y %H:%M:%S")
        except Exception:
            pub_date = datetime.now()

        if pub_date < cutoff:
            continue

        # Keyword filter
        if keywords:
            text = (title + " " + summary).lower()
            if not any(k.lower() in text for k in keywords):
                continue

        articles.append({
            "source": source_name,
            "date": pub_date.strftime("%Y-%m-%d"),
            "title": title,
            "summary": summary,
            "url": link,
        })

    print(f"  [{source_name}] {len(articles)} relevant articles")
    return articles


def scrape_all_news():
    print("Scraping news sources...")
    all_articles = []

    # --- CBC Calgary ---
    all_articles += fetch_rss(
        "https://www.cbc.ca/cmlink/rss-canada-calgary",
        "CBC Calgary",
        keywords=["housing", "real estate", "creb", "rent", "mortgage", "home price",
                  "oil", "energy", "interest rate", "inflation", "economy",
                  "power", "pipeline", "market", "price", "cost of living"]
    )

    # --- CREB Newsroom ---
    all_articles += fetch_rss(
        "https://www.creb.com/news/rss/",
        "CREB",
        keywords=None  # All CREB news is relevant
    )

    # --- OilPrice.com ---
    all_articles += fetch_rss(
        "https://oilprice.com/rss/main",
        "OilPrice.com",
        keywords=["WTI", "crude", "OPEC", "Iran", "Strait", "Hormuz",
                  "Alberta", "oil sands", "Canada", "price", "barrel"]
    )

    # --- Reuters Energy (via RSS) ---
    all_articles += fetch_rss(
        "https://feeds.reuters.com/reuters/businessNews",
        "Reuters Business",
        keywords=["oil", "crude", "OPEC", "Iran", "energy", "housing",
                  "Canada", "interest rate", "mortgage", "Alberta"]
    )

    # --- Better Dwelling ---
    all_articles += fetch_rss(
        "https://betterdwelling.com/feed/",
        "Better Dwelling",
        keywords=["Calgary", "Alberta", "housing", "real estate", "mortgage",
                  "price", "market", "canada"]
    )

    if not all_articles:
        print("No articles fetched — check network/RSS availability")
        return []

    # Deduplicate by title
    seen = set()
    unique = []
    for a in all_articles:
        if a["title"] not in seen:
            seen.add(a["title"])
            unique.append(a)

    # Sort by date descending
    unique.sort(key=lambda x: x["date"], reverse=True)

    # Save CSV
    df = pd.DataFrame(unique)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(unique)} articles to {OUTPUT_CSV}")

    return unique


def build_news_markdown(articles):
    """Format articles as clean markdown for MiroFish seed input."""
    now = datetime.now().strftime("%B %d, %Y")

    lines = [
        f"# Calgary Housing & Oil Market — News Brief",
        f"**Generated:** {now} | **Sources:** CBC Calgary, CREB, OilPrice.com, Reuters, Better Dwelling",
        f"**Coverage:** Last {LOOKBACK_DAYS} days",
        "",
        "---",
        "",
    ]

    # Group by source
    sources = {}
    for a in articles:
        sources.setdefault(a["source"], []).append(a)

    for source, arts in sources.items():
        lines.append(f"## {source}")
        for a in arts:
            lines.append(f"### {a['title']}")
            lines.append(f"*{a['date']}* | [Link]({a['url']})")
            if a["summary"]:
                lines.append(f"> {a['summary']}")
            lines.append("")
        lines.append("---")
        lines.append("")

    md = "\n".join(lines)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Saved news markdown to {OUTPUT_FILE}")
    return md


if __name__ == "__main__":
    articles = scrape_all_news()
    if articles:
        md = build_news_markdown(articles)
        print(f"\n--- PREVIEW (first 2000 chars) ---")
        print(md[:2000])
