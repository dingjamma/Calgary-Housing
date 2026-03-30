"""
Translate the latest MiroFish simulation report from Chinese to English.
Uses the Qwen API (same key as MiroFish).
Output: report_en.md alongside report.md in the simulation directory.
"""

import os
import sys
import glob
from openai import OpenAI
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY  = os.environ.get("QWEN_API_KEY") or os.environ.get("LLM_API_KEY")
BASE_URL = os.environ.get("LLM_BASE_URL",
           "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
MODEL    = os.environ.get("LLM_MODEL_NAME", "qwen-plus")

_repo_dir     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_mirofish_dir = os.environ.get(
    "MIROFISH_DIR",
    os.path.join(_repo_dir, "..", "MiroFish"),
)
SIM_BASE = os.path.join(_mirofish_dir, "simulations")


def find_latest_report():
    """Find the most recently created report.md in any simulation subdir."""
    pattern = os.path.join(SIM_BASE, "*", "report.md")
    reports = glob.glob(pattern)
    if not reports:
        raise FileNotFoundError(f"No report.md found under {SIM_BASE}")
    # Sort by directory name (date-prefixed) descending
    reports.sort(reverse=True)
    return reports[0]


def translate_report(report_path: str) -> str:
    """Translate report.md → report_en.md using Qwen."""
    if not API_KEY:
        raise ValueError("QWEN_API_KEY / LLM_API_KEY not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    with open(report_path, encoding="utf-8") as f:
        content = f.read()

    # Split off the HTML comment header (metadata block) — don't translate that
    header, body = "", content
    if content.startswith("<!--"):
        end = content.find("-->")
        if end != -1:
            header = content[: end + 3]
            body   = content[end + 3 :].lstrip("\n")

    print(f"Translating {len(body):,} chars from {report_path}...")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional financial and geopolitical translator. "
                    "Translate the following Markdown report from Chinese to English. "
                    "Preserve all Markdown formatting exactly (headings, bold, blockquotes, "
                    "bullet points, tables). Translate naturally — do not transliterate. "
                    "Keep all numbers, dates, and proper nouns (Calgary, WTI, CREB, BoC, OPEC, "
                    "IRGC, Alberta) unchanged. Return only the translated Markdown, no commentary."
                ),
            },
            {"role": "user", "content": body},
        ],
        temperature=0.2,
    )

    translated = response.choices[0].message.content.strip()

    translation_note = (
        f"\n\n---\n*Translated from Chinese by qwen-plus · "
        f"{datetime.now().strftime('%Y-%m-%d')}*\n"
    )

    return header + "\n\n" + translated + translation_note


def main():
    report_path = find_latest_report()
    print(f"Source: {report_path}")

    translated = translate_report(report_path)

    output_path = os.path.join(os.path.dirname(report_path), "report_en.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(translated)

    print(f"Saved: {output_path}")
    print(f"Length: {len(translated):,} chars")
    print()
    print("--- PREVIEW ---")
    print(translated[:1500])


if __name__ == "__main__":
    main()
