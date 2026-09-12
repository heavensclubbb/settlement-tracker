"""
Pulls every settlement currently listed on Top Class Actions' "Open Settlements"
page, extracts structured fields with Claude, and keeps a running JSON file
(settlements.json) that the app reads from.

Run it on a schedule (e.g. once an hour) and it will:
  - add newly-listed settlements
  - re-check ones already in the file, and flag/update if the source page
    now shows the settlement as closed
  - never touch a record it can't confidently extract (low confidence -> "needs_review")

SETUP (one-time):
  1. pip install anthropic requests beautifulsoup4
  2. Get an API key at console.anthropic.com -> set it as an environment
     variable: export ANTHROPIC_API_KEY="sk-ant-..."
  3. python crawl_topclassactions.py
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from anthropic import Anthropic

LISTING_URL = "https://topclassactions.com/category/lawsuit-settlements/open-lawsuit-settlements/"
DATA_FILE = Path(__file__).parent / "settlements.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment

EXTRACTION_SCHEMA = """
Return ONLY a JSON object (no prose, no markdown fences) with this shape:

{
  "case_name": string,
  "administrator": string or null,
  "class_period": {"start": "YYYY-MM-DD" or null, "end": "YYYY-MM-DD" or null},
  "eligibility_criteria": string,          // transcribe the condition closely, do not broaden or paraphrase loosely
  "eligibility_source_quote": string,      // the exact sentence(s) this was pulled from
  "proof_required": "none" | "attestation" | "receipt" | "specific_doc",
  "proof_details": string or null,
  "payout_structure": "flat" | "range" | "pro_rata" | "unknown",
  "payout_amount": {"min": number or null, "max": number or null, "currency": "USD"},
  "claim_deadline": "YYYY-MM-DD" or null,
  "claim_form_url": string or null,
  "is_closed": boolean,                    // true if the page itself says the settlement is closed/expired
  "confidence": number                     // 0-1, your honest confidence in this extraction
}

Rules:
- If you cannot find explicit source text for a field, set it to null rather than guessing.
- confidence must be LOW (below 0.6) if eligibility language is ambiguous, state-specific,
  or the page format is unusual. Low confidence is expected and fine.
"""


def fetch(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.text


def get_open_settlement_links() -> list[str]:
    html = fetch(LISTING_URL)
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.select("a[href*='/lawsuit-settlements/open-lawsuit-settlements/']"):
        href = a.get("href", "")
        if re.search(r"/open-lawsuit-settlements/[^/]+/?$", href):
            links.add(href.split("?")[0])
    return sorted(links)


def page_text(url: str) -> str:
    html = fetch(url)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)[:12000]


def extract(url: str, text: str) -> dict:
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Here is the text of a class-action settlement notice page "
                    f"from {url}:\n\n{text}\n\n{EXTRACTION_SCHEMA}"
                ),
            }
        ],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


def load_existing() -> dict:
    if DATA_FILE.exists():
        return json.loads(DATA_FILE.read_text())
    return {}


def save(data: dict):
    DATA_FILE.write_text(json.dumps(data, indent=2, default=str))


def run():
    store = load_existing()
    links = get_open_settlement_links()
    print(f"Found {len(links)} settlements listed as open.")

    for url in links:
        try:
            text = page_text(url)
            fields = extract(url, text)
        except Exception as e:
            print(f"  FAILED  {url}  ({e})")
            continue

        fields["source_url"] = url
        fields["last_verified_at"] = datetime.now(timezone.utc).isoformat()
        fields["status"] = (
            "closed" if fields.get("is_closed") else
            "needs_review" if fields.get("confidence", 0) < 0.6 else
            "live"
        )

        prev = store.get(url)
        if prev and prev.get("status") != fields["status"]:
            print(f"  STATUS CHANGE  {fields.get('case_name')}: {prev.get('status')} -> {fields['status']}")

        store[url] = fields
        print(f"  ok  [{fields['status']:<12}] {fields.get('case_name', url)[:60]}")
        time.sleep(1)  # be polite to the source site

    save(store)
    print(f"\nSaved {len(store)} settlements to {DATA_FILE}")


if __name__ == "__main__":
    run()
