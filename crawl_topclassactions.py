"""
Pulls every settlement currently listed on Top Class Actions' "Open Settlements"
page, extracts structured fields with Claude, and keeps a running JSON file
(settlements.json) that the app reads from.

Requests are routed through ScraperAPI (scraperapi.com) because the source
site blocks requests coming directly from cloud/datacenter IP ranges,
including GitHub Actions runners.

To keep this well inside ScraperAPI's free monthly credit allowance, only the
listing page is fetched every run (cheap - 1 request). A settlement's own
detail page is only (re)fetched when it is new, or when it was last verified
more than STALE_HOURS ago. A settlement that disappears from the open-listing
page is marked closed without needing to fetch it again.

SETUP (one-time):
  1. pip install anthropic requests beautifulsoup4
  2. Get an API key at console.anthropic.com -> set it as an environment
     variable: export ANTHROPIC_API_KEY="sk-ant-..."
  3. Get a free API key at scraperapi.com -> set it as an environment
     variable: export SCRAPERAPI_KEY="..."
  4. python crawl_topclassactions.py
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
SCRAPERAPI_KEY = os.environ["SCRAPERAPI_KEY"]
STALE_HOURS = 20  # how long a "live" record is trusted before re-checking it

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
    """Fetch a URL through ScraperAPI so requests don't come from a blocked IP range."""
    r = requests.get(
        "http://api.scraperapi.com",
        params={"api_key": SCRAPERAPI_KEY, "url": url},
        timeout=60,
    )
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


def is_stale(record: dict) -> bool:
    try:
        last = datetime.fromisoformat(record["last_verified_at"])
    except (KeyError, ValueError):
        return True
    age_hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600
    return age_hours > STALE_HOURS


def run():
    store = load_existing()
    links = set(get_open_settlement_links())
    print(f"Found {len(links)} settlements listed as open.")

    checked = 0
    for url in links:
        prev = store.get(url)
        if prev and prev.get("status") in ("live", "needs_review") and not is_stale(prev):
            continue  # still fresh, skip the request entirely

        checked += 1
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

        if prev and prev.get("status") != fields["status"]:
            print(f"  STATUS CHANGE  {fields.get('case_name')}: {prev.get('status')} -> {fields['status']}")

        store[url] = fields
        print(f"  ok  [{fields['status']:<12}] {fields.get('case_name', url)[:60]}")
        time.sleep(1)  # be polite to the source site

    # anything we were tracking that no longer appears on the open-listing page
    # has been removed by the source, i.e. it is now closed - no fetch needed
    for url, record in store.items():
        if url not in links and record.get("status") != "closed":
            record["status"] = "closed"
            record["last_verified_at"] = datetime.now(timezone.utc).isoformat()
            print(f"  CLOSED (removed from listing)  {record.get('case_name', url)[:60]}")

    save(store)
    print(f"\nChecked {checked} settlement(s) this run. {len(store)} total on file.")


if __name__ == "__main__":
    run()
