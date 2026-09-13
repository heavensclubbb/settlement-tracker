"""
Pulls every settlement currently listed across multiple settlement-tracking
sites, extracts structured fields with Claude, and keeps a running JSON file
(settlements.json) that the app reads from.

Sources:
  - Top Class Actions: one open-settlement page per case, fetched and
    extracted individually (see crawl_topclassactions).
  - ClassAction.org: a single listing page already shows deadline, payout,
    proof requirement, and a short eligibility description for every open
    settlement, so the whole page is extracted in one batched call instead
    of visiting each case separately (see crawl_classaction_org).

Requests are routed through ScraperAPI (scraperapi.com) because at least one
of these sites blocks requests coming directly from cloud/datacenter IP
ranges, including GitHub Actions runners.

Efficiency, so this stays well inside ScraperAPI's and Claude's free/low-cost
tiers even running hourly:
  - Top Class Actions: only the listing page is fetched every run. A
    settlement's own detail page is only (re)fetched when it's new, or when
    it was last verified more than STALE_HOURS ago.
  - ClassAction.org: the listing page is fetched every run (cheap - 1
    request), but it's only sent to Claude for extraction when its content
    has actually changed since the last run.
  - Either way, a settlement that disappears from a source's open-listing
    page is marked closed without needing to fetch it again.

SETUP (one-time):
  1. pip install anthropic requests beautifulsoup4
  2. Get an API key at console.anthropic.com -> set it as an environment
     variable: export ANTHROPIC_API_KEY="sk-ant-..."
  3. Get a free API key at scraperapi.com -> set it as an environment
     variable: export SCRAPERAPI_KEY="..."
  4. python crawl_settlements.py
"""

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from anthropic import Anthropic

DATA_FILE = Path(__file__).parent / "settlements.json"
STATE_FILE = Path(__file__).parent / "crawl_state.json"
SCRAPERAPI_KEY = os.environ["SCRAPERAPI_KEY"]
STALE_HOURS = 20  # how long a "live" record is trusted before re-checking it

client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment

SINGLE_SCHEMA = """
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

BATCH_SCHEMA = """
This page lists MANY settlements at once. Return ONLY a JSON array (no prose,
no markdown fences), one object per settlement card, each with this shape:

{
  "case_name": string,
  "administrator": null,                   // not shown on this page, always null
  "class_period": {"start": "YYYY-MM-DD" or null, "end": "YYYY-MM-DD" or null},
  "eligibility_criteria": string,          // the card's own description, transcribed closely
  "eligibility_source_quote": string,      // the exact sentence(s) this was pulled from
  "proof_required": "none" | "attestation" | "receipt" | "specific_doc",
  "proof_details": string or null,
  "payout_structure": "flat" | "range" | "pro_rata" | "unknown",
  "payout_amount": {"min": number or null, "max": number or null, "currency": "USD"},
  "claim_deadline": "YYYY-MM-DD" or null,  // convert relative/short dates using this page's fetch date
  "claim_form_url": string,                // the "Visit Official Settlement Website" link - REQUIRED, skip any card without one
  "is_closed": false,
  "confidence": number
}

Rules:
- Every card on an "open settlements" page is open; only set is_closed true if a card explicitly says otherwise.
- If a field isn't shown for a card, set it to null rather than guessing.
- confidence must be LOW (below 0.6) if a card's information is ambiguous or incomplete.
- Skip any card that has no official settlement website link.
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


def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2, default=str))


def is_stale(record: dict) -> bool:
    try:
        last = datetime.fromisoformat(record["last_verified_at"])
    except (KeyError, ValueError):
        return True
    age_hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600
    return age_hours > STALE_HOURS


def status_for(fields: dict) -> str:
    if fields.get("is_closed"):
        return "closed"
    if fields.get("confidence", 0) < 0.6:
        return "needs_review"
    return "live"


# ---------------------------------------------------------------------------
# Source 1: Top Class Actions - one page per settlement
# ---------------------------------------------------------------------------
TCA_LISTING_URL = "https://topclassactions.com/category/lawsuit-settlements/open-lawsuit-settlements/"


def tca_open_links() -> list[str]:
    html = fetch(TCA_LISTING_URL)
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.select("a[href*='/lawsuit-settlements/open-lawsuit-settlements/']"):
        href = a.get("href", "")
        if re.search(r"/open-lawsuit-settlements/[^/]+/?$", href):
            links.add(href.split("?")[0])
    return sorted(links)


def tca_page_text(url: str) -> str:
    html = fetch(url)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)[:12000]


def extract_single(url: str, text: str) -> dict:
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Here is the text of a class-action settlement notice page "
                    f"from {url}:\n\n{text}\n\n{SINGLE_SCHEMA}"
                ),
            }
        ],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


def crawl_topclassactions(store: dict) -> int:
    links = set(tca_open_links())
    print(f"[topclassactions] {len(links)} settlements listed as open.")

    checked = 0
    for url in links:
        prev = store.get(url)
        if prev and prev.get("status") in ("live", "needs_review") and not is_stale(prev):
            continue

        checked += 1
        try:
            text = tca_page_text(url)
            fields = extract_single(url, text)
        except Exception as e:
            print(f"  FAILED  {url}  ({e})")
            continue

        fields["source_url"] = url
        fields["source"] = "topclassactions"
        fields["last_verified_at"] = datetime.now(timezone.utc).isoformat()
        fields["status"] = status_for(fields)

        prev = store.get(url)
        if prev and prev.get("status") != fields["status"]:
            print(f"  STATUS CHANGE  {fields.get('case_name')}: {prev.get('status')} -> {fields['status']}")

        store[url] = fields
        print(f"  ok  [{fields['status']:<12}] {fields.get('case_name', url)[:60]}")
        time.sleep(1)  # be polite to the source site

    now = datetime.now(timezone.utc).isoformat()
    for url, record in store.items():
        if record.get("source") == "topclassactions" and url not in links and record.get("status") != "closed":
            record["status"] = "closed"
            record["last_verified_at"] = now
            print(f"  CLOSED (removed from listing)  {record.get('case_name', url)[:60]}")

    return checked


# ---------------------------------------------------------------------------
# Source 2: ClassAction.org - one listing page holds every open settlement
# ---------------------------------------------------------------------------
CAO_LISTING_URL = "https://www.classaction.org/settlements"


def cao_listing_text() -> str:
    html = fetch(CAO_LISTING_URL)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    # keep link destinations inline as plain text so the extraction step can
    # see each card's official settlement URL without needing to know the
    # page's exact HTML structure
    for a in soup.find_all("a", href=True):
        a.replace_with(f"{a.get_text(strip=True)} [{a['href']}]")
    return soup.get_text(separator="\n", strip=True)


def extract_batch(url: str, text: str) -> list[dict]:
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Today's date is {datetime.now(timezone.utc).date().isoformat()}. "
                    f"Here is the text of a page listing many open class-action "
                    f"settlements, from {url}:\n\n{text}\n\n{BATCH_SCHEMA}"
                ),
            }
        ],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


def crawl_classaction_org(store: dict, state: dict) -> int:
    text = cao_listing_text()
    content_hash = hashlib.sha256(text.encode()).hexdigest()

    if state.get("classaction_org_hash") == content_hash:
        print("[classaction_org] Listing unchanged since last run, skipping re-extraction.")
        return 0

    # the page is sorted "ending soon" by default; capping here keeps each
    # extraction call a predictable size and favors the most time-sensitive
    # settlements
    text = text[:16000]

    try:
        items = extract_batch(CAO_LISTING_URL, text)
    except Exception as e:
        print(f"[classaction_org] FAILED to extract listing ({e})")
        return 0

    print(f"[classaction_org] {len(items)} settlements extracted from listing.")

    now = datetime.now(timezone.utc).isoformat()
    current_urls = set()
    for fields in items:
        url = fields.get("claim_form_url")
        if not url:
            continue
        current_urls.add(url)

        fields["source_url"] = CAO_LISTING_URL
        fields["source"] = "classaction_org"
        fields["last_verified_at"] = now
        fields["status"] = status_for(fields)

        prev = store.get(url)
        if prev and prev.get("status") != fields["status"]:
            print(f"  STATUS CHANGE  {fields.get('case_name')}: {prev.get('status')} -> {fields['status']}")

        store[url] = fields

    for url, record in store.items():
        if record.get("source") == "classaction_org" and url not in current_urls and record.get("status") != "closed":
            record["status"] = "closed"
            record["last_verified_at"] = now
            print(f"  CLOSED (removed from listing)  {record.get('case_name', url)[:60]}")

    state["classaction_org_hash"] = content_hash
    return len(items)


# ---------------------------------------------------------------------------
def run():
    store = load_json(DATA_FILE)
    state = load_json(STATE_FILE)

    tca_checked = crawl_topclassactions(store)
    cao_checked = crawl_classaction_org(store, state)

    save_json(DATA_FILE, store)
    save_json(STATE_FILE, state)
    print(
        f"\nChecked {tca_checked} (Top Class Actions) + {cao_checked} (ClassAction.org) "
        f"this run. {len(store)} total settlements on file."
    )


if __name__ == "__main__":
    run()
